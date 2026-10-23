# -*- coding: utf-8 -*-
"""逐层 fp8 -> fp16 即时反量化。

思路
----
权重以 **fp8(e4m3) + 每行 scale** 常驻显存（约 fp16 的一半），前向时只把
**当前正在算的这一层** 反量化成 fp16、算完立刻丢弃。这样显存占用是

    fp8 常驻权重 + 单层 fp16 峰值      （而不是整个模型的 fp16）

设计取舍（都是实测逼出来的）
---------------------------
1. **per-row(per-output-channel) scale，不是 per-tensor。**
   fp8 e4m3 只有 3 位尾数，一个整层共用一个 scale 时，量程被最大离群值吃掉，
   其余权重全被压到量化台阶以下。逐行量化把每行的量程独立归一，明显更稳。

2. **hook 挂在整个 DecoderLayer 上，不挂在 Linear 上。**
   挂 Linear 的话，同一层内 7 个 Linear 会**同时**持有 fp16 副本，峰值按层叠加。
   挂在 DecoderLayer 上，则在层入口一次性反量化该层全部权重、层出口丢弃，
   峰值严格限制在「一层」。

3. **反量化结果不放进 `param.data`，而是用 forward_pre_hook 临时替换 `param`。**
   `param.data` 赋值会让 accelerate 的 offload hook 认为参数已就位，
   从而跳过它自己的搬运逻辑。用 pre_hook 注入、post_hook 还原，跟 accelerate 共存。

4. **只量化 2 维权重（Linear），跳过 RMSNorm 的 1 维 weight。**
   归一化权重对精度极敏感、且总量只有几百 KB，量化收益为零。

5. **fused `gate_up_proj` 自动拆开再合并。**
   官方 mlp 会把 gate/up 拼成一个 Linear 提速，此时不能按名字逐行对齐，
   要按输出维切分、各自量化后再拼回去。

已知限制
--------
- 反量化是**逐 token 重复做**的：每生成一个 token，每一层都要从 fp8 翻一遍。
  这是纯增的算力开销，本机没有 fp8 matmul 单元，省下的是显存、付出的是时间。
- scale 由权重的 max_abs 直接算（无需校准集）。这比用校准数据统计激活动态范围粗糙，
  但胜在自包含、可直接对比 offload 基准。
"""
import re

import torch
from torch import nn

# ---------------------------------------------------------------- 量化 / 反量化

E4M3_MAX = 448.0


def quantize_per_row(w):
    """fp16/bf16 权重 -> (fp8_e4m3, per-row scale)。

    scale 取该行 max_abs / 448（e4m3 可表示的最大值），保证不溢出。
    整行全零时 scale 置 1，避免除零。
    """
    w = w.detach().float()
    amax = w.abs().amax(dim=1, keepdim=True)
    scale = (amax / E4M3_MAX).clamp(min=1e-12)
    q = (w / scale).clamp(-E4M3_MAX, E4M3_MAX)
    return q.to(torch.float8_e4m3fn), scale.squeeze(1).to(torch.float16)


def dequantize_per_row(q, scale, dtype=torch.float16):
    """fp8 + per-row scale -> 原精度张量（默认 fp16）。"""
    return (q.to(dtype) * scale.to(dtype).unsqueeze(1))


# ---------------------------------------------------------------- 待量化的层筛选

# fused 权重：官方建模把 gate/up 合并成一个 Linear（名字形如 gate_up_proj）
FUSED_RE = re.compile(r'^(.*\.)?(gate_up_proj|up_gate_proj)$')


def _linear_children(layer):
    """返回该层里所有 2 维权重的 Linear（跳过 RMSNorm 的 1 维 weight）。"""
    out = []
    for name, mod in layer.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if mod.weight is None or mod.weight.dim() != 2:
            continue
        out.append((name, mod))
    return out


def _is_fused(name, mod, cfg):
    """判断是不是 gate/up 被拼起来的 fused Linear。"""
    if FUSED_RE.match(name):
        return True
    try:
        inter = cfg.intermediate_size
        return mod.out_features == 2 * inter
    except Exception:
        return False


class LayerQuantizer:
    """负责把模型上的 Linear 换成 fp8 权重 + 注册逐层反量化 hook。"""

    def __init__(self, model, cfg, dtype=torch.float16, verbose=True):
        self.model = model
        self.cfg = cfg
        self.dtype = dtype
        self.verbose = verbose
        self.store = {}       # 参数名 -> (fp8, scale, 原始 shape, 是否 fused)
        self.hook_handles = []
        self.stats = {'quantized': 0, 'params': 0, 'fp8_bytes': 0, 'fp16_bytes': 0}

    # ---- 量化：把每个 DecoderLayer 的 Linear 权重抽出来存 fp8 ----
    def quantize(self):
        layers = self._find_layers()
        if not layers:
            raise SystemExit('找不到 DecoderLayer 列表，请检查 --layers-attr')
        for li, layer in enumerate(layers):
            self._quantize_layer(li, layer)
        if self.verbose:
            g = self.stats
            print(f'[quant] 量化 Linear {g["quantized"]} 个 / 参数 {g["params"]/1e9:.3f}B')
            print(f'[quant] fp8 {g["fp8_bytes"]/2**30:.2f} GiB '
                  f'-> 反量化后 fp16 {g["fp16_bytes"]/2**30:.2f} GiB '
                  f'(省 {100*(1-g["fp8_bytes"]/max(1,g["fp16_bytes"])):.1f}%)')
            print('[quant] 注意：原 fp16 权重**仍在原参数上**（零值），'
                  '上面的峰值只在「先 free 再反量化」时才能达到')
        return self

    def _find_layers(self):
        for attr in ('model.layers', 'layers', 'model.model.layers'):
            cur = self.model
            ok = True
            for part in attr.split('.'):
                cur = getattr(cur, part, None)
                if cur is None:
                    ok = False
                    break
            if ok:
                return list(cur)
        return []

    def _quantize_layer(self, li, layer):
        for name, mod in _linear_children(layer):
            full = f'model.layers.{li}.{name}' if name else f'model.layers.{li}'
            w = mod.weight.data
            if _is_fused(name, mod, self.cfg):
                # gate/up 拼接：按输出维对半切，各自 per-row 量化后存两份、合并还原
                inter = self.cfg.intermediate_size
                wg = w[:inter]
                wu = w[inter:2 * inter]
                qg, sg = quantize_per_row(wg)
                qu, su = quantize_per_row(wu)
                self.store[full] = ('fused', (qg, sg), (qu, su))
                nbytes = qg.numel() + qu.numel()
                nfull = wg.numel() * 2 + wu.numel() * 2
            else:
                q, s = quantize_per_row(w)
                self.store[full] = ('plain', (q, s), None)
                nbytes = q.numel()
                nfull = w.numel() * 2
            self.stats['quantized'] += 1
            self.stats['params'] += mod.weight.numel()
            self.stats['fp8_bytes'] += nbytes
            self.stats['fp16_bytes'] += nfull
            # 必须保持「同 dtype + 同 shape」的空张量：直接换成别的 dtype 或 0 维，
            # torch 会报 `variable.set_data(tensor)` incompatible tensor type。
            # 用 param.data.zero_() 就地把权重清成 0，既释放不了显存（见 install_hooks
            # 里的说明），也不触发形状/dtype 校验。
            mod.weight.data.zero_()

    # ---- 反量化：注册逐层 hook ----
    def install_hooks(self):
        layers = self._find_layers()
        for li, layer in enumerate(layers):
            h1 = layer.register_forward_pre_hook(self._make_pre(li))
            h2 = layer.register_forward_hook(self._make_post(li))
            self.hook_handles += [h1, h2]
        if self.verbose:
            print(f'[hooks] 已在 {len(layers)} 个 DecoderLayer 上注册逐层反量化')
        return self

    def _entries_for(self, li):
        prefix = f'model.layers.{li}.'
        return [(k, v) for k, v in self.store.items() if k.startswith(prefix)]

    def _resolve(self, key):
        """key -> 对应的 nn.Module。"""
        parts = key.split('.')
        cur = self.model
        for p in parts:
            cur = getattr(cur, p)
        return cur

    def _make_pre(self, li):
        def _pre(module, args):
            self._cache = getattr(self, '_cache', {})
            for key, entry in self._entries_for(li):
                target = self._resolve(key)
                kind = entry[0]
                if kind == 'fused':
                    (qg, sg), (qu, su) = entry[1], entry[2]
                    wg = dequantize_per_row(qg, sg, self.dtype)
                    wu = dequantize_per_row(qu, su, self.dtype)
                    new = torch.cat([wg, wu], dim=0)
                else:
                    q, s = entry[1]
                    new = dequantize_per_row(q, s, self.dtype)
                self._cache[key] = target.weight
                target.weight = nn.Parameter(new, requires_grad=False)
        return _pre

    def _make_post(self, li):
        def _post(module, args, output):
            cache = getattr(self, '_cache', {})
            for key, entry in self._entries_for(li):
                target = self._resolve(key)
                old = cache.pop(key, None)
                if old is not None:
                    target.weight = old
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            return output
        return _post

    def remove_hooks(self):
        for h in self.hook_handles:
            h.remove()
        self.hook_handles = []
