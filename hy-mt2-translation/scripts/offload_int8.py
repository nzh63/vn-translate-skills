# -*- coding: utf-8 -*-
"""量化权重分层常驻 + 逐层搬运管线（hy-mt2-translation 版；全程 WSL2 运行）。

实测（同机贪心受控 limit2/32，2026-09-23 终测；★ 一律报 median of ≥3，
完整记录见 references/perf-log.md）
------------------------------------------------------------------------
  int4-g128-fused（默认，32/32 全驻零搬运） :  257 ms/forward（5 次中位 133~308）← 推荐
  int8-fused（--quant int8 回退）           :  496~508 ms/forward
  int4 no-fused（无 triton 对照，未复测）    :  历史 int8 no-fused ~884 ms

批量入口 `translate_tsv.py` 走**同一套量化管线**（带断点续跑/术语/失败重译）。

量化：默认 int4 group128 非对称（uint32 直序打包 + 每组 scale/zp，0.523 B/param，
相对误差 ~0.10，磁盘缓存 3.40 GiB）；--quant int8 回退 per-row 对称
（0.0102，1 B/param，6.50 GiB）。字节减半 → 全层显存常驻 → 消灭 PCIe 层搬运。

架构
----
  - meta device 建结构，量化字节磁盘缓存 mmap 零拷贝（缓存在 ext4：
    ~/.cache/hy-mt2-quant，HY_MT2_CACHE_DIR 可覆盖）；
  - GPU 不留完整权重：装得下的层量化常驻（int4 ~109 MiB/层、int8 ~208 MiB/层），
    其余层每 token 从锁页内存异步搬进 GPU 槽——**每层一条单块，单次 enqueue**；
    int4 reserve1000 时 32/32 全驻、每 token 搬运 0；
  - 前向两路：fused=寄存器内反量化 + triton W4A16/W8A16 GEMV（推荐；int4 核
    按组约减，zp/scale 每组一次）；no-fused=反量化挂载 + rocBLAS（回退）；
  - tie 权重（embed↔lm_head）合并上传，未量化常驻 0.98 GiB。

入口与坑
--------
  - 入口是 int8_pipe.py：**必须先 `torch.cuda.init()` 再 import 本模块**
    （triton 导入链早于 CUDA 初始化会让后续大分配 hipErrorInvalidValue）；
  - `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` 在当前驱动下任何设备分配
    都会炸，已停用（勿恢复）；
  - triton 进程初始化 ~2-9.7s 已在 load 内预热（warmup 块），删掉会把 ★ 抬高；
  - venv/HF/量化/triton 缓存都在 ext4（/mnt/c 是 9p，顺序读仅 ~170-230 MB/s，
    冷启动会慢一个量级）；权重加载 local_files_only + HF_HUB_OFFLINE=1；
  - 排障：verify_cache / verify_prefetch 逐位校验

用法
----
    python int8_pipe.py --greedy --limit 2 --max-new-tokens 32   # 默认 int4
    python int8_pipe.py --quant int8 ...                         # int8 回退
    python int8_pipe.py --cache-info / --rebuild-cache / --dry-run
"""
import argparse
import gc
import os
import sys
import time

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
# 2026-09-22 环境回归：expandable_segments 在当前驱动 + WSL/ROCDXG 组合
# 组合下，任何 device 分配都直接 hipErrorInvalidValue（最小复现：
#   PYTORCH_HIP_ALLOC_CONF=expandable_segments:True python -c
#   "import torch; torch.cuda.init(); torch.empty(8, device='cuda')"
# 同机不设该变量、或设 max_split_size_mb 均正常）。
# 故不再默认开启；如驱动修复可自行 export 试验。
# os.environ.setdefault('PYTORCH_HIP_ALLOC_CONF', 'expandable_segments:True')

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)   # mt_common 等同目录模块（本副本位于 skill scripts/）

import torch  # noqa: E402
from torch import nn  # noqa: E402
import int8_cache  # noqa: E402
from int8_quant import (  # noqa: E402
    quantize_per_row_int8 as quantize_per_row,
    dequantize_per_row_int8 as dequantize_per_row,
    quantize_int4_g128 as quantize_int4,
    dequantize_int4_g128 as dequantize_int4,
)
# fp8_layerwise.py 保留作为公共工具（_linear_children/_is_fused/dequant）
from fp8_layerwise import _linear_children, _is_fused  # noqa: E402

# ---- triton 融合 W8A16/W4A16 GEMV/GEMM（RDNA2 可用，见 int8_gemm_triton.py）----
HAVE_FUSED = False
try:
    from int8_gemm_triton import int8_gemm_rows, int4_gemm_rows  # noqa: E402
    HAVE_FUSED = True
except Exception:
    pass


def _probe(tag):
    """1.05GB 显存分配探针：首个失败点即问题点（失败后 allocator 会坏）。"""
    try:
        g = torch.empty(128167, 4096, device='cuda', dtype=torch.float16)
        torch.cuda.synchronize()
        del g
        print(f'[probe:{tag}] OK', flush=True)
    except Exception as ex:
        print(f'[probe:{tag}] FAIL: {str(ex)[:100]}', flush=True)
        raise SystemExit(2)

# 非驻留层在前向那一刻，显存里要放下的东西（按 --prefetch-slots 计）：
#   - 预取槽缓冲（slots × 单层量化权重：int4 ~109 MiB/层、int8 ~208 MiB/层）
#   - fp16 挂载路径还要加反量化出的单层 fp16
# reserve 不覆盖它就会越界（load_meta_fp8 里按 transient 动态校验）。
PREFETCH_SLOTS = 2


# ================================================================ 权重量化仓库

class Int8Store:
    """把模型的 Linear 权重量化（int4 g128 或 int8 per-row），常驻在指定 device。

    quant='int4'（默认主路径）：uint32 直序打包 + 每组 scale/zp，0.523 B/param；
    quant='int8'：per-row 对称（回退选项），1 B/param。

    不依赖 accelerate。权重的读取分两条路：
      - 已经在 CPU/GPU 上的参数：直接读
      - meta 参数（低内存加载的占位）：从 safetensors 分片里按需读
    """

    def __init__(self, model, cfg, dtype=torch.float16, device='cpu', verbose=True,
                 pin=False, quant='int8'):
        self.model = model
        self.cfg = cfg
        self.dtype = dtype
        self.device = device
        self.pin = pin
        self.verbose = verbose
        self.quant = quant if quant in ('int4', 'int8') else 'int8'
        self.weights = {}     # 参数全名 -> {'q': tensor, 's': tensor, 'z':?, 'shape': ...}
        self.stats = dict(n=0, params=0, fp8_bytes=0, fp16_bytes=0, skipped=0)
        self._big = None      # 单一连续量化字节 buffer（packed/int8）
        self._big_off = 0

    def _put(self, t):
        """放到常驻设备；pinned 时用锁页内存，让 H2D 走真正的 DMA。"""
        t = t.to(self.device)
        if self.pin and self.device == 'cpu':
            t = t.pin_memory()
        return t

    # ---- 单一连续 buffer：避免 224 个碎片，也避免 cat 时二次分配 ----

    def _plan(self):
        """预扫描：算出大 buffer 的总字节数（只装 packed/int8，scale/zp 另存）。"""
        total = 0
        for layer in self._layers():
            for name, mod in _linear_children(layer):
                w = self._weight_shape(mod, name)
                if w is None:
                    continue
                n = w[0] * w[1]
                total += (n // 2) if self.quant == 'int4' else n  # int4 半字节/参
        return total

    def _weight_shape(self, mod, name):
        """拿到 Linear 权重的 (out, in) 形状（meta 上也能读）。"""
        w = getattr(mod, 'weight', None)
        if w is None or w.dim() != 2:
            return None
        return tuple(w.shape)

    # ---- 找到所有 DecoderLayer ----
    def _layers(self):
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
        raise SystemExit('找不到 DecoderLayer')

    def build(self, shard_reader=None):
        """遍历所有层，量化每个 Linear。

        shard_reader: callable(full_name) -> tensor 或 None。
          给 meta 参数用，从 safetensors 取真身。

        内存策略：量化字节写进**一块预分配的大 buffer**，而不是每个 Linear
        各存一个小张量。本机物理内存仅 13.69 GiB，224 个碎片的分配器开销
        和 cat 时的二次分配会直接打穿物理内存（实测 0xC0000005）。
        """
        # 一次性分配量化大 buffer（+5% 余量，防止形状统计与量化后不一致）
        total = int(self._plan() * 1.05) + (1 << 20)
        print(f'[store] 预分配 {self.quant} buffer {total/2**30:.2f} GiB', flush=True)
        self._big = torch.empty(total, dtype=torch.uint8, device=self.device)
        self._big_off = 0

        for li, layer in enumerate(self._layers()):
            for name, mod in _linear_children(layer):
                full = f'model.layers.{li}.{name}' if name else f'model.layers.{li}'
                w = self._fetch(mod, full, shard_reader)
                if w is None:
                    self.stats['skipped'] += 1
                    continue
                self._quant(full, name, w, mod)
                del w
                _to_empty_shell(mod)
            gc.collect()
        self._big = self._big[:self._big_off]
        if self.verbose:
            g = self.stats
            print(f'[store] 量化 {g["n"]} 个 Linear / {g["params"]/1e9:.3f}B 参数')
            print(f'[store] {self.quant} {g["fp8_bytes"]/2**30:.2f} GiB（含 scale/zp）-> '
                  f'原精度 {g["fp16_bytes"]/2**30:.2f} GiB '
                  f'(压缩到 {100*g["fp8_bytes"]/max(1,g["fp16_bytes"]):.1f}%)')
            if g['skipped']:
                print(f'[store] 跳过 {g["skipped"]} 个读不到权重的 Linear')
        # 关掉分片句柄，让 mmap 页面缓存可被回收（关键：否则 RSS 停留在 ~9 GiB）
        if shard_reader is not None:
            close = getattr(shard_reader, 'close', None)
            if close:
                close()
        gc.collect()
        return self

    def shell_out(self):
        """把已量化的 Linear 参数换成 0 元素空壳。

        `build()` 里会顺手做掉；但**走磁盘缓存时不会执行 build**，
        模型上那些 Linear 仍是带完整 numel 的 meta 参数。若不换成空壳，
        `_materialize_small_params` 会把它们当成「未量化参数」，
        把整整 13 GiB fp16 权重搬上显卡 → 直接 OOM（实测）。
        """
        n = 0
        for li, layer in enumerate(self._layers()):
            for name, mod in _linear_children(layer):
                full = f'model.layers.{li}.{name}' if name else f'model.layers.{li}'
                if full in self.weights:
                    _to_empty_shell(mod)
                    n += 1
        if self.verbose:
            print(f'[store] 已将 {n} 个 Linear 置为空壳（权重只存在于 {self.quant} 副本）',
                  flush=True)
        return n

    def _slot(self, q):
        """把量化结果拷进大 buffer 的当前偏移处，返回视图（保持 q 的 dtype/shape）。

        按**字节**推进偏移：uint32 packed 的 numel≠字节数（须 ×4），
        旧实现按 numel 记账会在 int4 下溢出/错位。
        """
        qb = q.detach().contiguous().reshape(-1).view(torch.uint8)
        n = qb.numel()
        start = self._big_off
        if start + n > self._big.numel():
            raise RuntimeError(f'量化 buffer 溢出: {start+n} > {self._big.numel()}')
        self._big[start:start + n] = qb
        self._big_off += n
        # int4 入口偏移恒为 4 的倍数（uint32 行字节数 = M*K/2，K%8==0）
        return self._big[start:start + n].view(q.dtype).view(q.shape)

    def _fetch(self, mod, full, shard_reader):
        w = getattr(mod, 'weight', None)
        if w is not None and w.device.type != 'meta' and w.numel() > 0:
            return w.detach().float().cpu()
        # meta：走 accelerate 的 weights_map，或外部 shard_reader
        h = getattr(mod, '_hf_hook', None)
        if h is not None:
            wm = getattr(h, 'weights_map', None)
            if wm is not None:
                try:
                    return wm['weight'].detach().float().cpu()
                except Exception:
                    pass
        if shard_reader is not None:
            # safetensors 的键是全名 + '.weight'
            t = shard_reader(full + '.weight')
            if t is not None:
                return t.detach().float().cpu()
        return None

    def _quant(self, full, name, w, mod):
        i4 = self.quant == 'int4'
        if _is_fused(name, mod, self.cfg):
            inter = self.cfg.intermediate_size
            # 打包只在行内沿 K，[:inter] / [inter:2*inter] 的行切片依然安全
            if i4:
                qg, sg, zg = quantize_int4(w[:inter])
                qu, su, zu = quantize_int4(w[inter:2 * inter])
            else:
                qg, sg = quantize_per_row(w[:inter])
                qu, su = quantize_per_row(w[inter:2 * inter])
            # 量化字节进大 buffer；scale/zp 很小，单独存
            qg_v = self._slot(qg)
            qu_v = self._slot(qu)
            ent = dict(kind='fused',
                       qg=qg_v, sg=self._put(sg),
                       qu=qu_v, su=self._put(su),
                       shape=(2 * inter, w.shape[1]))
            nbytes = (qg.numel() * qg.element_size()
                      + qu.numel() * qu.element_size()
                      + (sg.numel() + su.numel()) * 2)
            if i4:
                ent['zg'] = self._put(zg)
                ent['zu'] = self._put(zu)
                nbytes += zg.numel() + zu.numel()
            self.weights[full] = ent
            self.stats['fp8_bytes'] += nbytes
            del qg, qu
        else:
            if i4:
                q, s, z = quantize_int4(w)
            else:
                q, s = quantize_per_row(w)
                z = None
            q_v = self._slot(q)
            ent = dict(kind='plain', q=q_v, s=self._put(s),
                       shape=tuple(w.shape))
            nbytes = q.numel() * q.element_size() + 2 * s.numel()
            if i4:
                ent['z'] = self._put(z)
                nbytes += z.numel()
            self.weights[full] = ent
            self.stats['fp8_bytes'] += nbytes
            del q
        self.stats['n'] += 1
        self.stats['params'] += w.numel()
        self.stats['fp16_bytes'] += w.numel() * 2

    # ---- 反量化：量化张量 + scale(/zp) -> 原精度 ----
    def _dq_one(self, q, s, z=None):
        if self.quant == 'int4':
            return dequantize_int4(q, s, z, self.dtype)
        return dequantize_per_row(q, s, self.dtype)

    def layer_dequant_from_buffers(self, li, bufs):
        """从预取到的 GPU 缓冲反量化（零 PCIe 等待）。"""
        out = {}
        for k in self._layer_keys(li):
            e = self.weights[k]
            b = bufs[k]
            if e['kind'] == 'fused':
                out[k] = torch.cat([self._dq_one(b['qg'], e['sg'], e.get('zg')),
                                    self._dq_one(b['qu'], e['su'], e.get('zu'))],
                                   dim=0)
            else:
                out[k] = self._dq_one(b['q'], e['s'], e.get('z'))
        return out

    def layer_int8_refs(self, li, bufs=None):
        """fused 前向用：返回 {k: (wq_gpu, s_gpu, z_gpu|None)}，不做任何反量化。

        int4 时 wq 是 uint32 直序 packed、z 是 zero-point（均已在 GPU）；
        int8 时 z 恒为 None。bufs 给定时 wq 取预取缓冲视图（零拷贝）；
        否则取 e['q']，不在 GPU 就地搬运（同步模式每次前向都会搬，仅供对照）。
        """
        out = {}
        for k in self._layer_keys(li):
            e = self.weights[k]
            if e['kind'] == 'fused':
                raise RuntimeError('融合前向不支持 fused 权重')
            if bufs is not None:
                wq = bufs[k]['q']
            else:
                wq = e['q']
                if wq.device.type != 'cuda':
                    wq = wq.to('cuda', non_blocking=True)
            s = e['s']
            if s is None:
                raise RuntimeError(f'{k} 缺少 scale')
            if s.device.type != 'cuda':
                s = s.to('cuda')
            z = e.get('z')
            if z is not None and z.device.type != 'cuda':
                z = z.to('cuda')
            out[k] = (wq, s, z)
        return out

    def release_host_for(self, resident_li, verbose=True):
        """常驻层的宿主 int8 副本已不再需要 —— `.to('cuda')` 已把它替换成 GPU 张量，
        原 mmap 里那一段没有任何引用指向它了，页缓存可由 OS 回收。

        这里只做统计（不置空 `e[f]`：常驻层的 `e[f]` 就是 GPU 上的那份权重本身）。
        """
        held = 0
        for key, e in self.weights.items():
            parts = key.split('.')
            if len(parts) < 3 or parts[0] != 'model' or parts[1] != 'layers':
                continue
            if int(parts[2]) in resident_li:
                for f in FP8_FIELDS:
                    t = e.get(f)
                    if t is not None:
                        held += t.numel() * t.element_size()
        if verbose and held:
            print(f'[store] 常驻层权重已在 GPU（{held/2**20:.0f} MiB），宿主侧无副本', flush=True)
        return held

    def drop_big(self, verbose=True):
        """断开整块 mmap 引用。

        **不能顺手把 `e[f]` 置空** —— 常驻层的 `e[f]` 正是 GPU 上的权重，
        置空会把权重弄丢。到这一步，指向 mmap 的视图要么已被 `.to('cuda')`
        替换（常驻层），要么已被 pinned 副本取代（非驻留层），所以直接放掉
        `_big` 即可，剩下的映射由引用计数自然回收。
        """
        self._big = None
        gc.collect()
        if verbose:
            print(f'[store] 已断开 mmap 引用（宿主匿名内存不再持有 {self.quant} 大块）',
                  flush=True)

    def to_gpu_fp16(self, full):
        e = self.weights[full]
        if e['kind'] == 'fused':
            qg = e['qg'].to('cuda', non_blocking=True)
            sg = e['sg'].to('cuda', non_blocking=True)
            qu = e['qu'].to('cuda', non_blocking=True)
            su = e['su'].to('cuda', non_blocking=True)
            zg = e.get('zg')
            zu = e.get('zu')
            if zg is not None:
                zg = zg.to('cuda', non_blocking=True)
                zu = zu.to('cuda', non_blocking=True)
            return torch.cat([self._dq_one(qg, sg, zg),
                              self._dq_one(qu, su, zu)], dim=0)
        q = e['q'].to('cuda', non_blocking=True)
        s = e['s'].to('cuda', non_blocking=True)
        z = e.get('z')
        if z is not None:
            z = z.to('cuda', non_blocking=True)
        return self._dq_one(q, s, z)

    def layer_dequant_resident(self, li):
        """量化权重已在 GPU 的层：就地反量化，零 PCIe。"""
        out = {}
        for k in self._layer_keys(li):
            e = self.weights[k]
            if e['kind'] == 'fused':
                out[k] = torch.cat([self._dq_one(e['qg'], e['sg'], e.get('zg')),
                                    self._dq_one(e['qu'], e['su'], e.get('zu'))],
                                   dim=0)
            else:
                out[k] = self._dq_one(e['q'], e['s'], e.get('z'))
        return out

    def layer_to_gpu_fp16(self, li):
        """把整层搬到 GPU 并反量化。

        **不使用 torch.cat 拼连续 buffer**：本机物理内存只有 13.69 GiB，
        量化后 RSS 已 ~10 GiB，再 cat 出一条几百 MiB 的连续 CPU 块会直接
        打穿物理内存（实测 0xC0000005）。改为逐 tensor `.to('cuda')`，
        每个量化权重在其自身 storage 上直接 DMA，峰值额外内存为 0。
        """
        out = {}
        st = self.stats
        for k in self._layer_keys(li):
            e = self.weights[k]
            t0 = time.perf_counter()
            if e['kind'] == 'fused':
                qg = e['qg'].to('cuda', non_blocking=True)
                sg = e['sg'].to('cuda', non_blocking=True)
                qu = e['qu'].to('cuda', non_blocking=True)
                su = e['su'].to('cuda', non_blocking=True)
                zg = e.get('zg')
                zu = e.get('zu')
                if zg is not None:
                    zg = zg.to('cuda', non_blocking=True)
                    zu = zu.to('cuda', non_blocking=True)
                t1 = time.perf_counter()
                out[k] = torch.cat([self._dq_one(qg, sg, zg),
                                    self._dq_one(qu, su, zu)], dim=0)
            else:
                q = e['q'].to('cuda', non_blocking=True)
                s = e['s'].to('cuda', non_blocking=True)
                z = e.get('z')
                if z is not None:
                    z = z.to('cuda', non_blocking=True)
                t1 = time.perf_counter()
                out[k] = self._dq_one(q, s, z)
            t2 = time.perf_counter()
            st['t_h2d'] = st.get('t_h2d', 0.0) + (t1 - t0)
            st['t_dequant'] = st.get('t_dequant', 0.0) + (t2 - t1)
        return out

    def _layer_keys(self, li):
        if not hasattr(self, '_layer_index'):
            self._layer_index = {}
            for k in self.weights:
                p = k.split('.')
                if len(p) >= 3 and p[0] == 'model' and p[1] == 'layers':
                    self._layer_index.setdefault(int(p[2]), []).append(k)
        return self._layer_index.get(li, [])

    # ---- 分层常驻 GPU：把部分层的 fp16 直接放显存，免掉每次搬运 ----

    def layer_fp16_bytes(self, li):
        """该层反量化成 fp16 后的字节数。"""
        e = self.weights[self._layer_keys(li)[0]] if self._layer_keys(li) else None
        tot = 0
        for k in self._layer_keys(li):
            ent = self.weights[k]
            shape = ent['shape']
            tot += shape[0] * shape[1] * 2
        return tot

    def layer_fp8_bytes(self, li):
        return sum(self.transfer_bytes(k) for k in self._layer_keys(li))

    def resident_fp8_by_budget(self, budget_bytes, verbose=True, interleave=False):
        """把部分层的 **int8 权重**搬进显存常驻（不是 fp16）。

        这是最省显存的档位：
          - fp8 常驻省掉「每 token 一次的 H2D 搬运」，只花一半显存；
          - 前向时仍在本层内反量化成 fp16、用完即弃，峰值只多一层 fp16。

        各层完全同构（每层都是 208 MiB fp8），显存占用与选哪几层无关，
        所以真正的自由度在于**怎么摆**：
          - interleave=False：取前 k 层（0..k-1），非驻留层连成一片；
          - interleave=True ：把非驻留层均匀铺开（见 spread_positions），
            每次预取拷贝能躲过更多层计算。
        """
        n_layers = len(self._layers())
        per = self.layer_fp8_bytes(0)
        k = int(budget_bytes // per) if per else 0
        k = min(k, n_layers)
        if interleave and 0 < k < n_layers:
            nr = spread_positions(n_layers, n_layers - k)
            self.resident_li = set(range(n_layers)) - set(nr)
            self.nonresident_li = nr
        else:
            self.resident_li = set(range(k))
            self.nonresident_li = [li for li in range(n_layers) if li not in self.resident_li]
        self.resident = len(self.resident_li)
        # 把常驻层的数据搬到 GPU
        moved = 0
        for li in self.resident_li:
            for key in self._layer_keys(li):
                e = self.weights[key]
                for f in ('q', 's', 'z', 'qg', 'qu', 'sg', 'su', 'zg', 'zu'):
                    if e.get(f) is not None:
                        e[f] = e[f].to('cuda')
                        moved += e[f].numel() * e[f].element_size()
        if verbose:
            total = n_layers * per
            print(f'[i8pin] 常驻 {self.resident}/{n_layers} 层的 {self.quant} → 显存 {moved/2**20:.0f} MiB；'
                  f'每 token 仍需搬运 {(n_layers-self.resident)*per/2**20:.0f} MiB '
                  f'(原 {total/2**20:.0f} MiB)')
            if interleave:
                print(f'[i8pin] 交错布局：非驻留层 {self.nonresident_li}')
        return self.resident

    def pin_by_budget(self, budget_bytes, only_mlp=False, verbose=True):
        """按显存预算挑层常驻，优先选「单位显存省搬运最多」的层。

        评分 = 该层 fp8 搬运量 / 该层 fp16 显存占用。
        - attn 层：fp8 40 MiB / fp16 80 MiB = 0.5
        - mlp 层： fp8 168 MiB / fp16 336 MiB = 0.5
        两者相同，所以真正的区分度在于**层越大，PCIe 固定开销摊得越薄**，
        因此按「层显存占用从大到小」贪心装填：先用有限显存吃掉最大的层，
        把最昂贵的传输整段消掉。

        only_mlp=True 时只考虑 MLP 层（实践里 MLP 是传输大头）。
        """
        n_layers = len(self._layers())
        cands = []
        for li in range(n_layers):
            keys = self._layer_keys(li)
            if not keys:
                continue
            is_mlp = all('.mlp.' in k for k in keys)
            if only_mlp and not is_mlp:
                continue
            cands.append((li, self.layer_fp16_bytes(li), self.layer_fp8_bytes(li)))

        # 大层优先：整层常驻能整段消掉最贵的传输
        cands.sort(key=lambda x: -x[1])
        chosen, used = [], 0
        for li, f16, f8 in cands:
            if used + f16 <= budget_bytes:
                chosen.append((li, f16, f8))
                used += f16
        chosen.sort()
        self.pinned = chosen
        self.pinned_li = {li for li, _, _ in chosen}
        if verbose:
            print(f'[pin] 预算 {budget_bytes/2**20:.0f} MiB → 常驻 {len(chosen)}/{n_layers} 层 '
                  f'(占 {used/2**20:.0f} MiB)，省掉 {sum(f8 for _,_,f8 in chosen)/2**20:.0f} MiB/token 的搬运')
            for li, f16, f8 in chosen:
                print(f'      layer {li:2d}: 占 {f16/2**20:6.1f} MiB, 省搬运 {f8/2**20:6.1f} MiB')
        return chosen

    def transfer_bytes(self, full):
        """该 Linear 的实际搬运字节数（packed/int8 按 element_size 折字节 + scale + zp）。"""
        e = self.weights[full]
        if e['kind'] == 'fused':
            b = (e['qg'].numel() * e['qg'].element_size()
                 + e['qu'].numel() * e['qu'].element_size()
                 + (e['sg'].numel() + e['su'].numel()) * 2)
            if self.quant == 'int4':
                b += e['zg'].numel() + e['zu'].numel()
            return b
        b = e['q'].numel() * e['q'].element_size() + e['s'].numel() * 2
        if self.quant == 'int4':
            b += e['z'].numel()
        return b

    def layers_of(self, li):
        pre = f'model.layers.{li}.'
        return [k for k in self.weights if k.startswith(pre)]


def spread_positions(n, m, avoid0=True):
    """在 0..n-1 里均匀选 m 个位置，让非驻留层尽量被常驻层隔开。

    为什么要有这个函数
    ------------------
    预取拷贝能隐藏多久，取决于「这个非驻留层前面还有几层常驻计算」。
    若非驻留层连成一片（如 16..31），每次拷贝只有**零层**计算可以躲；
    均匀铺开后，每个非驻留层前后都有常驻层，隐藏时间成倍增加。

    起点刻意避开第 0 层：`reset()` 在第 0 层的 pre_hook 里才发起首批拷贝，
    若第 0 层本身就要预取，拷贝完全没时间完成，只能原地干等。
    """
    if m <= 0:
        return []
    if m >= n:
        return list(range(n))
    start = 1 if (avoid0 and m < n) else 0
    span = (n - 1) - start
    if m == 1:
        return [start + span // 2]
    step = (span - 1) / (m - 1)
    pos = [start + int(round(i * step)) for i in range(m)]
    used, out = set(), []
    for p in pos:
        p = max(0, min(p, n - 1))
        while p in used and p < n - 1:
            p += 1
        if p in used:                     # 尾部撞车，往回找空位
            p = 0
            while p in used:
                p += 1
        used.add(p)
        out.append(p)
    return sorted(out)


def _to_empty_shell(mod, name='weight'):
    old = getattr(mod, name, None)
    if old is None:
        return
    try:
        delattr(mod, name)
    except AttributeError:
        mod._parameters.pop(name, None)
    p = nn.Parameter(torch.empty(0, dtype=torch.float16), requires_grad=False)
    p.__dict__['orig_shape'] = tuple(old.shape)
    setattr(mod, name, p)


# ================================================================ 预取管线

FP8_FIELDS = ('q', 'qg', 'qu')


class PrefetchPipeline:
    """非驻留层的异步预取：pinned 单块源 + 单块 GPU 槽 + 独立 stream。

    为什么必须这么做（实测）
    ------------------------
    原来每次前向都做 `mmap_view.to('cuda', non_blocking=True)`，实测
    **639 ms / 208 MiB = 0.33 GB/s**，而同路径微基准能跑到 8.24 GB/s，
    慢了 25 倍。原因是：

      1. mmap 是 **pageable** 内存 —— `non_blocking=True` 在这种源上
         形同同步，每次拷贝都要等 GPU 流水线排空；
      2. 目的缓冲每次现分配，在 90%+ 显存占用下分配器频繁停顿。

    本管线把两者都消掉：
      - 源：一次性搬进 **pinned 内存**，真正可异步；
      - 目的：**预分配**缓冲，零分配开销；
      - 独立 copy stream + event，让「搬第 i+1 层」与「算第 i 层」重叠。

    缓冲布局（P0-1，2026-09-22 单块化）
    ---------------------------
    **每层一条连续单块**：pinned 源一块、GPU 槽一块，`_issue` 每层只
    enqueue **1 次** copy。旧实现每层 7 个张量 → 7 次 `copy_`，每次吃
    1.9~2.3 ms blit 背压，每 forward 累计 241~305 ms 全耗在发射侧。
    张量视图在 `fetch` 时按 offset 切出来。

    HIP 历史坑（只影响「写」方向）：`buf[off:off+n].copy_(src,
    non_blocking=True)` 在**切片目标**上会静默不生效（微基准：整块 dst
    ✓ / 切片 dst ✗ 全零 / narrow ✗）。单块化后 GPU 侧只有整块→整块
    拷贝，写方向绕开该坑；**读方向**（fetch 切视图给 dequant / triton）
    不走 copy_，历来安全。

    槽位保护（P0-2）
    --------
    - reset 首填、以及「本回合还没被读者入队过」的槽：`wait_stream(compute)`
      —— 兜底覆盖上一回合残留在 GPU 上的读取；
    - 稳态补发：只 `wait_event(该槽位读者入队事件)`（消费方在读者全部
      入队后调 `note_slot_read` record），不再等整条计算流排空 ——
      这是把 fused 补发从 post 挪到 pre（`issue_at='pre'`）的安全前提：
      pre 里补发覆盖的是「上一个非驻留层」的槽，其读者早已入队完毕。
    """

    def __init__(self, store, nr_layers, verbose=True, slots=None,
                 issue_at='post', inline_copy=False):
        self.store = store
        self.nr = list(nr_layers)
        self.stream = torch.cuda.Stream()
        self.slots = max(2, int(slots or PREFETCH_SLOTS))
        # inline_copy=True：拷贝直接入队当前（计算）流，废除跨流事件握手。
        # 依据（2026-09-23 三段诊断+窗口计时）：blit 与 compute 本就串行，
        # 跨流 wait_event 把 copy 流 80% 时间拖在等事件上（窗口 950ms vs
        # 纯数据 176ms），且 host 发射被背压锁死成正反馈。单流 FIFO 下
        # 「拷贝覆盖前读者已入队」由流内顺序天然保证。
        self.inline_copy = bool(inline_copy)
        # issue_at:
        #   'post'    —— 读者入队后立刻补发（fused 旧默认；覆盖当前槽）
        #   'pre'     —— reset 少填一槽，挂载后在 pre 补发上一个读者的槽
        #   'consume' —— 反量化即消费完缓冲，pre 里补发当前槽（no-fused 路径）
        if issue_at not in ('post', 'pre', 'consume'):
            raise SystemExit(f'[prefetch] 未知 issue_at={issue_at!r}')
        self.issue_at = issue_at
        self.reserve_free = 1 if issue_at == 'pre' else 0
        self.bufs = []        # [slot] -> 整层单块 uint8 GPU 缓冲
        self.pin_blocks = {}  # li -> 整层单块 uint8 pinned 源
        self.events = [torch.cuda.Event() for _ in range(self.slots)]
        self.slot_ev = [torch.cuda.Event() for _ in range(self.slots)]
        self.slot_ev_ok = [False] * self.slots
        self.li_slot = {}     # 层号 -> 槽位（每轮前向开始时清空）
        self.li_ready = {}    # 层号 -> event
        self.cursor = 0
        self.next_slot = 0
        self._layout = None
        self._layer_bytes = 0
        self.t_prefetch = 0.0
        self.n_prefetch = 0
        self.n_reset = 0
        self.verbose = verbose

    def reset(self):
        """每轮前向开始时调用。

        **必须**重置：否则第二轮起 cursor 已到末尾、不再发起拷贝，
        而 fetch 仍按上一轮的槽位映射去读 —— 槽里装的是别的层的权重，
        输出会直接变成乱码（实测踩过）。
        """
        # GPU 流窗口计时（enable_timing）：copy 流本轮起点，
        # post(最后一层) 里 mark_copy_end() 收尾 —— 用于把 wall 拆成
        # 「copy 流忙窗」与「compute 流忙窗」，量化两流串行化程度。
        if not hasattr(self, 'cpy_wins'):
            self.cpy_wins = []          # [(e0, e1), ...] 每 forward 一对
        if not hasattr(self, 'fwd_host_ms'):
            self.fwd_host_ms = []       # 每 forward 主机耗时（拆 prefill/decode）
        self._cw0 = torch.cuda.Event(enable_timing=True)
        if self.inline_copy:
            self._cw0.record()          # 单流：窗口与 compute 流同轨
        else:
            self._cw0.record(self.stream)
        self.cursor = 0
        self.li_slot.clear()
        self.li_ready.clear()
        self.slot_ev_ok = [False] * self.slots   # 新回合：首填退化为 wait_stream
        self.next_slot = 0
        self.n_reset += 1
        # 先填满流水（issue_at='pre' 留一个空槽给 pre 里的补发）
        for _ in range(self.slots - self.reserve_free):
            if self.cursor >= len(self.nr):
                break
            self._issue(self.nr[self.cursor], self.next_slot)
            self.next_slot = (self.next_slot + 1) % self.slots
            self.cursor += 1

    def mark_copy_end(self):
        """post(最后一层) 调用：收尾本 forward 的 copy 流窗口。"""
        if getattr(self, '_cw0', None) is None:
            return
        e1 = torch.cuda.Event(enable_timing=True)
        if self.inline_copy:
            e1.record()
        else:
            e1.record(self.stream)
        self.cpy_wins.append((self._cw0, e1))
        self._cw0 = None

    def mark_compute_start(self):
        """pre(第 0 层) 调用：compute 流本 forward 窗口起点。"""
        if not hasattr(self, 'cmp_wins'):
            self.cmp_wins = []
        self._gw0 = torch.cuda.Event(enable_timing=True)
        self._gw0.record()                       # 当前=compute 流
        self._hf0 = time.perf_counter()          # 主机侧 forward 起点

    def mark_compute_end(self):
        """post(最后一层) 调用：compute 流窗口收尾。"""
        e0 = getattr(self, '_gw0', None)
        if e0 is None:
            return
        e1 = torch.cuda.Event(enable_timing=True)
        e1.record()
        self.cmp_wins.append((e0, e1))
        self._gw0 = None
        self.fwd_host_ms.append((time.perf_counter() - self._hf0) * 1e3)

    def gpu_window_ms(self):
        """生成结束后调用：返回 (copy 窗 ms/fwd, compute 窗 ms/fwd)。
        需先 torch.cuda.synchronize()；窗口=流上首尾事件时间差，
        含流内空隙（正是「该流被卡住/空闲」的总时长）。"""
        torch.cuda.synchronize()
        cw = [(e0.elapsed_time(e1)) for e0, e1 in
              getattr(self, 'cpy_wins', [])]
        gw = [(e0.elapsed_time(e1)) for e0, e1 in
              getattr(self, 'cmp_wins', [])]
        mean = lambda xs: (sum(xs) / len(xs)) if xs else 0.0
        return mean(cw), mean(gw), len(cw), len(gw)

    def note_slot_read(self, li):
        """记录「li 所在槽位的读者已全部入队」。

        之后向该槽补发只需 wait 这个事件，不必 wait 整条计算流 ——
        wait_stream 会把 issue 时刻之后新入队的 kernel 也算进去，
        把拷贝白白推迟到本层算完。
        inline_copy 模式下无意义（流内 FIFO 即保证），直接空转。
        """
        if self.inline_copy:
            return
        slot = self.li_slot.get(li)
        if slot is None:
            return
        self.slot_ev[slot].record(torch.cuda.current_stream())
        self.slot_ev_ok[slot] = True

    # ---- 布局：用**相对后缀**（如 'self_attn.q_proj'）而不是绝对参数名 ----
    # 各层同构，后缀才能跨层复用同一份布局；用绝对名会让第 25 层匹配不上第 24 层。
    # 条目 (suffix, f, nbytes, shape, off)：off/nbytes 均为**字节**（P0-1；
    # int4 的 q 是 uint32，元素数≠字节数，按字节记账对 int8 恰好等价）。
    def _plan_layout(self):
        if self._layout is not None:
            return self._layout
        li0 = self.nr[0]
        pre = f'model.layers.{li0}.'
        layout = []
        off = 0
        for k in self.store._layer_keys(li0):
            suffix = k[len(pre):] if k.startswith(pre) else k
            e = self.store.weights[k]
            for f in FP8_FIELDS:
                t = e.get(f)
                if t is None:
                    continue
                nb = t.numel() * t.element_size()
                layout.append((suffix, f, nb, tuple(t.shape), off))
                off += nb
        self._layout = layout
        self._layer_bytes = off
        return layout

    def _key(self, li, suffix):
        return f'model.layers.{li}.{suffix}'

    def layer_bytes(self):
        self._plan_layout()
        return self._layer_bytes

    def setup(self, verbose=True):
        """① 把每层非驻留量化字节拼成**一条连续 pinned 单块**并丢弃原宿主视图。

        ② scale(/zp) 直接常驻显存：int4 每层 ~4.9 MB（组表 3 B/参数）、
        int8 每层 ~56 KB。让它们常驻可以省掉每次前向的 H2D 小拷贝
        —— 小张量 H2D 的固定开销（kernel 启动 + 隐式同步）比数据本身贵得多。
        """
        layout = self._plan_layout()
        lb = self._layer_bytes
        need = {(s, f): n for s, f, n, _, _ in layout}
        total = 0
        n_scale = 0
        n_zp = 0
        try:
            for li in self.nr:
                print(f'[prefetch] pin 层 {li} ...', flush=True)
                pre = f'model.layers.{li}.'
                have = {}
                for k in self.store._layer_keys(li):
                    e = self.store.weights[k]
                    suffix = k[len(pre):] if k.startswith(pre) else k
                    for f in FP8_FIELDS:
                        t = e.get(f)
                        if t is not None:
                            have[(suffix, f)] = t.numel() * t.element_size()
                if have != need:
                    raise SystemExit(
                        f'[prefetch] 层 {li} 与层 {self.nr[0]} 不同构：'
                        f'缺 {sorted(set(need) - set(have))} '
                        f'多 {sorted(set(have) - set(need))}')
                # 整层 clone 进一条锁页块：读一次 mmap（页缓存/磁盘），
                # 之后常驻锁页内存。CPU→锁页走普通 memcpy，切片目标安全
                # （HIP 的静默失败只在 device 侧异步 copy_）。
                blk = torch.empty(lb, dtype=torch.uint8).pin_memory()
                for suffix, f, numel, shape, off in layout:
                    e = self.store.weights[self._key(li, suffix)]
                    src = e[f].detach().reshape(-1)
                    if src.dtype != torch.uint8:
                        src = src.view(torch.uint8)
                    blk[off:off + numel].copy_(src)
                    e[f] = None          # 丢掉对 mmap 的引用
                self.pin_blocks[li] = blk
                total += lb
                # scale(/zp) 上卡常驻
                for k in self.store._layer_keys(li):
                    e = self.store.weights[k]
                    for sf in ('s', 'sg', 'su'):
                        st = e.get(sf)
                        if st is not None:
                            e[sf] = st.to('cuda')
                            n_scale += st.numel()
                    for zf in ('z', 'zg', 'zu'):
                        st = e.get(zf)
                        if st is not None:
                            e[zf] = st.to('cuda')
                            n_zp += st.numel()
        except torch.AcceleratorError as ex:
            # 锁页内存锁的是真实物理页：本机只有 13.69 GiB，锁数 GiB 会直接失败。
            # 实测上限约 16~20 层（int8 3.3~4.2 GiB）。非驻留层太多时给明确提示，
            # 而不是抛一个看不懂的 CUDA OOM。
            raise SystemExit(
                f'[prefetch] 锁页内存不足（已 pin {total/2**20:.0f} MiB，'
                f'非驻留 {len(self.nr)} 层）：{ex}\n'
                f'  请**调小** --reserve-mib，让更多层常驻显存、减少要 pin 的层数。\n'
                f'  实测可用组合：--reserve-mib 2600（16 层非驻留，pin 3328 MiB）')
        if verbose:
            print(f'[prefetch] pinned 源 {total/2**20:.0f} MiB'
                  f'（{len(self.nr)} 层 × 单块 {lb/2**20:.1f} MiB），'
                  f'scale/zp 常驻显存 {(n_scale*2 + n_zp)/2**20:.2f} MiB', flush=True)

        # ② 预分配 GPU 缓冲：每槽一条整层单块（整块→整块，见类文档）
        for _ in range(self.slots):
            self.bufs.append(torch.empty(lb, dtype=torch.uint8, device='cuda'))
        if verbose:
            print(f'[prefetch] GPU 缓冲 {self.slots * lb/2**20:.0f} MiB '
                  f'（单层 {lb/2**20:.1f} MiB × {self.slots} 槽，1 块/槽）',
                  flush=True)

        # ③ 首轮预取由 reset() 完成
        self.reset()
        return self

    # ---- 发起一次异步拷贝（整块 → 整块，每层仅 1 次 enqueue）----
    def _issue(self, li, slot):
        self._plan_layout()
        row = self.bufs[slot]
        src = self.pin_blocks[li]
        t0 = time.perf_counter()
        if self.inline_copy:
            # 单流模式：拷贝直接排进当前（计算）流尾部。
            # 覆盖安全由 FIFO 保证 —— issue 调用点总在「该槽上一批读者
            # 已入队」之后（pre: 上一层 post 之后；post: 本层 forward 之后；
            # consume: 本层 dequant 已入队），无需任何 wait。
            row.copy_(src, non_blocking=True)
            self.events[slot].record()
        else:
            # ★ 必须在切换 stream **之前**取计算流。
            #   一旦进入 `with torch.cuda.stream(self.stream)`，
            #   torch.cuda.current_stream() 返回的就是 self.stream，
            #   `self.stream.wait_stream(current)` 会退化成自我等待（空操作），
            #   拷贝就不会等计算流上刚排队的读者 —— 直接覆盖正在被读取的槽位，
            #   造成数据竞争、权重错乱、输出乱码（实测踩了很久）。
            compute = torch.cuda.current_stream()
            with torch.cuda.stream(self.stream):
                if self.slot_ev_ok[slot]:
                    # 稳态：只等该槽上一批读者入队时的事件（可能早已完成）
                    self.stream.wait_event(self.slot_ev[slot])
                else:
                    # 首填/新槽：等整条计算流 —— 覆盖上一回合残留的读取
                    self.stream.wait_stream(compute)
                row.copy_(src, non_blocking=True)      # 整块 → 整块
                self.events[slot].record(self.stream)
        self.li_slot[li] = slot
        self.li_ready[li] = self.events[slot]
        self.t_prefetch += time.perf_counter() - t0
        self.n_prefetch += 1

    def _issue_next(self):
        """把游标处的非驻留层预取到下一个轮转槽。"""
        if self.cursor >= len(self.nr):
            return
        self._issue(self.nr[self.cursor], self.next_slot)
        self.next_slot = (self.next_slot + 1) % self.slots
        self.cursor += 1

    # ---- 取用：等 event，然后返回 GPU 上的 fp8 视图 ----
    def fetch(self, li):
        if li not in self.li_ready:
            # reset 之后没排到这一层（不该发生），补发一次兜底
            self._issue(li, self.next_slot)
            self.next_slot = (self.next_slot + 1) % self.slots
        slot = self.li_slot[li]
        torch.cuda.current_stream().wait_event(self.li_ready[li])
        row = self.bufs[slot]
        i4 = self.store.quant == 'int4'
        out = {}
        for suffix, f, nbytes, shape, off in self._plan_layout():
            # 读侧切视图：安全（HIP 的坑只在写方向，见类文档）。
            # off/nbytes 为字节；int4 → uint32 直序 packed（off 恒 4 对齐），
            # int8 → 按位重解释成 int8。
            v = row[off:off + nbytes]
            if i4:
                v = v.view(torch.uint32)
            else:
                v = v.view(torch.int8)
            out.setdefault(self._key(li, suffix), {})[f] = v.view(shape)
        return out

    def free_source(self, model):
        """④ 清理：丢掉 pinned 源（数据已在 GPU 缓冲里滚动复用）。

        注意：pinned 内存不能像普通张量那样立刻归还，这里显式断开引用，
        让 Python 回收后再由驱动解锁。
        """
        self.pin_blocks.clear()


# ================================================================ 逐层搬运

def install_offload_hooks(model, store, verbose=True, prefetch=None,
                          fused=True):
    """在 DecoderLayer 上挂 pre/post hook。

    fused=True（triton 可用）时不反量化、不挂 fp16 权重：把量化权重引用
    （常驻张量或预取缓冲视图）挂到模块属性上，forward 走融合 W4A16/W8A16
    GEMV/GEMM kernel（寄存器内反量化，见 int8_gemm_triton.py）。

    三种层各走一条路：
      - 量化常驻层：引用在 install 时已挂好，pre 零工作；
      - 非驻留层：pre 取预取缓冲视图挂上，post 释放；
      - fp16 常驻层（pinned_li，与 fused 互斥）：原样走 F.linear。
    """
    import types
    layers = store._layers()
    handles = []
    is_i4 = store.quant == 'int4'

    class _QuietDict(dict):
        """INT8_QUIET=1 时吞掉所有计数写入 —— 用于 A/B 验证统计 hook 本身的开销。"""
        def __setitem__(self, k, v):
            if os.environ.get('INT8_QUIET') == '1':
                return
            super().__setitem__(k, v)

    counters = _QuietDict(transferred=0, peak_fp16=0, calls=0, transfer_s=0.0,
                          pinned_hits=0, resident_hits=0, prefetch_hits=0,
                          t_deq=0.0, t_mount=0.0, t_unmount=0.0, t_fwd=0.0,
                          t_issue=0.0, n_deq=0, t_gemv=0.0)
    pinned_li = getattr(store, 'pinned_li', set())
    resident_li = getattr(store, 'resident_li', set())
    use_i8 = fused and not pinned_li

    if use_i8:
        # scale(/zp) 全部常驻显存；常驻层引用现在就挂好
        n_sc_b = 0
        n_res = 0
        for li in range(len(layers)):
            if li in pinned_li:
                continue
            for k in store._layer_keys(li):
                e = store.weights[k]
                s = e.get('s')
                if s is not None and s.device.type != 'cuda':
                    e['s'] = s.to('cuda')
                    n_sc_b += e['s'].numel() * 2
                z = e.get('z')
                if z is not None and z.device.type != 'cuda':
                    e['z'] = z.to('cuda')
                    n_sc_b += e['z'].numel()
                if li in resident_li:
                    wq = e.get('q')
                    if wq is not None and wq.device.type == 'cuda':
                        mod = _resolve(model, k)
                        mod._i8_w = wq
                        mod._i8_s = e['s']
                        mod._i8_z = e.get('z')
                        n_res += 1
        if verbose:
            print(f'[fused] {store.quant} 前向已启用（triton）；scale/zp 常驻 '
                  f'{n_sc_b / 2**20:.2f} MiB，常驻层引用 {n_res} 个')

        def i8_forward(self, x):
            # t_gemv：融合 GEMV 调用的**主机**耗时（发射 + 分配器 + 隐式同步），
            # 用来拆解 fused 的 t_fwd 阻塞到底在 GEMV 还是 attention。
            _tg = time.perf_counter()
            if is_i4:
                y = int4_gemm_rows(self._i8_w, self._i8_s, self._i8_z, x)
            else:
                y = int8_gemm_rows(self._i8_w, self._i8_s, x)
            counters['t_gemv'] += time.perf_counter() - _tg
            return y

        for li in range(len(layers)):
            if li in pinned_li:
                continue
            for k in store._layer_keys(li):
                mod = _resolve(model, k)
                mod.forward = types.MethodType(i8_forward, mod)

    def make_pre(li):
        is_pinned = li in pinned_li
        is_resident = li in resident_li
        i8 = use_i8 and not is_pinned

        def _pre(module, args):
            if li == 0 and prefetch is not None:
                # 每轮前向开始：重置预取游标并重新填满缓冲
                prefetch.reset()
                prefetch.mark_compute_start()      # GPU 窗口：compute 起点
            if is_pinned:
                # fp16 常驻层：权重已在 GPU，整层都是纯计算
                counters['pinned_hits'] += 1
                counters['calls'] += 1
                module._t0 = time.perf_counter()
                return
            if i8 and is_resident:
                # int8 常驻层：引用已挂好，零工作
                counters['resident_hits'] += 1
                counters['calls'] += 1
                module._t0 = time.perf_counter()
                return
            t0 = time.perf_counter()
            if i8:
                if prefetch is not None:
                    # 非驻留层：int8 数据已被异步预取到 GPU 缓冲（零拷贝引用）
                    bufs = prefetch.fetch(li)
                    refs = store.layer_int8_refs(li, bufs)
                    counters['prefetch_hits'] += 1
                else:
                    refs = store.layer_int8_refs(li)
                counters['transfer_s'] += time.perf_counter() - t0
                t1 = time.perf_counter()
                counters['t_deq'] += t1 - t0
                counters['n_deq'] += 1
                total = 0
                for k, (wq, sg, sz) in refs.items():
                    t = _resolve(model, k)
                    t._i8_w = wq
                    t._i8_s = sg
                    t._i8_z = sz
                    total += wq.numel()
                t2 = time.perf_counter()
                counters['t_mount'] += t2 - t1
                counters['calls'] += 1
                counters['peak_fp16'] = max(counters['peak_fp16'], total)
                module._t0 = t2
                if prefetch is not None and prefetch.issue_at == 'pre':
                    # P0-2：补发挪到 pre —— 此刻覆盖的是「上一个非驻留层」的
                    # 槽，其读者事件已在上一层 post 里 record（note_slot_read），
                    # _issue 只等该事件，不与本层尚未入队的 kernel 竞争。
                    # （issue_at='post' 时这里绝不能补发：本层 kernel 还没入队，
                    #   补发会覆盖正在读的槽 → 乱码，实测。）
                    t4 = time.perf_counter()
                    prefetch._issue_next()
                    counters['t_issue'] += time.perf_counter() - t4
                return
            # ---- fp16 挂载路径（fused 关闭时的原逻辑）----
            if is_resident:
                tensors = store.layer_dequant_resident(li)
                counters['resident_hits'] += 1
            elif prefetch is not None:
                bufs = prefetch.fetch(li)
                tensors = store.layer_dequant_from_buffers(li, bufs)
                counters['prefetch_hits'] += 1
                counters['transfer_s'] += time.perf_counter() - t0
            else:
                tensors = store.layer_to_gpu_fp16(li)
                counters['transfer_s'] += time.perf_counter() - t0
            t1 = time.perf_counter()
            counters['t_deq'] += t1 - t0
            counters['n_deq'] += 1
            total = 0
            for k, w in tensors.items():
                target = _resolve(model, k)
                target.weight = nn.Parameter(w, requires_grad=False)
                total += w.numel() * w.element_size()
            t2 = time.perf_counter()
            counters['t_mount'] += t2 - t1
            counters['calls'] += 1
            counters['peak_fp16'] = max(counters['peak_fp16'], total)
            module._t0 = t2
            if prefetch is not None and not is_resident:
                # consume 语义：反量化已把槽位读完（读者=刚才入队的 dequant
                # kernel），record 事件后即可在 pre 里补发当前槽。
                t4 = time.perf_counter()
                prefetch.note_slot_read(li)
                prefetch._issue_next()
                counters['t_issue'] += time.perf_counter() - t4
        return _pre

    def make_post(li):
        is_pinned = li in pinned_li
        i8 = use_i8 and not is_pinned

        def _post(module, args, output):
            t3 = time.perf_counter()
            if getattr(module, '_t0', None) is not None:
                counters['t_fwd'] += t3 - module._t0
                module._t0 = None
            if prefetch is not None and li == len(layers) - 1:
                # 最后一层：收尾本 forward 的两条流窗口
                prefetch.mark_compute_end()
                prefetch.mark_copy_end()
            if is_pinned:
                return output                          # fp16 常驻层不释放
            if i8:
                if li not in resident_li:
                    # 常驻层的引用保留；非驻留层放下引用（缓冲被下一层复用）
                    for k in store._layer_keys(li):
                        t = _resolve(model, k)
                        t._i8_w = None
                        t._i8_s = None
                        t._i8_z = None
                    if prefetch is not None:
                        # 本层 kernel 已入队：先 record 槽位读者事件。
                        # 'post' 模式紧接着补发（_issue 只等该事件，
                        # 与旧 wait_stream 在同一点等价）；
                        # 'pre' 模式只 record，补发留给下一层的 pre。
                        prefetch.note_slot_read(li)
                        if prefetch.issue_at != 'pre':
                            t4 = time.perf_counter()
                            prefetch._issue_next()
                            counters['t_issue'] += time.perf_counter() - t4
                counters['t_unmount'] += time.perf_counter() - t3
                counters['transferred'] += 1
                return output
            for k in store._layer_keys(li):
                target = _resolve(model, k)
                target.weight = nn.Parameter(
                    torch.empty(0, dtype=torch.float16), requires_grad=False)
            counters['t_unmount'] += time.perf_counter() - t3
            counters['transferred'] += 1
            return output
        return _post

    # ---- 全驻快路（lite）：resident 覆盖全部层 + 无预取 + fused 时， ----
    # 旁路逐层 pre/post 回调（每 forward 64 次 Python 闭包 + 分支/计数开销），
    # 只保留 2 个钩子：layer0 pre 计 calls/resident_hits + 起 host 计时，
    # last post 记逐 forward 主机耗时。计数器语义不变（calls += n_layers/次）。
    n_all = len(layers)
    lite = (fused and not pinned_li and prefetch is None
            and len(resident_li) == n_all)
    if lite:
        host_ms = []
        counters['fwd_ms'] = host_ms
        _t = {}

        def _pre0(module, args):
            counters['calls'] += n_all
            counters['resident_hits'] += n_all
            _t['t0'] = time.perf_counter()

        def _postL(module, args, output):
            host_ms.append((time.perf_counter() - _t['t0']) * 1e3)
            return output

        handles.append(layers[0].register_forward_pre_hook(_pre0))
        handles.append(layers[-1].register_forward_hook(_postL))
        if verbose:
            print(f'[hooks] lite 全驻旁路：{n_all} 层逐层回调已裁撤'
                  f'（仅 layer0/last 计时钩 ×2）')
        return handles, counters

    for li, layer in enumerate(layers):
        handles.append(layer.register_forward_pre_hook(make_pre(li)))
        handles.append(layer.register_forward_hook(make_post(li)))
    if verbose:
        print(f'[hooks] {len(layers)} 层已挂载（常驻 {len(pinned_li)} 层，'
              f'搬运 {len(layers)-len(pinned_li)} 层）')
    return handles, counters


def materialize_pinned(model, store, verbose=True):
    """把常驻层的 fp16 权重一次性搬上 GPU 并挂好。"""
    total = 0
    for li, f16, f8 in getattr(store, 'pinned', []):
        tensors = store.layer_to_gpu_fp16(li)
        for k, w in tensors.items():
            target = _resolve(model, k)
            target.weight = nn.Parameter(w, requires_grad=False)
            total += w.numel() * w.element_size()
        # 常驻层不必再留着量化副本（省 CPU 内存）
        for k in store._layer_keys(li):
            e = store.weights[k]
            for kk in ('q', 's', 'z', 'qg', 'qu', 'sg', 'su', 'zg', 'zu'):
                if kk in e:
                    e[kk] = None
            e['resident_gpu'] = True
    if verbose:
        print(f'[pin] 常驻层已上卡，占显存 {total/2**30:.3f} GiB')
    return total


def _resolve(model, dotted):
    cur = model
    for p in dotted.split('.'):
        cur = getattr(cur, p)
    return cur


# ================================================================ 分片读取

def make_shard_reader(model_dir):
    """返回 name -> tensor 的读取器（惰性打开 safetensors 分片）。"""
    import json
    from safetensors import safe_open

    idx_path = os.path.join(model_dir, 'model.safetensors.index.json')
    if not os.path.exists(idx_path):
        return None
    with open(idx_path, encoding='utf-8') as f:
        idx = json.load(f)['weight_map']
    open_files = {}

    def reader(name):
        shard = idx.get(name)
        if shard is None:
            return None
        fh = open_files.get(shard)
        if fh is None:
            fh = safe_open(os.path.join(model_dir, shard), framework='pt', device='cpu')
            open_files[shard] = fh
        try:
            return fh.get_tensor(name)
        except Exception:
            return None

    def close():
        """关闭所有分片句柄，释放 mmap（让 OS 能回收页面缓存）。"""
        for fh in open_files.values():
            try:
                # safe_open 的句柄在 GC 时释放；显式删引用并 gc
                del fh
            except Exception:
                pass
        open_files.clear()
        gc.collect()

    reader.close = close
    reader.model_dir = model_dir
    return reader


def resolve_model_dir(repo_id):
    """把 HF repo id 换成本地缓存目录（不触发完整性校验）。

    snapshot_download 会校验整个快照，而本地缓存常缺 README 之类的非权重文件，
    导致 IncompleteSnapshotError。这里直接用 try_to_load_from_cache 定位
    model.safetensors.index.json，拿它所在目录即可。
    """
    from huggingface_hub import try_to_load_from_cache

    p = try_to_load_from_cache(repo_id, 'model.safetensors.index.json')
    if isinstance(p, str) and os.path.exists(p):
        return os.path.dirname(p)
    # 退化：单文件模型
    p = try_to_load_from_cache(repo_id, 'model.safetensors')
    if isinstance(p, str) and os.path.exists(p):
        return os.path.dirname(p)
    raise SystemExit(f'本地缓存里找不到 {repo_id} 的权重索引，请先下载')


# ================================================================ dry-run

def dry_run():
    print('=== int4 g128 分层管线显存账目（Hy-MT2-7B，32 层，推算）===')
    print()
    print('GPU 常驻：')
    print(f'  未量化参数（tie 合并后 embedding/lm_head/norm） ≈ 0.98 GiB')
    print(f'  常驻 int4 层（reserve 1000 时可全驻 32 层 × 109 MiB）≈ 3.5 GiB')
    print(f'  预取槽（slots × 109 MiB，slots=3，全驻时为 0）      ≈ 0.32 GiB')
    print(f'  KV cache（batch=1, 2048 上下文）                   ≈ 0.3 GiB')
    print(f'  激活 + 框架开销                                    ≈ 0.5 GiB')
    print(f'  ---------------------------------------------')
    print(f'  合计 ≈ 5.6 GiB，8GB 卡可容（--reserve-mib 留余量）')
    print()
    print('宿主内存：')
    print(f'  非驻留层 pinned int4（全驻时为 0）                  ≈ 0 GiB')
    print(f'  磁盘缓存 3.40 GiB（mmap 零拷贝，不占 RSS）')
    print()
    print('=== 每 token 搬运成本（非驻留层）===')
    single = 4096 * 14336
    print(f'  最大单层 int4 packed：{single/2/2**20:.0f} MiB (4096x14336)')
    print(f'  int4 全层常驻后搬运=0；int8 回退时非驻层仍需搬运（~208 MiB/层）')
    return 0


# ================================================================ 主流程

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='tencent/Hy-MT2-7B')
    ap.add_argument('--quant', choices=['int4', 'int8'], default='int4',
                    help='权重量化模式。int4=group128 非对称 uint32 直序打包'
                         '（默认，0.523 B/param，全层常驻消灭 PCIe 搬运）；'
                         'int8=per-row 对称（回退选项，1 B/param）')
    ap.add_argument('--tpl', default=os.path.join(
        HERE, '..', '..', 'example-ptl', 'tpl', '01.tsv'))
    ap.add_argument('--out', default=os.path.join(
        HERE, '..', '..', 'out', '02-int4.tsv'))
    ap.add_argument('--limit', type=int, default=8)
    ap.add_argument('--batch', type=int, default=1)
    ap.add_argument('--max-new-tokens', type=int, default=128)
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--pin', action='store_true', help='常驻小张量锁页内存')
    ap.add_argument('--pin-layers', type=int, default=0,
                    help='整层常驻 GPU 的层数（0=自动）')
    ap.add_argument('--fp8-resident', action='store_true', default=True,
                    help='【默认】把装得下的层的量化权重搬进显存常驻')
    ap.add_argument('--no-pin', action='store_true',
                    help='权重全留 CPU（最大搬运量，最省显存）')
    ap.add_argument('--pin-mlp', action='store_true',
                    help='只常驻 MLP 层（配合 --pin-layers）')
    ap.add_argument('--reserve-mib', type=int, default=2200,
                    help='给非驻层预取槽+激活预留的显存 MiB。'
                         'int4 全驻时 1000 即可；int8 回退 WSL2 实测 1000 最优'
                         '（常驻 21-22 层，424ms；默认取保守值 2200）')
    ap.add_argument('--no-cache', action='store_true',
                    help='不使用磁盘缓存（既不读也不写）')
    ap.add_argument('--rebuild-cache', action='store_true',
                    help='删掉当前量化模式的旧缓存重新量化并落盘')
    ap.add_argument('--cache-info', action='store_true',
                    help='只打印缓存状态后退出')
    ap.add_argument('--load-only', action='store_true',
                    help='只加载模型不生成（用于验证/计时）')
    ap.add_argument('--no-prefetch', action='store_true',
                    help='关闭异步预取管线（退回同步逐层搬运，用于对比）')
    ap.add_argument('--no-fused-gemm', action='store_true',
                    help='关闭 triton 融合 W4A16/W8A16 前向（退回反量化挂载路径）')
    ap.add_argument('--prefetch-slots', type=int, default=PREFETCH_SLOTS,
                    help='预取槽数（≥2）。每槽一条整层单块 GPU 缓冲（int4 约 '
                         '109 MiB/槽、int8 约 208 MiB/槽）；3~4 槽给拷贝更多超前余量')
    ap.add_argument('--issue-at', choices=['auto', 'pre', 'post'],
                    default='auto',
                    help='预取补发时点。auto=旧行为（fused 在 post、'
                         'no-fused 反量化后立即补发）；pre=fused 在 pre 里'
                         '覆盖「上一个非驻留层」的槽（P0-2，需 ≥2 槽）')
    ap.add_argument('--inline-copy', action='store_true',
                    help='拷贝直接入队计算流（废除跨流事件握手，单流 FIFO）。'
                         'A/B 验证 ROCDXG 跨流握手是否为窗口拉伸元凶')
    ap.add_argument('--greedy', action='store_true',
                    help='贪心解码（do_sample=False）。基准测试必须开：'
                         '采样会让每次生成长度不同，总时长根本不可比')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--interleave', action='store_true', default=True,
                    help='【默认】把非驻留层均匀铺开，让预取拷贝躲过更多层计算')
    ap.add_argument('--no-interleave', dest='interleave', action='store_false',
                    help='非驻留层取尾部连续一段（用于 A/B 对比）')
    ap.add_argument('--profile', action='store_true',
                    help='用 torch.profiler 剖析第一批生成的 GPU kernel 分布')
    a = ap.parse_args()

    if a.cache_info:
        mdir = resolve_model_dir(a.model)
        quant = getattr(a, 'quant', 'int4')
        idx = int8_cache.describe(a.model, HERE, int8_cache.revision_of(mdir),
                                  quant=quant)
        p = int8_cache.paths(a.model, HERE, quant=quant)
        files = ('q4', 'scale', 'zp', 'index') if quant == 'int4' \
            else ('fp8', 'scale', 'index')
        if idx is None:
            print(f'[cache] 无缓存（{int8_cache.scheme_of(quant)}）。'
                  f'目录：{p["root"]}')
        elif idx.get('stale'):
            print(f'[cache] 缓存过期：缓存 revision {idx.get("revision","")[:12]} '
                  f'≠ 当前 {int8_cache.revision_of(mdir)[:12]}')
        else:
            print(f'[cache] 有效缓存（{p["root"]}）')
            print(f'        revision   {idx.get("revision","")[:12]}')
            print(f'        scheme     {idx.get("scheme")}')
            print(f'        Linear     {idx.get("n_entries")} 个 / '
                  f'{idx.get("params",0)/1e9:.3f}B 参数')
            if quant == 'int4':
                print(f'        packed     {idx.get("fp8_bytes",0)/2**30:.2f} GiB（uint32 直序）')
                print(f'        scale      {idx.get("scale_elems",0)*2/2**20:.1f} MiB')
                print(f'        zp         {idx.get("zp_elems",0)/2**20:.1f} MiB')
            else:
                print(f'        int8       {idx.get("fp8_bytes",0)/2**30:.2f} GiB')
                print(f'        scale      {idx.get("scale_elems",0)*2/2**20:.1f} MiB')
            tot = sum(os.path.getsize(p[k]) for k in files
                      if os.path.exists(p[k]))
            print(f'        磁盘占用   {tot/2**30:.2f} GiB')
        return 0

    if a.dry_run:
        return dry_run()

    import mt_common as C
    from transformers import AutoConfig, AutoTokenizer

    props = torch.cuda.get_device_properties(0)
    # 真实可用显存：total_memory 是纸面容量，驱动/上下文先吃掉约 141 MiB
    _free0, _total0 = torch.cuda.mem_get_info()
    usable_gib = _free0 / 2**30
    print(f'[env] {torch.cuda.get_device_name(0)} | {props.gcnArchName} | '
          f'vram={_total0/2**30:.2f}GiB 可用={usable_gib:.2f}GiB', flush=True)

    # 本地已有全量权重缓存：local_files_only 免联网校验段（HF_HUB_OFFLINE 双保险）
    cfg = AutoConfig.from_pretrained(a.model, trust_remote_code=True,
                                     local_files_only=True)
    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True,
                                        local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'
    torch.manual_seed(a.seed)
    _probe('after-tokenizer')

    t0 = time.time()
    model, mode = load_meta_fp8(a, cfg, tok, props)
    print(f'[load] {time.time()-t0:.0f}s  mode={mode}', flush=True)

    torch.cuda.reset_peak_memory_stats()
    print(f'[mem] allocated={torch.cuda.memory_allocated()/2**30:.2f} GiB', flush=True)

    if a.load_only:
        print(f'[load-only] 跳过生成。用时 {time.time()-t0:.0f}s')
        return 0

    # ---- 数据 ----
    wf = C.WorkFile(a.tpl)
    rows = [r for r in wf.rows if C.stripped(wf.source_of(r[2]))][:a.limit]
    srcs = [wf.source_of(r[2]) for r in rows]
    print(f'[data] {len(srcs)} 条', flush=True)

    def translate(prompts):
        texts = [tok.apply_chat_template([{'role': 'user', 'content': p}],
                                         tokenize=False, add_generation_prompt=True)
                 for p in prompts]
        enc = tok(texts, return_tensors='pt', padding=True).to(model.device)
        kw = dict(max_new_tokens=a.max_new_tokens, pad_token_id=tok.pad_token_id)
        if a.greedy:
            kw['do_sample'] = False
        else:
            kw.update(do_sample=True, temperature=0.7, top_p=0.6,
                      top_k=20, repetition_penalty=1.05)
        with torch.inference_mode():
            out = model.generate(**enc, **kw)
        return tok.batch_decode(out[:, enc['input_ids'].shape[1]:],
                                skip_special_tokens=True)

    results = []
    peak = 0.0
    # prompt 构建（含占位符保护的 isin，单条可达数百 ms）移出计时窗口，
    # 否则会被摊进「每 forward」里污染跨配置比较。
    batches = []
    for i in range(0, len(srcs), a.batch):
        chunk = srcs[i:i + a.batch]
        protected = [C.protect(s) for s in chunk]
        prompts = [C.build_prompt(p, target_lang='简体中文', placeholders=bool(t))
                   for p, t in protected]
        batches.append((chunk, prompts, [t for _, t in protected]))
    t0 = time.time()
    for i, (chunk, prompts, tags_list) in enumerate(batches):
        if a.profile and i == 0:
            # 只剖析第一批：回答「GPU 时间到底花在哪种 kernel 上」
            from torch.profiler import profile, ProfilerActivity
            with profile(activities=[ProfilerActivity.CPU,
                                     ProfilerActivity.CUDA]) as prof:
                outs = translate(prompts)
            print()
            print('=== GPU kernel 耗时 Top 20 ===')
            # 按 GPU(device) 时间排序 —— 旧 sort_by='cuda_time_total' 在
            # 本 torch 版本上静默失效（回退成 CPU 列），必须用 device_time_total
            events = prof.key_averages()

            def _dev(e):
                return (getattr(e, 'device_time_total', 0)
                        or getattr(e, 'self_device_time_total', 0))

            rows = sorted((e for e in events if _dev(e) > 0),
                          key=_dev, reverse=True)[:20]
            print(f'{"GPU ms":>10}  {"CPU ms":>9}  {"calls":>7}  name')
            for e in rows:
                print(f'{_dev(e)/1000:10.1f}  {e.cpu_time_total/1000:9.1f}  '
                      f'{e.count:7d}  {e.key[:70]}')
            dev_total = sum(_dev(e) for e in events
                            if not e.key.startswith(('aten::', 'cuda::',
                                                      'hip::', 'c10::')))
            print(f'[profile] 非 aten 层 GPU 时间合计 ≈ {dev_total/1000:.0f} ms'
                  f'（单序列窗口，profiler 会放大 launch 开销，只看分布）')
        else:
            outs = translate(prompts)
        for s, raw, tags in zip(chunk, outs, tags_list):
            results.append((s, C.restore(C.sanitize(raw), tags)))
        peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
        print(f'\r[{min(i+a.batch,len(srcs))}/{len(srcs)}] {time.time()-t0:.0f}s '
              f'peak={peak:.2f}GiB', end='', flush=True)
    print()

    el = time.time() - t0
    print()
    print('=' * 60)
    print(f'模式      : {mode}')
    print(f'条数      : {len(results)}')
    print(f'用时      : {el:.0f}s  ({len(results)/max(1e-9,el)*60:.1f} 条/分)')
    print(f'峰值显存   : {peak:.2f} GiB / {usable_gib:.2f} GiB 可用（纸面 {props.total_memory/2**30:.2f}）')
    c = getattr(model, '_offload_counters', None)
    if c and c.get('calls'):
        n_layers = len(model.model.layers)
        n_fwd = c['calls'] / n_layers
        print(f'前向次数   : {n_fwd:.0f}（每层 hook 命中 {c["calls"]} 次 / {n_layers} 层）')
        print(f'★ 每 forward: {el/max(1e-9,n_fwd)*1000:.0f} ms   '
              f'← 唯一可跨配置比较的指标')
        print(f'  常驻命中 {c.get("resident_hits", 0)} / 预取 {c.get("prefetch_hits", 0)} '
              f'/ fp16常驻 {c.get("pinned_hits", 0)}')
        st = getattr(model._fp8_store, 'stats', {}) if hasattr(model, '_fp8_store') else {}
        th, td = st.get('t_h2d', 0.0), st.get('t_dequant', 0.0)
        pf = getattr(model, '_prefetch', None)
        if pf is not None:
            print(f'  预取发起 {pf.n_prefetch} 次（调用侧仅 {pf.t_prefetch:.2f}s，'
                  f'实际拷贝与计算重叠）')
        print(f'  同步搬运耗时 {c.get("transfer_s", 0.0):.1f}s '
              f'(H2D {th:.1f}s + 反量化 {td:.1f}s)')
        # 每 forward 的时间去向（host 侧墙钟）
        n = max(1, n_fwd)
        tdeq, tmnt, tunmnt = c.get('t_deq', 0.0), c.get('t_mount', 0.0), c.get('t_unmount', 0.0)
        tfwd, tiss = c.get('t_fwd', 0.0), c.get('t_issue', 0.0)
        tot = tdeq + tmnt + tunmnt + tfwd + tiss
        if tot > 0:
            print('  每 forward 时间去向:')
            print(f'    反量化(dequant)   {tdeq/n*1000:7.1f} ms  {100*tdeq/tot:5.1f}%')
            print(f'    权重挂载/卸载      {(tmnt+tunmnt)/n*1000:7.1f} ms  {100*(tmnt+tunmnt)/tot:5.1f}%')
            print(f'    层计算(matmul等)   {tfwd/n*1000:7.1f} ms  {100*tfwd/tot:5.1f}%')
            print(f'    预取发起(issue)    {tiss/n*1000:7.1f} ms  {100*tiss/tot:5.1f}%')
            if c.get('t_gemv', 0.0) > 0:
                print(f'    融合GEMV(主机)     {c["t_gemv"]/n*1000:7.1f} ms  '
                      f'{100*c["t_gemv"]/tot:5.1f}%  '
                      f'← 拆 t_fwd：GEMV 发射/分配/隐式同步')
        elif c.get('t_gemv', 0.0) > 0:
            # lite 全驻旁路：无逐层 t_fwd 分账，单独给 GEMV 主机耗时
            print(f'  融合GEMV(主机)     {c["t_gemv"]/n*1000:7.1f} ms/fwd'
                  f'（lite 旁路，层回调已裁撤）')
    # P0-3 测量纪律：单行固化本 run 全部可比性字段（跨天对比必须带上）
    _pf = getattr(model, '_prefetch', None)
    if _pf is not None and getattr(_pf, 'cpy_wins', None):
        _cw, _gw, _nc, _ng = _pf.gpu_window_ms()
        print(f'  GPU流窗口/forward: copy流 {_cw:7.1f} ms (×{_nc})  '
              f'compute流 {_gw:7.1f} ms (×{_ng})  '
              f'← 流上首尾事件，含流内空隙/被卡时间')
    # 逐 forward 主机序列：预取管线（部分驻留）或 lite 全驻旁路（counters['fwd_ms']）
    _fh = (getattr(_pf, 'fwd_host_ms', None) if _pf else None) \
        or (c.get('fwd_ms') if c else None) or []
    if _fh:
        print('  逐forward主机(ms): '
              + ' '.join(f'{x:.0f}' for x in _fh))
    _st = getattr(model, '_fp8_store', None)
    _n_res = len(getattr(_st, 'resident_li', set()) or set())
    _n_lay = len(model.model.layers) if hasattr(model, 'model') else 0
    print(f'[run] resident={_n_res}/{_n_lay} free0={usable_gib:.2f}GiB '
          f'quant={a.quant} '
          f'fused={getattr(model, "_fused_used", False)} '
          f'slots={_pf.slots if _pf else 0} '
          f'issue_at={_pf.issue_at if _pf else "-"} '
          f'inline={getattr(_pf, "inline_copy", False) if _pf else "-"} '
          f'reserve_mib={a.reserve_mib} '
          f'sdma={os.environ.get("HSA_ENABLE_SDMA", "-")} '
          f'serialize={os.environ.get("AMD_SERIALIZE_KERNEL", "-")} '
          f'quiet={os.environ.get("INT8_QUIET", "-")}')
    print('=' * 60)
    for s, cn in results[:6]:
        print(f'  {s[:30]!r} -> {cn[:54]!r}')

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, 'w', encoding='utf-8', newline='\n') as f:
        for s, cn in results:
            f.write(f'{s}\t{cn}\n')
    print(f'已写 {a.out}')
    return 0


def load_meta_fp8(a, cfg, tok, props):
    """用 meta device 建结构，量化存 CPU，不把 fp16 权重留在任何地方。

    量化结果按 `--cache` 走磁盘缓存：命中就 mmap 直接读回，跳过 36s 的量化。
    """
    from transformers import AutoModelForCausalLM

    t = time.time()
    with torch.device('meta'):          # torch 2.x 原生 meta 上下文，无需 accelerate
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)
    print(f'[meta] 空结构 {time.time()-t:.0f}s', flush=True)

    # torch.device('meta') 会把 **buffer**（如 rope 的 inv_freq）也变成 meta；
    # accelerate 的 init_empty_weights 默认 include_buffers=False 不受影响。
    # rope 模块的 __init__ 纯确定性（只依赖 config，无权重读写），原地重跑即可。
    n_buf = 0
    for name, mod in list(model.named_modules()):
        metas = [bn for bn, b in mod.named_buffers(recurse=False)
                 if b.device.type == 'meta']
        if not metas:
            continue
        try:
            mod.__class__.__init__(mod, cfg, device='cuda')
            n_buf += len(metas)
        except Exception as ex:
            raise RuntimeError(f'meta buffer 重建失败: {name} ({metas}): {ex}')
    if n_buf:
        print(f'[meta] 重建 meta buffer {n_buf} 个（rope inv_freq）', flush=True)

    mdir = resolve_model_dir(a.model)
    revision = int8_cache.revision_of(mdir)
    print(f'[meta] 分片索引就绪 {mdir}', flush=True)

    quant = getattr(a, 'quant', 'int4')
    store = Int8Store(model, cfg, device='cpu', pin=a.pin, quant=quant)
    hit = False
    if not a.no_cache:
        if a.rebuild_cache:
            int8_cache.remove(a.model, HERE, quant=quant)
        t = time.time()
        hit = int8_cache.load_into(store, a.model, revision, HERE)
        if hit:
            print(f'[cache] 载入耗时 {time.time()-t:.1f}s', flush=True)
            _probe('after-cache-load')
            store.shell_out()
            _probe('after-shell-out')

    if hit:
        pass
    else:
        t = time.time()
        store.build(shard_reader=make_shard_reader(mdir))
        print(f'[quant] {time.time()-t:.0f}s', flush=True)
        if not a.no_cache:
            t = time.time()
            int8_cache.save(store, a.model, revision, HERE)
            print(f'[cache] 落盘耗时 {time.time()-t:.1f}s', flush=True)

    # 未量化参数（embedding/lm_head/norm）上卡，占固定 ~0.98 GiB
    # 计时：此前这段是 [load] 里 ~65s 无账时段的主体（mmap 缺页/9p 读）
    t_sm = time.time()
    _materialize_small_params(model, a.model)
    print(f'[small] {time.time()-t_sm:.1f}s', flush=True)

    # ---- 分层常驻预算 ----
    # 关键：不能用 props.total_memory（纸面容量 8176 MiB）当预算基准 ——
    # 驱动/上下文先吃掉 ~141 MiB，真实可用只有 ~8035 MiB。
    # 用实测的「当前空闲显存」才准；低估 141 MiB 会让最后一层越过上限，
    # 触发 HIP 超卖（显存走 PCIe 迁移），速度反而从 2.7 掉到 1.7 条/分。
    free_now, usable_total = torch.cuda.mem_get_info()
    used = usable_total - free_now
    reserve = int(a.reserve_mib * 2**20)
    budget = free_now - reserve

    if budget <= 0:
        store.resident_li, store.resident = set(), 0
        store.pinned, store.pinned_li = [], set()
        print(f'[pin] 显存不足（空闲 {free_now/2**20:.0f} MiB - 预留 {reserve/2**20:.0f} '
              f'<= 0），{store.quant} 全留 CPU', flush=True)
    elif a.pin_layers > 0:
        # 用户显式指定：整层 fp16 常驻
        store.pin_by_budget(int(a.pin_layers) * 1024**3, only_mlp=a.pin_mlp)
    elif a.fp8_resident and not a.no_pin:
        # 推荐档位：把能装下的层的 fp8 搬进显存（比 fp16 常驻省一半显存）
        n_layers = len(store._layers())
        per = store.layer_fp8_bytes(0)
        # 非驻层的瞬时开销：预取槽缓冲（每槽一条整层单块）；
        # fp16 挂载路径还要加单层 fp16 峰值
        slots_n = max(2, int(getattr(a, 'prefetch_slots', PREFETCH_SLOTS)))
        fused = HAVE_FUSED and not a.no_fused_gemm
        transient = slots_n * per
        if not fused:
            transient += store.layer_fp16_bytes(0)
        if a.reserve_mib * 2 ** 20 < transient + 64 * 2 ** 20:
            reserve = transient + 64 * 2**20
            budget = free_now - reserve
            print(f'[pin] 自动把预留抬到 {reserve/2**20:.0f} MiB —— '
                  f'预取槽缓冲 + fp16 峰值需要这么多', flush=True)
        k = int(budget // per)
        if k < n_layers and reserve < transient:
            k -= 1
            print(f'[pin] 预留 {reserve/2**20:.0f} MiB < 非驻层瞬时 {transient/2**20:.0f} MiB，'
                  f'主动退一层以规避超卖', flush=True)
        # 计时：驻留上卡（H2D）+ 预测打印 —— 此前与 [small] 同为无账时段
        t_up = time.time()
        if k > 0:
            store.resident_fp8_by_budget(k * per, interleave=a.interleave)
        else:
            store.resident_li, store.resident = set(), 0
            store.nonresident_li = list(range(n_layers))
        store.pinned, store.pinned_li = [], set()
        print(f'[i8pin-up] {time.time()-t_up:.1f}s', flush=True)
        # 预测峰值：常驻 fp8 + 非驻层的瞬时开销，必须 <= 当前真正空闲的显存
        peak_extra = store.resident * per + (transient if k < n_layers else 0)
        print(f'[pin] 预测峰值 {peak_extra/2**20:.0f} MiB / 空闲 {free_now/2**20:.0f} MiB，'
              f'余量 {(free_now-peak_extra)/2**20:.0f} MiB', flush=True)
    else:
        store.resident_li, store.resident = set(), 0
        store.pinned, store.pinned_li = [], set()
        print(f'[pin] 按 --no-pin 运行：{store.quant} 全部留 CPU', flush=True)

    t = time.time()
    materialize_pinned(model, store)

    # ---- 预取管线：只留非驻留层在内存，且放进 pinned 内存 ----
    n_layers = len(store._layers())
    nr = [li for li in range(n_layers)
          if li not in getattr(store, 'resident_li', set())
          and li not in getattr(store, 'pinned_li', set())]
    fused = HAVE_FUSED and not a.no_fused_gemm
    issue_at = getattr(a, 'issue_at', 'auto')
    if issue_at == 'auto':
        # 2026-09-23 终测（预热后 ★424ms）后固化：fused 默认 pre 补发
        # （覆盖上一非驻留层槽 + 逐槽事件，同条件优于 post）；
        # no-fused 反量化即消费、pre 里补发（consume）
        issue_at = 'pre' if fused else 'consume'
    elif not fused:
        issue_at = 'consume'   # no-fused 无「就地读」竞态，pre 补发即 consume
    prefetch = None
    if nr and not a.no_prefetch:
        host_mib = len(nr) * store.layer_fp8_bytes(nr[0]) / 2**20
        prefetch = PrefetchPipeline(
            store, nr, slots=getattr(a, 'prefetch_slots', PREFETCH_SLOTS),
            issue_at=issue_at,
            inline_copy=getattr(a, 'inline_copy', False)).setup()
        store.release_host_for(getattr(store, 'resident_li', set()))
        # 非驻留层已全部进 pinned、常驻层已上 GPU —— mmap 可以整体还掉
        if hit:
            store.drop_big()
        print(f'[prefetch] 宿主侧只保留非驻留层 pinned {host_mib:.0f} MiB '
              f'（{len(nr)} 层），slots={prefetch.slots}，'
              f'issue_at={prefetch.issue_at}，'
              f'inline_copy={prefetch.inline_copy}', flush=True)

    _, counters = install_offload_hooks(model, store, prefetch=prefetch,
                                        fused=fused)
    print(f'[hooks] {time.time()-t:.0f}s', flush=True)

    if fused and HAVE_FUSED:
        # triton 进程级一次性初始化实测 ~6-9.7 s（ROCDXG 首次发射：
        # 运行时/模块装载），落在生成期会把 ★ 均值抬高 ~429 ms/forward
        # （14 个 forward 摊）。这里在 [load] 阶段预热掉（直接调 kernel，
        # 不经过模型 forward → 不触发 hook/预取），计入 [load]、移出计时窗。
        t_w = time.time()
        if store.quant == 'int4':
            # uint32 直序 packed (M, K/8)；K=1024 → 128 字/行。
            # 随机位型即可（仅热身）。rows={1,34} 覆盖 decode/gemm 两种 grid。
            _wr = torch.randint(-2**31, 2**31 - 1, (1024, 128), device='cuda',
                                dtype=torch.int32).view(torch.uint32)
            _ws = torch.ones(1024, 8, device='cuda', dtype=torch.float16)
            _wz = torch.randint(0, 16, (1024, 8), device='cuda', dtype=torch.uint8)
            int4_gemm_rows(_wr, _ws, _wz,
                           torch.randn(1, 1024, device='cuda', dtype=torch.float16))
            int4_gemm_rows(_wr, _ws, _wz,
                           torch.randn(34, 1024, device='cuda', dtype=torch.float16))
            torch.cuda.synchronize()
            del _wr, _ws, _wz
        else:
            _wr = torch.randint(-127, 128, (1024, 1024), device='cuda',
                                dtype=torch.int8)
            _ws = torch.ones(1024, device='cuda', dtype=torch.float16)
            int8_gemm_rows(_wr, _ws,
                           torch.randn(1, 1024, device='cuda', dtype=torch.float16))
            int8_gemm_rows(_wr, _ws,
                           torch.randn(32, 1024, device='cuda', dtype=torch.float16))
            torch.cuda.synchronize()
            del _wr, _ws
        print(f'[warmup] triton 进程初始化已预热 {time.time()-t_w:.1f}s'
              f'（计入 [load]，不进 ★）', flush=True)

    model.eval()
    model._fp8_store = store
    model._offload_counters = counters
    model._prefetch = prefetch              # 供 benchmark 报告用
    model._fused_used = fused               # 供 [run] 汇总行用
    src = '磁盘缓存' if hit else '现场量化'
    tag = ('融合W4A16前向' if store.quant == 'int4' else '融合W8A16前向') \
        if fused else '反量化挂载'
    return model, (f'{store.quant}-on-CPU + PCIe 搬运（{src}，{tag}，'
                   f'常驻 {len(store.resident_li)}/{len(store._layers())} 层 {store.quant}）')


def _materialize_small_params(model, repo_id):
    """把没被量化的参数（norm / embedding / lm_head）真正加载到 GPU。

    embedding 和 lm_head 各约 0.49 GiB，是 8GB 卡上必须保留的固定成本。
    """
    from safetensors import safe_open
    import json

    mdir = resolve_model_dir(repo_id)
    with open(os.path.join(mdir, 'model.safetensors.index.json'), encoding='utf-8') as f:
        idx = json.load(f)['weight_map']

    kept = 0
    kept_b = 0
    _open = {}
    try:
        # remove_duplicate=False：绑定权重（lm_head <-> embed_tokens）必须
        # 两个名字都显式上传 —— 否则 tied 的 lm_head 会一直留在 meta 上，
        # 模型能跑但 logits 全是垃圾（实测：logits 值域恰好 ±127）。
        for name, p in list(model.named_parameters(remove_duplicate=False)):
            if p.device.type != 'meta':
                continue                  # 已有真身（包括空壳 numel==0 的也跳过）
            if name not in idx:
                continue
            # 层内的 Linear 权重已由 store 量化并换成空壳，
            # 但层内的 RMSNorm 权重（1 维）没被量化，必须加载。
            # 判据：空壳是 numel==0，真 meta 才有 numel>0。
            if p.numel() == 0:
                continue
            shard = idx[name]
            fh = _open.get(shard)
            if fh is None:
                fh = safe_open(os.path.join(mdir, shard), framework='pt', device='cpu')
                _open[shard] = fh
            # WSL/ROCDXG 下大张量一次性 .to('cuda') 偶发 hipErrorInvalidValue
            # （转换+搬运合一的路径）；改为 CPU 上转 dtype，再分块 copy_。
            t_cpu = fh.get_tensor(name).to(torch.float16)
            g = torch.empty(t_cpu.shape, device='cuda', dtype=torch.float16)
            flat_s = t_cpu.reshape(-1)
            flat_g = g.reshape(-1)
            CH = 32 * 1024 * 1024                   # 32M 元素 = 64 MiB/块
            for i in range(0, flat_s.numel(), CH):
                flat_g[i:i + CH].copy_(flat_s[i:i + CH])
            t = g
            _set_param(model, name, t)
            kept += 1
            kept_b += t.numel() * 2
        # 硬检查：不允许任何 meta 参数/缓冲残留（有就是加载漏了，宁可炸）
        leftover = [n for n, p in model.named_parameters(remove_duplicate=False)
                    if p.device.type == 'meta' and p.numel() > 0]
        leftover += [n for n, b in model.named_buffers(remove_duplicate=False)
                     if b.device.type == 'meta']
        if leftover:
            raise RuntimeError(f'仍有 {len(leftover)} 个 meta 张量未上卡: '
                               f'{leftover[:6]}')
        # tied 权重合并（2026-09-22）：本模型 tie_word_embeddings=True，但
        # safetensors 的 index 里 embed 与 lm_head **各存一份**（逐位相同已验证），
        # _set_param 按名字分别上传 → 两份 ~0.978 GiB 全占显存，白扔 ≈4.7 个
        # 常驻 int8 层。重新绑定成同一 Parameter，省出的显存直接进常驻预算。
        emb = model.get_input_embeddings()
        lh = model.get_output_embeddings()
        if lh is not None and emb is not None:
            we, wl = emb.weight, lh.weight
            if we is not wl and we.numel() > 0 and we.device.type != 'meta':
                if wl.device.type == 'meta' or (
                        wl.shape == we.shape and torch.equal(wl, we)):
                    freed = (0 if wl.device.type == 'meta'
                             else wl.numel() * wl.element_size())
                    lh.weight = we
                    if freed:
                        kept_b -= freed      # 修正外层统计
                        kept -= 1
                        # 归还驱动：释放的副本先进 PyTorch 缓存池，不还驱动的话
                        # 后面 mem_get_info() 预算看不见这 ~1 GiB（实测常驻层数
                        # 反而少 1 层）。empty_cache 只清未使用块，正在用的不受影响。
                        torch.cuda.empty_cache()
                        print(f'[gpu] tie 合并：释放重复 lm_head 副本 '
                              f'{freed/2**30:.3f} GiB（已归还驱动）', flush=True)
                else:
                    print('[gpu] 警告：lm_head 与 embed 内容不同，保留两份',
                          flush=True)
    finally:
        _open.clear()
        gc.collect()
    print(f'[gpu] 常驻未量化参数 {kept} 个 / {kept_b/2**30:.3f} GiB', flush=True)


def _set_param(model, name, tensor):
    """把参数替换成给定张量。

    不能写 `.data = tensor`：meta 参数与目标张量「tensor type 不兼容」，
    torch 会抛 set_data 错误。直接用 setattr 换掉整个 Parameter。
    """
    parts = name.split('.')
    parent = _resolve(model, '.'.join(parts[:-1])) if len(parts) > 1 else model
    setattr(parent, parts[-1], nn.Parameter(tensor, requires_grad=False))


if __name__ == '__main__':
    sys.exit(main())
