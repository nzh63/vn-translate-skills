# -*- coding: utf-8 -*-
"""隔离校验预取缓冲的内容是否正确。

为什么必须单独测
----------------
整链路跑出来的乱码有两种可能：权重错了，或者别的地方错了。
这个脚本把预取管线单独拎出来，在**不跑模型**的情况下，逐层比对：

    prefetch 缓冲里的 fp8  ==  pinned 源里的 fp8   （逐位）

并连续模拟多轮（reset → 逐层 fetch），专门抓「槽位复用/陈旧数据」类 bug。

用法
----
    python verify_prefetch.py
"""
import os
import sys

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
# expandable_segments 已停用（2026-09-22 环境回归，见 offload_int8.py 注释）

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
# WSL/ROCDXG 坑（与 int8_pipe.py 相同）：offload_int8 的导入链（triton）
# 必须晚于 CUDA 初始化，否则首次大分配稳定 hipErrorInvalidValue。
torch.cuda.init()                                         # noqa: E402
import int8_cache  # noqa: E402
from offload_int8 import PrefetchPipeline, Int8Store, resolve_model_dir  # noqa: E402

MODEL = 'tencent/Hy-MT2-7B'
ROUNDS = 3


def main():
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
    # 与主管线一致：torch 原生 meta 上下文，不依赖 accelerate
    with torch.device('meta'):
        model = AutoModelForCausalLM.from_config(cfg, trust_remote_code=True)

    revision = int8_cache.revision_of(resolve_model_dir(MODEL))
    store = Int8Store(model, cfg, verbose=False)
    if not int8_cache.load_into(store, MODEL, revision, HERE, verbose=True):
        raise SystemExit('缓存未命中')
    store.shell_out()

    n_layers = len(store._layers())
    # 模拟「16 层常驻 / 16 层非驻留」
    resident = set(range(16))
    nr = [li for li in range(n_layers) if li not in resident]
    print(f'层数 {n_layers}，设常驻 {len(resident)}，非驻留 {len(nr)}: {nr}')

    # 用法：python verify_prefetch.py [post|pre|consume]（默认 post）
    mode = sys.argv[1] if len(sys.argv) > 1 else 'post'
    pf = PrefetchPipeline(store, nr, issue_at=mode).setup()
    print(f'issue_at={mode}，slots={pf.slots}，'
          f'单块 {pf.layer_bytes()/2**20:.1f} MiB/层')

    # 真值：非驻留层的 fp8 在 pinned 单块里，按布局 offset 切出来
    def truth(li):
        blk = pf.pin_blocks[li]
        out = {}
        for suffix, f, numel, shape, off in pf._plan_layout():
            out.setdefault(pf._key(li, suffix), {})[f] = blk[off:off + numel]
        return out

    bad = 0
    checked = 0
    for r in range(ROUNDS):
        pf.reset()
        for li in nr:
            bufs = pf.fetch(li)
            t = truth(li)
            for suffix, f, numel, shape, off in pf._plan_layout():
                key = pf._key(li, suffix)
                # 两边都压成一维再比：buffers 是二维视图，pinned 源是一维，
                # torch.equal 会连形状一起比，不压平会全部误报为不符
                got = bufs[key][f].reshape(-1).view(torch.uint8).cpu()
                want = t[key][f][:numel].cpu()
                checked += 1
                if got.shape != want.shape or not torch.equal(got, want):
                    bad += 1
                    if bad <= 6:
                        print(f'  ✗ 第{r+1}轮 layer {li} {suffix}.{f}: '
                              f'形状 {tuple(got.shape)} vs {tuple(want.shape)}，'
                              f'不符 {int((got != want).sum())} 字节')
            pf._issue_next()          # 模拟前向里紧接着发起下一层

    # ---- 第二轮校验：反量化后的 fp16 是否逐位一致（含 scale）----
    print()
    print('=== 反量化结果比对（含 scale）===')
    from fp8_layerwise import dequantize_per_row
    pf.reset()
    n_dq = 0
    bad_dq = 0
    max_diff = 0.0
    for li in nr:
        got = store_dq = store.layer_dequant_from_buffers(li, pf.fetch(li))
        for k in store._layer_keys(li):
            e = store.weights[k]
            shape = tuple(e['shape'])
            # 从 pinned 单块按布局偏移取出该权重当真值
            pre_l = f'model.layers.{li}.'
            suffix = k[len(pre_l):] if k.startswith(pre_l) else k
            off = next(o for s, f, n, sh, o in pf._plan_layout()
                       if s == suffix and f == 'q')
            q = pf.pin_blocks[li][off:off + shape[0] * shape[1]]
            q = q.view(torch.int8).view(shape).to('cuda')
            want = dequantize_per_row(q, e['s'], torch.float16)
            n_dq += 1
            same = torch.equal(got[k], want)
            if not same:
                bad_dq += 1
                max_diff = max(max_diff, float((got[k].float() - want.float()).abs().max()))
                if bad_dq <= 4:
                    print(f'  ✗ layer {li} {k.split(".", 2)[-1]}: 反量化结果不同，'
                          f'最大偏差 {float((got[k].float()-want.float()).abs().max()):.3e}')
        pf._issue_next()
    print(f'比对 {n_dq} 个张量，不同 {bad_dq} 个，最大偏差 {max_diff:.3e}')
    if bad_dq:
        bad += bad_dq
    print()
    print('=' * 56)
    print(f'校验 {checked} 个张量，不符 {bad} 个')
    print('=' * 56)
    print('结论：' + ('缓冲内容全部正确 ✓' if bad == 0 else '存在错误 ✗'))
    return 0 if bad == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
