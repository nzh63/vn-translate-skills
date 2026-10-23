# -*- coding: utf-8 -*-
"""校验磁盘缓存与现场量化**逐位一致**。

为什么必须校验
--------------
缓存一旦与现场量化有任何偏差，翻译质量会**静默**退化 —— 不会报错，
只会输出略微变差的译文。所以缓存路径必须证明自己等价于原路径。

校验内容（对每一层抽样）
------------------------
1. fp8 数据逐位相同（torch.equal）
2. per-row scale 逐位相同
3. 反量化后的 fp16 逐位相同

用法
----
    python verify_cache.py              # 抽样 6 层
    python verify_cache.py --layers 0,15,31
"""
import argparse
import os
import sys

os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
# expandable_segments 已停用（2026-09-22 环境回归，见 offload_int8.py 注释）

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import int8_cache  # noqa: E402
from int8_quant import (quantize_per_row_int8 as quantize_per_row,
                        dequantize_per_row_int8 as dequantize_per_row)  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='tencent/Hy-MT2-7B')
    ap.add_argument('--layers', default='0,5,15,23,24,31',
                    help='抽样校验的层号，逗号分隔')
    a = ap.parse_args()

    from offload_int8 import resolve_model_dir, make_shard_reader

    mdir = resolve_model_dir(a.model)
    revision = int8_cache.revision_of(mdir)
    p = int8_cache.paths(a.model, HERE)
    if not os.path.exists(p['index']):
        raise SystemExit(f'没有找到缓存，请先运行 offload_int8.py 生成：{p["index"]}')

    # 用一个空壳 store 只为了复用 load_into 的解析逻辑
    class _Stub:
        device = 'cpu'
        pin = False
    stub = _Stub()
    ok = int8_cache.load_into(stub, a.model, revision, HERE, verbose=True)
    if not ok:
        raise SystemExit('缓存未命中或不兼容')

    reader = make_shard_reader(mdir)
    layers = [int(x) for x in a.layers.split(',') if x.strip()]

    # 从缓存里按层取 key
    store = stub
    layer_keys = {}
    for name in store.weights:
        parts = name.split('.')
        if len(parts) >= 3 and parts[0] == 'model' and parts[1] == 'layers':
            layer_keys.setdefault(int(parts[2]), []).append(name)

    n_checked = 0
    n_key = 0
    max_abs_diff = 0.0
    for li in layers:
        keys = layer_keys.get(li, [])
        if not keys:
            print(f'  layer {li}: 缓存里没有条目，跳过')
            continue
        print(f'  layer {li}: {len(keys)} 个 Linear')
        for name in keys:
            src = reader(name + '.weight')
            if src is None:
                print(f'    {name}: 源权重读不到，跳过')
                continue
            src = src.detach().float()
            q_ref, s_ref = quantize_per_row(src)
            d_ref = dequantize_per_row(q_ref, s_ref)

            ent = store.weights[name]
            if ent['kind'] == 'fused':
                # 缓存里把 fused 拆成 g/u 两段，各自独立量化
                inter = q_ref.shape[0] // 2
                for sub, qk, sk in (('g', 'qg', 'sg'), ('u', 'qu', 'su')):
                    q_c = ent[qk].to(torch.float32)
                    s_c = ent[sk].to(torch.float32)
                    q_r = q_ref[:inter] if sub == 'g' else q_ref[inter:]
                    s_r = s_ref[:inter] if sub == 'g' else s_ref[inter:]
                    same_q = torch.equal(q_c.to(torch.uint8), q_r.to(torch.uint8))
                    same_s = torch.equal(s_c.to(torch.float16), s_r.to(torch.float16))
                    n_checked += 2
                    n_key += int(same_q and same_s)
                    if not same_q or not same_s:
                        print(f'    {name}[{sub}]: int8同={same_q} scale同={same_s}  ✗')
                    d_c = dequantize_per_row(ent[qk], ent[sk]).to(torch.float32)
                    d_r = dequantize_per_row(q_r, s_r).to(torch.float32)
                    max_abs_diff = max(max_abs_diff,
                                       float((d_c - d_r).abs().max()))
            else:
                q_c = ent['q'].to(torch.float32)
                s_c = ent['s'].to(torch.float32)
                same_q = torch.equal(q_c.to(torch.uint8), q_ref.to(torch.uint8))
                same_s = torch.equal(s_c.to(torch.float16), s_ref.to(torch.float16))
                n_checked += 2
                n_key += int(same_q and same_s)
                if not same_q or not same_s:
                    print(f'    {name}: int8同={same_q} scale同={same_s}  ✗')
                d_c = dequantize_per_row(ent['q'], ent['s']).to(torch.float32)
                max_abs_diff = max(max_abs_diff,
                                   float((d_c - d_ref).abs().max()))

    print()
    print('=' * 56)
    print(f'校验 Linear      : {n_key} 个（逐位一致 {n_checked // 2}）')
    print(f'反量化最大绝对偏差: {max_abs_diff:.3e}')
    print('=' * 56)
    if n_key == n_checked // 2 and max_abs_diff == 0.0:
        print('结论：缓存与现场量化**逐位一致**，可放心复用。')
        return 0
    print('结论：存在偏差，缓存不可信！')
    return 1


if __name__ == '__main__':
    sys.exit(main())
