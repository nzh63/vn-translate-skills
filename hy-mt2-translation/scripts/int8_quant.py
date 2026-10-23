# -*- coding: utf-8 -*-
"""INT8 对称逐行量化 + INT4 g128 非对称量化原语（默认主路径 int4）。

量化方案
--------
- int4 g128（默认）：沿 K 每 128 列一组，非对称 zero-point：
    s = (amax - amin) / 15，zp = clamp(round(-amin / s), 0, 15)
    q = clamp(round(w / s) + zp, 0, 15)（uint8 值 0..15），打包成 **uint32 直序**
    （straight order：列 k 的 nibble 位于 word >> ((k%8)*4) & 0xF；即一个 dword
    装 8 个连续列，kernel 加载指令数 ÷8）。小端下该字节流与旧 uint8 打包
    （偶数列低 4 位）**逐位相同**，旧 v1 缓存可无损重解释——但线格式已升
    scheme `int4-g128-v2`，load 拒绝旧 scheme 防混读。
    约 0.523 B/param，7B ≈3.40 GiB（int8 是 6.50 GiB）。
    真实权重实测相对 Frobenius ~0.10（int8 per-row ~0.011，int4 对称 g128 ~0.12）。
    注：规格稿写 zp=round(amin/s)，在 amin<0 时恒为 0、负权重全被截到 0
    （实测 rel ~0.71）；与离线标定 0.102 对得上的是 round(-amin/s)，按后者实现。
- int8（回退）：对称均匀量化 q = round(w / s)，s = amax(row) / 127，q ∈ [-127, 127]。

与 fp8 e4m3 的本质区别
----------------------
- fp8 e4m3：浮点编码，3 位尾数，**相对误差恒定 ~2^-4**（实测均值 2.64%），
  无论权重多大，相对精度一样差；
- int8：均匀台阶，**绝对误差恒定 = s/2**。一行里大权重的相对误差极小
  （~0.4%），小权重相对误差大，但小权重对输出贡献也小。
  对高斯分布的权重，理论 RMS 相对误差 ≈ 1/(127·√3)·(σ_row/amax_row)·...
  实测见 bench_accuracy.py——预期比 fp8 好 4~6 倍。

反量化：q.to(fp16) * scale —— int8→fp16 是原生 cast，不存在 fp8 的
「类型提升不支持」问题，且 cast 路径与 fp8 相同（两段式 kernel），
所以搬运/反量化成本与 fp8 完全一致。
"""
import torch

INT8_MAX = 127.0


def quantize_per_row_int8(w):
    """fp16/bf16 权重 -> (int8, per-row fp16 scale)。

    scale 取该行 max_abs / 127，保证 ±amax 都落在 [-127,127]。
    整行全零时 scale 置 1，避免除零。
    """
    w = w.detach().float()
    amax = w.abs().amax(dim=1, keepdim=True)
    scale = (amax / INT8_MAX).clamp(min=1e-12)
    q = torch.round(w / scale).clamp(-INT8_MAX, INT8_MAX)
    return q.to(torch.int8), scale.squeeze(1).to(torch.float16)


def dequantize_per_row_int8(q, scale, dtype=torch.float16):
    """int8 + per-row scale -> 原精度张量（默认 fp16）。"""
    return q.to(dtype) * scale.to(dtype).unsqueeze(1)


# ================================================================ int4 g128 非对称

INT4_GROUP = 128


def quantize_int4_g128(w):
    """权重 (M, K) -> (packed uint32 (M, K/8), scale fp16 (M, G), zp uint8 (M, G))。

    沿最后一维 K 分组，group=128（K % 8 == 0 必须成立，group128 天然覆盖）：
      s  = (amax - amin) / 15            （每组，clamp min=1e-12）
      zp = clamp(round(-amin / s), 0, 15)（每组 int zero-point）
      q  = clamp(round(w / s) + zp, 0, 15)（uint8 值 0..15）

    打包 **uint32 直序**：列 k 的 nibble 位于 word >> ((k%8)*4) & 0xF。
    实现先按字节流（偶数列低 4 位）再小端重解释为 uint32——两者逐位等价，
    且字节流与 scheme v1 完全相同（可无损对照）。K%8=0 保证行内字节数
    是 4 的倍数，uint8→uint32 view 对齐成立。
    行内打包，按行切片（fused-Mlp 的 [:inter] 拆分）依然安全。
    """
    w = w.detach().float()
    M, K = w.shape
    if K % INT4_GROUP:
        raise ValueError(f'K={K} 不是 int4 group{INT4_GROUP} 的倍数')
    if K % 8:
        raise ValueError(f'K={K} 不是 8 的倍数（uint32 直序要求）')
    G = K // INT4_GROUP
    wg = w.view(M, G, INT4_GROUP)
    amin = wg.amin(dim=2)
    amax = wg.amax(dim=2)
    s = ((amax - amin) / 15.0).clamp(min=1e-12)          # (M, G)
    zp = torch.round(-amin / s).clamp(0.0, 15.0)          # (M, G)
    q = torch.round(wg / s.unsqueeze(2)) + zp.unsqueeze(2)
    q = q.clamp(0.0, 15.0).to(torch.uint8).reshape(M, K)
    # 字节流（偶列低 nibble）→ 小端 uint32 = 直序 word（列 8j+i 在 bit i*4）
    packed8 = torch.bitwise_or(q[:, 0::2],
                               torch.bitwise_left_shift(q[:, 1::2], 4))
    packed = packed8.contiguous().reshape(-1).view(torch.uint32).reshape(M, K // 8)
    return packed.contiguous(), s.to(torch.float16), zp.to(torch.uint8)


def dequantize_int4_g128(packed, scale, zp, dtype=torch.float16):
    """packed uint32 (M, K/8) + scale (M, G) + zp (M, G) -> 原精度张量 (M, K)。

    直序解包：列 k = (word[k>>3] >> ((k&7)*4)) & 15；反量化 (q - zp) * s。
    word 先转 int64 再算术右移——uint32 值域非负，位模式无损。
    """
    M, W = packed.shape
    K = W * 8
    if K % INT4_GROUP:
        raise ValueError(f'packed 宽 {W} 对应 K={K} 不是 {INT4_GROUP} 的倍数')
    b = packed.to(torch.int64)
    sh = torch.arange(0, 8, device=packed.device, dtype=torch.int64) * 4
    q = ((b[:, :, None] >> sh) & 15).to(torch.uint8).reshape(M, K)
    G = K // INT4_GROUP
    out = (q.reshape(M, G, INT4_GROUP).to(torch.float32)
           - zp.to(torch.float32).unsqueeze(2)) * scale.to(torch.float32).unsqueeze(2)
    return out.reshape(M, K).to(dtype)
