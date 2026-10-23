# -*- coding: utf-8 -*-
"""W8A16 / W4A16 融合 GEMV/GEMM（triton，RDNA2 可用 —— 不用 tl.dot，纯 SIMT FMA）。

int8：y = (W_int8 ⊙ s_row) @ x
- 权重只读一遍 int8（fp16 的一半流量），反量化在寄存器内完成；
- 无中间 fp16 权重的「写+读」（现两段式反量化多花 2.4x 显存流量）；
- fp32 累加，行末乘 per-row scale，精度无损（实测核误差 ~1e-7）。

int4 g128（默认主路径）：按组约减（参考 AWQ g128 kernel 的做法）
    Σ_n x_n·(q_{m,n} − z_{m,g})·s_{m,g}
        = s_{m,g} · ( Σ_{n∈g} x_n·q_{m,n} − z_{m,g} · Σ_{n∈g} x_n )
- zp/scale 从「每权重一次」降到「每组一次」：(BLOCK_M,BLOCK_N) 的逐列 gather
  消掉，只 load (BLOCK_M, BLOCK_G) 组表；
- 组内 Σx 由 Python 侧预算（xsum (rows, G)），kernel 直接查表；
- 权重流量 0.5 B/param，配合全层显存常驻消灭 PCIe 层搬运。
  逐列 gather 版实测 decode 8.2ms/层 → 本版 0.71ms/层（×32=23ms，反超 int8 38ms）。

decode (rows=1) 时是纯带宽问题：int8 实测 144-213 GB/s，逼近硬件上限。
"""
import torch
import triton
import triton.language as tl

# 超过这个行数（prompt 处理）就退回「反量化 + rocBLAS GEMM」：
# SIMT GEMM 在大 rows 下不如 rocBLAS 的优化 GEMM。
INT8_GEMM_MAX_ROWS = 256
INT4_GEMM_MAX_ROWS = 256
INT4_GROUP = 128
# BLOCK 配置按实测扫描固定（RX 6800S / gfx1030，rows=1 与 rows=34 扫描均为该组最优）：
#   小 BLOCK_N + 少 warps（AWQ g128 经验在本卡成立）；BN=256=2 组，奇偶/组界天然对齐
I4_GEMV_BLOCK_M, I4_GEMV_BLOCK_N, I4_GEMV_WARPS = 8, 256, 1
I4_GEMM_BLOCK_M, I4_GEMM_BLOCK_N, I4_GEMM_WARPS = 8, 256, 1


@triton.jit
def _gemv_i8(w_ptr, s_ptr, x_ptr, y_ptr, M, N,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """y[m] = s[m] * Σ_n W[m,n]*x[n]。grid=(cdiv(M,BLOCK_M),)，纵向扫 N。"""
    pid = tl.program_id(0)
    rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mm = rm < M
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        rn = n0 + tl.arange(0, BLOCK_N)
        w = tl.load(w_ptr + rm[:, None] * N + rn[None, :],
                    mask=mm[:, None], other=0).to(tl.float32)
        x = tl.load(x_ptr + rn).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    s = tl.load(s_ptr + rm, mask=mm, other=0.0).to(tl.float32)
    tl.store(y_ptr + rm, (acc * s).to(tl.float16), mask=mm)


@triton.jit
def _gemm_i8(w_ptr, s_ptr, x_ptr, y_ptr, M, N,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr):
    """Y: (R, M) = (W*s) @ X^T，X: (R, N)。grid=(cdiv(M,BM), R)。"""
    pid_m = tl.program_id(0)
    r = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mm = rm < M
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        rn = n0 + tl.arange(0, BLOCK_N)
        w = tl.load(w_ptr + rm[:, None] * N + rn[None, :],
                    mask=mm[:, None], other=0).to(tl.float32)
        x = tl.load(x_ptr + r * N + rn).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    s = tl.load(s_ptr + rm, mask=mm, other=0.0).to(tl.float32)
    tl.store(y_ptr + r * M + rm, (acc * s).to(tl.float16), mask=mm)


# ---- int4 g128 非对称：按组约减（uint32 直序打包；s/zp 为 (M, K/128) 组表）----

@triton.jit
def _gemv_i4(w_ptr, s_ptr, z_ptr, xs_ptr, x_ptr, y_ptr, M, N, G,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
             BLOCK_G: tl.constexpr):
    """packed W4 GEMV（按组约减）。grid=(cdiv(M,BLOCK_M),)。

    xs_ptr: (G,) 组内 Σx（Python 预算）。BLOCK_N = BLOCK_G * 128。
    权重按 uint32 直序加载：每行每块只 load BLOCK_N/8 个 dword（÷8 指令），
    shift 表展开 → (BM, BN/8, 8) → reshape (BM, BN)；负 word 算术右移
    & 0xF 取位仍正确（已全位验证）。
    """
    pid = tl.program_id(0)
    rm = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mm = rm < M
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for g0 in range(0, G, BLOCK_G):
        rn = g0 * 128 + tl.arange(0, BLOCK_N)          # 块内列号（组界对齐）
        wr = (g0 * 16) + tl.arange(0, BLOCK_N >> 3)     # 对应 dword 号（128列=16字）
        b = tl.load(w_ptr + rm[:, None] * (N >> 3) + wr[None, :],
                    mask=mm[:, None], other=0)          # (BM, BN/8) uint32
        sh = tl.arange(0, 8) * 4
        nib = (b.to(tl.int32)[:, :, None] >> sh[None, None, :]) & 15
        q = nib.reshape(BLOCK_M, BLOCK_N).to(tl.float32)
        x = tl.load(x_ptr + rn).to(tl.float32)
        # 每组约减：Σ_{n∈g} x·q → (BLOCK_M, BLOCK_G)
        contrib = tl.sum(q.reshape(BLOCK_M, BLOCK_G, 128)
                         * x.reshape(BLOCK_G, 128)[None, :, :], axis=2)
        gsel = g0 + tl.arange(0, BLOCK_G)
        z = tl.load(z_ptr + rm[:, None] * G + gsel[None, :],
                    mask=mm[:, None], other=0).to(tl.float32)
        sf = tl.load(s_ptr + rm[:, None] * G + gsel[None, :],
                     mask=mm[:, None], other=0.0).to(tl.float32)
        xsg = tl.load(xs_ptr + gsel)[None, :]
        acc += tl.sum((contrib - z * xsg) * sf, axis=1)
    tl.store(y_ptr + rm, acc.to(tl.float16), mask=mm)


@triton.jit
def _gemm_i4(w_ptr, s_ptr, z_ptr, xs_ptr, x_ptr, y_ptr, M, N, G,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
             BLOCK_G: tl.constexpr):
    """packed W4 GEMM（按组约减）：Y (R, M) = dequant(W) @ X^T。

    xs_ptr: (R, G) 每输入行的组内 Σx。解包同 _gemv_i4（uint32 直序）。
    """
    pid_m = tl.program_id(0)
    r = tl.program_id(1)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mm = rm < M
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for g0 in range(0, G, BLOCK_G):
        rn = g0 * 128 + tl.arange(0, BLOCK_N)
        wr = (g0 * 16) + tl.arange(0, BLOCK_N >> 3)
        b = tl.load(w_ptr + rm[:, None] * (N >> 3) + wr[None, :],
                    mask=mm[:, None], other=0)
        sh = tl.arange(0, 8) * 4
        nib = (b.to(tl.int32)[:, :, None] >> sh[None, None, :]) & 15
        q = nib.reshape(BLOCK_M, BLOCK_N).to(tl.float32)
        x = tl.load(x_ptr + r * N + rn).to(tl.float32)
        contrib = tl.sum(q.reshape(BLOCK_M, BLOCK_G, 128)
                         * x.reshape(BLOCK_G, 128)[None, :, :], axis=2)
        gsel = g0 + tl.arange(0, BLOCK_G)
        z = tl.load(z_ptr + rm[:, None] * G + gsel[None, :],
                    mask=mm[:, None], other=0).to(tl.float32)
        sf = tl.load(s_ptr + rm[:, None] * G + gsel[None, :],
                     mask=mm[:, None], other=0.0).to(tl.float32)
        xsg = tl.load(xs_ptr + r * G + gsel)[None, :]
        acc += tl.sum((contrib - z * xsg) * sf, axis=1)
    tl.store(y_ptr + r * M + rm, acc.to(tl.float16), mask=mm)


def int8_gemm_rows(wq, s, x):
    """Linear 前向：x 任意形状 (..., K) fp16 -> (..., M) fp16。

    wq: (M, K) int8（必须连续），s: (M,) fp16（GPU）。
    """
    K = x.shape[-1]
    rows = x.numel() // K
    x2 = x.contiguous().reshape(rows, K)
    M = wq.shape[0]
    if not (wq.stride(1) == 1 and wq.stride(0) == K):
        wq = wq.contiguous()
    y = torch.empty(rows, M, device=wq.device, dtype=torch.float16)
    if rows == 1:
        BLOCK_M = 16
        BLOCK_N = 1024
        assert K % BLOCK_N == 0, f'K={K} 不是 {BLOCK_N} 的倍数'
        _gemv_i8[(triton.cdiv(M, BLOCK_M),)](
            wq, s, x2[0], y[0], M, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    elif rows <= INT8_GEMM_MAX_ROWS:
        BLOCK_M = 16
        BLOCK_N = 1024
        assert K % BLOCK_N == 0, f'K={K} 不是 {BLOCK_N} 的倍数'
        _gemm_i8[(triton.cdiv(M, BLOCK_M), rows)](
            wq, s, x2, y, M, K, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N)
    else:
        # prompt 很长：退回反量化 + rocBLAS（大 GEMM 上 rocBLAS 更快）
        from int8_quant import dequantize_per_row_int8 as _dq
        wf = _dq(wq, s, torch.float16)
        y = (x2 @ wf.T).to(torch.float16)
    return y.reshape(*x.shape[:-1], M)


def int4_gemm_rows(wq, s, z, x):
    """int4 g128 非对称 Linear 前向：x 任意形状 (..., K) fp16 -> (..., M) fp16。

    wq: (M, K/8) uint32 packed（直序），s: (M, K/128) fp16，z: (M, K/128) uint8
    zero-point（均在 GPU、行连续）。组内 Σx 在此预算后传入 kernel。
    """
    K = x.shape[-1]
    rows = x.numel() // K
    x2 = x.contiguous().reshape(rows, K)
    M = wq.shape[0]
    G = K // INT4_GROUP
    assert K % 8 == 0, f'K={K} 不是 8 的倍数（uint32 直序要求）'
    if not (wq.stride(1) == 1 and wq.stride(0) == K >> 3):
        wq = wq.contiguous()
    if not (s.stride(1) == 1 and s.stride(0) == G):
        s = s.contiguous()
    if not (z.stride(1) == 1 and z.stride(0) == G):
        z = z.contiguous()
    if rows > INT4_GEMM_MAX_ROWS:
        # 超长 prompt：退回反量化 + rocBLAS（大 GEMM 上 rocBLAS 更快）
        from int8_quant import dequantize_int4_g128 as _dq
        wf = _dq(wq, s, z, torch.float16)
        return (x2 @ wf.T).to(torch.float16).reshape(*x.shape[:-1], M)
    # 组内 Σx 预算：dtype=fp32 直接以 fp32 累加，免去整条 x 的 fp16→fp32 物化
    xsum = x2.reshape(rows, G, INT4_GROUP).sum(dim=2, dtype=torch.float32).contiguous()
    y = torch.empty(rows, M, device=wq.device, dtype=torch.float16)
    if rows == 1:
        BLOCK_M = I4_GEMV_BLOCK_M
        BLOCK_N = I4_GEMV_BLOCK_N
        BLOCK_G = BLOCK_N >> 7
        assert K % BLOCK_N == 0 and G % BLOCK_G == 0, \
            f'K={K} G={G} 与 BLOCK_N={BLOCK_N} 不兼容'
        _gemv_i4[(triton.cdiv(M, BLOCK_M),)](
            wq, s, z, xsum[0], x2[0], y[0], M, K, G,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_G=BLOCK_G,
            num_warps=I4_GEMV_WARPS)
    else:
        BLOCK_M = I4_GEMM_BLOCK_M
        BLOCK_N = I4_GEMM_BLOCK_N
        BLOCK_G = BLOCK_N >> 7
        assert K % BLOCK_N == 0 and G % BLOCK_G == 0, \
            f'K={K} G={G} 与 BLOCK_N={BLOCK_N} 不兼容'
        _gemm_i4[(triton.cdiv(M, BLOCK_M), rows)](
            wq, s, z, xsum, x2, y, M, K, G,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_G=BLOCK_G,
            num_warps=I4_GEMM_WARPS)
    return y.reshape(*x.shape[:-1], M)
