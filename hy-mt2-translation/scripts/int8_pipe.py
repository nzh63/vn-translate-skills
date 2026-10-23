"""量化权重路径入口（WSL2；2026-09-23 起 int4 主路径 + int8 回退）。

实测（贪心同机受控 limit2/32，2026-09-23 终测；★ 报 median of ≥3）：
  int4-g128-fused（默认，32/32 全驻零搬运） :  257 ms/forward（5 次中位 133~308）← 推荐
  int8-fused（--quant int8 回退）           :  496~508 ms/forward
  int8 no-fused（历史对照）                 : ~884 ms/forward

批量入口 `translate_tsv.py` 走**同一套量化管线**（带断点续跑/术语/失败重译）。

- 默认 int4 group128 非对称：0.523 B/param、相对误差 ~0.10，字节减半使
  32/32 层全常驻（int4 ~109 MiB/层）、每 token PCIe 搬运 = 0；
- int8 回退 per-row 对称：0.0102、1 B/param（非驻层仍需搬运 ~208 MiB/层）；
- tie 权重（embed↔lm_head）合并后未量化常驻仅 0.98 GiB；
- 磁盘缓存在 ext4 `~/.cache/hy-mt2-quant`（scheme `int4-g128-v2` /
  `per-row-int8-v1`），命中 `[load] 22s`（int4），首跑量化另计。

实现全在 offload_int8.py（Int8Store + PrefetchPipeline 单块预取 + 磁盘缓存）；
本文件是薄入口，**先 `torch.cuda.init()` 再 import**（ROCDXG 初始化坑）。

用法（WSL 内；HSA 环境变量见 references/rocm-setup.md）：
  python int8_pipe.py --greedy --limit 2 --max-new-tokens 32     # 默认 int4
  bash _run_wsl_fused.sh                 # 一键：env + 胜出参数（reserve 1000）
  python int8_pipe.py --cache-info / --rebuild-cache / --dry-run

注意：
- venv/HF/量化/triton 缓存全部在 ext4（/mnt/c 是 9p，冷启动慢一个量级）。
- `--reserve-mib` int4 全驻 1000 即可；int8 回退 1000 最优（2200 → 更厚余量）。
- 融合前向需要 triton；`TRITON_CACHE_DIR` 必须持久路径（勿用 /tmp，WSL 停机
  清零 → 每次冷编译 ★ 虚高）。triton 进程初始化已在 [load] 内预热
  （warmup 块），勿删。
- 勿设 `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`（当前驱动下任何
  设备分配都会 hipErrorInvalidValue）。
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# WSL/ROCDXG 的坑：offload_int8 的导入链（连带 triton）若早于 CUDA 初始化，
# 之后的大显存分配会稳定报 hipErrorInvalidValue。先初始化再 import。
import torch                                              # noqa: E402
torch.cuda.init()                                         # noqa: E402

from offload_int8 import main                              # noqa: E402

if __name__ == '__main__':
    sys.exit(main())
