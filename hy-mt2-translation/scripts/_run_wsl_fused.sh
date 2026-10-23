#!/bin/bash
# int4-g128-fused 快线（WSL2 + ROCDXG + triton；2026-09-23 ★中位257ms·5次133~308，32/32 全驻零搬运）
# 环境：venv 在 ext4 的 ~/.local/share/hy-mt2/.venv（setup_env.py 用 uv 幂等重建）
# 前置（librocdxg / 五个 HSA 变量）见 references/rocm-setup.md
PY=$HOME/.local/share/hy-mt2/.venv/bin/python
export HSA_ENABLE_DXG_DETECTION=1
export HSA_OVERRIDE_GFX_VERSION=10.3.0
export HSA_ENABLE_SDMA=0
export HSA_ENABLE_PEER_SDMA=0
export LIBROCDXG_ALLOC_USER_QUEUE_FROM_UMD=1
export HF_HOME=$HOME/.cache/huggingface
# 本地已有全量权重缓存：离线模式省 ~8s 联网段并消掉 unauthenticated 警告
export HF_HUB_OFFLINE=1
# 持久缓存（ext4；勿用 /tmp：WSL 停机清零 → 每次冷编译 ≈ ★+430ms）
export TRITON_CACHE_DIR=$HOME/.cache/triton
export PYTHONUNBUFFERED=1
cd "$(dirname "${BASH_SOURCE[0]}")" || exit 1
# 配置：quant=int4 + issue_at=pre + slots3 + reserve1000（triton 预热计入 [load]）
timeout 900 "$PY" int8_pipe.py --quant int4 --greedy --limit 2 --max-new-tokens 32 --batch 1 \
  --issue-at pre --prefetch-slots 3 --reserve-mib 1000 \
  --out ../../out/02-int4-wsl-fused.tsv 2>&1
echo "exit=$?"
