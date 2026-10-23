---
name: hy-mt2-translation
description: 用本地部署的腾讯混元 Hy-MT2 翻译模型（1.8B/7B/30B-A3B）批量机翻 Galgame 译文 TSV，替代容易"只思考不干活"的通用 LLM subagent 翻译。负责 AMD ROCm WSL2（RDNA2/gfx103x）环境搭建、行内标签 [r]/[p] 与字面 \n 的占位保护、批量分隔符指令、断点续跑与结构校验。凡涉及"本地模型翻译 TSV""机翻初稿""Hy-MT2 / Hunyuan-MT""别再让 subagent 翻译了"等，即使没明说 Hy-MT2 也应使用本技能。
---

# 本地 Hy-MT2 机翻译文 TSV

> **全部命令在 WSL2（Ubuntu）内执行，推理统一走 int4 g128 分层常驻管线（★0.26 s/forward 中位，int8 可回退）。**
> 首次使用先按 `references/rocm-setup.md` 搭环境（librocdxg + 五个 HSA 变量 + uv 配方）。

## 定位：它是"填译文列"的发动机，不是完整汉化流程

本技能**只做一件事**：把译文 TSV 里的空译文列填满。
解包/回写/校对/封包仍是 `kirikiri-translation` / `siglus-translation` /
`tyrano-translation` 三个技能的活。三者产出的 work TSV 列格式一致，所以本技能
一套脚本通吃三个引擎。

| 阶段 | 谁负责 |
| --- | --- |
| 解包、建 tpl/work/proof 工作区 | 引擎技能（`*_tool.py make_template`） |
| **填初译（本技能）** | `hy-mt2-translation/translate_tsv.py` |
| 人工/代理校对、术语冻结、结构终检、回写封包 | 引擎技能（`make_proof` / `check_structure` / `rebuild`） |

为什么不用 LLM subagent：subagent 翻译是"按条推理"，几千条极易退化成只写计划不落盘、
漏条、标签乱改、中途停摆。Hy-MT2 是**专门的翻译模型**，无思考过程、指令遵循强，
一次加载后可无人值守跑完，输出可校验（标签序列逐字比对）。

## TSV 契约（三个引擎通用，改动前必读）

```
索引 \t 说话人 \t 日文原文 \t 译文          # 4 列 work 骨架（最常见）
索引 \t 说话人 \t 日文原文                  # 3 列 tpl 模板（本技能追加第 4 列）
索引 \t 说话人 \t 原文 \t 初译 \t 校对 \t 备注   # 6 列对照稿（本技能填"初译"列）
```

硬约束（引擎技能 `check_structure.py` 的门禁，违反就回写失败）：
- 前缀列（索引/说话人/原文）**逐字节不变**，行序不变，`#` 注释行原样保留
- 译文单元格内**不能有真实换行符、不能有 Tab**（换行必须写字面 `\n` 记号或 `[r]` 标签）
- 无 BOM、无 CRLF
- 行内标签 `[r]` `[p]` `[vs]` 等是**结构不是文字**：序列必须与原文完全一致

本技能对上述每一点都有对应保护：标签与字面 `\n` 先换成 `⟨N⟩` 占位符，译后还原并
比对序列；不一致的条自动降级逐条重译，仍失败就留空并记进失败清单——**绝不写脏数据**。

## 工具速查

```bash
# 均在 WSL2 (Ubuntu) 内执行；仓根示例 /mnt/c/<path-to>/vn-translate-skills（按需替换）
S=/mnt/c/<path-to>/vn-translate-skills/hy-mt2-translation/scripts
PY=$HOME/.local/share/hy-mt2/.venv/bin/python                             # ext4 venv（setup_env.py 用 uv 建/修）
export HF_HOME=$HOME/.cache/huggingface                                   # 权重缓存（ext4；配合 HF_HUB_OFFLINE=1 离线加载）
# 前置：五个 HSA_*/LIBROCDXG_* 变量已写入 ~/.bashrc（见 references/rocm-setup.md）

# 0) 建/修环境（uv，幂等。权重不预下，首次翻译自动拉取）
python $S/setup_env.py                       # 默认建到 ~/.local/share/hy-mt2；--root 可自建独立环境
# 1) 环境体检 + 真跑一条翻译（排 ROCm 问题时先跑这个）
"$PY" $S/env_check.py --model tencent/Hy-MT2-7B
# 2) 只看 prompt 不加载模型（调指令/术语时用）
"$PY" $S/translate_tsv.py tpl/01.tsv --dry-run --limit 6 --batch 3
# 3) 正式翻译（同一量化管线，默认 int4；4 列 work 原地补空 / 3 列 tpl 生成到 --out-dir）
"$PY" $S/translate_tsv.py work/*.tsv
"$PY" $S/translate_tsv.py tpl/*.tsv --out-dir work
# 4) 回到引擎技能做结构终检（必跑）
python <引擎技能>/scripts/check_structure.py work/*.tsv --tpl=tpl
python <引擎技能>/scripts/check_terms.py terms.tsv work/*.tsv
# 5) 单文件快线 / 基准：同一量化管线的一键入口（★0.26 s/forward 中位，int4 全驻）
bash $S/_run_wsl_fused.sh
"$PY" $S/int8_pipe.py --cache-info          # 查量化缓存（scheme int4-g128-v2；--quant int8 看回退）
# 自测（改动脚本后必跑；不需要 GPU）
python $S/_selftest.py
```

## 流程

1. **建环境**：`setup_env.py`（uv；配方是仓库真文件 `env/pyproject.toml`，
   torch/triton/rocm **精确钉版本**）。环境首次搭建的完整前置
   （librocdxg RDNA2 fork、五个 HSA 环境变量、python3.12-dev）见
   `references/rocm-setup.md`。装完**必须**跑 `env_check.py` 确认 ROCm 起来了
   ——它顺带触发一次真实模型加载，**权重此时由 transformers 自动下载**到 HF 缓存
   （建议 `HF_HOME` 指向共享盘缓存，避免重复下载）。
2. **冒烟**：`env_check.py` 会用一条含 `[r][p]` 的日文实测标签是否被吃掉。
3. **小批试译**：挑一个剧本文件，先 `--limit 30`，人工看 20 条。重点看：
   名字行要不要译、`[r]` 位置、称呼/口吻、引号风格。
4. **定指令**：把风格、术语、额外要求固化下来（见 `references/prompting.md`）：
   `--style` + `--terms terms.tsv` + `--extra`。术语表与各引擎技能的 `check_terms.py` 同格式。
5. **全量**：整目录跑，脚本每 20 批自动落盘，中断后重跑自动跳过已译条（续跑）。
6. **终检**：回引擎技能跑 `check_structure` + `check_terms`，再进校对流程。
   机翻初稿必须过人工/代理校对，不要直接封包。

## 8GB 卡跑 7B：两个入口，同一套 int4 管线

| 入口 | 实测速度 | 适用 |
| --- | --- | --- |
| `translate_tsv.py`（批量） | 按长度分桶组批（24 行混合输入 54→**57 条/分**），同管线量级 | **正式批量**：断点续跑、术语、失败重译 |
| `_run_wsl_fused.sh` / `int8_pipe.py`（单文件） | **0.26 s/forward（5 次中位；区间 133~308ms）** | 快速出稿、跑基准；无续跑 |

两个入口底层都是同一套 **int4 g128 分层常驻管线**（详见 `references/rocm-setup.md` §4）：
权重离线量化成 **int4 group128 非对称**（uint32 直序打包 + 每组 scale/zp，0.523 B/param、
相对误差 ~0.10），磁盘缓存 mmap 秒载（scheme `int4-g128-v2`，落盘 3.40 GiB、224 Linear）；
字节减半后 **32/32 层全部显存常驻（~3.4 GiB，109 MiB/层）、每 token PCIe 搬运 = 0**，
前向走 triton 融合 W4A16 GEMV（按组约减 + uint32 直序解包；全驻时启用 lite 旁路，
逐层 Python 回调裁到 2 个计时钩）。tie 权重（embed↔lm_head）
已合并，未量化常驻仅 0.98 GiB。输出可与贪心基准（`out/*.tsv`）逐字对拍。
`--reserve-mib`：int4 全驻 **1000 即可**。回退选项：`--quant int8`（per-row 对称、
1 B/param、缓存 6.50 GiB，非驻层需搬运，实测 0.50 s/forward），管线同源、译文同基准。
★ 跨 run 方差在 GPU 侧（同码复测 133~308ms，层主机时间恒定 ~45ms），**一律报 median of ≥3**。

## 参考文档

| 文件 | 读它当你要…… |
| --- | --- |
| references/rocm-setup.md | AMD ROCm WSL2 环境搭建与排障、uv 配方、量化管线原理与调参 |
| references/prompting.md | 指令模板、术语表、风格/个性化、批量分隔符、失败条处理 |
| env/pyproject.toml | 环境配方（torch/triton/rocm 钉版本），改依赖直接改它 |
