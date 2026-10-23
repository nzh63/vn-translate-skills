# Hy-MT2-7B 本地推理优化 — 进度与下一步

> **归档说明（2026-09-23）**：本文自原 `int8-layerwise` 实验目录归档进 skill。
> 运行环境已迁移：venv → `hy-mt2-translation/runtime-wsl/.venv`，一键入口 →
> `scripts/_run_wsl_fused.sh` / `scripts/int8_pipe.py`，环境搭建说明 → 同目录
> `wsl2-rocm-pytorch-triton.md`。文中历史路径与 `bench_*` / `diag_*` 实验脚本
> 按当时记录保留（实验脚本已随原目录移出，可在 git 历史中找回）。

> 更新：2026-09-23 ｜ 机器：AMD Radeon RX 6800S（gfx1032，RDNA2，8 GB 显存，13.69 GiB 内存）
> 两条路线并存：**WSL2 + ROCDXG int8-fused（当前最快，★424ms，超 Windows fp8）** 与 **Windows 原生 fp8（743ms，09-13 旧数，待 P1-2 复测）**

---

## 0. 状态速览（2026-09-23 终测，交流电，贪心 limit2/32token 同口径）

| 路线 | 配置 | 每 forward | 常驻 | 状态 |
| --- | --- | --- | --- | --- |
| **WSL int8 fused（推荐）** | slots3 + issue_at=pre + reserve1000 + 预热 | **424 / 462 ms**（两测） | 21-22/32 | ✅ 最快，超 Windows fp8 1.75× |
| WSL int8 fused，同 Windows reserve | slots3 + pre + **reserve2200** | **555 ms** | 16/32 | ✅ 公平口径仍超 743（Windows 18 常驻） |
| **Windows int8 no-fused（skill 入口）** | `hy-mt2-.../scripts/int8_pipe.py`，r2200 | **762 ms** | 18/32 | ✅ 已迁入 skill 替换 fp8（r1000=4484ms 反例，U 形曲线） |
| WSL int8 no-fused | reserve1000 | 884~926 ms | 21-22/32 | ✅ 受反量化+fp16 地板限制 |
| WSL int8 no-fused，r2200 | 默认双流 | 993~1063 ms | 13-16/32 | ✅ 可用 |
| Windows 原生 fp8（历史） | W8A16+rocBLAS，18/32，reserve2200 | 743 ms | 18/32 | ⚠️ 09-13 旧数，已被 int8 替换 |

- **译文一致性**：本次全部 10+ 份 `out/10-int8-p0-*.tsv` 与历史 `01/07` **逐字节一致**
- **量化精度**：per-row int8 rel-RMS 0.0102，优于 fp8 e4m3 的 0.0265（2.6×），同为 1 B/param
- 一键复现胜出配置：`_run_pipe.sh`（已内置 pre/slots3/r1000 + 持久 triton 缓存 + 预热）

---

## 1. 目标

在 8 GB 显存机器上稳定跑 Hy-MT2-7B（`hunyuan_v1_dense`，32 层，tied embed/lm_head），
追求最低 **每 forward 延迟**（decode 单 token 前向，唯一可跨配置比较的指标）。
当前子目标「WSL int8 超越 Windows fp8 的 743ms」**已达成**（424-555ms，视 reserve）。

---

## 2. 已完成（累计）

### 2.1 管线能力

- **分层常驻 + 交错布局**：按预算选常驻层（非驻留层 `spread_positions` 均匀铺开）。
- **P0-1 单块拷贝（2026-09-22）**：每层一条 208 MiB 连续 pinned 源 + 每槽一条整层 GPU 单块，
  `_issue` 每层仅 1 次 enqueue（旧：7 张量 ×7 次 ×1.9-2.3ms 背压）。fetch 按 offset 切**读侧**
  视图（HIP「切片目标写不进去」的坑只在写方向）。**实测：issue 626→1.5 ms/fwd，挂载 309→≤6.5 ms**
  （P0-2 的「<50ms」验收线被 P0-1 直接打穿）。
- **P0-2 槽位/补发时点**：`--prefetch-slots ≥2`、`--issue-at {auto,pre,post,consume}`、
  逐槽读者事件 `note_slot_read`（稳态只等该槽读者事件，不 `wait_stream` 整流）。
  fused pre = reset 少填一槽、pre 里覆盖「上一非驻留层」的槽。**终测固化 auto：fused→pre、nofused→consume**。
- **P0-3 测量纪律行** `[run] resident=… free0=… fused=… slots=… issue_at=… inline=… reserve/sdma/serialize/quiet`；
  另有 `GPU流窗口`（copy/compute 两流首尾事件）与 `逐forward主机(ms)` 列表。
- **tie 权重合并（09-22）**：`tie_word_embeddings=True` 且 safetensors 双存 embed/lm_head（逐位相同已验证），
  旧代码两份都上卡白占 0.978 GiB。合并后**必须 `torch.cuda.empty_cache()`**——否则释放进缓存池，
  `mem_get_info` 预算看不见，常驻反而少 1 层（实测踩过）。收益：常驻 +4~6 层。
- **triton 预热 + 持久缓存（09-23，最大单项）**：
  - ROCDXG 上 **triton 进程级一次性初始化 ~6-7.8s**（探针：进程首调 5970/6048 ms，同进程次调 1ms），
    全落在 fwd1 → 摊进 ★ = **+429 ms/forward 的税**，历史所有 fused ★ 均虚高。
  - `TRITON_CACHE_DIR` 从 `/tmp`（WSL 停机即清）迁至**仓内 `.triton_cache/`**（已 gitignore）；
  - `load_meta_fp8` 尾部直接调 `int8_gemm_rows` 预热（不走 forward/hook），计入 `[load]`、移出计时窗。
  - 预热后：fwd1 9119→~700ms，★ 1004→**424**。
- **`--inline-copy` 单流模式**：拷贝直接入计算流、废除跨流事件（FIFO 保正确）。
  实测与双流**持平**（424 vs 429），保留为开关；默认仍双流。
- 融合 W8A16 GEMV/GEMM（triton SIMT FMA）、meta 构建、分片惰性读取、失败可观测（沿用）。

### 2.2 关键修复/诊断记录（本轮新增）

| 症状 | 根因 | 处置 |
| --- | --- | --- |
| 任何 device 分配 `hipErrorInvalidValue` | `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` 在当前驱动/ROCDXG 组合**整体损坏**（最小复现：设它连 `torch.empty(device='cuda')` 都炸；不设/设其他值正常） | `int8-layerwise` 三入口停用并留注释 |
| ★ 每次实验数字乱跳、fused 首 forward 8-10s | ① triton 进程初始化 6-7.8s；② `/tmp` 缓存随 WSL 停机清零 → 每次冷编译 | 持久缓存 + load 内预热（见 2.1） |
| tie 合并后常驻反少 1 层 | 释放的 0.978 GiB 进了 PyTorch 缓存池，`mem_get_info` 看不见 | `empty_cache()` 归还驱动 |
| 电池供电跑出 1431ms 废数据 | 未接电源触发降频 | 测量纪律：**必须交流电**（`Win32_Battery.BatteryStatus=2`） |
| `torch.profiler` 排序列失效 | `cuda_time_total`/`device_time_total` 在此 ROCm 构建恒为 0（ROCTracer 归因坏） | 放弃 profiler；改用微基准+三段诊断+流窗口 |
| `verify_prefetch` hipErrorInvalidValue | 导入链（triton）早于 `torch.cuda.init()`（与 int8_pipe 同坑） | 脚本内补 `torch.cuda.init()`；accelerate 依赖改 `torch.device('meta')` |
| fused `t_fwd`≈650ms 主机阻塞 | 三段诊断 T1 发射 25µs/T2 含 empty 43µs/**T3 加 copy 流乒乓 1833µs** → copy↔compute 事件握手把两流拖成串行（copy 窗 950ms vs 纯数据 176ms，80% 在等） | 预热后窗口回落 317/342ms；根因主体是 fwd1 税 + 串行化本身（blit 无 SDMA 本就难重叠） |

---

## 3. 09-13/09-22 性能疑案（保留作背景）

- 统计 hook 无罪；跨天数字不可比（可用显存 6.6~7.25 漂移 → 常驻 11~16）；
- WSL fused 曾「无收益」的旧结论（1520 vs 1111）**已被推翻**：其中大部分是
  issue/挂载背压（P0-1 修掉）+ fwd1 的 triton 税（预热修掉）。同日热态 fused 424 ≪ nofused 884。

---

## 4. 结构性约束（仍然成立）

| 事实 | 实测 | 影响 |
| --- | --- | --- |
| SDMA 不可用 | `HSA_ENABLE_SDMA=0` 硬性要求 | H2D 走 blit，与计算难重叠 |
| blit 背压 | ~12.9 GiB/s，enqueue≈完成 | 拷贝地板 ≈ 非驻层数 × 16ms |
| SDPA math 回退 | 每 forward ~2078 launch | 下一个可压缩项（P1-3） |
| triton 发射 | 23-43 µs/launch（热态） | 224 次 ≈ 5-10 ms，可接受 |
| **triton 进程初始化** | **6-7.8 s/进程（ROCDXG）** | 必须 load 期预热，否则 ★ +429ms |
| GPU 纯计算地板 | 32 层融合 GEMV ≈ 32 ms | 远未饱和；瓶颈=拷贝+发射+同步 |
| nofused 反量化地板 | 两段式 cast+mul+fp16 GEMV ≈ 33ms/层 ≈ 1060ms/fwd（微基准） | nofused 永远打不破 743，除非融合 dequant |

---

## 5. 已知问题 / 未决项

1. Windows fp8 743 为 09-13 跨天旧数（且 Windows 驱动已变、expandable 疑似同样受影响）→ **P1-2 复测未做**。
2. reserve1000 的峰值余量仅 350-780 MiB（随桌面负载），极端情况靠 k 自动回退保护；保守场合用 2200（555ms）。
3. SDPA math 的 ~2078 launch/forward 未压缩（P1-3 `torch.compile` 未试）。
4. nofused 两段式 dequant 未融合（单 triton 核 cast+mul 可省一半 dequant 流量）——nofused 已非最优路线，优先级低。
5. `--inline-copy` 与双流持平，保底开关；未证明有收益。
6. ~~`hy-mt2-translation/SKILL.md` 未同步 int8 路线~~ ✅ 2026-09-23 完成：
   int8 七件套已迁入 skill 替换 fp8（Windows E2E ★762ms@r2200 译文一致）；
   缓存实体移入 `scripts/cache/`（**非硬链接**，独立副本）；SKILL/rocm-setup §4
   已重写；`_run_wsl_fused.sh` 快线随 skill 发布。fp8 四件套（fp8_pipe/
   offload_fp8/fp8_cache/tune）已从 worktree 与索引移除——仓库无提交，
   仅存 `.git` 悬垂 blob 可 `git fsck` 找回。
7. 每次 WSL 会话首次跑含 78-101s `[load]`（含 warmup 7-8s、pin 数十秒）——服务化可再省。
8. **事故记录**：2026-09-23 00:56 前后 `out/` 目录被误清（元凶未定位，迁移
   操作窗口内）。已按本会话逐字节对拍的相同内容重建 21 份 tsv 并 `git add`；
   各 run 的 `.log`（gitignore）不可恢复，关键数字以本文件为准。

---

## 6. 下一步操作

### P0 — 已全部完成 ✅
- P0-1 单块拷贝 ✅（issue 626→1.5ms）
- P0-2 槽位/补发时点 ✅（挂载 309→6.5ms；auto 固化 pre）
- P0-3 测量纪律 ✅（`[run]` 行 + 流窗口 + 逐 forward 列表 + 交流电纪律）

### P1 — 结构性改进（重排优先级）
**P1-2 Windows fp8 基线复测（升为第一优先）**：同口径（limit2/32/记常驻/交流电）复核 743；
驱动已变且 Windows 侧 expandable 同样设着，旧数不可信。若 Windows 也掉到 ~800+，本项目结论更强。

**P1-1 融合 GEMV 移植 Windows（HIP C++）**：WSL 已证融合路径的威力；Windows 原生无 triton，
移植后目标 **< 350 ms**（Windows 无 SDMA 限制，拷贝可真重叠）。收益仍然最大，但已从「翻身仗」变「锦上添花」。

**P1-3 `torch.compile`（WSL）**：压缩 SDPA math 的 2078 launch；目标 ★424 → ~350。风险：hook 图断裂，
建议只 compile attention 子模块或 `mode="reduce-overhead"` 保留回退开关。

### P2 — 可选
- **P2-1** `rows>256` 回退路径压测（长 prompt）；当前 prompt 仅 29/34 token 未触发。
- ~~**P2-2** int8 结论同步进 `hy-mt2-translation/SKILL.md`~~ ✅ 已完成（2026-09-23）。
- **P2-3** 服务化常驻进程（一次预热、多批翻译），把 `[load]`+warmup 摊薄到忽略不计。

---

## 7. 复现命令

### WSL（int8，胜出配置已内置一键脚本）
```bash
wsl -e bash /mnt/c/<path-to>/vn-translate-skills/int8-layerwise/_run_pipe.sh          # fused ★424-462
wsl -e bash /mnt/c/<path-to>/vn-translate-skills/int8-layerwise/_run_pipe_nofused.sh  # nofused 对照

# A/B 矩阵：bash _run_ab_p0.sh [fused|nofused] [slots] [issue_at] [reserve_mib] [额外参数...]
bash _run_ab_p0.sh fused 3 pre 1000 --inline-copy

# 手动
source .../int8-layerwise/.venv/bin/activate
export HSA_ENABLE_DXG_DETECTION=1 HSA_OVERRIDE_GFX_VERSION=10.3.0
export HSA_ENABLE_SDMA=0 HSA_ENABLE_PEER_SDMA=0
export LIBROCDXG_ALLOC_USER_QUEUE_FROM_UMD=1
export HF_HOME=/mnt/c/Users/<user-name>/.cache/huggingface
export TRITON_CACHE_DIR=.../int8-layerwise/.triton_cache    # 必须持久路径！
python int8_pipe.py --greedy --limit 2 --max-new-tokens 32 --batch 1 \
  --issue-at pre --prefetch-slots 3 --reserve-mib 1000
```
> ⚠️ 不要加 `AMD_SERIALIZE_KERNEL/COPY`；不要开 `expandable_segments`（当前驱动下必炸）。
> ⚠️ `TRITON_CACHE_DIR` 不要用 `/tmp`（WSL 停机清零 → 每次冷编译 ≈ ★+430ms）。
> ⚠️ fused 首次发射有 6-7.8s 进程初始化——已在 load 内预热，**勿删 warmup 块**。

### 诊断工具
```bash
python diag_gemv_host.py        # T1 发射 / T2 +empty / T3 +copy流乒乓 三段拆解
python bench_triton_gemv.py     # GEMV 地板（32ms/fwd）与发射开销
python bench_fused_gemv.py      # 融合 vs 两段式反量化逐 shape 对照
python verify_prefetch.py [post|pre|consume]   # 单块预取逐位校验（336/336）
```

### Windows（fp8 原生，待 P1-2 复测）
```powershell
cd C:\Users\<user-name>\repo\vn-translate-skills\hy-mt2-translation\scripts
..\runtime\.venv\Scripts\python.exe fp8_pipe.py --greedy --limit 2 --max-new-tokens 32 --batch 1
```

---

## 8. 文件地图

### `int8-layerwise/`（WSL int8 主线）

| 文件 | 作用 |
| --- | --- |
| `int8_pipe.py` | 入口（必须先 `torch.cuda.init()`） |
| `offload_int8.py` | 核心：`Int8Store` + `PrefetchPipeline`（单块/事件/inline/窗口计时）+ hook + tie 合并 + warmup + CLI |
| `int8_gemm_triton.py` | 融合 W8A16 GEMV/GEMM |
| `int8_quant.py` / `int8_cache.py` | per-row int8 量化；三文件磁盘缓存（mmap） |
| `diag_gemv_host.py` | **新**：GEMV 主机阻塞三段诊断 |
| `bench_launch_overhead.py` / `bench_triton_gemv.py` / `bench_fused_gemv.py` / `bench_accuracy.py` | 微基准 |
| `verify_cache.py` / `verify_in_situ.py` / `verify_prefetch.py` | 正确性校验（均已补 `cuda.init`/去 accelerate/单块源） |
| `_run_pipe.sh` / `_run_pipe_nofused.sh` | 一键基线（**已内置胜出参数 + 持久缓存路径**） |
| `_run_ab_p0.sh` | A/B 矩阵（slots/issue/reserve/额外参数透传，日志带 `[run]`） |
| `.triton_cache/` | 持久 triton 编译缓存（gitignore） |
| `cache/` | int8 磁盘缓存 6.50 GiB |

### 输出
- `out/10-int8-p0-*.tsv|.log`：P0 矩阵/消融/终测（全部与历史译文逐字节一致）
- `out/11-int8-prof-*.log`：profiler 尝试（设备时间列不可用，留档）

---

## 9. 测量纪律（避免不可比数字）

1. **指标**：每 forward ms = 生成墙钟 ÷ (hook 数 ÷ 32)；prompt 构建在窗外。
2. **必须交流电**（`BatteryStatus=2`）；电池数据一律作废。
3. **必须带 `[run]` 行**：常驻层数、free0、fused/slots/issue/inline/reserve、三环境开关。
4. **triton 预热与持久缓存**是 ★ 的一部分：换 `TRITON_CACHE_DIR` 或删 warmup 会把 fused 抬高 ~430ms。
5. 同 reserve 才能与 Windows 对表：**555ms（r2200）vs 743ms（旧数）** 是唯一公平跨机比较格。
6. 方差 ±5-8%（桌面负载）：关键结论至少两测（如 424/462）；单点 1326 之类离散值勿下结论。
7. 加载时间看 `[load]`（含 warmup 7-8s），不进 ★。

---

## 10. int4 g128 升级（2026-09-23 终测，贪心 limit2/32 同口径）

> 本节为 skill 化之后的 int4 升级实测，**追加于文末**；上方归档横幅与历史数字原样保留。

| 路线 | 配置 | 每 forward | 常驻 | 状态 |
| --- | --- | --- | --- | --- |
| **WSL int4-g128 fused（当前默认）** | quant=int4 + reserve1000 + 预热 + ext4 缓存 | **193 ms** | 32/32（搬运 0） | ✅ 达成 ≤200 目标，对 int8 基线（496~508）提速 2.6× |
| WSL int8 fused（回退 `--quant int8`） | slots3 + pre + reserve1000 | 496~508 ms | 20-22/32 | ✅ 保留可回退，译文同基准 |

- **量化**：int4 group128 非对称（`s=(amax-amin)/15`、`zp=round(-amin/s)`、nibble 打包偶数列在低 4 位），0.523 B/param。真实权重相对 Frobenius 实测 **0.103**（int8 per-row 0.0117、int4 对称 g128 0.122；规格稿字面 `zp=round(amin/s)` 实测 0.71，已证伪并按标准 az 实现）。缓存 scheme `int4-g128-v1` 落盘 **3.40 GiB / 224 Linear**（packed 3.25 GiB + scale 104 MiB + zp 52 MiB）。
- **全驻留证据**：`[i8pin] 常驻 32/32 层的 int4 → 显存 3484 MiB；每 token 仍需搬运 0 MiB` + `[run] resident=32/32 … slots=0`——PCIe 层搬运清零（int8 回退时非驻 10-12 层仍搬 2.1~2.5 GiB/token）。峰值显存 4.42 GiB / 7.10 GiB 可用。
- **kernel 两轮迭代**：逐列 zp/scale gather 版首测 ★1998 ms（decode 8.2 ms/层、prefill ×32≈9.8 s）；改按组约减 `Σx·(q−z)·s = s·(Σ_g x·q − z·Σ_g x)` + Python 预算组内 Σx + BLOCK 扫描（decode/gemm 均 BM=8、BN=256、warps=1 最优）后：decode **0.71 ms/层**（×32=23 ms，反超 int8 的 38 ms）、prefill(34 tok) ×32≈586 ms，最终 ★193 ms。主耗时转为层间计算外的发射/attention（层计算 139 ms/fwd、GEMV 主机 58 ms/fwd）。
- **译文门禁**：`out/02-int4-wsl-fused.tsv` 与 int8 基准 `out/01-int8-wsl-fused.tsv` `cmp` **逐字节一致**（`「嗯……」` / `「不，嗯，没关系的」`；基准缺失时已按同参 int8 重跑重建并对照历史口径确认）。
- **冷启动（9p → ext4 迁移）**：`import transformers` **47.3s → 2.5s**；`import torch` 10.9s → 2.1s；快线整墙钟 **~160s → 32.4s**；`[load]` **91s → 22s**。新增无账段计时 `[small] 2.2s`、`[i8pin-up] 16.3s`（原 ~65s 无账时段定位为 safetensors 缺页 + 驻留 H2D，迁 ext4 后合计 18.5s）。venv / HF 权重 / 量化缓存 / triton 缓存全部迁至 ext4（`~/.local/share/hy-mt2`、`~/.cache/{huggingface,hy-mt2-quant,triton}`），删除旧数据释放 C: 约 **35 GB**（df used 429G→394G，可用 47G→81G）。
- 一键复现：`bash scripts/_run_wsl_fused.sh` → ★193 ms；int8 回退 `int8_pipe.py --quant int8`（0.50 s/forward，同基准译文）。

---

## 11. 吞吐专项：uint32 直序 + 长度分桶 + lite 全驻旁路（2026-09-23 第二轮）

> 本节为第二轮吞吐改造实测，**追加于文末**；第 10 节的 ★193 与 `int4-g128-v1`
> 是该轮单次/当时 scheme 的记录，**以本节为准**（方差排查后 ★ 改报 median of ≥3）。

- **A. packed 沿 K 升级 uint32 直序**（一个 dword 装 8 列，权重 load 指令 ÷8）：
  选「scheme 升版 `int4-g128-v2` + 从 safetensors 重建」路线（quant 82s、落盘 10.9s，HF/量化缓存均在 ext4）。小端字节流与 v1 的 uint8 打包逐位相同（单测钉住），但 **load 拒绝旧 scheme**（selftest 伪造 v1 index → `load_into` 必须 False ✓），重建后旧 v1 目录已删。kernel：`(BLOCK_M, BLOCK_N/8)` dword + shift 表展开 → reshape 直序；负 word 算术右移 `&0xF` 全位验证通过。微基准（7 线程全形状合计）：decode **27ms/层组 ×32**（旧 uint8 分组版 23）、prefill(34tok) **557ms**（旧 586）——解包成本基本持平、prefill 略升为代价换 load 指令减半。
- **B. `translate_tsv.py` 按长度分桶**：chunk（原序连续块）内按真实 token 长度稳定排序、以 `--batch` 开窗逐窗 generate，同窗 `max_len≈min_len` 砍 padding。**续跑不变量**：分桶只发生在 chunk 内，chunk 边界=flush 边界（未 flush 结果整体丢失→整块回到 pending），故 resume 看到的 chunk 序列与一次跑完逐一致 → 每条输入/桶成员相同。实测 24 行长短混合：**54 → 57 条/分**；interrupt（`--limit 8` = 首个 flush 边界）→ resume 与一次跑完 `cmp` **RESUME_BYTE_IDENTICAL**；输出行序=输入行序（按 `wf.lines[li]` 原行号回填）。分桶独立增益有限的原因：词表头 lm_head 成本 ∝ batch（不吃 padding）+ 贪心早停使 decode 短，padding 只作用于 backbone 段。
- **C1. ★ 方差排查（同码 5 连跑）**：**133 / 251 / 257 / 258 / 308 → median 257**（min 133、max 308，spread 2.3×）。逐 forward 主机序列跨跑几乎恒定（prefill 619~677ms、decode 42~57ms），层窗口内时间不变 → 摆动全部位于层窗口**之外的 GPU 侧**（attention、词表投影、时钟/调度态），与独立复测 270 同源；此前单次 193 判定为 favorable-tail 离群（硬件在同一构建下确实能到 133）。**此后 ★ 一律报 median of ≥3**；层内肥肉已基本清完（见 C2），剩余大头在 GPU 侧方差与 SDPA/词表头。
- **C2. 全驻快路 lite 旁路**：`resident=32/32 且 prefetch=None 且 fused` 时裁掉逐层 pre/post 回调（每 forward 64 → 2 个钩子：layer0 计数+起表、last 记主机耗时），仅保留 Linear.forward 包装走 kernel。实测层主机 **139 → 45 ms/fwd**、GEMV 主机 **58 → 20 ms/fwd**（叠加 xsum `sum(dtype=fp32)` 免去整条 x 的 fp16→fp32 物化）。计数器语义保持：`calls/resident_hits += n_layers`，★ 计算与「常驻命中」打印数值口径不变；逐 forward 主机序列在 lite 下由 `counters['fwd_ms']` 输出。
- **C3. 顺手项**：warmup 追加 rows=34 覆盖 gemm grid；0 搬运下 prefetch 管线本就不实例化（`[run] slots=0`，无逐槽事件，确认无浪费）；batch>1 decode 走 `_gemm_i4` 的 BM=8/BN=256/warps=1 即小-rows 扫描最优。CUDA graphs（动态步长+词表头在栈外，收益不明确）与 split-K（参考实现已证更慢）按指示**不做**。
- **门禁复跑全绿**：golden `cmp out/01-int8-wsl-fused.tsv out/02-int4-wsl-fused.tsv` **GOLDEN_BYTE_IDENTICAL**（uint32 为无损重解释，分桶不触碰 int8_pipe 两行路径）；`--quant int8 --limit 1` **exit=0**（★663 含首条 prefill 口径，仅冒烟）；`--cache-info` → `scheme int4-g128-v2 / 224 Linear / packed 3.25 + scale 104 + zp 52 = 3.40 GiB`（0.523 B/param）；`_selftest` ALL PASS（新增 uint32 形状/直序/旧 scheme 拒载断言）。
- 一键复现：`bash scripts/_run_wsl_fused.sh` → ★ median 257ms（5 次 133~308，报 median）。
