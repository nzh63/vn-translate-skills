# AMD ROCm（WSL2）部署 Hy-MT2

> 本文是本 skill 的环境速览 + 排障手册，**全部流程跑在 WSL2（Ubuntu）内，
> 推理统一走 int8 分层常驻管线**。完整首次搭建步骤（librocdxg RDNA2 fork
> 全过程、宿主机一次性前置）见[这篇博客](https://blog.nzh21.site/rdna2%e7%82%bc%e4%b8%b9%e7%8e%af%e5%a2%83%e6%90%ad%e5%bb%ba%ef%bc%88wsl2%e7%af%87%ef%bc%89/)。

## 1. 环境组成与配方

| 层 | 组件 | 说明 |
|---|---|---|
| 宿主机 | WSL2 + Ubuntu 24.04、显卡驱动 | 一次性前置，详见 wsl2 文档 |
| WSL | librocdxg（RDNA2 社区 fork） | ROCm↔dxg 桥接；官方版不含 RDNA2，会挂死 |
| WSL | Python 3.12 + `python3.12-dev` | triton 现场编译要 `Python.h` |
| venv | torch + triton + rocm-sdk | AMD TheRock nightly（gfx103X 索引）精确钉版本 |

配方是仓库真文件 **`env/pyproject.toml`**，`setup_env.py` 把它落到环境目录后
`uv sync`（默认同步本 skill 自带的 **`runtime-wsl/`**，`--root` 可自建）。
要点：

```toml
# torch 与 triton 必须成对钉死，裸装会解析到互相冲突的 nightly 组合：
"torch==2.12.0+rocm7.14.0a20260612",
"triton==3.7.0+gitb4e20bbe.rocm7.14.0a20260612",

[[tool.uv.index]]                       # AMD nightly（gfx103X = Navi 2x，内核按 gfx1030 编译）
name = "amd"
url = "https://rocm.nightlies.amd.com/v2-staging/gfx103X-all/"
explicit = true                          # 只有 [tool.uv.sources] 点名的包走它
```

- 其余纯 Python 依赖走国内镜像（tencent index）。
- 代理：`export https_proxy=http://127.0.0.1:<port> UV_HTTP_TIMEOUT=600`。

**五个必需环境变量**（RDNA2，写入 `~/.bashrc`；所有运行命令都依赖它们）：

```bash
export HSA_ENABLE_DXG_DETECTION=1
export HSA_OVERRIDE_GFX_VERSION=10.3.0      # 设备以 gfx1030 身份出现（wheel 内核按它编译）
export HSA_ENABLE_SDMA=0
export HSA_ENABLE_PEER_SDMA=0
export LIBROCDXG_ALLOC_USER_QUEUE_FROM_UMD=1   # 缺它：显存分配直接挂死
```

## 2. 排障清单（按顺序试）

先跑 `env_check.py`，它会打出设备名、`gcnArchName`、显存、以及一条真实翻译。

1. **`torch.cuda.is_available()` 为 False / 分配挂死**
   - librocdxg 没装或没 `sudo ldconfig`；RDNA2 必须用社区 fork（官方版只枚举、
     第一次分配就死）——安装见 wsl2 文档 §二。
   - 五个 HSA 变量缺一不可（尤其 `LIBROCDXG_ALLOC_USER_QUEUE_FROM_UMD=1`）。
   - `Cannot load librocdxg.so` → 装 fork 后 `sudo ldconfig`。
2. **import 阶段报 DLL/库加载失败**
   - venv 里 `rocm`/`rocm-sdk-*` 版本必须与 torch 的 `rocmX.Y` 后缀一致
     （这里是 `7.14.0a20260612`）。版本漂移就 `uv sync` 整片重装。
   - 确认没混进非 ROCm 版 torch：`python -c "import torch;print(torch.__version__)"` 应含 `rocm`。
3. **triton 编译报 `Python.h: No such file`**
   - `sudo apt install -y python3.12-dev`。
4. **速度慢得离谱**
   - 跑 §4 的 int8 快线（`_run_wsl_fused.sh`）；**基准必须交流电**
     （电池降频会把数据整体拉高 30%+）。
   - `--batch` 甜点是 8（再大分隔符失败率上升，整批逐条重译反而更慢）；
     `--micro-batch` 控制单次 generate 并行批数，OOM 时降回来。
   - venv 放在 NTFS（`/mnt/c`）上能跑但偏慢；介意就 `UV_PROJECT_ENVIRONMENT`
     指到 ext4（`~/`）。
5. **attention 相关报错** → 不用管 `memory efficient attention` 降级告警（math
   路径可用）；真报错再回退 transformers 的 `--attn eager` 兼容开关（仅影响
   非 int8 入口如 `env_check.py`）。
6. **`tl.dot` 编译失败** → RDNA2 无 WMMA/MFMA 指令；本管线的融合 kernel
   已全部手写 FMA，不依赖 `tl.dot`，正常跑不该出现——若来自你自己的 kernel，
   改 `tl.sum(w * x)` 广播乘累加。

## 3. 显存与 KV cache 速算

- 权重：Hy-MT2-7B 全精度约 15 GiB——8GB 卡装不下，**必须走 §4 的 int8 分层
  管线**。批量入口 `translate_tsv.py` 与单文件快线**共用同一套管线与磁盘缓存**。
- KV cache：7B（32 层）每 token 数十 KB 量级，8GB 卡留 ~3 GiB 给 KV 与激活，
  足够 `--batch 8 --micro-batch 4`。
- `translate_tsv.py`：`--batch`（一次请求合并条数）甜点 8；`--micro-batch`
  （单次 generate 的并行批数）默认 4，OOM 再降。

## 4. 量化分层管线（scripts/；默认 int4-g128，`--quant int8` 回退）

权重离线量化成 **int4 group128 非对称（uint32 直序打包 + 每组 scale/zp，
0.523 B/param）**，32/32 层全部显存常驻（~109 MiB/层）、每 token PCIe 搬运 0；
回退 `--quant int8`：**int8 + per-row scale**，能装进显存的层常驻，其余层每
token 从锁页内存**按层单块**搬进 GPU 双槽、用时反量化 + triton 融合 GEMV。
入口：批量 `translate_tsv.py`（带断点续跑、按长度分桶组批）/ 单文件 `int8_pipe.py`，
一键脚本 `_run_wsl_fused.sh` 内置全部 env 与胜出参数。

**实测**（贪心受控 limit2/32，RX 6800S 8GB，★ = 生成墙钟 ÷ forward 数）：

| 配置 | ★/forward | 常驻 | 备注 |
| --- | --- | --- | --- |
| **int8 + triton 融合（推荐）** | **424 / 462 ms**（两测） | 21-22/32 | `_run_wsl_fused.sh`，reserve 1000 |
| int8 融合，reserve 2200 | **555 ms** | 16/32 | 显存紧时的保守档 |
| int8 反量化挂载（无 triton 对照） | 884~926 ms | 21-22/32 | 回退路径 |

输出可与贪心基准（`out/*.tsv`）**逐字对拍**；int8 per-row 相对误差 0.0102
（旧 e4m3 方案为 0.0265），同为 1 B/param。磁盘缓存命中时 `[load] ~15s`，
首跑量化 ~40s。

```bash
bash $S/_run_wsl_fused.sh                 # 单文件一键：env + 胜出参数
"$PY" $S/translate_tsv.py tpl/*.tsv --out-dir work   # 批量（同管线 + 续跑）
"$PY" $S/int8_pipe.py --cache-info        # 查缓存（scheme per-row-int8-v1）
"$PY" $S/int8_pipe.py --rebuild-cache     # 模型/量化方案变了强制重建
```

原理与关键数字：
- gfx1032 无 int8 matmul 单元，走 triton **融合 W8A16 GEMV**：寄存器内反量化、
  权重只读一遍 int8（流量减半）；★424 中 decode 段主机仅 ~45 ms/forward。
- **tie 权重合并**：config 声明 embed↔lm_head 绑定而权重文件双存，合并上传省
  0.98 GiB（未量化常驻仅 0.98 GiB），常驻层数 +4~6。
- 显存账：未量化 0.98 GiB + 每常驻层 int8 208 MiB + 预取槽（slots × 208 MiB）。
- **`--reserve-mib`：1000 最优**（常驻 21-22，424 ms）；2200 → 555 ms 但余量更厚。
- 宿主侧只留非驻留层的 pinned int8（~2~4 GiB），mmap 整体释放；锁页上限
  ~4.5 GiB（全层 pin 会 OOM，报错会提示调 reserve）。
- 缓存落在 `<skill>/scripts/cache/`（6.5 GiB，mmap 零拷贝），revision/scheme
  不符自动失效重建；`verify_cache.py` / `verify_prefetch.py` 逐位校验
  （实测 42/42、336/336 通过）。

踩过的坑（改这套代码前必读）：
- `wait_stream` 必须在**切换 stream 之前**取计算流引用：进了
  `with torch.cuda.stream(s):` 之后 `current_stream()` 返回的就是 s 自身，
  等于没等 → 预取覆盖正在计算的槽位 → 输出乱码。
- 本 HIP 上 `buf[off:off+n].copy_(src, non_blocking=True)` 在**切片目标**上
  不可靠 → 预取已改为**每层一条单块**（GPU 侧只有整块→整块拷贝，读侧才切视图）。
- **勿设 `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`**：当前驱动 +
  ROCDXG 组合下任何设备分配都直接 `hipErrorInvalidValue`（代码已停用）。
- **triton 有 ~7s/进程初始化**，已放在 `[load]` 期预热；`TRITON_CACHE_DIR`
  必须用持久路径（仓内 `.triton_cache/`），用 `/tmp` 会随 WSL 停机清零 →
  每次冷编译，★ 虚高 ~430 ms。
- 量化阶段会把 13 GiB 权重读进物理内存做逐行 scale，RSS 冲 ~10 GiB——
  13.69 GiB 物理内存的机器上这是段错误的根源，缓存落盘后平时不再触发。
- pinned 内存在 WSL/librocdxg 下可用，但全层 pin 会超锁页上限（见上）。
