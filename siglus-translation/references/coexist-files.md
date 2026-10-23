# SiglusEngine 汉化文件共存（中/日两版并存 + 汉化图集 .chs 扩展名交换）

> 适用：在**不改动原版文件**的前提下，让中文版与日文版共存、汉化图像资源优先加载。
> 两个主题：① 数据文件改名（Scene.pck → Scene.chs）双版并存；② g00 图像**扩展名交换**
> （汉化图 `.chs` 优先、原版 `.g00` 兜底）。文字渲染异常（叠字/缺字/字体）见 rendering-fonts.md。

## 1. 数据文件改名：让中/日两版并存

思路：改 exe 里的**数据文件引用**，让中文 exe 读一个改名后的数据文件，原版文件保留。

- 做法：
  1. exe 里按 UTF-16LE 搜 `Scene.pck`，**通常有两处**：**第一处是实际打开的文件名**（改它）；**第二处是错误提示** `シーンデータ（Scene.pck）が開けません`（别改）。
  2. 把第一处改成**等长**新名，如 `Scene.chs`（9 字符 = 9 字符，可原地替换）。
  3. 把译好的数据文件改名/复制为对应名（如 `Scene.pck` → `Scene.chs`），并从原版备份还原原 `Scene.pck`。
  4. `Gameexe.dat` 的引用有**多处**（多为错误提示：`Gameexe.dat が開けません` 等）；教程说"可一个个试"。汉化通常不动它，两版可共用。
- 結果：`SiglusEngine_chs.exe`（已改引用）读 `Scene.chs` = 中文；原版 `SiglusEngine.exe`（引用 `Scene.pck`）读 `Scene.pck` = 日文。
- **Gameexe.dat 同样处理**：exe 里按 UTF-16LE 搜 `\Gameexe.dat`（**带反斜杠**的才是实际打开路径），
  改成等长 `\Gameexe.chs`（12 码元）；`Gameexe.dat`/`Gameexe.dat が開けません` 等裸串是错误提示别改。
  把配置好的 `Gameexe.dat`（含字体/显示补丁）复制为 `Gameexe.chs`，原版 `Gameexe.dat` 保留给日文 exe。
- 銀色、遥か 实测偏移：`Scene.pck` 文件名引用 @`0x6384D0`，错误提示 @`0x6384F2`；
  `\Gameexe.dat` 文件名引用 @`0x63e37c`（→ `\Gameexe.chs`），错误提示 `0x63d9c6 / 0x63e398 / 0x642758`（别改）。
  cdb 启动验证：引擎打开 `Gameexe.chs` + `Scene.chs`，游戏正常进标题。

## 2. g00 扩展名交换：汉化图 .chs 优先、原版 .g00 兜底

### 2.1 引擎加载 g00 的扩展名回退顺序

```
g00 → bmp → png → jpg → dds
```

（銀色、遥か 实测 cdb 顺序；老教程写作 g00→jpg→png→bmp→dds，以实际构建为准。）
找不到任何后缀 → 弹"找不到文件"错误框。

**坑点**：`g00` 同时是后缀和文件夹名。直接把"g00"push 地址改"ggg"会破坏文件夹路径查找。
**定位方法**：扩展名不是裸 push——它们是 .rdata 的 UTF-16LE 字符串，代码通过**绝对地址** push
（ImageBase 0x400000 + 段 RVA；注意 section header 的 VirtualAddress 是 RVA 不是 VA！）。
银色的回退序列在 `tnm_find_g00_sub`（5 个连续 push，各带 `mov ebx, 1..5` 的 pct_type）。

### 2.2 标准做法：扩展名交换（.chs 与 .g00 共存）

目标：查找顺序变为 `chs → bmp → png → jpg → g00`，汉化优先、原版兜底。

1. 找到回退序列中 push "g00" 的指令（首选位置）和 push "dds" 的指令（末位位置）。
2. 交换两者的 push 立即数（g00 位置 → 指向 dds 字符串；dds 位置 → 指向 g00 字符串）。
3. 把原 `dds` 字符串内容改为 `chs`（UTF-16LE 等长，3 码元）。
4. **关键额外步骤**：末位（现在试 .g00）的 `mov ebx, N` 改为 `mov ebx, 1`——
   否则原版 .g00 回退时 pct_type=N（dds=5），引擎按 dds 解析 g00 失败。
   引擎源码 `switch(file_type){case 1: load_g00; default: FATAL}` 只实现 g00 解析！
5. 把汉化图片命名为 `xxx.chs`，放进 `g00/` 文件夹（原版 `xxx.g00` 保留）。

效果（cdb 文件追踪实测验证）：
- 汉化 `_mj_SE20_0007.chs` → 引擎打开 `.chs`（12 次命中）✓
- 未汉化图片（`_mw00_*.g00` 等）→ 回退 `.g00` 正常加载（pct_type=1）✓
- 无 `.dds` 尝试、无加载错误 ✓

**优点**：原版图片不动、汉化图片可单独管理、可随时删除 .chs 还原日文。

### 2.3 銀色、遥か chs 构建实测偏移（file 偏移，等长字节补丁）

| 修改 | 位置 | 原字节 → 新字节 |
| --- | --- | --- |
| 位置1 push g00 → chs | `0x1e33ed` | `68 8c 90 a3 00` → `68 cc 90 a3 00` |
| 位置5 push dds → g00 | `0x1e365e` | `68 cc 90 a3 00` → `68 8c 90 a3 00` |
| 位置5 pct_type 5 → 1 | `0x1e3666` | `bb 05 00 00 00` → `bb 01 00 00 00` |
| "dds" 字符串 → "chs" | `0x6376cc` | `64 00 64 00 73 00 00 00` → `63 00 68 00 73 00 00 00` |

备份：`SiglusEngine_chs.exe.extswap`（交换前）。

### 2.4 验证方法

cdb 断 `kernelbase!CreateFileW` 打印路径（`du poi(esp+4) L 60`）：
- 打开手机界面 → 应看到 `_mj_SE20_0007.chs`（汉化命中），**不是** `.g00`。
- 未汉化 UI（`_mw00_*.g00`）→ 应正常回退 `.g00`，无报错。
- 全程不应出现 `.dds` 尝试。

## 3. 位图字形图集共存工作流

```
# 1) 恢复原版图集（如曾覆盖为汉化版）
mv g00/_mj_SE20_*.g00.orig g00/_mj_SE20_*.g00
# 2) 重生成汉化图集为 .chs（脚本已支持；原 .g00 保留不动）
python g00_tool.py regen g00 dat/font01.ttf
# 3) 取消汉化 = 删除 g00/_mj_SE20_*.chs；恢复汉化 = 重新 regen
```

图集格式、chip 映射、重生成对齐校准见 rendering-fonts.md 附录 A。
