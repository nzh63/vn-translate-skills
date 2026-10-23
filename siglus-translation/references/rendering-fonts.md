# SiglusEngine 中文显示修复（文字重叠 / 缺字 / 字体选择）

> 适用：SiglusEngine 汉化后，中文文本在真机上显示异常（叠字、缺字变方框、字体选不到中文）。
> 一句话核心：**消息正文用的不是 `dat/font01.ttf`，而是系统字体**；缺字触发字体回退，破坏消息窗的等宽单元格排版 → 叠字。修正在 **exe 字符串层**，不在 `.ss` 内容层。
> 汉化文件共存（中/日两版并存、g00 扩展名交换）见 **coexist-files.md**。

## 1. 症状与根因

- **症状**：打包后**部分**文本叠字/重叠（典型：名字行被正文"划穿"，看起来像加了删除线）；或**个别简体字变方框/豆腐**。
- **根因**：
  1. 消息正文用的是**日文字体**，缺**简体字形**（`嗯` `间` `拢` `优` `贝`…，以及 U+2014 `—`；日文字体用 U+2015）。
  2. 缺字触发引擎**字体回退**：回退字形的步进宽度/基线与原字体不一致，破坏 SiglusEngine 消息窗依赖的**等宽（等幅）单元格排版** → 自动折行的游标漂移、相邻两行叠在一起，正文窜上名字行，故名字看着被划穿。
  3. 因此**只换 `dat/font01.ttf` 没有任何作用**——消息字体根本不是它。
- **为什么"部分文字重叠"**：只有含这些缺字字形的句子才触发；纯日文/繁形体（如名字 `優子`、`雪兎`）字形成在，不触发叠字。

## 2. 排查方法

1. **确认消息字体是系统字体**：在 `SiglusEngine.exe` 里按 UTF-16LE 搜 `cfg_wnd_base_font`，附近就是消息字体选项（`ＭＳ ゴシック` 等）；同时 `window.ini` 里有对应字体设置窗，`config.sav`（加密）存用户的选择。
2. **确认哪些字形缺字**：解析目标字体（`dat/font01.ttf`、或系统字体 `/c/Windows/Fonts/simhei.ttf` 等）的 cmap，逐字符判断。工具见 §7。
3. 留意：`dat/font01.ttf`/`font02.ttf` 是 Motoya **等宽日文**字体（内部名 `モトヤLマルベリ3等幅` / `モトヤLシータ゛3等幅`），恰好印证"消息窗按等宽单元格排版"；它们是打包的日文字体，不是消息字体。

## 3. 修复：语言/间距补丁 —— 能解决"重叠"

让引擎按**中文排布/间距**处理文本。

- 做法：在 exe 里搜 **UTF-16LE 小写 `japanese`**，整串替换为 `chinese` 并**填充 1 个 0x0000 补回等长**（`japanese`=8 码元 → `chinese`+1 空码元=8 码元，16 字节正好不变长）。
- ⚠ 关键坑：这个串是**小写、UTF-16LE**。教程常写成大写 `Japanese`，直接用 UTF-16LE 搜大写是搜不到的（本作实测：大写 0 命中，小写 1 命中）。
- **偏移因构建而异**，别写死。銀色、遥か 实测：`japanese` 位于 `0x642788`，紧邻 `エラー %d 行目` 与 `riched32.dll`（文本/语言相关区）；此构建该串只有 1 处（无教程说的"第二个错误提示"）。
- 效果：本作实测**解决了叠字**。

## 4. 字体替换的两层坑（缺字/字体选不到中文）

改 exe 里的字体名前，先认清**三条字体路径**，各管各的：

1. **消息窗配置字体**（`cfg_wnd_base_font` 附近 3 个选项：`ＭＳ ゴシック`/`ＭＳ 明朝`/`メイリオ`）—— 游戏设置里可选。
2. **GDI 默认字体**（`D3DXCreateFont` 邻近的 `MS ゴシック`）—— 不指定字体的 UI 用它。
3. **引擎默认字体**（`Gameexe.dat` 的 `#CONFIG.FONT.NAME`，如 `モトヤLシータ゛3等幅`）—— 手机等不读配置的 UI 用它。

**坑 A：字体名必须 ASCII**。引擎创建字体时 `lfCharSet` 固定 = 0x80（SHIFTJIS）；GDI 在 SHIFTJIS charset 下**只认英文/ASCII 字体名**（`Noto Sans SC`/`SimSun`/`NSimSun`/`Microsoft YaHei UI` 正常解析），**中文名一律被静默替换成日文字体 MS PGothic**（`黑体`/`宋体`/`微软雅黑`/`楷体` 全踩坑）→ 部分汉字空白（JIS 只有部分简体字形）。实测见下：

| 请求 (face + SHIFTJIS 0x80) | GDI 实际返回 |
|---|---|
| `Noto Sans SC` / `SimSun` / `NSimSun` / `Microsoft YaHei UI` | ✅ 正常解析 |
| `黑体` / `宋体` / `微软雅黑` / `楷体` / `等线` / `仿宋`（中文名） | ❌ 替换成 MS PGothic（日文字体） |

**坑 B：charset 要改 0x86（GB2312）**。官方（number201724/SiglusEngine 仓库 README）修法：
1. setlocale 的 `japanese` → `chinese`（§3）
2. `CreateFontW` 的字体名 + charset → 黑体 + 0x86
3. `GDI32.CreateFontIndirectW` 的字体名 + charset → 黑体 + 0x86

銀色、遥か 实测（两处字体包装函数，等长字节补丁）：
- 主包装 `0x29c261`（file `0x29b661`）：`8B 45 0C 83 C4 0C 88 45 B7` → `83 C4 0C C6 45 B7 86 8B C3`（保留栈清理 `add esp,0xc`！强制 `lfCharSet=0x86`）
- 第二包装 `0x5b4a4`（file `0x5a8a4`）：`8A 48 14 88 4D B7` → `C6 45 B7 86 90 90`（强制 0x86）
- ⚠ **补丁错位教训**：若只覆盖 7/9 字节导致 `add esp,0xc` 被 nop 掉 → 载入时创建字体直接崩（栈错乱）。打完必反汇编回看 + 启动冒烟。

**坑 C：字体名长度受限**。exe 选项字段很小（如 `メイリオ` 槽位仅 4 码元），放不下 ASCII 名时**宁可不改**（该选项失效但别崩）；短字段用 `0x0000` 补位保持等长。

## 5. 通用纪律与坑

- **改 exe 前先备份**（如 `SiglusEngine.exe.bak`），并保留"逐字节 diff 只差预期字节"的自检。
- **游戏进程在跑时 exe/数据文件被锁**：写补丁会 `PermissionError`。先结束进程（`taskkill /IM SiglusEngine_chs.exe /F`，注意 `//IM` 语法在 Git Bash 下）再改。
- 补丁尽量**等长原地替换**（UTF-16LE 下想换不同长度会平移后续数据、风险高、易崩）。
- **别把偏移写进技能**：不同构建偏移完全不同；文档里的偏移仅作銀色、遥か 参考，其它游戏一律**按字符串搜索**定位。
- 判据：文本是否用系统字体 → 看 `cfg_wnd_base_font`；是否缺字 → 看 cmap；是否走了教程的 `cmp 0x80` → 动态断点确认（不要预设它存在）。

## 6. 动态调试（WinDbg/cdb 脚本要点）

- `cdb.exe` 在 `C:\Program Files\WindowsApps\Microsoft.WinDbg_*/amd64/`；用 `-o -G -c '...; g'` 启动子进程调试。
- **`$$>a<` 别名文件会语法错误**；用 `-c` 内联命令。WOW64 断点要 `sxi 4000001f; sxi 80000003` 跳过，否则 cdb 停在 loader 断点后 stdin 关闭即退出。
- 断字体创建：`bp gdi32!CreateFontIndirectW "dd poi(esp+4) L1; du poi(esp+4)+1c; db poi(esp+4)+17 L1; g"`（LOGFONT+0x17 = lfCharSet）。
- 断字形请求：`bp gdi32full!GetGlyphOutlineW "dd esp+8 L1; kb L6; g"`（**只有时间戳等系统文本命中 = 该 UI 不走 GDI 字形**）。
- 断文件打开：`bp kernelbase!CreateFileW "du poi(esp+4) L60; g"`（追踪 g00 图集加载/扩展名回退；长路径会跨两行显示）。
- 日志是 **UTF-16LE**，用 Python `raw.decode('utf-16-le')` 解析；日志会混入 ModLoad/NatVis 噪音，过滤后按 `地址 "字符串"` 模式提取。

## 7. cmap 字形校验脚本（纯 stdlib）

```python
# 用法: python check_cmap.py <字体文件> "字符串"
import struct, sys
def load_cmap(path):
    d = open(path,'rb').read(); num = struct.unpack('>H', d[4:6])[0]; off = 12; cmap = None
    for _ in range(num):
        if d[off:off+4] == b'cmap': cmap = struct.unpack('>I', d[off+8:off+12])[0]
        off += 16
    subt = struct.unpack('>H', d[cmap+2:cmap+4])[0]; s = None
    for i in range(subt):
        pid, eid, o = struct.unpack('>HHI', d[cmap+4+i*8:cmap+12+i*8]); ss = cmap + o
        fmt = struct.unpack('>H', d[ss:ss+2])[0]
        if fmt == 12 and pid == 3:
            n = struct.unpack('>I', d[ss+12:ss+16])[0]; gr = []
            for j in range(n):
                sc, ec, sg = struct.unpack('>III', d[ss+16+j*12:ss+28+j*12]); gr.append((sc, ec, sg))
            def have(c):
                for sc, ec, sg in gr:
                    if sc <= c <= ec: return sg + (c-sc) != 0
                return False
            return have
        if fmt == 4 and pid == 3 and eid == 1: s = ss
    if s is None: return lambda c: False
    seg = struct.unpack('>H', d[s+6:s+8])[0] // 2
    end = [struct.unpack('>H', d[s+14+i*2:s+16+i*2])[0] for i in range(seg)]
    start = [struct.unpack('>H', d[s+16+seg*2+i*2:s+18+seg*2+i*2])[0] for i in range(seg)]
    delta = [struct.unpack('>h', d[s+16+seg*4+i*2:s+18+seg*4+i*2])[0] for i in range(seg)]
    ro = [struct.unpack('>H', d[s+16+seg*6+i*2:s+18+seg*6+i*2])[0] for i in range(seg)]
    gd = s + 16 + seg*8
    def have(c):
        for i in range(seg):
            if start[i] <= c <= end[i]:
                if ro[i] == 0: g = (c + delta[i]) & 0xFFFF
                else:
                    gi = gd + ro[i] + (c-start[i])*2
                    if gi+2 > len(d): return False
                    g = struct.unpack('>H', d[gi:gi+2])[0]
                    if g != 0: g = (g + delta[i]) & 0xFFFF
                return g != 0
        return False
    return have

have = load_cmap(sys.argv[1])
miss = [c for c in sys.argv[2] if not have(ord(c))]
print('缺字:', ''.join(miss) if miss else '(无)')
```

注意：`simhei.ttf` 等单字体文件可直接解析；`simsun.ttc`/`msyh.ttc` 是 **TTC 集合**，需先按 `ttcf` 头取第一个字体表偏移再解析（脚本默认处理单字体）。

---

# 附录 A: g00 容器与位图字形图集

> 手机聊天（SMH/LIME 场景）的**空白缺字**走的是这套位图图集，不是 GDI。**汉化共存（.chs 扩展名交换）见 coexist-files.md §2。**

## g00 容器 type 2 结构（cut data）

```
byte  type = 2
u16   width
u16   height
u32   cut_cnt
G00_CUT_DATABASE[cut_cnt]  (24B/entry，PSD 还原元数据，引擎忽略)
LZSS 压缩的 cut 表（注意 1 字节对齐偏移）:
  [u32 arc_size][u32 org_size][LZSS data]
  LZSS 变体：12-bit 索引 + 4-bit 长度, BREAK_EVEN=1, LSB-first
解压后 = cut 表:
  u32 cut_cnt
  (cut_cnt × u32 offset, u32 size)   // 偏移相对解压缓冲起点
  cut 块:
    G00_CUT_HEADER (变长, MSVC 对齐):
      type(pad1) count(int8) + 后续元数据
    CHIP_HEADER + 原始 RGBA8888 (xl*yl*4 字节)
```

解码工具：`scripts/g00_tool.py`（decode/dump/diff/probe/regen 子命令）。

## 手机聊天文本：位图字形图集（关键发现！）

**Siglus 引擎有两套文字系统**：
1. **GDI 路径**（`GetGlyphOutlineW` + 文本度量）—— 主对话、消息窗等正常文本
2. **g00 位图字形图集**——手机聊天（SMH/LIME 场景）专用

### 文件模式

```
g00/_mj_SE20_0000..0025.g00   聊天 20px 位数字体图集（26 文件，2601 cut/文件 = 51×51 网格）
g00/_moji_SE48_0000..0148.g00  48px 位数字体图集（149 文件，441 cut/文件 = 21×21 网格）
                              与 _mj_SE20 同结构（位图字体），但字号更大（48px），
                              用于对话/UI 中需要大字的文本（标题、强调文字等）。
g00/_smh_*.g00                 手机 UI 贴图
g00/_lm_*.g00                  LIME（聊天应用）贴图
```

### 索引映射（已用游戏内实测 28/28 验证 + OCR 破译）

```
file_no = charcode // 2601   (0..25)
chip    = charcode % 2601    (每文件 2601 个 cut = 51×51 网格)
```

验证方法：OCR 图集 chip（白底 ×6 条带，PP-OCR 可读）破译 chip→字形；
再用用户游戏内实测（显示 vs 空白）核对 28/28 全中。
关键锚点：一(0x4E00)→f7/c1761、今(0x4ECA)→f7/c1963、先(0x5148)→f8/c0、
值(0x503C)→f7/c2333=EMPTY(空白)。

**空白原理**：图集只含日文 JIS 字形。简体独有字（值晚雏酱她等）对应的
chip 是空的（无字形数据）→ 引擎渲染空白。与主对话的 GDI 路径无关。

### 修复：重生成图集（scripts/g00_tool.py regen 子命令）

```
python g00_tool.py regen <g00目录> <CJK字体.ttf>
```

用 CJK 字体（如 dat/font01.ttf = Noto Sans SC）按 code//2601 渲染全码位
字形重建 26 个图集。**共存模式输出为 _mj_SE20_*.chs**（原版 .g00 保留不动），
配合 coexist-files.md §2 的扩展名交换补丁实现「汉化 .chs 优先、原版 .g00 兜底」。
- 字形格式：16×20 RGBA，RGB=0 黑 + alpha（与原件一致，引擎用 alpha 出字）
- 空 cut = 116B（count=0，引擎渲染无物）
- **对齐校准（重要）**：用字号 15 渲染 + anchor='la' + em_top=-2 → 墨迹中心 ≈ 10.4
  （原版图集实测 9.5-11.0，极差 1.5；墨高 ~13px 与原版 12-14 一致）。
  不可用 ink 底部贴齐（一 会变下划线），也不可用 font.getbbox（CJK 返回整 em 盒）。
  用 `draw.text((0,0), anchor='la')` 后 `img.getbbox()` 取真实墨迹盒再水平居中。
- 注意：重生成后全聊天文本（含日文原文场景）都用该 CJK 字形渲染
- 取消汉化 = 删除 g00/_mj_SE20_*.chs；恢复汉化 = 重新 regen（原版 .g00 不受影响）
- **同样适用于 _moji_SE48**（大号位数字体）：与 _mj_SE20 同结构，文件号公式
  `file = code // 441, chip = code % 441`（每文件 441 cut = 21×21）；同样只含 JIS 字形，
  简体独有字也会空白。**需 g00_tool.py 扩展 regen 支持不同 stride（441 vs 2601）才能重生成。
  当前 g00_tool.py 仅支持 2601 步长。**如果游戏大号文字也用这个图集，扩展 regen 后同样走
  扩展名交换（_moji_SE48_*.chs 优先、_moji_SE48_*.g00 兜底）。

# 附录 B: 工具脚本

| 脚本 | 用途 |
|------|------|
| `scripts/pck_tool.py` | Scene.pck 容器解包/回封 |
| `scripts/ss_tool.py` | .ss 场景解包（字符串提取等）|
| `scripts/crypt.py` | LZSS 解压 / 加密算法 |
| `scripts/g00_tool.py` | g00 type-2 容器解码/PNG 导出/差异比较/图集重生成(.chs) |
