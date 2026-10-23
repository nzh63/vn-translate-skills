---
name: siglus-translation
description: SiglusEngine 系视觉小说游戏文本汉化的完整工具链与流程：解包 Scene.pck 场景数据库（游戏密钥+固定表双层解密、LZSS）、提取 .ss 脚本字符串表、生成翻译工作区（模板/译文/校对三分离）、行宽门禁、字符串原位回写、内容级验证、回封并冒烟验证；含多代理并行翻译的分派纪律。凡涉及 Scene.pck/.pck 解包封包、.ss 文本提取或回写、Siglus/Key 系 Galgame 汉化翻译流程，即使没有明说"Siglus"也应使用本技能。
---

# SiglusEngine 系游戏文本汉化

通用工具链 + 经过真机验证的流程纪律（验证样本：銀色、遥か，442 个场景全量回环）。
全部脚本仅用 Python 标准库，位于 `scripts/`；格式细节与历史事故见 `references/`。
**不要**把任何具体游戏的密钥、译文、路径等写进脚本——它们属于项目工作区，不属于技能。

## 与 KiriKiri 系的根本差异（先读这个）

| | KiriKiri (.xp3+.scn) | SiglusEngine (Scene.pck+.ss) |
| --- | --- | --- |
| 容器 | 散文件归档，patch.xp3 叠加 | **单一场景数据库**：一个包内 N 个 .ss 槽位 |
| 加密 | 无（个别混淆头部） | 双层：16 字节**游戏密钥** XOR + 引擎固定表 XOR |
| 压缩 | zlib(mdf) | 自定义 LZSS（头 [压缩长][解压长]） |
| 文本定位 | 树结构 texts 表 + 说话人标签 | 平铺字符串表；**索引序=剧情序**；说话人靠启发式推断 |
| 换行 | 字面 `\n` 记号 | **没有换行记号**——消息窗自动折行、％Ｂ 串翻页 |

## 工具速查

```bash
S=<skill目录>/scripts
K=<16字节游戏密钥>          # 项目工作区提供（如 workspace/key.txt）；本仓库银色遥か的见 workspace/key.txt
# 并发：批量脚本（make_template / extract_text / validate_text / check_structure /
# check_terms / check_punct / check_quotes_mirror / merge_proof / pck_tool unpack·pack）
# 支持 --jobs=N 多进程跑多文件，产物与串行逐字节一致、输出保序。缺省自动（条目≥8 才起
# 进程池，worker≤8 按 CPU 数）；--jobs=2 及以上强制指定并发；少量小文件用 --jobs=1 串行更快。
# Scene.pck 封包
python $S/pck_tool.py probe  Scene.pck --key-file $K        # 先探测；顺带验证密钥对不对
python $S/pck_tool.py unpack Scene.pck unpacked/ --key-file $K   # → *.ss + manifest.tsv
python $S/pck_tool.py pack   Scene.pck Scene_new.pck unpacked/ --key-file $K [--store]
# 密钥没有现成时：自己逆（详见 references/key-recovery.md）
python $S/find_key.py Scene.pck SiglusEngine.exe --key-out workspace/key.txt [--verbose]
# .ss 字符串表
python $S/ss_tool.py info    unpacked/a.ss                  # 头部各区段（排查用）
python $S/ss_tool.py strings unpacked/a.ss [--all]          # 索引\t内容（索引序=剧情序）
python $S/ss_tool.py rebuild unpacked/a.ss out/a.ss work/a.tsv proof/a.tsv
python $S/ss_tool.py verify  unpacked/a.ss out/a.ss work/a.tsv proof/a.tsv   # 必须 MATCH
# 工作区
python $S/make_template.py   unpacked目录 work区             # tpl/(原文)+work/(译文骨架)
python $S/extract_text.py    unpacked目录 text目录           # 剧情顺序可读全文
python $S/make_proof.py      unpacked/a.ss work/a.tsv proof/a.tsv   # 对照稿+preview/read(幂等)
# 门禁与校对（引擎无关，与 kirikiri 技能同源）
python $S/validate_text.py   work/*.tsv --width=N --lines=1 [--soft] # 换行记号=致命；超宽默认警告，--soft 不影响退出码（N 真机实测校准；自动折行引擎如 Siglus 可放宽）
python $S/check_structure.py work/*.tsv --tpl=<tpl目录>     # 结构终检（封包前必跑）
python $S/check_terms.py     terms.tsv work/*.tsv [--strict]
python $S/edit_proof.py      proof/a.tsv <索引> <改稿> [批注] / --file edits.json
python $S/check_punct.py     proof/*.tsv
python $S/merge_proof.py     work目录 proof目录
# 自测（改动脚本后必跑）
python $S/_selftest.py
```

## 流程

1. **适配探测**：probe 确认容器与密钥（解不出=密钥不对，见 references/pck-format.md §3）
   → 全量解包确认 0 失败 → 定位文本结构 → **实测消息窗每行字数上限** → 用伪译文走完
   链路进真机，确认部署方式（直接替换 Scene.pck？有无 patch 机制？）并备份原版。
2. **译前通读**（workflow.md §1）：extract_text 导出全文 → 少量读者代理逐章通读汇报 →
   主控汇总**统一定名**，产出剧情梗概 + 翻译指南 + terms.tsv。术语冻结的唯一来源；
   读者需要全书视野，译者不需要。
3. **建工作区**：tpl/work/proof 三分离；生成器绝不覆盖已有 work。生效译文 = 校对列非空
   优先于初译列；多文件合并后者覆盖前者。unpacked/ 的 manifest.tsv 是回封槽位依据，
   **不得增删改名**。
4. **翻译**：按章分派（大工程多代理并行，见下节）。换行不要写任何记号——Siglus 自动
   折行，写 `\n` 或字面 `\n` 都会被 rebuild 直接拒绝；索引列是唯一锚点，改稿前回验。
5. **校对**：make_proof 合成六列对照稿 + preview/read 两份阅读稿；五维总清单执行：
   信达雅 / 上下文与人称 / 译法一致性 / 伏笔双关梗 / 标点换行（workflow.md §4）；
   edit_proof 写第 5/6 列 → check_punct 自查 → merge_proof 回合或让 rebuild 直接吃
   work+proof（校对优先自动生效）。
6. **重建与验证**：ss_tool rebuild 只改字符串内容（全等长→字节同构的原位覆盖；有长度
   变化→重排布局并平移头字段）；verify 做「预期 vs 实际」逐串比对 + 结构串零改动 +
   字节码尾块零改动，**必须 MATCH 才许封包**。
7. **终检与封包**：集中跑 validate_text（逐文件）+ check_structure（全量 vs tpl）+
   check_terms → pck_tool pack 回封（未改动槽位逐字节复用，只重压改动过的）→ 回读解包
   逐串比对 → 上真机抽查对白/翻页/既读判定。

## 关键坑（详见 references/）

- **密钥错误是第一大坑**：症状是解压报"非法回引"或 probe 的"压缩长≠数据块长"。
  密钥属项目工作区；获取与逆向的完整决策树见 key-recovery.md（社区表 → 同族 key →
  find_key.py 明文滑窗 → 反汇编重组 → 动态工具）。
- **％Ａ/％Ｂ 开头的串是引擎控制标记**（翻页/语音挂接），绝不可翻译；它们不进模板，
  但若用 `--all` 兜底模式导出则必须人工跳过。
- **说话人列是启发式推断**（短文本+下条以「开头），可能误判；它只作语气参考，
  不参与重建。对白也有【名字】「台词」合一形态，此时说话人列为空。
- **字符串长度可以变**，但条目数与顺序绝对不能变——字节码按索引引用字符串；
  verify 的「数量不变+尾块零改动」就是这条纪律的门禁。
- 回封后 Scene.pck 变小是正常的：原包场景块之间有填充间隙，pack 紧凑重排。
  首次适配时务必真机确认能正常读取。
- `.ss` 里纯 ASCII 串（命令名、标识符）是脚本结构，改了就崩；verify 会拦
  「ASCII 结构串被改动」。
- validate_text 多文件合并的"合计"按索引跨文件去重，看各文件的"已译 N 条"。
- **图像资源汉化用 .chs 共存**（不改原版 .g00）：引擎 g00 查找有扩展名回退
  （g00→bmp→png→jpg→dds），改 exe 交换扩展名后可让 `.chs` 优先、`.g00` 兜底；
  手机聊天的位图字形图集 `_mj_SE20_*.g00` 的简体缺字也靠这套共存机制修复。
  完整做法见 references/coexist-files.md（含实测偏移、pct_type 坑、cdb 验证）。

## 多代理并行翻译（大工程提速，实测纪律）

- **简报先行**：一份共享 BRIEF.md 固化术语总表、称呼对照、角色声纹/拟声词方案、高风险
  规则与各章上下文；所有代理只读它 + 翻译指南 + 本章 extract_text 全文。没有简报不要开
  并行——术语漂移的返工成本远高于写简报。简报由译前通读阶段产出。
- **按章分派、高并行**：每个 .ss 一个执行者互不依赖，可一次并行 10+；超长章在提示里
  明确"逐条完成不得跳行漏译"。超时不代表失败：以工作区文件实际状态为准，未完成的章直接重派。
- **分派提示必须写死三条**：① 只填最后一列（译文列）；② 不要写任何换行记号——Siglus
  自动折行，`\n` 会被拒绝；写入一律用脚本文件，禁止 heredoc/命令行直写大段译文；
  ③ 完成后自跑 `validate_text --width=N --lines=1` 并汇报输出。
- **不信任自报，集中终检**：全部汇报完成后由主控统一跑 validate_text（逐文件）+
  check_structure（全量 vs tpl）+ check_terms；终检脚本的比对逻辑本身也要先用已知好样本
  回归再信任其结果。

## 参考文档

| 文件 | 读它当你要…… |
| --- | --- |
| references/workflow.md | 走完整流程 / 新游戏首次适配清单 / 并行分派纪律 / 事故速查 |
| references/pck-format.md | 探测 Scene.pck、理解加密层、回封与部署 |
| references/ss-format.md | 回写 .ss 字符串、排查 verify 失败、理解说话人推断 |
| references/key-recovery.md | 拿不到游戏密钥时：决策树 / 已知明文推导 / find_key.py 原理 / 动态兜底 |
| references/rendering-fonts.md | 汉化后中文显示异常：文字重叠/缺字/字体选择/位图字形图集（exe 字符串补丁、japanese→chinese、charset 0x86、cmap 校验、_mj_SE20 图集重生成） |
| references/coexist-files.md | 汉化文件共存：Scene.pck 改名双版并存、g00 扩展名交换（汉化 .chs 优先 .g00 兜底）、图集共存工作流 |
