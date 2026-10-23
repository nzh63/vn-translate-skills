---
name: kirikiri-translation
description: KiriKiri/KAG 系视觉小说游戏文本汉化的完整工具链与流程：解包 xp3、提取 .scn/PSB 剧本文本、生成翻译工作区（模板/译文/校对三分离）、行宽门禁、字符串表原位回写、内容级验证、封包 patch.xp3 并冒烟验证；含多代理并行翻译的分派纪律。凡涉及 xp3 解包封包、patch.xp3 制作、scn/psb 文本提取或回写、Galgame 汉化翻译流程，即使没有明说"KiriKiri"也应使用本技能。
---

# KiriKiri 系游戏文本汉化

通用工具链 + 经过真机验证的流程纪律。全部脚本仅用 Python 标准库，位于 `scripts/`；
格式细节与历史事故见 `references/`。**不要**把任何具体作品的译文、路径偏移等写进脚本——
它们属于项目工作区，不属于技能。

## 工具速查

```bash
S=<skill目录>/scripts
# XP3 封包
python $S/xp3_tool.py probe  data.xp3                    # 先探测格式（标准 or 定制变体）
python $S/xp3_tool.py unpack data.xp3 unpacked/
python $S/xp3_tool.py pack   patch_files patch.xp3 --patch --template data.xp3
# PSB/.scn 剧本
python $S/psb_tool.py json    scn/A.scn out.json         # 树 → JSON（定位结构用）
python $S/psb_tool.py strings scn/A.scn                  # 字符串表全量（索引\t内容）
python $S/make_template.py    scn目录 work区             # 生成 tpl/(原文) + work/(译文骨架)
python $S/extract_text.py     scn目录 text目录           # 剧情顺序可读文本（译者语境通读）
# 文本门禁（上限按目标游戏实测校准，典型 27字×3行）
python $S/validate_text.py   work/*.tsv --width=27 --lines=3          # 行宽/行数
python $S/check_structure.py work/*.tsv --tpl=<tpl目录>               # 结构终检（封包前必跑）
python $S/check_terms.py     terms.tsv work/*.tsv [--strict]          # 术语冻结检查
# 校对工作流
python $S/make_proof.py   scn/A.scn work/A.tsv proof/A.tsv  # 对照稿+preview/read两份阅读稿(幂等)
python $S/edit_proof.py   proof/A.tsv <索引> <改稿> [批注]   # 按索引写第5/6列；--file 批量JSON
python $S/check_punct.py  proof/*.tsv                       # 引号配平/弯引号/凭空增丢引号/半角点
python $S/merge_proof.py  work目录 proof目录                # 改稿回合 work（proof 留存）
# 回写与内容级验证
python $S/psb_tool.py rebuild scn/A.scn out/A.scn work/A.tsv proof/A.tsv
python $S/psb_tool.py verify  scn/A.scn out/A.scn work/A.tsv proof/A.tsv   # 必须 MATCH
# 自测（改动脚本后必跑）
python $S/_selftest.py
```

## 流程

1. **适配探测**：probe 封包格式 → 解包校验 → 定位文本结构 → 实测行宽上限与字体覆盖
   → 小规模伪翻译冒烟确认补丁挂载行为（裸名 vs 相对路径 / stored vs 压缩 / 头部同款）。
2. **译前通读**（workflow.md §1）：extract_text 导出全文 → 少量读者代理逐章通读并按固定
   结构汇报 → 主控汇总**统一定名**，产出剧情梗概 + 翻译指南 + terms.tsv。这是术语冻结与
   伏笔规则的唯一来源；读者需要全书视野，译者不需要（也不应该有）。
3. **建工作区**：tpl（原文，只读）/ work（译文）/ proof（对照稿）三分离；生成器绝不覆盖
   已有 work。生效译文 = 校对列非空优先于初译列；多文件合并后者覆盖前者。
4. **翻译**：把译前三产物整理成《翻译简报》，再按章分派（大工程可多代理高并行，见下节）；
   换行写字面 `\n` 记号，真实换行会被 rebuild 直接拒绝；索引列是唯一锚点，改稿前回验目标行。
5. **校对**：make_proof 合成六列对照稿，并默认产出两份剧情顺序阅读稿——`*.preview.txt`
   （无索引纯文本，当读者通读文气/术语一致性）与 `*.read.txt`（每条带 #索引 的原文⇒译文
   对照，当校对逐句精读、按索引改稿）。校对范围按五维总清单执行：信达雅 / 上下文与人称 /
   译法一致性 / 伏笔双关梗 / 标点换行（workflow.md §4.1）；改稿经 edit_proof 写第 5 列
   （改稿）/第 6 列（批注），配 check_punct 做引号/标点自查 → merge_proof 回合或直接让
   rebuild 吃 work+proof（校对优先自动生效）。
6. **重建与验证**：rebuild 只改字符串内容、树区字节零动、履历副本自动派生；verify 做
   「预期 vs 实际」逐串比对 + 结构串零改动检查，**必须 MATCH 才许封包**。
7. **终检与封包**：全部章节交付后集中跑 validate_text（逐文件）+ check_structure（全量 vs
   tpl）+ check_terms；patch 只含改动文件，回读解包逐字节比对后上真机抽查对白/履历/既读判定。

## 关键坑（详见 references/）

- **`\n` 记号被解码成真实换行**是最高频事故（shell/heredoc/普通字符串写入都会触发），
  多代理协作下尤其高发：约三分之一执行者会踩一次。纪律：用脚本文件 + raw 字符串写入；
  解析器对"数据行首列不是整数"必须**致命报错**而非静默跳过——否则台词在游戏内被悄悄截断；
  交付前集中跑 check_structure 兜底。（references/workflow.md §3、§6）
- PSB 数组宽度标记 `0x0F` = **3 字节**不是 4；写错则整表错位、进章即崩，且树区校验查不出
  ——所以门禁是内容级 verify，不是"能解析"。（references/psb-scn-format.md §2）
- 多行文本在字符串表里有显示版（含 `\n` 记号）+ 履历版两个副本，只译显示版，履历版由
  rebuild 自动派生；漏派生会破坏履历/跳过/既读判定。（同上 §3）
- 部分剧本存在不经 texts 表、由 lines 直接引用的对白；可读文本提前断尾即此因，
  用 `strings` 全量兜底。（同上 §6）
- 引擎按根目录裸名查询剧本时，补丁条目带目录前缀会**静默**回退日文，无任何报错。
- validate_text 多文件合并的"合计"按索引跨文件去重，不是全量条数——看各文件的"已译 N 条"。
- `.sig` 完整性签名不参与运行时加载，可忽略。

## 多代理并行翻译（大工程提速，实测纪律）

- **简报先行**：一份共享 BRIEF.md 固化术语总表、称呼对照、角色声纹/拟声词方案、高风险场景
  规则与各章上下文，所有代理只读它 + 翻译指南 + 本章剧情顺序原文。没有简报不要开并行——
  术语漂移的返工成本远高于写简报。简报不是凭空写的，它由译前通读阶段产出
  （workflow.md §1：读者代理逐章通读 → 主控汇总定名 → 梗概/指南/terms.tsv）。
- **按章分派、高并行**：每章一个执行者互不依赖，可一次并行 10+；超长章（数百条以上）在提示
  里明确"逐条完成不得跳行漏译"。超时不代表失败：以工作区文件实际状态为准，未完成的章直接重派。
- **分派提示必须写死三条**：① 只填最后一列；② `\n` 是字面两字符，必须用脚本文件 + raw
  字符串写入，禁止 heredoc/命令行直写；③ 完成后自跑 validate_text 并汇报输出。
- **不信任自报，集中终检**：全部汇报完成后由主控统一跑 validate_text（逐文件）+
  check_structure（全量 vs tpl）。实测有执行者在自检环节发现并修复了自己的坏文件，但仍有
  漏网需要集中终检兜住；终检脚本的比对逻辑本身也要先用已知好样本回归再信任其结果。

## 参考文档

| 文件 | 读它当你要…… |
| --- | --- |
| references/workflow.md | 走完整流程 / 新游戏首次适配清单 / 并行分派纪律 / 事故速查 |
| references/xp3-format.md | 探测定制封包、理解 patch 挂载行为、整包重建兜底 |
| references/psb-scn-format.md | 回写 .scn 字符串、排查乱序/崩溃、生成模板 |
