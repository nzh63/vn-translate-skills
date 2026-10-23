# 工作流：Tyrano 系游戏汉化全流程

0 → 7 走完即交付一轮。与 kirikiri/siglus 技能同构：三分离工作区、生效译文规则、
门禁前置、集中终检、不信任自报。差异只在第 0 步（适配）和第 7 步（部署）。

## 0. 适配探测（新游戏首次）

1. **识别引擎**：`exe + resources/app.asar + <projectID>_sf.sav + readme 写
   TyranoScript` ⇒ Tyrano + NW.js。projectID 读 `data/system/Config.tjs`（asar 内）。
2. `asar_tool.py probe` → 全量 `extract`（哈希校验）→ `ks_tool.py info/strings`
   抽样定位文本结构（详见 ks-format.md）。
3. 语料普查：行内标签种类（决定 rebuild 门禁覆盖面，本样本只有
   p/vs/r/l/ruby）、名字行数量、`[iscript]` 比重。
4. **小规模全链路冒烟**：1 个剧本伪翻译（如「屯屯屯」）→ rebuild → verify →
   打 tpatch → 模拟合并逐文件比对 → （可以的话）真机应用。链路哪里断修哪里，
   不要带着未知上全量。
5. 记录消息窗配置（`Config.tjs` + `system/message_window.ks` 的
   position/ptext/字体），确认消息窗自动折行 ⇒ 翻译不需要对齐日文断行。

## 1. 译前通读

`extract_text.py <scn目录> <text目录>` 导出剧情顺序全文 → 少量读者代理逐章通读
汇报 → 主控汇总**统一定名**，产出剧情梗概 + 翻译指南 + terms.tsv。
读者需要全书视野，译者不需要。Tyrano 特有决策要在指南里写死：

- 名字行（#【…】）译不译（backlog 一致性 vs 名牌图是 PNG 改不了）；
- `[ruby]` 的注音方案（中文无假名注音时的替代：删 ruby、改注拼音、或注译文同音字）；
- 旁白与对白的引号风格（原文「」直引是保留还是转「」内不加）。

## 2. 建工作区

`ks_tool.py make_template <scn目录> <work区>` 生成 `tpl/`（原文 TSV，只读）+
`work/`（译文骨架）。生成器**绝不覆盖已有 work**（分章推进时重复运行安全；
但个别文件重来时要先手动删旧 work 文件）。out/ 回写目录按 `data/scenario`
相对路径镜像，payload 只收「已应用译文 > 0」的文件。

## 3. 翻译（可多代理并行）

分派提示必须写死（与 kirikiri/siglus 相同的三条 + 一条引擎特有）：

1. 只填最后一列（初译列）；索引列是唯一锚点，改稿前回验目标行。
2. **换行只写 `[r]` 标签**；真实换行和字面 `\n` 都会被 rebuild 拒绝；
   写入一律用脚本文件 + raw 字符串，禁止 heredoc/命令行直写。
3. **逐字保留行内标签序列**（[p][vs][r][l][ruby …]），原文行尾的 `[p][vs]` 不能丢。
4. 完成后自跑 `check_structure`（单文件 vs tpl）并汇报输出。

## 4. 校对

`make_proof.py` 合成六列对照稿 + preview/read 两份阅读稿 → 校对五维总清单
（信达雅 / 上下文与人称 / 译法一致性 / 伏笔双关梗 / 标点与标签）→
`edit_proof.py` 写第 5/6 列 → `check_punct.py work/*.tsv --src=<scenario目录>`（标点 +
**块级引号净差**：一条消息内每种引号的开-闭净差必须与原版一致——日文多行对白把「开在
首行、」闭在末行，行级不配对是正常的，所以按 tb_start_text 块算；能拦下「续行行首凭空
多开『」和「丢了收尾」这类真错，而不误报成对强调增补与 "…"→『…』 转换）→ 有违规跑
`fix_quotes.py --src=<scenario目录> work/*.tsv` 自动修（删凭空多开的开号/补漏掉的收尾，
成对增补不动）→ `merge_proof.py` 回合 work，或 rebuild 直接吃 work+proof（校对列非空优先）。

## 5. 重建与验证（每文件）

```
ks_tool.py rebuild <tpl.ks> <out.ks> <work.tsv> [proof.tsv]   # 拒绝项必须清零
ks_tool.py verify  <tpl.ks> <out.ks> <work.tsv> [proof.tsv]   # 必须 MATCH
```

verify 校验：预期 vs 实际逐行、结构行（含 iscript、标签行）逐字节零改动、
名字行带 `#` 回写、总行数不变。

## 6. 集中终检（封包前必跑）

```
check_structure.py work/*.tsv --tpl=<tpl目录>     # 全量结构终检
check_punct.py     work/*.tsv --src=<scenario目录> # 标点 + 块级引号净差（须 0 处）
check_terms.py    terms.tsv work/*.tsv            # 术语冻结
```

## 7. 打包部署

- **tpatch**（推荐，官方机制）：payload 只放改动文件、按 `data/scenario/` 相对路径
  镜像 → `make_tpatch.py` → 放 exe 旁 → 真机冒烟（清单见 deploy-and-coexist.md）。
- **直接替换**：`asar_tool.py pack out目录 new.asar --template 原asar` → 备份原包 →
  替换 `resources/app.asar` → 真机抽查。
- 交付物清单：tpatch（或 new.asar）+ 改动文件清单 + 已验证的回滚方法
  （备份的 original asar 改回文件名即可）。

## 事故速查

| 症状 | 原因 | 处置 |
| --- | --- | --- |
| work TSV 行数凭空变多、出现裸 `\t` 行 | 源 .ks 的 `\r\n` 未归一，`\r` 被 universal newlines 当行分隔（87→151 实案） | 已在 `load_ks` 修死；自造代码读 .ks 必须 `replace('\r\n','\n')` |
| rebuild 只应用了少量条译文（760 条只应用 99） | 用 work+proof 时，proof 校对列为空的行把 work 初译覆盖成了空（effective_translation 旧逻辑） | 已修：校对列为空则沿用初译、不覆盖；遇此症状先确认重建用的事实源文件 |
| 对话里凭空冒出未闭合的 `「` / 结尾丢 `」`（游戏内一段话中间出现落单引号） | 多行对白续行行首多开了 `「` 或漏了收尾 `」`；旧 check_punct 只查旁白行，对白行漏检 | `check_punct.py --src=` 块级净差门禁拦 + `fix_quotes.py` 自动修（实测 scene1 60 / scene3 1 / scene05 1 处） |
| rebuild 拒绝「标签序列不一致」 | 译文丢/改/加了 `[p][vs]` 等标签 | 按原文标签序列修正译文，**不要**给 rebuild 加旁路 |
| rebuild 拒绝「译文不能以 ;/@/*/_ 开头」 | 行首字符会被引擎解析成注释/标签/转义 | 改写措辞 |
| verify 失败「结构行被改动」 | 手工编辑动了模板结构区 | 从 tpl 重新 rebuild |
| make_template 报「work 已存在，绝不覆盖」但内容是旧的/残缺的 | 上次中断留下半成品 | 确认后删该 work+tpl 文件重跑 |
| 测试副本写文件改坏原版 | 对会写的文件用了硬链接（实案：原版 app.asar 被穿透覆盖） | app.asar/.sav 一律 copy；static 资源才可 link |
| 真机弹「パッチを適応しました」后闪退/白屏 | 引擎 extractAll 竞态或磁盘不足 | 检查 update_tmp 残留与磁盘余量；重放 tpatch；必要时直接换手动回封的 asar |

## 多代理并行（大工程提速）

与 kirikiri/siglus 完全同源：BRIEF.md 先行（术语/称呼/声纹/高风险规则）→
按剧本文件分派、可一次并行 10+、超时以工作区文件实际状态为准 →
**不信任自报**：全部汇报后主控集中跑 check_structure（全量 vs tpl）+ check_terms，
终检逻辑本身先用已知好样本回归。
