---
name: tyrano-translation
description: TyranoScript/TyranoBuilder（NW.js，resources/app.asar）系视觉小说游戏文本汉化的完整工具链与流程：解包回封 asar、提取 .ks 剧本显示文本（tb_start_text 块 + # 说话人行）、生成翻译工作区（模板/译文/校对三分离）、行内标签序列门禁、字符串原位回写、内容级验证、打官方 tpatch 补包并真机冒烟；含两版共存方案评估与多代理并行翻译的分派纪律。凡涉及 app.asar/.ks/.tpatch 解包封包、Tyrano/ TyranoBuilder 系 Galgame 汉化翻译流程，即使没有明说"Tyrano"也应使用本技能。
---

# Tyrano 系游戏文本汉化

通用工具链 + 经过真机验证的流程纪律（验证样本：ユリアリス，NW.js 24.2.0，
3159 条可译行全量回环 + tpatch 模拟合并）。全部脚本仅用 Python 标准库，位于
`scripts/`；格式细节与历史事故见 `references/`。**不要**把任何具体游戏的译文、
projectID、路径写进脚本——它们属于项目工作区，不属于技能。

## 与 KiriKiri / Siglus 的根本差异（先读这个）

| | KiriKiri (.xp3+.scn) | Siglus (Scene.pck+.ss) | **Tyrano (app.asar+.ks)** |
| --- | --- | --- | --- |
| 容器 | 散文件归档 | 单一场景数据库 | **NW.js asar 包**（zip 式树+JSON 头），整个游戏在包里 |
| 加密 | 无/混淆头 | 双层 XOR+LZSS | **无**，纯 UTF-8 明文 |
| 文本定位 | 树 texts 表 | 平铺字符串表 | **按结构**：tb_start_text 块内物理行，索引序=剧情序 |
| 说话人 | texts 内标签 | 启发式推断 | **# 名字行**（引擎 chara_ptext，进名字栏+backlog） |
| 换行 | 字面 `\n` 记号 | 无记号自动折行 | **`[r]` 行内标签**；译文出现真实换行/字面 `\n` 都被拒 |
| 部署 | patch.xp3 叠加 | 替换 Scene.pck | **官方 tpatch 补包**（引擎自解自封）或直接换 app.asar |

## 工具速查

```bash
S=<skill目录>/scripts
# asar 封包（NW.js）
python $S/asar_tool.py probe   resources/app.asar            # 头部体检（先跑）
python $S/asar_tool.py extract resources/app.asar unpacked/  # 全量解包（带哈希校验）
python $S/asar_tool.py pack    <目录> new.asar --template resources/app.asar  # 回封
python $S/asar_tool.py verify  <asar> <目录>                 # asar vs 目录
# .ks 剧本
python $S/ks_tool.py info         unpacked/data/scenario/scene1.ks   # 文本块统计
python $S/ks_tool.py strings      unpacked/.../scene1.ks             # 索引\t说话人\t原文
python $S/ks_tool.py make_template unpacked/data/scenario work区      # tpl/+work/ 三分离
python $S/ks_tool.py rebuild       tpl/A.ks out/A.ks work/A.tsv [proof/A.tsv]
python $S/ks_tool.py verify        tpl/A.ks out/A.ks work/A.tsv [proof/A.tsv]  # 必须 MATCH
python $S/extract_text.py  unpacked/data/scenario text目录               # 剧情顺序可读全文
python $S/make_tpatch.py   <payload目录> <projectID>.tpatch              # 官方补丁包（zip）
# 校对与门禁（引擎无关，与其他技能同源）
python $S/make_proof.py      tpl/A.ks work/A.tsv proof/A.tsv   # 对照稿+preview/read(幂等)
python $S/edit_proof.py      proof/A.tsv <索引> <改稿> [批注] / --file edits.json
python $S/check_punct.py     work/*.tsv [--src=<scenario目录>]  # 标点+块级引号净差门禁（配平原版）
python $S/fix_quotes.py      --src=<scenario目录> work/*.tsv   # 自动修：删续行凭空「/补漏收尾」
python $S/merge_proof.py     work目录 proof目录
python $S/voice_namefix.py   <映射.tsv> <ks...>   # 语音键修复：voconfig 注册名同步译文（详情 ks-format.md）
python $S/check_structure.py work/*.tsv --tpl=<tpl目录>         # 结构终检（封包前必跑）
python $S/check_terms.py     terms.tsv work/*.tsv [--strict]
# 自测（改动脚本后必跑）
python $S/_selftest.py
```

## 流程

1. **适配探测**：认引擎（exe+app.asar+`*_tyrano_data.sav`）→ probe/解包 → info 普查
   标签种类与名字行 → 1 个剧本「屯屯屯」伪翻译走通 rebuild→verify→tpatch→模拟合并
   →（可以的话）真机冒烟，再上全量。
2. **译前通读**：extract_text 全文 → 读者代理通读 → 主控统一定名，产出梗概 + 指南 +
   terms.tsv。Tyrano 特有决策写进指南：名字行译不译、ruby 注音方案、引号风格。
3. **建工作区**：tpl（原文只读）/ work（译文）/ proof（对照稿）三分离；生成器绝不
   覆盖已有 work；生效译文 = 校对列非空优先于初译列。
4. **翻译**：按剧本文件分派（大工程多代理并行）。索引列唯一锚点；换行只用 `[r]`；
   **逐字保留行内标签序列**，行尾 `[p][vs]` 不能丢。
5. **校对**：make_proof 合成六列对照稿 + preview/read 阅读稿；五维总清单 →
   edit_proof 写第 5/6 列 → check_punct → merge_proof 回合或让 rebuild 吃 work+proof。
6. **重建与验证**：rebuild 只改显示文本行（门禁：标签序列全等、行首危险字符、
   真实换行/字面 `\n`、名字行补 `#`）；verify 必须 MATCH（结构行零改动）。
7. **终检与部署**：集中跑 check_structure（全量 vs tpl）+ check_terms →
   payload 只收改动文件（镜像 data/scenario 树）→ make_tpatch → 真机冒烟
   （deploy-and-coexist.md 清单）。**部署前必须备份原 resources/app.asar**。

## 关键坑（详见 references/）

- **原始 .ks 是 `\r\n`**：`.ks` 读入必须先归一换行（load_ks 已内置），否则 `\r` 混进
  TSV 被 universal newlines 拆行，87 行读成 151 行（实案）。TSV 内出现 `\r` 会被
  check_structure 拦下。
- **标签是结构**：rebuild 要求译文标签序列与原文完全一致；`[p][vs]` 行尾结构对、
  既读/语音/翻页行为就全保住；缺了会被静默改变行为，所以是致命门禁而非警告。
- **名字行（#【…】）是显示文本**：进 backlog 说话人栏；裸 `#`（旁白清名）是结构不进
  模板；消息窗名牌是 PNG 图，译不译 # 行是项目一致性决策。
- **名字行是语音键**：voiceplay 语音注册表按 # 行精确内容查角色（`[brainlabo_voconfig
  name="【ユリ】"]`，boot/结局剧本注册）。译文名字行对不上注册名 → 语音静默不播且
  编号不再推进；用 voice_namefix.py 把注册名同步成译文名（详见 ks-format.md）。
- **测试副本禁硬链接会写的文件**：app.asar、.sav 用 copy（实案：硬链接穿透改坏
  原版）；静态 exe/dll/pak 才可硬链接。
- **tpatch 是覆盖不是叠加**：应用后单版本生效；引擎 extractAll 未 await 有理论竞态，
  必须真机冒烟；应用过程需要 ~整包大小的临时磁盘和写权限。
- 引擎没有文件回退机制，单安装内无法让同名文件两版并存——共存方案评估见
  deploy-and-coexist.md（推荐双目录安装或备份回退）。
- UI/系统文本在 `tyrano/lang.js`、`tyrano/html/*.html`、`config.ks` 的 iscript 里，
  本工具链不提取（ks-format.md 有清单）；图片文本需改图。

## 多代理并行翻译（大工程提速，实测纪律）

- **简报先行**：共享 BRIEF.md 固化术语、称呼、声纹、ruby 方案、名字行决策；由译前
  通读阶段产出。没有简报不要开并行。
- **按文件分派、高并行**：每个 .ks 一个执行者互不依赖，可一次并行 10+；分派提示
  写死四条（见 workflow.md §3）：只填末列 / 换行只用 [r] / 标签序列逐字保留 /
  完成后自跑 check_structure 汇报。
- **不信任自报，集中终检**：全部汇报后主控统一跑 check_structure（全量 vs tpl）+
  check_terms；终检脚本先用已知好样本回归。

## 参考文档

| 文件 | 读它当你要…… |
| --- | --- |
| references/workflow.md | 走完整流程 / 新游戏首次适配 / 并行分派纪律 / 事故速查 |
| references/asar-format.md | 探测/回封 asar、理解头部与 integrity、pack 纪律 |
| references/ks-format.md | 回写 .ks 文本、标签规则、ruby、UI 外部文本清单 |
| references/deploy-and-coexist.md | tpatch 制作与应用细节、两版共存方案评估、真机冒烟清单 |
