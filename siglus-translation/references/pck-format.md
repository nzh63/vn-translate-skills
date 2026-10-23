# Scene.pck 封包格式与适配笔记

> 目标读者：需要解包/回封 SiglusEngine `Scene.pck` 的执行者。先 `probe`，再动手。
> 工具：`scripts/pck_tool.py`（probe / unpack / pack），底层算法在 `scripts/crypt.py`。

## 1. 容器布局

Scene.pck 是**场景数据库**（不是散文件归档）：一个文件内按区段存放全局变量表、
命令表、场景名与全部场景脚本。

```
头 92 字节:
  u32 头部长(=92)
  u32 ×10 对 (off, count)，依次为:
    varInfo / varNameIdx / varNames      全局变量
    cmdInfo / cmdNameIdx / cmdNames      命令名
    scnNameIdx / scnNames                场景名（UTF-16LE，偏移按"词"计）
    scnInfo   (off, count)               场景槽位表: 每槽 u32 数据偏移 + u32 长度
    scnData   (off, count)               场景数据区，从 off 到文件尾
  u32 ExtraKeyUse                        1 = 场景数据还要过一层游戏密钥
  u32 SourceHeaderLength                 原始头部长（原样保留即可）
```

- 区段按上列顺序紧邻排布；scnData 是最后一段。
- **实测不变量**：每个场景块的 LZSS 头「压缩长」== 该块存储长度。这既是完整性校验点，
  也是密钥快速验证点（见 §3）。
- 原包场景块之间普遍存在填充间隙 → 回封后紧凑重排会使文件变小，属正常现象；
  引擎只按 scnInfo 表寻址。

## 2. 场景数据的三层包装

```
明文 .ss 脚本
  ← lzss_decompress          流头 [u32 压缩长][u32 解压长]，控制位 LSB 在前，
                             1=字面量字节；0=2字节回引 (offset<<4 | len-2)，
                             offset 12 位(1..0xFFF)，len 4 位(2..17)
  ← xor_table (256B 固定表, type0)   decrypt2 —— 引擎内置，与游戏无关
  ← xor_key16 (16B 游戏密钥循环 XOR) decrypt1 —— 仅当 ExtraKeyUse=1；因游戏而异
= 包内存储
```

Gameexe.dat 同族不同参：头部 `[u32 magic][u32 needKey]` 之后用 type1 固定表 +
（needKey 时）同款游戏密钥；dat/*.dbs 用另一组派生算法——本技能暂不覆盖系统文本，
需要时参考社区工具（hiroshil/SiglusEngine 的 GameexeUnpacker/dbsDecrypt）。

## 3. 游戏密钥（项目工作区的事）

16 字节，每游戏一份（同一游戏的 DL/PK/HD 版本可能不同）。完整决策树——社区 key 表、
同族游戏试 key、`find_key.py` 明文滑窗/反汇编重组、动态工具兜底——见
**`key-recovery.md`**。要点：

- 验证：`pck_tool.py probe --key-file ...` 会试解第一个场景。判定依据：
  「压缩长 != 数据块长」→ 密钥前 4 字节就不对（明文前 4 字节恰为块长）；
  解压报「非法回引」→ 前 8 字节碰巧对上但整体不对。两者都通过才算密钥正确。
- 新引擎普遍把密钥**打散**成单字节藏进数据区（本作实测如此），连续明文滑窗搜不到，
  直接上 `find_key.py` 的反汇编模式（已在真机一键恢复成功）。

## 4. 回封行为与部署

- `pack` 以 unpack 产出的 manifest.tsv（序号\t场景名\t文件名）对回槽位；槽位不全直接拒绝。
- 内容未变的槽位**逐字节复用**原数据块（不重压缩）；只有改动过的槽位重压。
  因此小补丁的回封是秒级的，且回读 diff 最小化。
- 部署方式必须首次适配实测：把回封包改名替换原 Scene.pck（**务必备份原版**）进真机看是否
  正常读取；同时确认有无官方 patch 叠加机制。未实测前不要整包铺开。
- 冒烟清单：改 1~2 个场景为可辨识伪译文 → 回封 → 真机确认显示、翻页（％Ｂ 行为）、
  既读判定/跳过不受影响，再换真译文。

## 5. 已知坑速查

| 症状 | 根因 | 处置 |
| --- | --- | --- |
| probe 报压缩长不符 / 解压非法回引 | 密钥错误或漏了 ExtraKeyUse 层 | §3 验证流程 |
| 回封后游戏读不出场景 | 槽位对齐被破坏（manifest 被手工增删） | 用 unpack 原产物目录封包 |
| 文件变小而内容没少 | 原包填充间隙被紧凑掉 | 正常，真机确认一次即可 |
| 个别场景解包失败但其余正常 | 该槽历史上就损坏/特殊 | 记录并跳过该章翻译，勿硬解 |
