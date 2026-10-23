# 部署与两版共存（Tyrano / NW.js）

## 引擎官方补丁机制（tpatch）

`kag.js` 的 `checkUpdate` 在启动时检查 **exe 目录**下的
`<projectID>.tpatch`（projectID 见 `data/system/Config.tjs` 的 `;projectID=`，
本仓库样本为 `yurialice`）。`applyPatch` 的 asar 分支流程：

1. 把**当前整个 app.asar** 解到 `<exe目录>/update_tmp/`；
2. 把 tpatch（标准 zip）解压覆盖到同一目录（zip 内是相对路径，既能覆盖也能新增文件）；
3. `asar.createPackage(update_tmp → resources/app.asar)` 重封；
4. 弹窗「パッチを適応しました」→ 点确定 → 删除 tpatch 和 update_tmp → 关闭窗口；
5. 玩家再启动即为补丁版。

制作：`python $S/make_tpatch.py <payload目录> <输出.tpatch>`
（payload 按相对路径进 zip 根，如 `payload/data/scenario/scene1.ks`）。

注意：
- 解 update_tmp 需要约等于整包大小（样本 ~2GB）的临时磁盘空间，且要对
  `resources/app.asar` 有写权限。
- 引擎代码里 `asar.extractAll` 是 fire-and-forget（未 await 就开始覆盖/重封），
  存在理论竞态——**这正是必须真机冒烟的原因**。
- 补丁应用成功后 tpatch 被删除；对已打补丁的包重复应用 tpatch 是幂等覆盖。

## 两版共存评估（结论）

**结论：引擎没有其他引擎那种"同路径多扩展名回退"的查找机制**——
`loadScenario` 以固定前缀 `./data/scenario/` 经 jQuery `$.ajax` 加载唯一路径，
无法让同一文件名按语言回退。可行方案按改动量排序：

| 方案 | 改动 | 共存形态 | 评价 |
| --- | --- | --- | --- |
| A. 双目录安装 | 零（拷贝目录+替换 asar 或放 tpatch） | 两个独立安装 | **推荐**。零风险；exe/dll/pak 等静态文件可硬链接省空间，**app.asar 与 .sav 必须独立拷贝**（会写，硬链接穿透是真实事故）；存档随目录天然隔离 |
| B. 单安装 + 备份回退 | 打 tpatch 前把 `resources/app.asar` 备份为 `app.asar.original` | 单版本生效，可随时改名切回 | 折中。因为工具链保证剧本行结构不变，**两版存档互通**（标签/行号一致），既读标记按文件名记账互不污染 |
| C. 单安装双语切换（评估可行，未采用） | 只改 `index.html` 一处注入 hook + `data/scenario/chs/` 全量镜像 | 一包两语，运行时切换 | 技术依据已核实：`$.loadText` 走 `$.ajax`，可在 jQuery 加载后注册 `$.ajaxPrefilter`，把 `./data/scenario/` 前缀改写成 `chs/` 子目录（前提是 chs 镜像全量，不做存在性回退）；语言旗标存 localStorage，F9 切换重载。放弃原因：改动面大、需真机验证，tpatch 已满足需求。留作未来的扩展路径 |

无论哪种方案，**部署前必须先备份原 `resources/app.asar`**。

## 真机冒烟清单

1. **模拟合并**（不发真机）：整包解出 → tpatch 覆盖 → 回封 → 与原包逐文件比对，
   差异文件清单应与 payload 完全一致（asar_tool 可完成全部三步）。
2. **真机应用**：tpatch 放 exe 旁 → 启动 → 出现补丁弹窗 → 确定 → 窗口关闭
   → 确认 tpatch 已被删除、update_tmp 已清理。
3. **真机验证**：再次启动进新游戏，抽查对白/翻页/语音挂接/既读判定/backlog
   说话人；再用 asar_tool 解包重封后的 app.asar 确认剧本内容与 out/ 一致。
4. 存档回归：打补丁前的存档能否正常读取（结构不变 ⇒ 应能）。
