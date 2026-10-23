# asar 封包格式（NW.js/Electron）

TyranoBuilder 商店版游戏的主体是一个 NW.js 运行时（本仓库样本 24.2.0），
游戏内容全部在 `resources/app.asar` 里，无加密、无压缩。

## 头部布局

```
[0:4]   UInt32 = 4                    （pickle 长度字段宽，恒为 4）
[4:8]   UInt32 = header_size          （头部总长；文件数据区起点 = 8 + header_size）
[8:12]  UInt32 = pickle 载荷长        （= 4 + json长 + 对齐填充）
[12:16] UInt32 = json 长度
[16:..] JSON（UTF-8），按 4 字节对齐补 \0
```

- JSON 是一棵目录树：目录节点 `{"files": {...}}`，文件节点
  `{"size": n, "offset": "十进制字符串", "integrity": {...}}`。
- **offset 是相对数据区起点（8+header_size）的字符串**，与头部长度无关。
- integrity：`{"algorithm":"SHA256","hash":<整文件>,"blockSize":4194304,"blocks":[<每4MB块>]}`。
  解包时必须校验 hash（asar_tool extract 已内置）。
- 头部 JSON 没有统一排版规范，解析时不要假设任何空格/顺序；本工具 pack 用
  `separators=(',',':')` 紧凑输出，回封大小与原包相差百字节级属正常。

## 包结构（TyranoBuilder 商店版典型）

| 路径 | 内容 |
| --- | --- |
| `data/scenario/*.ks` | 剧本（根级=章节，`system/`=引擎包装脚本） |
| `data/other*/plugin/`、`data/system/` | 插件与配置（Config.tjs / KeyConfig.js） |
| `tyrano/` | 引擎本体（kag.parser.js 等在 `tyrano/plugins/kag/`） |
| `tyrano/html/*.html` | 设置/存档/鉴赏等 UI 页面（含日文 UI 文本） |
| `tyrano/lang.js` | 引擎系统弹窗文案 |
| `index.html` / `main.js` / `preload.js` / `package.json` | NW.js 入口 |

注意 `data/scenario/system/` 是**子目录**，`_scene*.ks`/`_title_screen.ks` 等包装脚本在
那里；剧本里 `[tb_system_call storage=system/_scene2.ks]` 引用的就是它。

## pack 的纪律

- `pack <目录> <new.asar> --template <原.asar>`：文件序按原包保持，目录里缺原包文件
  → **致命拒绝**（防删减）；新增文件追加包尾并警告。未改动文件逐字节复用，
  全量回封 2.5k 文件 / 2GB 约 13 秒。
- 回封后必须 `verify`（asar vs 目录）+ 与原包逐文件比对确认"只有预期文件不同"。
- 打出来的 asar 与原包的差异应能逐文件列清；列不清就不要部署。

## 事故记录

- **硬链接穿透（已实际发生）**：做测试副本时对 `resources/app.asar`、`*.sav` 用了硬链接，
  之后往副本写文件穿透改坏了原版。规则：**会被写的东西（asar、存档）只能 copy，不能
  link**；静态资源（exe/dll/pak）才可硬链接省空间。
- 存档：`configSave=file`，`.sav` 文件在 exe 目录（`<projectID>_sf.sav` 等），
  与包体无关，替换 asar 不影响存档。
