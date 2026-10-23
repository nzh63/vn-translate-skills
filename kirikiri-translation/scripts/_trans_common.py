# 共享库：译文 TSV 解析、生效译文规则、行宽检查（供同目录脚本 import）
# 仅用标准库；TSV 一律 UTF-8（无 BOM）、Tab 分隔
import os, sys

NL = chr(92) + 'n'   # 换行记号：字面两个字符（反斜杠+n），与引擎内部存储一致。
                     # 任何环节都不允许出现真实换行符（0x0A）。

def iter_rows(path):
    """逐行产出 (行号, index:int|None, cols:list[str])。跳过空行与 # 注释行。"""
    for ln, line in enumerate(open(path, encoding='utf-8'), 1):
        line = line.rstrip('\r\n')
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        cols = line.split('\t')
        try:
            yield ln, int(cols[0]), cols
        except ValueError:
            yield ln, None, cols

def jp_of(cols):
    """日文原文列位置（按列数自适应）：4列=第3列，3列=第2列，其余无原文列"""
    if len(cols) >= 4: return cols[2]
    if len(cols) == 3: return cols[1]
    return None

def speaker_of(cols):
    """说话人列（新版 4/6 列格式为第 2 列；旧版无说话人返回空串）。空＝旁白/独白。"""
    if len(cols) >= 4: return cols[1].strip()
    return ''

def cn_of(cols):
    """生效译文：
    - >=6 列视为"对照稿"（index/说话人/原文/初译/校对/备注）：校对列(5)非空优先，否则初译列(4)
    - 其余取最后一列"""
    if len(cols) >= 6:
        return cols[4] if cols[4].strip() else cols[3]
    return cols[-1]

def load_map(paths):
    """多文件按序合并（后者覆盖前者）→ {idx: (jp|None, cn)}。译文为空的条目跳过。
    ★ 数据行首列不是整数 = 文件被真实换行拆坏或错位，必须视为致命错误：
      静默跳过会让该条译文悄悄变短/丢失，游戏内表现为台词被截断。"""
    out = {}
    for p in paths:
        for ln, idx, cols in iter_rows(p):
            if idx is None:
                raise SystemExit(
                    f'{p}:{ln}: 数据行首列不是整数索引 —— 文件很可能被真实换行符'
                    f'拆坏（字面 \\n 被解码），先修复 TSV 再继续:\n    {cols[:3]}...')
            cn = cn_of(cols).strip()
            if not cn:
                continue
            out[idx] = (jp_of(cols), cn)
    return out

def check_lines(text, width=27, lines=3, lead_quote='「'):
    """行宽/行数检查，返回违规列表。换行记号是字面 NL；首行行首引号不计宽。"""
    errs = []
    if '\n' in text or '\r' in text:
        errs.append(f'含真实换行符（应写字面 {NL} 记号）')
    parts = text.split(NL)
    if len(parts) > lines:
        errs.append(f'行数超限: {len(parts)} > {lines} 行')
    for i, ln in enumerate(parts):
        n = len(ln)
        if i == 0 and lead_quote and n > 0 and ln.startswith(lead_quote):
            n -= 1   # 引擎消息窗首行行首引号悬挂，不占宽度
        if n > width:
            errs.append(f'第{i+1}行超宽: {n} > {width} 字')
    return errs

def count_translated(path):
    return sum(1 for _, idx, cols in iter_rows(path)
               if idx is not None and cn_of(cols).strip())
