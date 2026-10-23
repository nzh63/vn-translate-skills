# -*- coding: utf-8 -*-
# 翻译工作区生成器：.scn → tpl(原文模板,只读) + work(译文骨架,不覆盖已有文件)
# 适用: KAGParserEx 风格场景树（顶层 scenes[*].texts，条目形如
#       [说话人, [[null, "显示文本", 字数, ...履历副本...]], 语音, ...]）
#
# 用法:
#   python make_template.py <scn目录或单个.scn> <输出目录> [--all]
# 产物（按 scn 主名命名）:
#   <out>/tpl/<name>.tsv   索引 \t 说话人 \t 日文原文          （只读参考, 可随时重生成）
#   <out>/work/<name>.tsv  索引 \t 说话人 \t 日文原文 \t 译文  （只填最后一列; 已存在则跳过）
#   <out>/report.tsv       全部章节统计
# --all: 未知引擎时的兜底模式，转储字符串表全部条目（索引 \t 内容）到 tpl/，自行甄别待译串。
#
# 索引 = PSB 字符串表下标，回写按它定位。说话人取 texts 条目自带标签（空=旁白/独白）；
# 若目标引擎无此标签，需另行从剧本树分析（见 references/workflow.md §说话人）。
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from psb_tool import load_scn, StrRef

def _is_strish(x):
    # keep_strref=True 时字符串叶子是 StrRef，否则是已解析的 str
    return isinstance(x, (str, StrRef))

def find_texts(node, out):
    """递归收集所有形似 texts 表的列表: list of [speaker, segs, ...]"""
    if isinstance(node, list):
        looks = (len(node) > 0 and all(
            isinstance(e, list) and len(e) >= 2 and isinstance(e[1], list) and
            any(isinstance(s, list) and len(s) >= 2 and _is_strish(s[1]) for s in e[1])
            for e in node))
        if looks:
            out.append(node)
            return
        for x in node:
            find_texts(x, out)
    elif isinstance(node, dict):
        for v in node.values():
            find_texts(v, out)

def collect_rows(psb):
    """→ {str_idx: set(说话人)}，只收"显示版"字符串（含换行记号或被 texts 引用的短句）"""
    from _trans_common import NL
    holder = []
    find_texts(psb.value, holder)

    def resolve(x):
        if isinstance(x, StrRef):
            return psb.strings[x.idx] if x.idx < len(psb.strings) else ''
        return x if isinstance(x, str) else ''

    rows = {}
    for texts in holder:
        for e in texts:
            spk = resolve(e[0])
            for seg in e[1]:
                if isinstance(seg, list) and len(seg) >= 2:
                    ref = seg[1]
                    if isinstance(ref, StrRef):
                        rows.setdefault(ref.idx, set()).add(spk)
    return rows

def process(path, outdir, all_mode=False):
    name = os.path.splitext(os.path.basename(path))[0]
    if name.endswith('.txt'):
        name = name[:-4]
    psb = load_scn(path, keep_strref=True)
    os.makedirs(os.path.join(outdir, 'tpl'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'work'), exist_ok=True)

    if all_mode:
        rows = [(i, '', s) for i, s in enumerate(psb.strings) if s]
    else:
        m = collect_rows(psb)
        rows = [(i, '/'.join(sorted(x for x in spks if x)),
                 psb.strings[i]) for i, spks in sorted(m.items())]
    rows.sort(key=lambda r: r[0], reverse=True)   # 与常见工作区习惯一致：大索引在前

    tpl_path = os.path.join(outdir, 'tpl', f'{name}.tsv')
    with open(tpl_path, 'w', encoding='utf-8', newline='\n') as f:
        for i, spk, jp in rows:
            f.write(f'{i}\t{spk}\t{jp}\n')
    work_path = os.path.join(outdir, 'work', f'{name}.tsv')
    made = '跳过(已存在)'
    if not os.path.exists(work_path):
        with open(work_path, 'w', encoding='utf-8', newline='\n') as f:
            for i, spk, jp in rows:
                f.write(f'{i}\t{spk}\t{jp}\t\n')
        made = '新建'
    print(f'{name}: {len(rows)} 条 -> tpl/{name}.tsv, work/{name}.tsv {made}')
    return len(rows)

def main():
    a = [x for x in sys.argv[1:] if x != '--all']
    all_mode = '--all' in sys.argv
    if len(a) != 2:
        raise SystemExit(__doc__)
    src, outdir = a
    targets = [src] if os.path.isfile(src) else \
        sorted(os.path.join(src, n) for n in os.listdir(src) if n.endswith('.scn'))
    if not targets:
        raise SystemExit(f'{src} 中没有 .scn')
    total = 0
    for t in targets:
        total += process(t, outdir, all_mode)
    print(f'合计 {total} 条')

if __name__ == '__main__':
    main()
