# -*- coding: utf-8 -*-
# 校对改稿辅助：按索引向校对文件第 5 列（改稿）/第 6 列（批注）写入。
# 手改 TSV 易踩两类事故：列错位、字面 \n 记号被解码成真实换行把一行拆成两行——
# 本脚本对后者直接拒绝写入，从源头拦住。
# 用法:
#   python edit_proof.py <proof.tsv> <索引> <改稿> [批注]
#   python edit_proof.py <proof.tsv> --note <索引> <批注>     # 只加批注不改稿
#   python edit_proof.py <proof.tsv> --file <json>            # 批量: [{"i":..,"fix":..,"note":..},...]
# 语义：JSON/命令行里**给出**的键才写入；键值为空串即清除该列；未给出的键保持不变。
# 退出码：全部命中=0；存在未命中索引=1（便于脚本化校验）。
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _trans_common import NL

def load(path):
    rows = []
    for raw in open(path, encoding='utf-8'):
        line = raw.rstrip('\r\n')
        if not line.strip() or line.lstrip().startswith('#'):
            rows.append(('raw', line))
        else:
            rows.append(('row', line.split('\t')))
    return rows

def save(path, rows):
    out = [v if k == 'raw' else '\t'.join(v) for k, v in rows]
    open(path, 'w', encoding='utf-8', newline='\n').write('\n'.join(out) + '\n')

def check_field(val):
    """字段里出现真实换行符 = 字面 \\n 被解码，写进去会把 TSV 拆成两行。"""
    if '\n' in val or '\r' in val:
        raise SystemExit(
            f'拒绝写入：内容含真实换行符（会把 TSV 拆坏）。字面换行记号 {NL} '
            f'在 JSON 里应写成 "{NL}"（反斜杠转义）：\n    {val[:40]!r}')

def apply(path, items):
    rows = load(path)
    idxmap = {}
    for kind, v in rows:
        if kind == 'row':
            idxmap.setdefault(v[0], []).append(v)
    hit, miss = 0, []
    for it in items:
        key = str(it['i'])
        targets = idxmap.get(key)
        if not targets:
            miss.append(key)
            continue
        for v in targets:                       # 同索引多行时一并更新
            while len(v) < 6:
                v.append('')
            if 'fix' in it:
                check_field(str(it['fix']))
                v[4] = str(it['fix'])
            if 'note' in it:
                check_field(str(it['note']))
                v[5] = str(it['note'])
        hit += 1
    save(path, rows)
    print(f'{os.path.basename(path)}: 更新 {hit} 条'
          + (f'，未命中: {", ".join(miss)}' if miss else ''))
    return 1 if miss else 0

def main():
    a = sys.argv[1:]
    if len(a) < 2:
        raise SystemExit(__doc__)
    path = a[0]
    if not os.path.exists(path):
        raise SystemExit(f'no such file: {path}')
    if '--file' in a:
        i = a.index('--file')
        items = json.load(open(a[i + 1], encoding='utf-8'))
    elif '--note' in a:
        i = a.index('--note')
        items = [{'i': a[i + 1], 'note': a[i + 2]}]
    else:
        items = [{'i': a[1], 'fix': a[2],
                  **({'note': a[3]} if len(a) > 3 else {})}]
    sys.exit(apply(path, items))

if __name__ == '__main__':
    main()
