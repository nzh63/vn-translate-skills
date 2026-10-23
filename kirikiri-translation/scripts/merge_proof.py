# -*- coding: utf-8 -*-
# 把校对文件(proof)的改稿合并回译文文件(work)；proof 保留作为校对存档
# 用法: python merge_proof.py <work.tsv> <proof.tsv>
#       python merge_proof.py <work目录> <proof目录>      （按同名文件逐个合并）
# 规则: proof 第 5 列（校对）非空的条目 → 覆盖 work 第 4 列（译文）
#   重复合并是安全的/幂等；合并后对行宽违规给警告（不阻断，正式门禁仍是 validate_text.py）
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _trans_common import iter_rows, check_lines

def load_proof_fixes(path):
    """proof → {index: 校对稿}（仅非空；兼容旧版 5 列格式）"""
    fixes = {}
    for ln, idx, cols in iter_rows(path):
        if len(cols) >= 6 and cols[4].strip():
            fixes[idx] = cols[4].strip()
        elif len(cols) == 5 and cols[3].strip():
            fixes[idx] = cols[3].strip()
    return fixes

def merge_one(work_path, proof_path):
    fixes = load_proof_fixes(proof_path)
    if not fixes:
        print(f'{work_path}: proof 无改稿，跳过')
        return 0, []
    out, merged, warns = [], 0, []
    for ln, line in enumerate(open(work_path, encoding='utf-8'), 1):
        line = line.rstrip('\n')
        if not line.strip() or line.lstrip().startswith('#'):
            out.append(line)
            continue
        parts = line.split('\t')
        try:
            idx = int(parts[0])
        except ValueError:
            out.append(line)
            continue
        if idx in fixes:
            col = 3 if len(parts) >= 4 else 2   # 新版第4列 / 旧版第3列
            while len(parts) <= col:
                parts.append('')
            parts[col] = fixes[idx]
            merged += 1
            for e in check_lines(fixes[idx]):
                warns.append(f'{os.path.basename(work_path)}:{ln} [#{idx}] {e}')
        out.append('\t'.join(parts))
    tmp = work_path + '.tmp'
    with open(tmp, 'w', encoding='utf-8', newline='\n') as f:
        f.write('\n'.join(out) + '\n')
    os.replace(tmp, work_path)
    return merged, warns

def main():
    args = sys.argv[1:]
    if len(args) != 2:
        raise SystemExit(__doc__)
    w, p = args
    pairs = []
    if os.path.isdir(w) or os.path.isdir(p):
        for pf in sorted(glob.glob(os.path.join(p, '*.tsv'))):
            stem = os.path.basename(pf)
            wf = os.path.join(w, stem)
            if os.path.exists(wf):
                pairs.append((wf, pf))
            else:
                print(f'跳过 {stem}: work 目录中无对应文件')
    else:
        pairs.append((w, p))
    total, all_warns = 0, []
    for wf, pf in pairs:
        n, warns = merge_one(wf, pf)
        total += n
        all_warns += warns
        if n:
            print(f'{wf}: 合并改稿 {n} 条')
    for msg in all_warns:
        print('行宽警告:', msg)
    tail = '，存在行宽警告，请运行 validate_text.py 复查' if all_warns else ''
    print(f'合计合并 {total} 条{tail}')
    sys.exit(1 if all_warns else 0)

if __name__ == '__main__':
    main()
