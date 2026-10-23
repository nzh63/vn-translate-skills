# -*- coding: utf-8 -*-
# 译文文件结构终检：与原文模板逐行比对 + 编码/转义完整性审计
# 用法: python check_structure.py <work.tsv...> [--tpl <tpl目录>] [--allow-partial] [--jobs=N]
# 检查项（任何一项失败即退出码 1）：
#   - 无 BOM、无 CRLF
#   - 每个数据行首列为整数索引；全文件列数一致且等于模板列数+1
#   - 与模板按行对齐：索引集合与顺序一致，说话人列/日文原文列零差异
#   - 译文列内无真实换行符、无制表符（换行必须写字面 \n 记号）
#   - 索引无重复
# 未译条目默认放行（分章推进）；--allow-partial 关闭"存在未译"提示。
# 典型用法：全部章节交付后跑一遍 work 全量 + --tpl，作为封包前的结构门禁。
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _trans_common import iter_rows, cn_of, arg_jobs, run_jobs

def check_file(path, tpl_path=None):
    errs, notes = [], []
    raw = open(path, 'rb').read()
    if raw.startswith(b'\xef\xbb\xbf'):
        errs.append('含 UTF-8 BOM')
    if b'\r\n' in raw or b'\r' in raw:
        errs.append('含 CRLF/CR 行尾')

    rows = list(iter_rows(path))
    seen = {}
    ncol = None
    for ln, idx, cols in rows:
        if ncol is None:
            ncol = len(cols)
        elif len(cols) != ncol:
            errs.append(f'{path}:{ln} 列数 {len(cols)} != 本文件其他行 {ncol}')
            break
        if idx in seen:
            errs.append(f'{path}:{ln} 索引 {idx} 重复（首次出现在:{seen[idx]}）')
        seen[idx] = ln
        cn = cn_of(cols)
        if cn:
            if '\n' in cn or '\r' in cn:
                errs.append(f'{path}:{ln} [#{idx}] 译文含真实换行符（应写字面 \\n 记号）')
            if '\t' in cn:
                errs.append(f'{path}:{ln} [#{idx}] 译文含制表符')

    if tpl_path and os.path.exists(tpl_path):
        trows = list(iter_rows(tpl_path))
        if len(rows) != len(trows):
            errs.append(f'{path}: 数据行数 {len(rows)} != 模板 {len(trows)}')
        for i, ((_, widx, wcols), (_, tidx, tcols)) in enumerate(zip(rows, trows)):
            if widx != tidx:
                errs.append(f'{path}: 第{i+1}条数据行 索引 {widx} != 模板 {tidx}（行序错位？）')
                break
            # work 列数 = 模板列数 + 1（追加译文列），前缀各列须一致
            if wcols[:-1] != list(tcols):
                errs.append(f'{path}: [#{widx}] 前几列与模板不一致: '
                            f'{wcols[:-1]!r} != {tcols!r}')
                break
    elif tpl_path:
        notes.append(f'  （无同名模板 {tpl_path}，跳过模板比对）')
    n_tr = sum(1 for _, _, c in rows if (cn_of(c) or '').strip())
    return errs, notes, n_tr, len(rows)

def _check_job(t):
    return check_file(*t)

def main():
    args, jobs = arg_jobs(sys.argv[1:])
    tpl_dir = None
    allow_partial = '--allow-partial' in args
    opts = [a for a in args if a.startswith('--')]
    args = [a for a in args if not a.startswith('--')]
    for a in opts:
        if a.startswith('--tpl='):
            tpl_dir = a.split('=', 1)[1]
        elif a == '--tpl':
            raise SystemExit('--tpl 需要目录参数，写作 --tpl=<目录>')
    files = []
    for a in args:
        hit = sorted(glob.glob(a))
        if not hit:
            raise SystemExit(f'找不到文件: {a}')
        files += hit
    if not files:
        raise SystemExit(__doc__)

    job_args = [(f, os.path.join(tpl_dir, os.path.basename(f)) if tpl_dir else None)
                for f in files]
    bad = 0
    total_trans = 0
    total_rows = 0
    partial = False
    for f, (errs, notes, n_tr, n_rows) in zip(
            files, run_jobs(_check_job, job_args, jobs)):
        total_trans += n_tr
        total_rows += n_rows
        for msg in notes:
            print(msg)
        if n_tr < n_rows:
            partial = True
            print(f'{f}: 已译 {n_tr}/{n_rows} 条（未译 {n_rows - n_tr}）')
        else:
            print(f'{f}: 已译 {n_tr}/{n_rows} 条')
        for e in errs:
            bad += 1
            print('  FAIL:', e)
    note = ''
    if partial and not allow_partial:
        note = '（存在未译条目；全部完成前请知悉）'
    print(f'结构检查: 问题 {bad} 处，共 {total_trans}/{total_rows} 条已译{note}')
    sys.exit(1 if bad else 0)

if __name__ == '__main__':
    main()
