# 译文行宽/行数检查：换行记号违规=致命；行宽超限默认为警告（--strict 时才致命）
# 用法: python validate_text.py <a.tsv> <b.tsv> ... [--width=N] [--lines=M] [--quote=「] [--soft] [--jobs=N]
#   多文件按序合并，后者覆盖前者 —— "work 校对稿在前、proof 在后"即"改稿优先"
# 行宽规则默认 27 字 × 3 行（多数 KiriKiri 文本窗的典型上限），务必先实测目标游戏校准。
# SiglusEngine 等消息窗自动折行的引擎已实测可安全容纳超长串：超宽只提示不阻断，
#   需要恢复"超宽即失败"的旧口径时加 --strict。真实换行符/字面 \n 永远是致命错误。
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _trans_common import check_lines, iter_rows, cn_of, jp_of, arg_jobs, run_jobs

def parse_tsv(path):
    """并发 worker：单文件解析 → ({索引: (原文|None, 生效译文)}, 已译条数)。
    坏行（首列非整数）即致命，报错文案与旧 load_map 保持一字不差。"""
    out, n = {}, 0
    for ln, idx, cols in iter_rows(path):
        if idx is None:
            raise SystemExit(
                f'{path}:{ln}: 数据行首列不是整数索引 —— 文件很可能被真实换行符'
                f'拆坏（字面 \\n 被解码），先修复 TSV 再继续:\n    {cols[:3]}...')
        cn = cn_of(cols).strip()
        if not cn:
            continue
        out[idx] = (jp_of(cols), cn)
        n += 1
    return out, n

def main():
    args = sys.argv[1:]
    W, L, Q = 27, 3, '「'
    soft = '--soft' in args
    args, jobs = arg_jobs(args)
    opts = [a for a in args if a.startswith('--')]
    args = [a for a in args if not a.startswith('--')]
    for a in opts:
        if a.startswith('--width='): W = int(a.split('=', 1)[1])
        elif a.startswith('--lines='): L = int(a.split('=', 1)[1])
        elif a.startswith('--quote='): Q = a.split('=', 1)[1]

    files = []
    for a in args:
        hit = sorted(glob.glob(a))
        if not hit: raise SystemExit(f'找不到文件: {a}')
        files += hit
    if not files:
        raise SystemExit(__doc__)

    per = run_jobs(parse_tsv, files, jobs)
    eff = {}     # 多文件按序合并，后者覆盖前者 —— 与旧 load_map 合并语义一致
    for m, _ in per:
        eff.update(m)
    fatal = []   # 换行记号类：永远致命
    warn = []    # 超宽/超行：--soft 时仅警告
    for idx in sorted(eff):
        jp, cn = eff[idx]
        for e in check_lines(cn, W, L, Q):
            item = f'[#{idx}] {e}  <<{cn[:40]}>>'
            if '换行' in e:
                fatal.append(item)
            else:
                warn.append(item)

    for f, (_, n) in zip(files, per):
        print(f'{f}: 已译 {n} 条')
    head = (f'合计生效译文 {len(eff)} 条（多文件时按索引去重，勿当全量读）—— '
            f'致命 {len(fatal)} 处，超宽警告 {len(warn)} 处 (宽度≤{W}, 行数≤{L}'
            + ('，软模式)' if soft else ')'))
    print(head)
    for e in warn:
        print('  [warn]', e)
    for e in fatal:
        print('  [fatal]', e)
    sys.exit(1 if fatal or (not soft and warn) else 0)

if __name__ == '__main__':
    main()
