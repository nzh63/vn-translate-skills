# 译文行宽/行数门禁：封包前必须全绿（退出码 0）
# 用法: python validate_text.py <a.tsv> <b.tsv> ... [--width=N] [--lines=M] [--quote=「]
#   多文件按序合并，后者覆盖前者 —— "work 校对稿在前、proof 在后"即"改稿优先"
# 行宽规则默认 27 字 × 3 行（多数 KiriKiri 文本窗的典型上限），务必先实测目标游戏校准：
#   游戏内截图量出可容纳字数，或查引擎字体渲染宽度；半角字符按 1 字保守计入。
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _trans_common import load_map, check_lines, count_translated, iter_rows, cn_of

def main():
    args = sys.argv[1:]
    W, L, Q = 27, 3, '「'
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

    eff = load_map(files)
    errs = []
    for idx in sorted(eff):
        jp, cn = eff[idx]
        for e in check_lines(cn, W, L, Q):
            errs.append(f'[#{idx}] {e}  <<{cn[:40]}>>')

    for f in files:
        print(f'{f}: 已译 {count_translated(f)} 条')
    print(f'合计生效译文 {len(eff)} 条（多文件时按索引去重，勿当全量读），'
          f'问题 {len(errs)} 处 (宽度≤{W}, 行数≤{L})')
    for e in errs:
        print('  ', e)
    sys.exit(1 if errs else 0)

if __name__ == '__main__':
    main()
