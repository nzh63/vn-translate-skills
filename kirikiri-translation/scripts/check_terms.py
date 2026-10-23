# -*- coding: utf-8 -*-
# 术语冻结检查：按术语表扫描译文，抓"禁用表记"与"该译未译"
# 用法: python check_terms.py <terms.tsv> <work.tsv...> [--strict]
# terms.tsv 格式（Tab 分隔，UTF-8；# 开头为注释）：
#   原文片段 \t 规范译文 [\t 禁用形式1[,禁用形式2…]]
#   例:  サクラ \t 樱 \t 櫻
#        ラーメン \t 拉面
# 规则（对每条已译文本）：
#   A. 日文原文含"原文片段"而译文缺"规范译文" → 警告（可能漏译术语）
#   B. 译文含任一"禁用形式" → 错误
#   C. --strict 时警告也计为失败。
# 术语表建议与翻译简报同源维护：角色名/称呼/拟声词/专名/回环台词全部入表并全程冻结。
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _trans_common import iter_rows, jp_of, cn_of

def load_terms(path):
    rules = []
    for ln, line in enumerate(open(path, encoding='utf-8'), 1):
        line = line.rstrip('\n')
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        cols = line.split('\t')
        if len(cols) < 2 or not cols[0].strip() or not cols[1].strip():
            print(f'警告: {path}:{ln} 格式异常，已跳过', file=sys.stderr)
            continue
        bad = [x.strip() for x in cols[2].split(',') if x.strip()] if len(cols) >= 3 else []
        rules.append((cols[0].strip(), cols[1].strip(), bad))
    return rules

def main():
    args = [a for a in sys.argv[1:] if a != '--strict']
    strict = '--strict' in sys.argv
    if len(args) < 2:
        raise SystemExit(__doc__)
    rules = load_terms(args[0])
    if not rules:
        raise SystemExit(f'{args[0]}: 没有有效规则')
    files = []
    for a in args[1:]:
        hit = sorted(glob.glob(a))
        if not hit:
            raise SystemExit(f'找不到文件: {a}')
        files += hit

    n_err = n_warn = n_checked = 0
    for f in files:
        for ln, idx, cols in iter_rows(f):
            cn = cn_of(cols)
            jp = jp_of(cols)
            if not cn or not cn.strip():
                continue
            n_checked += 1
            cn = cn.strip()
            for src, dst, banned in rules:
                if src in (jp or ''):
                    if dst not in cn:
                        n_warn += 1
                        print(f'WARN {f}:{ln} [#{idx}] 原文含「{src}」但译文缺规范译法「{dst}」 <<{cn[:30]}>>')
                hit = [b for b in banned if b in cn]
                if hit:
                    n_err += 1
                    print(f'ERR  {f}:{ln} [#{idx}] 译文含禁用表记 {hit} （应为「{dst}」） <<{cn[:30]}>>')
    print(f'术语检查: 已扫 {n_checked} 条 × {len(rules)} 条规则, '
          f'错误 {n_err}, 警告 {n_warn}' + (' (--strict)' if strict else ''))
    sys.exit(1 if n_err or (strict and n_warn) else 0)

if __name__ == '__main__':
    main()
