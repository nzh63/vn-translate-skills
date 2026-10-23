# -*- coding: utf-8 -*-
# 块级引号配平自动修复：把译文的引号净差恢复到与原版一致（check_punct --src 的配套修复器）
# 仅用标准库。
#
# 背景（详见 check_punct.py 与 references/ks-format.md）:
#   多行对白在日文里把「开在首行、」闭在末行，行级各自不配对是正常的。真错误是
#   「在续行行首凭空多开了一个 『」或丢了收尾」，在游戏里渲染成一段话中间冒出
#   未闭合引号。本脚本只做两件事：
#     1) 删掉续行行首凭空多出的开号（该行日文不以开号起头，且不是消息首行）；
#     2) 消息尾行漏了闭号时，在尾行可见文本末尾补上闭号。
#   成对增补（叙述强调『…』）与 "…"→『…』 成对转换不产生净差，原样保留。
#
# 用法: python fix_quotes.py --src=<.ks目录> <work.tsv...>
#   只改写 work 文件最后一列（初译/生效译文），其他列/注释原样保留；
#   重复运行安全（已配平的不再动）。
import os, re, sys, glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ks_tool import load_ks, iter_blocks, TAG_RE
from _trans_common import iter_rows

QTYPES = (('「', '」'), ('『', '』'), ('"', '"'), ("'", "'"), ('“', '”'), ('‘', '’'))
TRAIL = re.compile(r'(?:[ \t]*\[[^\]]*\])*[ \t]*$')


def vis(s):
    return TAG_RE.sub('', s).strip()


def find_ks(src_dir, stem):
    for root, _d, fns in os.walk(src_dir):
        if (stem + '.ks') in fns:
            return os.path.join(root, stem + '.ks')
    return None


def load_cn(path):
    out = {}
    for ln, idx, cols in iter_rows(path):
        if idx is not None and len(cols) >= 2:
            out[idx] = cols[-1]  # 生效译文列（work=初译；proof 校对已合并过的文件亦然）
    return out


def net(s, ti):
    op, cl = QTYPES[ti]
    if op == cl:
        return s.count(op)
    return s.count(op) - s.count(cl)


def fix_block(block, cn, changed):
    """对单个 tb 块把 CN 净差修到与 JP 一致。返回 (是否还有未决, 说明)"""
    text_rows = [(idx, jp) for (_n, idx, _sp, jp, kind) in block if kind == 'text']
    jnet = [0] * len(QTYPES)
    for _i, jp in text_rows:
        for t in range(len(QTYPES)):
            jnet[t] += net(jp, t)
    # 逐类型处理
    notes = []
    for t in range(len(QTYPES)):
        op, cl = QTYPES[t]
        if op == cl:
            continue  # 同字符类型：成对转换合法，不在本修复器职责内
        for _guard in range(40):
            cnet = sum(net(cn.get(i, ''), t) for i, _jp in text_rows)
            if jnet[t] == cnet:
                break
            delta = jnet[t] - cnet
            if delta < 0:
                need = -delta
                # 情形1：尾行丢了闭号（JP 尾行以闭号收尾而 CN 没收）
                last_jp = text_rows[-1][1]
                last_i = text_rows[-1][0]
                if vis(last_jp).endswith(cl) and not vis(cn.get(last_i, '')).endswith(cl):
                    cn[last_i] = add_close(cn[last_i], cl)
                    changed.append((last_i, t, f'补{cl}'))
                    continue
                # 情形2：删续行行首多开的开号
                done = 0
                for i, jp in text_rows[1:]:  # 跳过消息首行
                    if done >= need:
                        break
                    cur = cn.get(i)
                    if not cur or not vis(cur).startswith(op):
                        continue
                    if vis(jp).startswith(op):
                        continue  # 日文本行也以开号起头，不是凭空加
                    cn[i] = remove_first(cur, op)
                    changed.append((i, t, f'删行首{op}'))
                    done += 1
                if done:
                    continue
                # 情形3：内部多余成对开号（罕见），人工
                notes.append(f'类型{op}{cl} 盈余 {need} 无法自动定位（人工）')
                break
            else:
                notes.append(f'类型{op}{cl} 缺 {delta} 个开号（人工）')
                break
    return notes


def remove_first(s, ch):
    """删掉可见文本开头的 ch（若行首有空位/标签则跳到第一个可见字符处再删）"""
    p = 0
    while p < len(s) and (s[p] in ' \t' or s[p] == '['):
        if s[p] == '[':
            m = TAG_RE.match(s, p)
            if not m:
                break
            p = m.end()
        else:
            p += 1
    if p < len(s) and s[p] == ch:
        return s[:p] + s[p + 1:]
    # 兜底：找不到可见起始 ch，则删首次出现
    q = s.find(ch)
    if q >= 0:
        return s[:q] + s[q + 1:]
    return s


def add_close(s, ch):
    """在可见文本末尾、尾随标签之前插入闭号"""
    m = TRAIL.search(s)
    start = m.start() if m else len(s)
    return s[:start] + ch + s[start:]


def fix_file(path, src_dir):
    stem = os.path.splitext(os.path.basename(path))[0]
    ks = find_ks(src_dir, stem)
    if not ks:
        print(f'{path}: 找不到 {stem}.ks，跳过')
        return
    text, _ = load_ks(ks)
    cn = load_cn(path)
    changed = []
    unresolved = []
    for block in iter_blocks(text):
        unresolved += fix_block(block, cn, changed)
    if not changed and not unresolved:
        print(f'{path}: 已配平，无改动')
        return
    # 写回
    lines = open(path, encoding='utf-8').read().split('\n')
    out = []
    for line in lines:
        cols = line.split('\t')
        try:
            idx = int(cols[0])
        except (ValueError, IndexError):
            out.append(line)
            continue
        if idx in cn:
            # 只改最后一列
            cols[-1] = cn[idx]
            out.append('\t'.join(cols))
        else:
            out.append(line)
    open(path, 'w', encoding='utf-8', newline='\n').write('\n'.join(out))
    print(f'{path}: 自动修复 {len(changed)} 处' + ('；' + '；'.join(unresolved) if unresolved else ''))
    for i, t, note in changed[:8]:
        print(f'   #{i} {note}')
    if len(changed) > 8:
        print(f'   …共 {len(changed)} 处')


def main():
    args = sys.argv[1:]
    src_dir = None
    for a in args:
        if a.startswith('--src='):
            src_dir = a.split('=', 1)[1]
            args.remove(a)
            break
    if not src_dir or not os.path.isdir(src_dir):
        raise SystemExit('需要 --src=<.ks 所在目录>')
    files = []
    for a in args:
        files += sorted(glob.glob(a))
    if not files:
        raise SystemExit('没有输入 work tsv')
    for p in files:
        fix_file(p, src_dir)


if __name__ == '__main__':
    main()
