# -*- coding: utf-8 -*-
# 标点/引号规范扫描（校对辅助 + 块级引号配平门禁；行宽/行数由 validate_text.py 把关）。
# 检查项：
#   1) 首尾符号对齐：日语原文行的开头/结尾若为 「」『』【】 之一，则译文的
#      开头/结尾必须是同一符号。不做全行计数对齐——中文在行中可能因内层引用
#      （『』）、语序调整等与日文计数不同，属正常；只卡首尾这种影响阅读的
#      结构性位置。原文首尾无引号时不检查（避免旁白/无引号行误报）。
#   2) 弯引号/直引号残留：U+201C/D、U+2018/9 与 ASCII " '
#   3) 凭空加引号：说话人列为空（旁白/独白）且原文不以「开头，译文却以「开头。
#      判据是"与原文对齐"而非说话人本身——旁白转述台词时原文自带引号，保留是正确的。
#   4) 半角标点 ,.!?;: 紧跟非 ASCII 字符之后——CJK 文本里混入半角点号；
#      "OK," "1,000" 这类英文/数字后缀不算（前一字符是 ASCII 就放过）
#   5) 块级引号净差（需要 --src=<.ks 所在目录>）：同一个 [tb_start_text] 块（一条
#      消息）内，每种引号的 开-闭 净差必须与原文一致。日文常把多行对白的「开在首行、
#      」闭在末行，行级各自不配对是正常的，所以按块算净差。能拦住的真错误：
#      凭空在续行行首多开「、丢了收尾」，而不误报合法的成对增补（叙述强调『』）
#      与 "…"→『…』 成对转换。
# 用法: python check_punct.py <work或proof的tsv>... [--src=<scenario目录>] [--jobs=N]
# 有问题退出码 1；输出 文件:行号 #索引 [类型] 片段。
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _trans_common import iter_rows, cn_of, jp_of, speaker_of, arg_jobs, run_jobs
from ks_tool import load_ks, iter_blocks, KsParse

CURLY = '\u201c\u201d\u2018\u2019"\''
HALF = ',.!?;:'
LQ = '「'
# 首尾对齐用的成对符号：日语行首/尾若是这些，译文首/尾须同符号
PAIRS = (('「', '」'), ('【', '】'))
# 引号类型：块级净差按类型分别统计（ASCII 双引号也会出现在日文里当引号用）
QTYPES = (('「', '」'), ('『', '』'), ('"', '"'), ("'", "'"), ('“', '”'), ('‘', '’'))


def qnet(s, types=QTYPES):
    """返回 {type_index: 净差}；同字符类型（ASCII " ' 等开闭同字）返回总出现次数。"""
    d = [0] * len(types)
    for ch in s:
        for i, (op, cl) in enumerate(types):
            if ch == op:
                d[i] += 1 if op != cl else 0  # 同字符走末尾统一 +1
            elif ch == cl:
                d[i] -= 1
    for i, (op, cl) in enumerate(types):
        if op == cl:
            d[i] = s.count(op)  # 同字符：总次数（配对性由调用方用奇偶判断）
    return d


def _pair_ok(i, jp_n, cn_n):
    """同字符类型要求出现次数奇偶一致（'…'→『…』/‘…’ 这类成对转换允许）；异字符要求净差相等。"""
    op, cl = QTYPES[i]
    if op == cl:
        return (jp_n % 2) == (cn_n % 2)
    return jp_n == cn_n


def find_ks(src_dir, stem):
    for root, _d, fns in os.walk(src_dir):
        if (stem + '.ks') in fns:
            return os.path.join(root, stem + '.ks')
    return None


def block_problems(path, tsv_map, src_dir):
    """块级净差检查。tsv_map: idx -> (jp_of, cn_of 生效译文) 的 dict，来自本文件。"""
    stem = os.path.splitext(os.path.basename(path))[0]
    ks_path = find_ks(src_dir, stem)
    if not ks_path:
        return []
    text, _ = load_ks(ks_path)
    probs = []
    for bi, block in enumerate(iter_blocks(text)):
        jnet = [0] * len(QTYPES)
        cnet = [0] * len(QTYPES)
        rows_used = []
        for no, idx, sp, line, kind in block:
            if kind == 'name':
                continue
            jnet = [a + b for a, b in zip(jnet, qnet(line))]
            item = tsv_map.get(idx)
            if item is not None and (item[1] or '').strip():
                cnet = [a + b for a, b in zip(cnet, qnet(item[1]))]
                rows_used.append((idx, sp, line, item[1]))
            else:
                rows_used.append((idx, sp, line, ''))
        bad = [i for i in range(len(QTYPES))
               if not _pair_ok(i, jnet[i], cnet[i])]
        if not bad:
            continue
        name = f'{os.path.basename(path)} 块{bi+1}'
        for i in bad:
            op, cl = QTYPES[i]
            same = op == cl
            if same:
                kind = f'引号 {op}{cl} 原{jnet[i]} 译{cnet[i]}（奇偶不一致，成对增/删才算数）'
            else:
                delta = jnet[i] - cnet[i]
                kind = f'引号净差 {op}{cl} 原{jnet[i]:+d} 译{cnet[i]:+d}（{"凭空增引号" if delta < 0 else "引号缺失"}）'
            frag = rows_used[0][3][:50] if rows_used else ''
            for idx, sp, jp, cn in rows_used:
                if idx is not None and (cn or '') and qnet(cn)[i] != 0:
                    frag = cn[:50]
                    break
            probs.append((f'{os.path.basename(path)} #{rows_used[0][0] if rows_used else bi}', kind, frag))
            if i == bad[0] and len(bad) > 1:
                pass  # 类型循环里每条独立输出
    return probs


def scan(path, src_dir=None):
    problems = []
    tsv_map = {}
    for ln, idx, cols in iter_rows(path):
        if idx is None:
            continue
        cn = cn_of(cols)
        if not cn or not cn.strip():
            continue
        tsv_map[idx] = (jp_of(cols) or '', cn)
        spk = speaker_of(cols)
        jp = jp_of(cols) or ''
        tag = f'{os.path.basename(path)}:{ln} #{idx}'
        frag = cn if len(cn) <= 44 else cn[:44] + '…'
        for op, cl in PAIRS:
            if jp.startswith(op) and not cn.startswith(op):
                problems.append((tag, f'开头{op}缺失', frag))
            if jp.endswith(cl) and not cn.endswith(cl):
                problems.append((tag, f'结尾{cl}缺失', frag))
        if any(ch in CURLY for ch in cn):
            problems.append((tag, '弯引号/直引号残留', frag))
        if not spk and cn.startswith(LQ) and not jp.startswith(LQ):
            problems.append((tag, '凭空加引号(旁白)', frag))
        for i, ch in enumerate(cn):
            if ch in HALF and i > 0 and ord(cn[i - 1]) > 0x7f:
                problems.append((tag, '半角标点?', cn[max(0, i - 10):i + 8]))
                break
    if src_dir:
        problems.extend(block_problems(path, tsv_map, src_dir))
    return problems

def _scan_job(args):
    return scan(*args)


def main():
    args = sys.argv[1:]
    src_dir = None
    for a in args:
        if a.startswith('--src='):
            src_dir = a.split('=', 1)[1]
            args.remove(a)
            break
    paths, jobs = arg_jobs(args)
    if not paths:
        raise SystemExit(__doc__)
    for p in paths:
        if not os.path.exists(p):
            raise SystemExit(f'no such file: {p}')
    if src_dir and not os.path.isdir(src_dir):
        raise SystemExit(f'--src 目录不存在: {src_dir}')
    total = 0
    for probs in run_jobs(_scan_job, [(p, src_dir) for p in paths], jobs):
        total += len(probs)
        for tag, kind, frag in probs:
            print(f'{tag} [{kind}] {frag}')
    print(f'--- 共 {total} 处')
    sys.exit(1 if total else 0)

if __name__ == '__main__':
    main()

