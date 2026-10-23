# -*- coding: utf-8 -*-
# 标点/引号规范扫描（校对辅助，非封包门禁；行宽/行数由 validate_text.py 把关）。
# 检查项（对每条生效译文）：
#   1) 引号配平：「」 计数相等、『』 计数相等
#   2) 弯引号/直引号残留：U+201C/D、U+2018/9 与 ASCII " '
#   3) 凭空加引号：说话人列为空（旁白/独白）且原文不以「开头，译文却以「开头。
#      判据是"与原文对齐"而非说话人本身——旁白转述台词时原文自带引号，保留是正确的。
#   4) 凭空丢引号：原文以「开头（游戏 UI 的对白引号），译文却没保留。
#   5) 半角标点 ,.!?;: 紧跟非 ASCII 字符之后——CJK 文本里混入半角点号；
#      "OK," "1,000" 这类英文/数字后缀不算（前一字符是 ASCII 就放过）
# 用法: python check_punct.py <work或proof的tsv>...
# 有问题退出码 1；输出 文件:行号 #索引 [类型] 片段。
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _trans_common import iter_rows, cn_of, jp_of, speaker_of

CURLY = '\u201c\u201d\u2018\u2019"\''
HALF = ',.!?;:'
LQ = '「'

def scan(path):
    problems = []
    for ln, idx, cols in iter_rows(path):
        if idx is None:
            continue
        cn = cn_of(cols)
        if not cn or not cn.strip():
            continue
        spk = speaker_of(cols)
        jp = jp_of(cols) or ''
        tag = f'{os.path.basename(path)}:{ln} #{idx}'
        frag = cn if len(cn) <= 44 else cn[:44] + '…'
        if cn.count(LQ) != cn.count('」'):
            problems.append((tag, '「」不配平', frag))
        if cn.count('『') != cn.count('』'):
            problems.append((tag, '『』不配平', frag))
        if any(ch in CURLY for ch in cn):
            problems.append((tag, '弯引号/直引号残留', frag))
        if not spk and cn.startswith(LQ) and not jp.startswith(LQ):
            problems.append((tag, '凭空加引号(旁白)', frag))
        if jp.startswith(LQ) and not cn.startswith(LQ):
            problems.append((tag, '凭空丢引号', frag))
        for i, ch in enumerate(cn):
            if ch in HALF and i > 0 and ord(cn[i - 1]) > 0x7f:
                problems.append((tag, '半角标点?', cn[max(0, i - 10):i + 8]))
                break
    return problems

def main():
    paths = sys.argv[1:]
    if not paths:
        raise SystemExit(__doc__)
    total = 0
    for p in paths:
        if not os.path.exists(p):
            raise SystemExit(f'no such file: {p}')
        probs = scan(p)
        total += len(probs)
        for tag, kind, frag in probs:
            print(f'{tag} [{kind}] {frag}')
    print(f'--- 共 {total} 处')
    sys.exit(1 if total else 0)

if __name__ == '__main__':
    main()
