# -*- coding: utf-8 -*-
# 跨行引号镜像检查（check_punct 的补充，非封包门禁）。
# 背景：Siglus 一句台词可拆成多条字符串，开/收引号可能分属两条（日文原文亦然）。
#   逐行"配平"检查会产生大量误报；本脚本改为【中日镜像】口径：
#     对每条生效译文，比较其与日文原文的 引号总数（「＋『 为开，」＋』 为闭）：
#       · 日文无任何引号而译文有          → suspect（凭空加）
#       · 日文有而译文全无                → suspect（凭空丢）
#       · 开/闭总数不相等（按方向分别比） → suspect（结构不镜像）
#   另输出三类可直接机械修复的子集（--fixes）：
#       A 补闭：日文以收括号结尾、译文结尾无收括号且闭数偏少
#       B 补开：日文以开括号开头、译文开头无开括号且开数偏少
# 用法:
#   python check_quotes_mirror.py <work或proof的tsv>... [--jobs=N]  # 列 suspect
#   python check_quotes_mirror.py <tsv>... --fixes                 # 只列 A/B 可机械修复项
# 注意：『』在译文中常作内层引用（对应日文 “ ” 或嵌套「），故两侧都合并计数；
#       早期文件存在"每条消息自带「…」"的合法风格，其多开/多闭属风格差异，
#       是否改为镜像由各批次校对代理按所在文件内部一致性决定，本工具只报告不强制。
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _trans_common import iter_rows, arg_jobs, run_jobs

OPEN = '「『'
CLOSE = '」』'

def counts(s):
    return s.count('「') + s.count('『'), s.count('」') + s.count('』')

def scan(path, want_fixes=False):
    out = []
    for ln, idx, cols in iter_rows(path):
        if idx is None or len(cols) < 6:
            if len(cols) >= 4:  # work 4列格式也支持
                pass
            else:
                continue
        eff = (cols[4] if len(cols) > 4 and cols[4].strip() else cols[3]).strip()
        if not eff:
            continue
        jp = cols[2] if len(cols) > 2 else ''
        jo, jc = counts(jp)
        co, cc = counts(eff)
        tag = f'{os.path.basename(path)}:{ln} #{idx}'
        if want_fixes:
            if jp.rstrip().endswith(tuple(CLOSE)) and not eff.rstrip().endswith(tuple(CLOSE)) and cc < jc:
                out.append((tag, 'A补闭', eff.rstrip() + '」'))
            elif jp.lstrip().startswith(tuple(OPEN)) and not eff.lstrip().startswith(tuple(OPEN)) and co < jo:
                out.append((tag, 'B补开', '「' + eff.lstrip()))
            continue
        kinds = []
        if jo == jc == 0 and (co or cc):
            kinds.append('凭空加')
        elif (jo or jc) and co + cc == 0:
            kinds.append('凭空丢')
        else:
            if co != jo:
                kinds.append(f'开数 {co}≠{jo}')
            if cc != jc:
                kinds.append(f'闭数 {cc}≠{jc}')
        if kinds:
            frag = eff if len(eff) <= 40 else eff[:40] + '…'
            out.append((tag, '/'.join(kinds), frag))
    return out

def _scan_job(t):
    return scan(*t)


def main():
    rest, jobs = arg_jobs(sys.argv[1:])
    want_fixes = '--fixes' in rest
    paths = [a for a in rest if not a.startswith('--')]
    if not paths:
        raise SystemExit(__doc__)
    total = 0
    for probs in run_jobs(_scan_job, [(p, want_fixes) for p in paths], jobs):
        for tag, kind, frag in probs:
            total += 1
            print(f'{tag} [{kind}] {frag}')
    print(f'--- 共 {total} 处')
    sys.exit(1 if total else 0)

if __name__ == '__main__':
    main()
