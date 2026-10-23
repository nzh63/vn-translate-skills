# -*- coding: utf-8 -*-
# 翻译工作区生成器：.ss → tpl(原文模板,只读) + work(译文骨架,不覆盖已有文件)
#
# 用法:
#   python make_template.py <ss目录或单个.ss> <输出目录> [--all] [--jobs=N]
# 产物（按 ss 主名命名）:
#   <out>/tpl/<name>.tsv   索引 \t 说话人 \t 日文原文          （只读参考, 可随时重生成）
#   <out>/work/<name>.tsv  索引 \t 说话人 \t 日文原文 \t 译文  （只填最后一列; 已存在则跳过）
# --all: 兜底模式，转储字符串表全部非空条目（索引 \t 内容），自行甄别待译串。
#
# 索引 = .ss 字符串表下标（按索引升序 = 剧情顺序）。说话人列是启发式推断
# （见 extract_text.looks_speaker），空＝旁白/独白；误判不影响重建，只影响参考。
# 控制标记（％开头）与纯 ASCII 命令串不进模板——它们必须保持原样。
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ss_tool import Ss
from extract_text import translatable, looks_speaker
from _trans_common import arg_jobs, run_jobs


def collect_rows(strings):
    """→ [(idx, speaker, text)] 升序。说话人=前一条人名形且本条以「开头。"""
    entries = [(i, t) for i, t in enumerate(strings) if t and translatable(t)]
    rows = []
    for j, (idx, t) in enumerate(entries):
        spk = ''
        if j > 0 and looks_speaker(entries[j - 1][1], t):
            spk = entries[j - 1][1]
        rows.append((idx, spk, t))
    return rows


def process(path, outdir, all_mode=False):
    name = os.path.splitext(os.path.basename(path))[0]
    ss = Ss(open(path, 'rb').read())
    os.makedirs(os.path.join(outdir, 'tpl'), exist_ok=True)
    os.makedirs(os.path.join(outdir, 'work'), exist_ok=True)

    if all_mode:
        rows = [(i, '', s) for i, s in enumerate(ss.strings)
                if s and not s.startswith('％')]
    else:
        rows = collect_rows(ss.strings)

    tpl_path = os.path.join(outdir, 'tpl', f'{name}.tsv')
    with open(tpl_path, 'w', encoding='utf-8', newline='\n') as f:
        for i, spk, jp in rows:
            f.write(f'{i}\t{spk}\t{jp}\n')
    work_path = os.path.join(outdir, 'work', f'{name}.tsv')
    made = '跳过(已存在)'
    if not os.path.exists(work_path):
        with open(work_path, 'w', encoding='utf-8', newline='\n') as f:
            for i, spk, jp in rows:
                f.write(f'{i}\t{spk}\t{jp}\t\n')
        made = '新建'
    return len(rows), f'{name}: {len(rows)} 条 -> tpl/{name}.tsv, work/{name}.tsv {made}'


def _job(t):
    return process(*t)


def main():
    rest, jobs = arg_jobs(sys.argv[1:])
    all_mode = '--all' in rest
    a = [x for x in rest if x != '--all']
    if len(a) != 2:
        raise SystemExit(__doc__)
    src, outdir = a
    targets = [src] if os.path.isfile(src) else \
        sorted(os.path.join(src, n) for n in os.listdir(src) if n.endswith('.ss'))
    if not targets:
        raise SystemExit(f'{src} 中没有 .ss')
    total = 0
    for n, msg in run_jobs(_job, [(t, outdir, all_mode) for t in targets], jobs):
        total += n
        print(msg)
    print(f'合计 {total} 条')


if __name__ == '__main__':
    main()
