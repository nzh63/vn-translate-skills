# -*- coding: utf-8 -*-
# 从 .ss 提取按剧情顺序排列的可读文本（译者把握语境用）。
# Siglus 的字符串表索引序即剧情顺序：旁白/对白依次排布，说话人名通常紧贴在
# 对白串的前一条；「全角％＋全角字母」的纯标记串（％Ｂ、％Ｄ…）是引擎控制标记，
# 不是文本。注意：％标记**开头**但后面跟正文的串（名字栏「％Ｂ・雛多」、旁白
# 「％Ｂくんのことを考えてると」）是会显示的文本，必须进模板——占位符原样保留，
# 只译后面的正文。
#
# 用法: python extract_text.py <.ss目录或单个.ss> <输出目录> [--jobs=N]
#
# 本文件同时是"场景文本模型"的共享库：
#   is_control(s)    是否控制标记（％开头）
#   has_prose(s)     是否含可译正文（全角字符判定）
#   translatable(s)  进翻译模板的判据
#   looks_speaker()  说话人启发式判定
#   story_lines()    → [(idx, speaker|'', text)] 按索引顺序的剧情文本
import sys, os, glob, re, unicodedata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ss_tool import Ss
from _trans_common import arg_jobs, run_jobs

MAX_SPEAKER_LEN = 12          # 说话人名候选的最大长度（超过按旁白处理）


# 纯控制标记：全角％＋全角字母/数字（％Ｂ、％Ｄ、％Ｅ…）或孤立％，整串都是标记。
# ％标记后跟正文的串（％Ｂ・雛多／％Ｂくん…）不是控制标记，是可译文本。
_PCT_CTRL_RE = re.compile(r'^％[Ａ-Ｚａ-ｚ０-９]*$')


def is_control(s):
    """全角％开头的纯引擎控制标记。绝不可翻译。"""
    return bool(_PCT_CTRL_RE.match(s))


def has_prose(s):
    """含全角（宽/全宽）字符即视为可译文本；纯 ASCII 是命令/标识符，不进模板。"""
    s2 = s.replace('\u3000', '')
    return bool(s2.strip()) and any(
        unicodedata.east_asian_width(c) in ('W', 'F') for c in s2)


def translatable(s):
    return (not is_control(s)) and has_prose(s)


def looks_speaker(text, next_text):
    """短、无引号、无控制记号，且下一条以「开头 → 判为下一条的说话人。
    这是启发式：误判只影响阅读稿展示与模板说话人列，不影响重建数据本身。"""
    if not text or len(text) > MAX_SPEAKER_LEN or '\n' in text:
        return False
    if any(c in text for c in '「」『』％【】'):
        return False
    return bool(next_text) and next_text.startswith('「')


def story_lines(strings):
    """→ [(idx, speaker, text)]：仅可译条目；speaker 为推断说话人（可为空）。
    当前条的对白若前一条是"说话人形"，就把那条人名挂到本条上；
    人名本体仍是独立条目（角色名同样需要翻译）。"""
    entries = [(i, t) for i, t in enumerate(strings) if t and translatable(t)]
    lines = []
    for j, (idx, text) in enumerate(entries):
        if j > 0 and looks_speaker(entries[j - 1][1], text):
            spk = entries[j - 1][1]
        else:
            spk = ''
        lines.append((idx, spk, text))
    return lines


def render_line(spk, body):
    """阅读稿行渲染：内嵌说话人（【名】…）原样；自带引号直接拼接；否则补全角引号。"""
    if not spk:
        return body
    if body.startswith(('「', '『')):
        return spk + body
    if body.startswith('【'):
        return body
    return f'{spk}「{body}」'


def process(path, outdir):
    ss = Ss(open(path, 'rb').read())
    name = os.path.splitext(os.path.basename(path))[0]
    out_lines = [f'◆---- name: {name}']
    for idx, spk, text in story_lines(ss.strings):
        out_lines.append(render_line(spk, text))
    open(os.path.join(outdir, name + '.txt'), 'w', encoding='utf-8',
         newline='\n').write('\n'.join(out_lines) + '\n')
    return len(out_lines) - 1


def _job(t):
    return process(*t)


def main():
    a, jobs = arg_jobs(sys.argv[1:])
    if len(a) != 2:
        raise SystemExit(__doc__)
    src, dst = a
    os.makedirs(dst, exist_ok=True)
    targets = [src] if os.path.isfile(src) else \
        sorted(glob.glob(os.path.join(src, '*.ss')))
    if not targets:
        raise SystemExit(f'{src} 中没有 .ss')
    ns = run_jobs(_job, [(t, dst) for t in targets], jobs)
    for t, n in zip(targets, ns):
        print(f'{os.path.splitext(os.path.basename(t))[0]}: {n} 行文本')
    print(f'TOTAL lines: {sum(ns)}')


if __name__ == '__main__':
    main()
