# -*- coding: utf-8 -*-
# 剧情顺序可读全文导出 + 阅读稿行渲染（供 make_proof.py 复用）
# 仅用标准库。
#
# 用法: python extract_text.py <scn目录> <text目录>
import os, re, sys, glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ks_tool import load_ks, KsParse, TAG_RE, NAME_PREFIX


def render_line(spk, body):
    """阅读稿行渲染：内嵌说话人（【名】…）原样；自带引号直接拼接；否则补全角引号。"""
    if not spk:
        return body
    if body.startswith(('「', '『')):
        return spk + body
    if body.startswith('【'):
        return body
    return f'{spk}「{body}」'


def _scn_files(scn_dir):
    files = sorted(glob.glob(os.path.join(scn_dir, '**', '*.ks'), recursive=True))
    if not files:
        raise SystemExit(f'{scn_dir}: 找不到 .ks 文件')
    return files


def render_story(scn_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    total = 0
    for path in _scn_files(scn_dir):
        text, _ = load_ks(path)
        stem = os.path.splitext(os.path.basename(path))[0]
        rel = os.path.relpath(path, scn_dir)
        out_path = os.path.join(out_dir, os.path.splitext(rel)[0].replace(os.sep, '_') + '.txt')
        with open(out_path, 'w', encoding='utf-8', newline='\n') as w:
            def hdr(spk):
                if not spk:
                    return '【旁白】'
                return spk if spk.startswith('【') else f'【{spk}】'

            def flush(cur):
                if cur and cur[1]:
                    w.write(f'◆{rel} ' + hdr(cur[0]) + '\n'
                            + '\n'.join(cur[1]) + '\n\n')
                    return len(cur[1])
                return 0
            cur, cnt = None, 0
            for no, idx, sp, line, kind in KsParse(text).rows:
                if kind == 'name':
                    cnt += flush(cur)
                    cur = (line.strip(), [])   # 名字行本体即后续文本的说话人
                    continue
                body = TAG_RE.sub('', line).strip()
                if cur is None or sp != cur[0]:
                    cnt += flush(cur)
                    cur = (sp, [])
                cur[1].append(body)
            cnt += flush(cur)
        total += cnt
        print(out_path)
    print(f'合计 {total} 条')


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    render_story(sys.argv[1], sys.argv[2])


if __name__ == '__main__':
    main()
