# -*- coding: utf-8 -*-
# 从 .scn 提取按剧情顺序排列的可读文本（译者把握语境用）
# 输出每行: 说话人「对白」 或 独白/旁白（无前缀）；场景边界写 ◆---- label 行（非游戏文本）
#
# 用法: python extract_text.py <scn目录或单个.scn> <输出目录>
# 依赖 KAGParserEx 风格结构（scenes[].lines 以 1-based 序号引用 texts 表，
# startline 命令携带说话人标签）——见 references/psb-scn-format.md §6。
import sys, os, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from psb_tool import load_scn

def find_root(psb):
    """顶层值里找含 scenes 键的字典；找不到就取第一个顶层值"""
    for v in psb.value:
        if isinstance(v, dict) and 'scenes' in v:
            return v
    return psb.value[0] if psb.value else {}

def scene_texts(scene):
    """按 lines 顺序产出 (说话人, 正文)。int 行引用 texts[tid-1]；
    startline 命令(ln[3]=='name')切换当前说话人并延续到后续无标签条目。"""
    texts = scene.get('texts') or []
    out = []
    speaker = ''
    for ln in scene.get('lines') or []:
        if isinstance(ln, int):
            tid = ln - 1
            if 0 <= tid < len(texts):
                t = texts[tid]
                spk = t[0] if isinstance(t[0], str) else ''
                out.append((spk or speaker, extract_body(t)))
        elif isinstance(ln, list) and ln and ln[0] == 'startline':
            if len(ln) >= 6 and ln[3] == 'name':
                speaker = ln[4] or ''
    return out

def extract_body(entry):
    """texts 条目的显示正文 = 各分段 seg[1] 连接（履历副本 seg[3:] 不参与）"""
    parts = []
    for seg in entry[1]:
        if isinstance(seg, list) and len(seg) >= 2 and isinstance(seg[1], str):
            parts.append(seg[1])
    return ''.join(parts)

def process(path, outdir):
    psb = load_scn(path)
    d = find_root(psb)
    name = d.get('name') or os.path.splitext(os.path.basename(path))[0]
    out_lines = []
    for sc in d.get('scenes', []):
        label = sc.get('label', '')
        title = sc.get('title')
        out_lines.append(f'◆---- label: {label}' + (f'  title: {title}' if title else ''))
        for sp, body in scene_texts(sc):
            if sp:
                out_lines.append(f'{sp}「{body}」' if not body.startswith('「') else f'{sp}{body}')
            else:
                out_lines.append(body)
    text = '\n'.join(out_lines)
    open(os.path.join(outdir, name), 'w', encoding='utf-8', newline='\n').write(text + '\n')
    n = sum(1 for l in out_lines if l and not l.startswith('◆'))
    print(f'{name}: {n} 行文本')
    return n

def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, dst = sys.argv[1:3]
    os.makedirs(dst, exist_ok=True)
    targets = [src] if os.path.isfile(src) else \
        sorted(glob.glob(os.path.join(src, '*.scn')))
    if not targets:
        raise SystemExit(f'{src} 中没有 .scn')
    total = 0
    for t in targets:
        total += process(t, dst)
    print(f'TOTAL lines: {total}')

if __name__ == '__main__':
    main()
