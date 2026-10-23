# -*- coding: utf-8 -*-
# 语音注册名与译文说话人名的绑定修复（voiceplay_ex 系）
# 仅用标准库。
#
# 背景（详见 references/ks-format.md「语音挂接」）:
#   引擎语音按「# 说话人行 的精确内容」查注册表 map_vo.vochara。注册发生在剧本里形如
#     [brainlabo_voconfig type=".ogg" number="1" buf="1" start="true" name="【ユリ】" storage="vo/lily"]
#   的调用（通常集中在 boot 剧本，如 first.ks；结局/分支可能另文件重注册，如 end2.ks）。
#   一旦 # 行被翻译，说话人名与 name= 键失配 → 语音静默不播、编号也不再推进（后续全错位）。
#   本脚本把「注册名」改成映射表里的译文名，使查找恢复。映射表由译文说话人名普查得出，
#   是项目产物，不属于脚本。
#
# 用法:
#   python voice_namefix.py <map.tsv> <ks...>
#     map.tsv: 每行 「日文注册名 \t 译文名」（如 【ユリ】\t【尤莉】；不含 #）
#     仅改写 [brainlabo_voconfig ...] / [voconfig ...] 标签的 name="..."，其余一律不动。
#     自动识别 \r\n / \n 行尾并在写回时保持；幂等（译文名已匹配则跳过）。
import os, re, sys

NAME_RE = re.compile(r'(\[(?:brainlabo_)?voconfig\b[^\]]*?\bname=")([^"]*)(")')
EOL = None


def read_map(path):
    m = {}
    for ln, line in enumerate(open(path, encoding='utf-8'), 1):
        line = line.rstrip('\r\n')
        if not line.strip() or line.startswith('#'):
            continue
        cols = line.split('\t')
        if len(cols) != 2:
            raise SystemExit(f'{path}:{ln} 需两列（注册名\t译文名）')
        m[cols[0].strip()] = cols[1].strip()
    if not m:
        raise SystemExit(f'{path}: 空映射表')
    return m


def fix_file(path, name_map):
    raw = open(path, 'rb').read()
    bom = raw.startswith(b'\xef\xbb\xbf')
    if bom:
        raw = raw[3:]
    eol = '\r\n' if b'\r\n' in raw else '\n'
    text = raw.decode('utf-8').replace('\r\n', '\n').replace('\r', '\n')
    hits, changes = [], 0

    def repl(m):
        nonlocal changes
        jp, zh = m.group(2), name_map.get(m.group(2))
        if zh and zh != jp:
            changes += 1
            hits.append((jp, zh))
            return m.group(1) + zh + m.group(3)
        return m.group(0)

    out = NAME_RE.sub(repl, text)
    data = (('\ufeff' if bom else '') + out.replace('\n', eol)).encode('utf-8')
    open(path, 'wb').write(data)
    for jp, zh in hits:
        print(f'  {path}: name="{jp}" → name="{zh}"')
    if not hits:
        print(f'  {path}: 无改动（未匹配到映射表中的注册名）')
    return changes


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    name_map = read_map(sys.argv[1])
    total = 0
    for ks in sys.argv[2:]:
        total += fix_file(ks, name_map)
    print(f'共改写 {total} 处')


if __name__ == '__main__':
    main()
