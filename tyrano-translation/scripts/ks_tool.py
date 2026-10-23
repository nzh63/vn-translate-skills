# -*- coding: utf-8 -*-
# TyranoScript/TyranoBuilder 剧本 .ks 文本工具
# 仅用标准库。
#
# 用法:
#   python ks_tool.py info        <file.ks>                     # 文本块统计（排查用）
#   python ks_tool.py strings     <file.ks>                     # 全部可译行: 索引\t说话人\t内容
#   python ks_tool.py make_template <scn目录> <work区>           # tpl/<名>.tsv + work/<名>.tsv
#   python ks_tool.py extract_text <scn目录> <text目录>          # 剧情顺序可读全文
#   python ks_tool.py rebuild     <tpl.ks> <out.ks> <work.tsv> [proof.tsv]
#   python ks_tool.py verify      <tpl.ks> <out.ks> <work.tsv> [proof.tsv]   # 必须 MATCH
#
# 模型（详见 references/ks-format.md）:
#   - TyranoBuilder 的显示文本都在 [tb_start_text ...] ... [_tb_end_text] 块内；
#     '#' 开头行是说话人标签（#【ユリ】 / 空的 # = 旁白），其余含"标签外文字"的行是显示文本行。
#   - 一条译文 = 一个物理行（多行对白被 [r] 标签拆成多条连续行），索引序 = 剧情序。
#   - 行内 [xxx] 标签是结构（[r] 换行 [p] 翻页等），回写时标签序列必须与原文完全一致；
#     译文里绝不能出现真实换行或字面 \n。
#   - TSV: 索引 \t 说话人 \t 日文原文 \t 初译 [\t 校对 \t 备注]；生效译文 = 校对列非空优先。
import os, re, sys, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _trans_common import iter_rows, cn_of, speaker_of

TAG_RE = re.compile(r'\[[A-Za-z_][^\[\]]*\]')   # 行内标签：以字母/下划线开头，不跨行
NAME_PREFIX = '#'


def strip_tags(line):
    return TAG_RE.sub('', line)


def tags_of(line):
    return TAG_RE.findall(line)


class KsParse:
    """解析一个 .ks：产出按序的行记录。
    rows 元素: (行号0-based, 索引, 说话人, 原文, kind)；kind='text' 显示文本行 / 'name' 说话人标签行。
    name 行的原文不带 '#' 前缀（引擎 chara_ptext：显示在名字栏与 backlog 说话人栏）；
    空的裸 '#'（旁白名栏清空）是结构，不进表。"""

    def __init__(self, text):
        self.lines = text.split('\n')
        self.rows = []
        self._parse()

    def _parse(self):
        in_block = False
        in_iscript = False
        speaker = ''
        idx = 0
        for no, line in enumerate(self.lines):
            s = line.strip()
            if '[iscript]' in s:
                in_iscript = True
                continue
            if '[endscript]' in s:
                in_iscript = False
                continue
            if in_iscript:
                continue
            if s.startswith(';'):          # KAG 注释
                continue
            if '[tb_start_text' in s:
                in_block = True
                speaker = ''
                continue
            if '[_tb_end_text]' in s:
                in_block = False
                speaker = ''
                continue
            if not in_block:
                continue
            if s.startswith(NAME_PREFIX):
                nm = s[1:].strip()
                if nm:                     # 裸 '#' 是清空名字栏的结构
                    self.rows.append((no, idx, '', nm, 'name'))
                    idx += 1
                speaker = nm
                continue
            body = strip_tags(line)
            if body.strip():
                self.rows.append((no, idx, speaker, line, 'text'))
                idx += 1


def load_ks(path):
    with open(path, 'rb') as f:
        raw = f.read()
    if raw.startswith(b'\xef\xbb\xbf'):
        raw = raw[3:]
    for enc in ('utf-8', 'cp932'):
        try:
            # 统一换行：原始 .ks 多为 \r\n。行内 \r 若不归一，写进 TSV 后被
            # universal newlines 拆成两行（87 行读回变 151 行的实际事故）。
            # 引擎解析器对每行 $.trim，写回 \n 行尾安全。
            text = raw.decode(enc).replace('\r\n', '\n').replace('\r', '\n')
            return text, raw.startswith(b'\xef\xbb\xbf')
        except UnicodeDecodeError:
            continue
    raise SystemExit(f'{path}: 无法解码（既非 UTF-8 也非 CP932）')


def effective_translation(work_path, proof_path):
    """读 work(+proof)，返回 {索引: (生效译文, 说话人, 原文)}；数据行首列非整数 → 致命。"""
    out = {}
    for path in [work_path] + ([proof_path] if proof_path else []):
        for ln, idx, cols in iter_rows(path):
            if idx is None:
                raise SystemExit(f'{path}:{ln} 数据行首列不是整数（索引是唯一锚点，禁止损坏）: {cols[:2]}')
            if len(cols) < 4:
                raise SystemExit(f'{path}:{ln} 列数不足 4')
            trans = cols[3]
            if path == proof_path:
                # proof: 1索引 2说话人 3原文 4初译 5校对 6备注
                if len(cols) < 5:
                    raise SystemExit(f'{path}:{ln} 校对文件列数不足 6')
                # 校对列非空才覆盖；空=该条未校，沿用 work 初译，不得用空串顶掉
                if cols[4].strip():
                    trans = cols[4]
                else:
                    continue
            out[idx] = (trans, cols[1] if len(cols) > 1 else '', cols[2] if len(cols) > 2 else '')
    return out


def iter_blocks(text):
    """按 [tb_start_text]…[_tb_end_text] 把可译行分组。
    每块产出 [row, ...]，row=(行号0-based, 索引, 说话人, 原文行, kind)。
    iscript/注释行不计入任何块；name 行保留在块内（调用方按需跳过）。
    """
    ks = KsParse(text)
    by_no = {r[0]: r for r in ks.rows}
    blocks, cur, in_iscript = [], None, False
    for no, line in enumerate(ks.lines):
        s = line.strip()
        if '[iscript]' in s:
            in_iscript = True
            continue
        if '[endscript]' in s:
            in_iscript = False
            continue
        if in_iscript or s.startswith(';'):
            continue
        if '[tb_start_text' in s:
            cur = []
            continue
        if '[_tb_end_text]' in s:
            if cur:
                blocks.append(cur)
            cur = None
            continue
        if cur is not None:
            r = by_no.get(no)
            if r is not None:
                cur.append(r)
    if cur:
        blocks.append(cur)
    return blocks


def cmd_info(args):
    text, _ = load_ks(args[0])
    ks = KsParse(text)
    speakers = {}
    for _no, _i, sp, _t, _k in ks.rows:
        speakers[sp or '(旁白)'] = speakers.get(sp or '(旁白)', 0) + 1
    print(f'{args[0]}: 可译行 {len(ks.rows)} 条, 总行数 {len(ks.lines)}')
    for k, v in sorted(speakers.items(), key=lambda x: -x[1]):
        print(f'  {k}: {v}')


def cmd_strings(args):
    text, _ = load_ks(args[0])
    for _no, idx, sp, line, kind in KsParse(text).rows:
        print(f'{idx}\t{sp}\t{line}')


def _scn_files(scn_dir):
    files = sorted(glob.glob(os.path.join(scn_dir, '**', '*.ks'), recursive=True))
    if not files:
        raise SystemExit(f'{scn_dir}: 找不到 .ks 文件')
    return files


def cmd_make_template(args):
    scn_dir, work_dir = args[0], args[1]
    tpl_dir = os.path.join(work_dir, 'tpl')
    wdir = os.path.join(work_dir, 'work')
    os.makedirs(tpl_dir, exist_ok=True)
    os.makedirs(wdir, exist_ok=True)
    total = 0
    for path in _scn_files(scn_dir):
        stem = os.path.splitext(os.path.basename(path))[0]
        text, _ = load_ks(path)
        ks = KsParse(text)
        tpl_path = os.path.join(tpl_dir, stem + '.tsv')
        work_path = os.path.join(wdir, stem + '.tsv')
        if os.path.exists(work_path):
            print(f'跳过（work 已存在，绝不覆盖）: {work_path}')
            total += len(ks.rows)
            continue
        with open(tpl_path, 'w', encoding='utf-8', newline='\n') as t, \
             open(work_path, 'w', encoding='utf-8', newline='\n') as w:
            for _no, idx, sp, line, kind in ks.rows:
                row = f'{idx}\t{sp}\t{line}\n'
                t.write(row)
                w.write(row[:-1] + '\t\n')
        print(f'{stem}: {len(ks.rows)} 条 → {work_path}')
        total += len(ks.rows)
    print(f'合计 {total} 条可译行')


def cmd_extract_text(args):
    import extract_text
    extract_text.render_story(args[0], args[1])


def cmd_rebuild(args):
    tpl_path, out_path, work_path = args[0], args[1], args[2]
    proof_path = args[3] if len(args) > 3 else None
    text, _bom = load_ks(tpl_path)
    ks = KsParse(text)
    trans = effective_translation(work_path, proof_path)

    lines = text.split('\n')
    applied, skipped_bad = 0, []
    for no, idx, sp, orig, kind in ks.rows:
        if idx not in trans:
            continue
        eff = trans[idx][0]
        if not eff.strip():
            continue
        # 门禁 1：真实换行 / 字面 \n
        if '\n' in eff or '\r' in eff or chr(92) + 'n' in eff:
            skipped_bad.append(f'#{idx} 译文含换行或字面 \\n（Tyrano 换行用 [r] 标签）')
            continue
        if kind == 'name':
            # 名字行：回写时补 #；不允许标签（chara_ptext 的 name 参数）
            if tags_of(eff):
                skipped_bad.append(f'#{idx} 名字行不允许含标签')
                continue
            lines[no] = NAME_PREFIX + eff
        else:
            # 门禁 2：标签序列必须与原文完全一致（标签是结构）。
            # [ruby text=…] 的 text= 属可译显示内容（BRIEF §4.4），比较时把 ruby 归一为
            # 标签名本身：只校验种类/数量/位置，不校验注音值。
            def _norm(tags):
                return ['[ruby]' if t.startswith('[ruby') else t for t in tags]
            if _norm(tags_of(eff)) != _norm(tags_of(orig)):
                skipped_bad.append(f'#{idx} 标签序列不一致: 原{tags_of(orig)} vs 译{tags_of(eff)}')
                continue
            # 门禁 3：行首危险字符（会被引擎解析成注释/标签/转义）
            head = eff.lstrip()[:1]
            if head in (';', '@', '*', '_'):
                skipped_bad.append(f'#{idx} 译文不能以 {head!r} 开头（会被引擎误解析）')
                continue
            lines[no] = eff
        applied += 1

    if skipped_bad:
        raise SystemExit('rebuild 拒绝（以下条目未回写，译文列请修正后重跑）:\n  '
                         + '\n  '.join(skipped_bad))

    out_text = '\n'.join(lines)
    with open(out_path, 'w', encoding='utf-8', newline='\n') as w:
        w.write(out_text)
    print(f'rebuild: {out_path}  可译行 {len(ks.rows)}, 已应用译文 {applied} 条')


def cmd_verify(args):
    tpl_path, out_path, work_path = args[0], args[1], args[2]
    proof_path = args[3] if len(args) > 3 else None
    tpl_text, _ = load_ks(tpl_path)
    out_text, _ = load_ks(out_path)
    trans = effective_translation(work_path, proof_path)

    tpl_ks, out_ks = KsParse(tpl_text), KsParse(out_text)
    errs = []
    if len(tpl_ks.lines) != len(out_ks.lines):
        errs.append(f'总行数变化: {len(tpl_ks.lines)} → {len(out_ks.lines)}（结构被改动）')
    if len(tpl_ks.rows) != len(out_ks.rows):
        errs.append(f'可译行数变化: {len(tpl_ks.rows)} → {len(out_ks.rows)}')
    # 逐条比对生效译文
    expect, name_expect = {}, {}
    for no, idx, sp, orig, kind in tpl_ks.rows:
        eff = trans.get(idx, ('', '', ''))[0]
        val = eff if eff.strip() else orig
        if kind == 'name' and eff.strip():
            # KsParse 产出行不含 #，按剥皮值比对；带 # 与否另行行级检查
            name_expect[idx] = NAME_PREFIX + val
        expect[idx] = val
    actual = {idx: line for _no, idx, _sp, line, _k in out_ks.rows}
    nmismatch = 0
    for idx in sorted(expect):
        if actual.get(idx) != expect[idx]:
            errs.append(f'#{idx} 期望 {expect[idx]!r} vs 实际 {actual.get(idx)!r}')
            nmismatch += 1
        if nmismatch >= 10:
            errs.append('（仅显示前 10 条）')
            break
    for idx, line in name_expect.items():
        if line not in out_ks.lines:
            errs.append(f'#{idx} 名字行未带 # 回写: {line!r}')
    # 非可译行必须逐字节一致
    tpl_rows = {no for no, *_ in tpl_ks.rows}
    for no, (a, b) in enumerate(zip(tpl_ks.lines, out_ks.lines)):
        if no in tpl_rows:
            continue
        if a != b:
            errs.append(f'结构行被改动（第{no+1}行）: {a!r} → {b!r}')
            if len(errs) > 20:
                break
    if errs:
        raise SystemExit('verify 失败:\n  ' + '\n  '.join(errs[:20]))
    print(f'MATCH: {len(tpl_ks.rows)} 条可译行全部符合预期，结构行零改动')


def main():
    table = {'info': cmd_info, 'strings': cmd_strings, 'make_template': cmd_make_template,
             'extract_text': cmd_extract_text, 'rebuild': cmd_rebuild, 'verify': cmd_verify}
    if len(sys.argv) < 2 or sys.argv[1] not in table:
        raise SystemExit(__doc__)
    table[sys.argv[1]](sys.argv[2:])


if __name__ == '__main__':
    main()
