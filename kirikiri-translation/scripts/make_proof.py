# -*- coding: utf-8 -*-
# 合成/刷新校对对照文件（6 列，带说话人列），并默认同时输出两种剧情顺序阅读稿。
# 用法: python make_proof.py <原scn> <work.tsv> <proof.tsv>
# 产物（后两者路径由 proof 文件名自动派生）:
#   proof.tsv         对照稿: index \t 说话人 \t 日文原文 \t 初译 \t 校对 \t 备注
#     校对员只填第 5 列（改稿，留空=采用初译）与第 6 列（批注）；1-4 列由本工具生成。
#   <stem>.preview.txt 纯剧情顺序文本（说话人「对白」/旁白），未译句标【未译】。
#                      面向文气通读：上下文衔接、术语一致性、语气归属。
#   <stem>.read.txt    带索引对照稿，每条两行：
#                      #索引 [说话人|旁白] 日文原文 ＋ ⇒ 生效译文（字面 \n 显示为 ⏎），
#                      场景间有分隔线。面向逐句精读改稿：按 #索引 直接改 proof 第 5 列。
# 两种阅读稿共用同一套"生效译文"（初译 ∪ 既有改稿）；不在字符串表内的行
# （lines 直引对白）分别标【未译】/ ????。
#
# 幂等：重复运行时已有 proof 的改稿/批注列按索引保留，只刷新说话人/日文/初译列与两份阅读稿。
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _trans_common import iter_rows, cn_of, jp_of, speaker_of, NL
from psb_tool import load_scn
from extract_text import find_root, scene_texts

def main():
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    scn_path, work_path, proof_path = sys.argv[1:4]
    stem = os.path.splitext(proof_path)[0]
    preview_path, read_path = stem + '.preview.txt', stem + '.read.txt'

    # 已有校对文件的改稿与批注（6 列新版 / 5 列旧版）
    old_proof = {}
    if os.path.exists(proof_path):
        for ln, idx, cols in iter_rows(proof_path):
            if len(cols) >= 6:
                old_proof[idx] = (cols[4], cols[5])
            elif len(cols) == 5:
                old_proof[idx] = (cols[3], cols[4])

    # 初译（生效译文）与 work 行序（= 模板行序）
    work = {}
    rows = []
    for ln, idx, cols in iter_rows(work_path):
        rows.append((ln, idx, cols))
        cn = cn_of(cols)
        if cn is not None and cn.strip():
            work[idx] = cn.strip()

    os.makedirs(os.path.dirname(proof_path) or '.', exist_ok=True)
    with open(proof_path, 'w', encoding='utf-8', newline='\n') as f:
        f.write('# 校对文件：第5列"校对"填改稿（留空则采用初译），第6列填批注；行宽规则同译文文件\n')
        f.write('#index\t说话人\t日文原文\t初译\t校对\t备注\n')
        for ln, idx, cols in rows:
            fix, note = old_proof.get(idx, ('', ''))
            f.write(f'{idx}\t{speaker_of(cols)}\t{jp_of(cols) or ""}\t'
                    f'{work.get(idx, "")}\t{fix}\t{note}\n')
    n_fix = sum(1 for idx in work if old_proof.get(idx, ('', ''))[0].strip())
    print(f'{os.path.basename(proof_path)}: {len(rows)} 条, 已译 {len(work)}, '
          f'保留既有改稿 {n_fix} 条')

    # ---- 阅读稿共通部分：剧情顺序遍历 + 生效译文（改稿优先） ----
    psb = load_scn(scn_path)
    d = find_root(psb)
    content2idx = {}
    for i, s in enumerate(psb.strings):
        if s and s not in content2idx:
            content2idx[s] = i
    eff = dict(work)
    for idx, (fix, _) in old_proof.items():
        if fix.strip():
            eff[idx] = fix.strip()

    pv_lines, rd_lines, n_miss = [], [], 0
    for sc in d.get('scenes', []):
        for sp, body in scene_texts(sc):
            idx = content2idx.get(body)
            translated = idx is not None and idx in eff
            raw = eff.get(idx, '') if translated else ''
            cn = raw.replace(NL, '\n')          # 预览稿用真实换行
            rd_cn = raw.replace(NL, ' ⏎ ')      # 对照稿保留换行可见
            if sp and not (translated and cn.startswith('「')):
                pv_lines.append(f'{sp}「{cn}」' if translated else f'【未译】{sp}「{body}」')
                rd_head = f'#{idx} [{sp}]' if idx is not None else f'???? [{sp}]'
            elif sp:
                pv_lines.append(f'{sp}{cn}' if translated else f'【未译】{sp}{body}')
                rd_head = f'#{idx} [{sp}]' if idx is not None else f'???? [{sp}]'
            else:
                pv_lines.append(cn if translated else f'【未译】{body}')
                rd_head = f'#{idx} [旁白]' if idx is not None else '???? [旁白]'
            rd_lines.append(rd_head + ' ' + body)
            if translated:
                rd_lines.append(f'      ⇒ {rd_cn}')
            else:
                n_miss += 1
                rd_lines.append('      ⇒ 【未译】')
        pv_lines.append('')
        rd_lines.append('―――――――――― scene break ――――――――――')
    while pv_lines and pv_lines[-1] == '':
        pv_lines.pop()
    open(preview_path, 'w', encoding='utf-8', newline='\n').write('\n'.join(pv_lines) + '\n')
    open(read_path, 'w', encoding='utf-8', newline='\n').write('\n'.join(rd_lines) + '\n')
    tail = f'，未译 {n_miss} 条' if n_miss else ''
    print(f'{os.path.basename(preview_path)}: 剧情顺序预览（{len(pv_lines)} 行，未译标【未译】）')
    print(f'{os.path.basename(read_path)}: 索引对照稿（{len(rd_lines)} 行{tail}）')

if __name__ == '__main__':
    main()
