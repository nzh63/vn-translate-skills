# -*- coding: utf-8 -*-
# 技能全链路自测：不依赖任何游戏文件，用内置迷你 PSB 写入器合成一个最小 .scn 夹具，
# 走完 模板→初译→行宽门禁→校对合成→改稿优先→重建→内容级验证→合并→结构终检→
#       术语检查→可读文本导出→xp3 封包回环 全流程。
# 改动本目录任何脚本后必须重跑本文件，退出码 0 才算通过。
# 用法: python _selftest.py
import sys, os, subprocess, tempfile, shutil, struct, zlib, json

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
NL = '\\n'   # 字面换行记号（反斜杠+n）

# ================= 迷你 PSB 写入器 =================

class MiniPsb:
    """只支持本夹具所需的子集：None/bool/非负 int/str/list/dict。
    名字表按"每名独占深度带"编码（charset 全 0，字符 = 位置低字节），与解析器算法互逆。"""

    def __init__(self):
        self.strings = ['']          # 约定首串为空串（offset[0]=0）
        self.smap = {'': 0}
        self.names_order = []
        self.nmap = {}

    def sid(self, s):
        if s not in self.smap:
            self.smap[s] = len(self.strings)
            self.strings.append(s)
        return self.smap[s]

    def nid(self, k):
        if k not in self.nmap:
            self.nmap[k] = len(self.names_order)
            self.names_order.append(k)
        return self.nmap[k]

    def add_unused(self, strs):
        for s in strs:
            self.sid(s)

    # ---- 值序列化 ----
    def _w_int(self, v):
        assert isinstance(v, int) and not isinstance(v, bool) and 0 <= v < (1 << 64), v
        if v == 0:
            return b'\x04'
        n = 1
        while v >= (1 << (8 * n)):
            n += 1
        return bytes([0x04 + n]) + v.to_bytes(n, 'little')

    def _w_arr(self, ints):
        cnt, mx = len(ints), max(ints, default=0)
        cb = max(1, (cnt.bit_length() + 7) // 8)
        eb = max(1, (mx.bit_length() + 7) // 8)
        out = bytearray([0x0c + cb])
        out += cnt.to_bytes(cb, 'little')
        out.append(0x0c + eb)
        for v in ints:
            out += v.to_bytes(eb, 'little')
        return bytes(out)

    def w_val(self, v):
        if v is None:
            return b'\x00'
        if v is False:
            return b'\x02'
        if v is True:
            return b'\x03'
        if isinstance(v, int):
            return self._w_int(v)
        if isinstance(v, float):
            return b'\x1e' + struct.pack('<f', v)
        if isinstance(v, str):
            i = self.sid(v)
            n = max(1, (i.bit_length() + 7) // 8)
            return bytes([0x14 + n]) + i.to_bytes(n, 'little')
        if isinstance(v, list):
            body, offs = bytearray(), []
            for it in v:
                offs.append(len(body))
                body += self.w_val(it)
            return b'\x20' + self._w_arr(offs) + bytes(body)
        if isinstance(v, dict):
            names = [self.nid(k) for k in v.keys()]
            body, offs = bytearray(), []
            for k, val in v.items():
                offs.append(len(body))
                body += self.w_val(val)
            return b'\x21' + self._w_arr(names) + self._w_arr(offs) + bytes(body)
        raise TypeError(type(v))

    # ---- 名字表（与解析器互逆的平凡编码）----
    # 解码方向：从名字【末字符】节点沿父指针走向首字符，边走边收集、最终反转。
    # 因此节点链条是 p_j -> p_{j-1}（j 为字符下标），首字符节点值为 0；
    # charset 全 0 时字符 = 节点位置低字节；每个名字独占深度带防跨名碰撞。
    def _names_arrays(self):
        BAND = 8
        ndata = {}
        maxp = 0
        lasts = []
        for ni, name in enumerate(self.names_order):
            cs = name.encode('utf-8')
            ps = [c + 256 * (ni * BAND + j) for j, c in enumerate(cs)]
            for j, p in enumerate(ps):
                parent = ps[j - 1] if j > 0 else 0
                assert p not in ndata or ndata[p] == parent
                ndata[p] = parent
                maxp = max(maxp, p)
            lasts.append(ps[-1])
        size = maxp + 2 + len(lasts)
        arr = [0] * size
        for p, nxt in ndata.items():
            arr[p] = nxt
        nameidx = []
        for i, pl in enumerate(lasts):
            slot = maxp + 1 + i
            arr[slot] = pl
            nameidx.append(slot)
        return [0] * size, arr, nameidx   # charset 全 0

    def dump(self, roots):
        for r in roots:
            self.w_val(r)                       # pass1: 注册字符串与键名
        charset, ndata, nameidx = self._names_arrays()
        front = self._w_arr(charset) + self._w_arr(ndata) + self._w_arr(nameidx)
        tree = b''.join(self.w_val(r) for r in roots)   # pass2（幂等）
        strdata = bytearray(b'\x00')
        offs = [0]
        for s in self.strings[1:]:
            offs.append(len(strdata))
            strdata += s.encode('utf-8') + b'\x00'
        stroff = self._w_arr(offs)
        off_strings = 0x2C + len(front) + len(tree)
        off_strdata = off_strings + len(stroff)
        end = off_strdata + len(strdata)
        hdr = struct.pack('<10I', 3, 0, 0, off_strings, off_strdata,
                          end, end + 2, end + 4, 0, len(front) + len(tree))
        return b'PSB\x00' + hdr + front + tree + stroff + bytes(strdata) + b'\x0d\x00\x0d\x00'

def wrap_mdf(raw):
    return b'mdf\x00' + struct.pack('<I', len(raw)) + zlib.compress(raw, 9)

# ================= 夹具 =================

def build_fixture(scn_path):
    mp = MiniPsb()
    disp_a = '「てすとのセリフです。' + NL + 'にぎょうめです。」'
    plain_a = '「てすとのセリフです。にぎょうめです。」'
    disp_b = '「ふふっ」。'
    shared = '「はい」'
    disp_d = '「うん」'
    texts = [
        ['Alice', [[None, disp_a, len(plain_a), plain_a]], None, len(plain_a)],
        ['',      [[None, disp_b, len(disp_b), disp_b]], None, len(disp_b)],
        ['Bob',   [[None, shared, len(shared), shared]], None, len(shared)],
        ['Alice', [[None, shared, len(shared), shared]], None, len(shared)],
        ['Bob',   [[None, disp_d, len(disp_d), disp_d]], None, len(disp_d)],
    ]
    scene = {'label': 'sc1', 'title': None, 'texts': texts,
             'lines': [['startline', 0, 0, 'name', 'Bob', '', 0], 1, 2, 3, 4, 5]}
    root = {'name': 'fix.txt', 'meta': {'build': 'Private.vfs'}, 'scenes': [scene]}
    mp.add_unused(['sys_ascii_decoy_01'])
    open(scn_path, 'wb').write(wrap_mdf(mp.dump([root])))
    return dict(disp_a=disp_a, plain_a=plain_a, disp_b=disp_b,
                shared=shared, disp_d=disp_d)

def run(*args, expect=None):
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    r = subprocess.run([sys.executable, *args], capture_output=True,
                       text=True, encoding='utf-8', errors='replace', env=env, cwd=HERE)
    out = (r.stdout or '') + (r.stderr or '')
    if expect is not None and r.returncode != expect:
        raise AssertionError(f'exit={r.returncode} (expect {expect})\n命令: {args}\n输出:\n{out}')
    return r.returncode, out

# ================= 用例 =================

def main():
    tmp = tempfile.mkdtemp(prefix='krkr_skill_test_')
    try:
        scn = os.path.join(tmp, 'src', 'fix.txt.scn')
        os.makedirs(os.path.dirname(scn), exist_ok=True)
        fx = build_fixture(scn)

        # [1] 模板生成 + 防覆盖
        wsdir = os.path.join(tmp, 'ws')
        _, out = run(os.path.join(HERE, 'make_template.py'), os.path.dirname(scn), wsdir, expect=0)
        assert os.path.exists(f'{wsdir}/tpl/fix.tsv') and os.path.exists(f'{wsdir}/work/fix.tsv'), out
        assert 'Alice/Bob' in open(f'{wsdir}/tpl/fix.tsv', encoding='utf-8').read(), '说话人未合并'
        with open(f'{wsdir}/work/fix.tsv', 'a', encoding='utf-8') as f:
            f.write('# 进度标记\n')
        _, out = run(os.path.join(HERE, 'make_template.py'), os.path.dirname(scn), wsdir, expect=0)
        assert '跳过' in out and '# 进度标记' in open(f'{wsdir}/work/fix.tsv', encoding='utf-8').read()
        print('[1] 模板生成、说话人列、防覆盖 OK')

        # [2] 填初译（含一条字面 \n 多行译文；留一条未译）
        tpl_rows = [l.rstrip('\n').split('\t') for l in open(f'{wsdir}/tpl/fix.tsv', encoding='utf-8')
                    if l.strip() and not l.startswith('#')]
        cn_map = {fx['disp_a']: '「测试台词第一行。' + NL + '第二行内容。」',
                  fx['disp_b']: '「嘿嘿」。'}
        filled = 0
        with open(f'{wsdir}/work/fix.tsv', 'w', encoding='utf-8', newline='\n') as f:
            for cols in tpl_rows:
                jp = cols[2]
                if jp in cn_map:
                    f.write('\t'.join(cols[:3]) + '\t' + cn_map[jp] + '\n')
                    filled += 1
                elif jp == fx['shared']:
                    f.write('\t'.join(cols[:3]) + '\t' + '「好的」。' + '\n')
                    filled += 1
                else:
                    f.write('\t'.join(cols[:3]) + '\t\n')   # 未译也保持 4 列（空译文列）
        assert filled == 3, filled   # 4 个索引、3 条独立字符串；故意留 1 条未译
        rc, out = run(os.path.join(HERE, 'validate_text.py'), f'{wsdir}/work/fix.tsv', expect=0)
        print('[2] 初译填写与行宽门禁 OK:', out.strip().splitlines()[-1])

        # [3] 行宽拦截
        bad = os.path.join(tmp, 'bad.tsv')
        with open(bad, 'w', encoding='utf-8', newline='\n') as f:
            f.write('5\tAlice\t「あ」\t「' + '超' * 30 + '」\n')
            f.write('6\tBob\t「い」\t「一' + NL + '二' + NL + '三' + NL + '四」\n')
        rc, out = run(os.path.join(HERE, 'validate_text.py'), bad)
        assert rc == 1 and '超宽' in out and '行数超限' in out, out
        print('[3] 超宽/超行拦截 OK')

        # [4] 校对合成（默认同时产出 preview + read）+ 改稿优先
        proof = f'{wsdir}/proof/fix.tsv'
        prev = f'{wsdir}/proof/fix.preview.txt'
        read = f'{wsdir}/proof/fix.read.txt'
        _, out = run(os.path.join(HERE, 'make_proof.py'), scn, f'{wsdir}/work/fix.tsv',
                     proof, expect=0)
        assert os.path.exists(prev) and os.path.exists(read), '阅读稿未产出'
        ptext = open(proof, encoding='utf-8').read()
        assert '初译\t校对\t备注' in ptext and '【未译】' not in ptext
        pvtext = open(prev, encoding='utf-8').read()
        assert '【未译】' in pvtext and '测试台词第一行' in pvtext.replace(NL, ''), '预览缺译文/未译标记'
        rdtext = open(read, encoding='utf-8').read()
        shared_idx = next(c[0] for c in tpl_rows if c[2] == fx['shared'])
        assert f'#{shared_idx} [Bob]' in rdtext and '⇒' in rdtext and 'scene break' in rdtext, \
            f'read 稿缺索引/译文/分景:\n{rdtext[:300]}'
        assert ('⇒ 「测试台词第一行。 ⏎ 第二行内容。」') in rdtext, 'read 稿换行应显示为 ⏎'
        plines = ptext.splitlines()
        with open(proof, 'w', encoding='utf-8', newline='\n') as f:
            for line in plines:
                if line.startswith('#'):
                    f.write(line + '\n')
                    continue
                p = line.split('\t')
                if len(p) >= 6 and p[2] == fx['shared']:
                    p[4], p[5] = '「了解了」。', '更贴角色口吻'
                    f.write('\t'.join(p) + '\n')
                else:
                    f.write(line + '\n')
        rc, out = run(os.path.join(HERE, 'validate_text.py'),
                      f'{wsdir}/work/fix.tsv', proof, expect=0)
        print('[4] 校对合成/预览/改稿列 OK')

        # [5] 重建 + 内容级验证 + 校对稿优先生效 + 履历派生
        outf = os.path.join(tmp, 'out', 'fix.txt.scn')
        os.makedirs(os.path.dirname(outf), exist_ok=True)
        _, out = run(os.path.join(HERE, 'psb_tool.py'), 'rebuild', scn, outf,
                     f'{wsdir}/work/fix.tsv', proof, expect=0)
        rc, out = run(os.path.join(HERE, 'psb_tool.py'), 'verify', scn, outf,
                      f'{wsdir}/work/fix.tsv', proof, expect=0)
        assert 'MATCH' in out, out
        from psb_tool import load_scn
        got = dict(enumerate(load_scn(outf).strings))
        vals = list(got.values())
        assert '「了解了」。' in vals, f'校对稿未优先生效: {vals!r}'
        cn_plain = '「测试台词第一行。第二行内容。」'
        assert cn_plain in vals, f'履历版未派生: {vals!r}'
        assert fx['plain_a'] not in vals   # 日文履历版已被派生译文覆盖
        assert '「测试台词第一行。' + NL + '第二行内容。」' in vals
        print('[5] 重建/内容级 MATCH/改稿优先/履历派生 OK')

        # [6] 重跑 make_proof 幂等（保留批注），merge 回 work 后仅用 work 重建结果一致
        _, out = run(os.path.join(HERE, 'make_proof.py'), scn, f'{wsdir}/work/fix.tsv', proof, expect=0)
        assert '更贴角色口吻' in open(proof, encoding='utf-8').read()
        _, out = run(os.path.join(HERE, 'merge_proof.py'), f'{wsdir}/work/fix.tsv', proof, expect=0)
        assert '「了解了」。' in open(f'{wsdir}/work/fix.tsv', encoding='utf-8').read()
        out2 = os.path.join(tmp, 'out', 'fix_workonly.txt.scn')
        run(os.path.join(HERE, 'psb_tool.py'), 'rebuild', scn, out2, f'{wsdir}/work/fix.tsv', expect=0)
        assert open(outf, 'rb').read() == open(out2, 'rb').read(), '合并前后重建不一致'
        print('[6] make_proof 幂等 / merge_proof 合并一致 OK')

        # [6b] edit_proof：单条写入 / 真实换行拒绝 / JSON 批量与清列 / 未命中退出码
        _, out = run(os.path.join(HERE, 'edit_proof.py'), proof, shared_idx,
                     '「改稿乙」。', '批注乙', expect=0)
        ptext = open(proof, encoding='utf-8').read()
        assert '「改稿乙」。' in ptext and '批注乙' in ptext, out
        rc, out = run(os.path.join(HERE, 'edit_proof.py'), proof, shared_idx, '坏\n稿')
        assert rc != 0 and '真实换行' in out, f'真实换行未被拒绝:\n{out}'
        jf = os.path.join(tmp, 'edits.json')
        with open(jf, 'w', encoding='utf-8') as f:
            json.dump([{'i': shared_idx,
                        'fix': '「恢复' + NL + '记号」。', 'note': ''}],
                      f, ensure_ascii=False)
        _, out = run(os.path.join(HERE, 'edit_proof.py'), proof, '--file', jf, expect=0)
        ptext = open(proof, encoding='utf-8').read()
        assert ('「恢复' + NL + '记号」。') in ptext and '批注乙' not in ptext, \
            f'批量写入/清除批注失败:\n{ptext[:400]}'
        rc, out = run(os.path.join(HERE, 'edit_proof.py'), proof, '99999', 'x')
        assert rc == 1 and '未命中' in out, out
        print('[6b] edit_proof（单条/换行拒绝/JSON 批量/清列/未命中）OK')

        # [6c] check_punct：干净文件全绿；坏样本各命中对应检查项
        rc, out = run(os.path.join(HERE, 'check_punct.py'),
                      f'{wsdir}/work/fix.tsv', proof, expect=0)
        badp = os.path.join(tmp, 'badpunct.tsv')
        with open(badp, 'w', encoding='utf-8', newline='\n') as f:
            f.write('21\t\tふふっ。\t「旁白不该加引号」\n')       # 旁白凭空加引号（原文无「）
            f.write('22\tBob\t「あ」\t“弯引号”\n')               # 弯引号 + 凭空丢引号（原文有「）
            f.write('22\tBob\t「あ」\t「保留正确」\n')           # 对照行：原文有「译文有「 = 合规
            f.write('23\tBob\t「い」\t「配平不良\n')             # 「」不配平（单物理行）
            f.write('24\tBob\t「う」\t「好吃,真好」\n')          # 半角逗号紧跟 CJK
            f.write('25\tBob\t「え」\t「ＯＫ,好的」\n')          # 全角字母后半角点 → 也该报
        rc, out = run(os.path.join(HERE, 'check_punct.py'), badp)
        assert rc == 1, out
        for kw in ('凭空加引号', '弯引号', '凭空丢引号', '「」不配平', '半角标点'):
            assert kw in out, (kw, out)
        print('[6c] check_punct（干净全绿/五类命中）OK')

        # [7] 结构终检：好文件过，坏文件各命中对应检查项
        rc, out = run(os.path.join(HERE, 'check_structure.py'),
                      f'{wsdir}/work/fix.tsv', f'--tpl={wsdir}/tpl', expect=0)
        bom = os.path.join(tmp, 'bom.tsv')
        with open(bom, 'wb') as f:
            f.write(b'\xef\xbb\xbf' + open(f'{wsdir}/work/fix.tsv', 'rb').read())
        rc, out = run(os.path.join(HERE, 'check_structure.py'), bom, f'--tpl={wsdir}/tpl')
        assert rc == 1 and 'BOM' in out, out
        nl_bad = os.path.join(tmp, 'realnl.tsv')
        src_line = [l for l in open(f'{wsdir}/work/fix.tsv', encoding='utf-8')
                    if '测试台词第一行' in l][0].rstrip('\n')
        broken = src_line.replace(NL, '\n')           # 字面记号 → 真实换行（高频事故）
        with open(nl_bad, 'w', encoding='utf-8', newline='\n') as f:
            f.write(broken.split('\n')[0] + '\n' + ''.join(broken.split('\n')[1:]) + '\n')
        rc, out = run(os.path.join(HERE, 'check_structure.py'), nl_bad, f'--tpl={wsdir}/tpl')
        assert rc == 1 and ('列数' in out or '模板' in out or '真实换行' in out), out
        jp_dir = os.path.join(tmp, 'jpbad')
        os.makedirs(jp_dir, exist_ok=True)
        jp_bad = os.path.join(jp_dir, 'fix.tsv')   # 与模板同名，模板比对才会生效
        with open(jp_bad, 'w', encoding='utf-8', newline='\n') as f:
            for l in open(f'{wsdir}/work/fix.tsv', encoding='utf-8'):
                if l.startswith('#'):
                    continue
                p = l.rstrip('\n').split('\t')
                if len(p) >= 4 and p[2] == fx['disp_b']:
                    p[2] = '「改过的原文」'
                f.write('\t'.join(p) + '\n')
        rc, out = run(os.path.join(HERE, 'check_structure.py'), jp_bad, f'--tpl={wsdir}/tpl')
        assert rc == 1 and '模板不一致' in out, out
        # 被真实换行拆坏的 TSV 不允许进入重建（load_map 必须致命报错而非静默跳过）
        rc, out = run(os.path.join(HERE, 'psb_tool.py'), 'rebuild', scn,
                      os.path.join(tmp, 'out', 'fatal.scn'), nl_bad)
        assert rc != 0 and '首列不是整数' in out, out
        print('[7] 结构终检（BOM/真实换行/原文篡改/重建拒绝坏表）OK')

        # [8] 术语冻结检查（专用样本）
        terms = os.path.join(tmp, 'terms.tsv')
        sample = os.path.join(tmp, 'terms_sample.tsv')
        with open(terms, 'w', encoding='utf-8', newline='\n') as f:
            f.write('# 术语表示例\nサクラ\t樱\t櫻\nラーメン\t拉面\n')
        with open(sample, 'w', encoding='utf-8', newline='\n') as f:
            f.write('11\tサクラ\t「サクラだよ」\t「櫻哦」\n')          # 禁用表记 + 缺规范译法
            f.write('12\t\tラーメンを食べた。\t吃了拉面。\n')           # 合规
        rc, out = run(os.path.join(HERE, 'check_terms.py'), terms, sample)
        assert rc == 1 and '禁用表记' in out and '缺规范译法' in out, out
        terms2 = os.path.join(tmp, 'terms2.tsv')
        sample2 = os.path.join(tmp, 'terms_sample2.tsv')
        with open(terms2, 'w', encoding='utf-8', newline='\n') as f:
            f.write('てすと\t测试乙\n')
        with open(sample2, 'w', encoding='utf-8', newline='\n') as f:
            f.write('13\t\tてすとです。\t这是一段例句。\n')           # 仅触发警告级"漏译"
        rc, out = run(os.path.join(HERE, 'check_terms.py'), terms2, sample2)
        assert rc == 0 and '缺规范译法' in out, out                   # 默认警告不阻断
        rc, out = run(os.path.join(HERE, 'check_terms.py'), terms2, sample2, '--strict')
        assert rc == 1, out                                           # --strict 下警告也算失败
        print('[8] 术语检查（禁用表记/漏译警告/--strict）OK')

        # [9] 可读文本导出
        txt_dir = os.path.join(tmp, 'text')
        _, out = run(os.path.join(HERE, 'extract_text.py'), outf, txt_dir, expect=0)
        t = open(os.path.join(txt_dir, 'fix.txt'), encoding='utf-8').read()
        assert '◆---- label: sc1' in t and 'Bob「' in t and '【' not in t, t
        print('[9] 剧情顺序文本导出 OK')

        # [10] xp3 封包回环（patch 模式 + 整包模式）
        pdir = os.path.join(tmp, 'patch_files')
        os.makedirs(pdir)
        shutil.copy(outf, os.path.join(pdir, 'fix.txt.scn'))
        pxp3 = os.path.join(tmp, 'patch.xp3')
        _, out = run(os.path.join(HERE, 'xp3_tool.py'), 'pack', pdir, pxp3, '--patch', expect=0)
        vdir = os.path.join(tmp, 'vfy')
        _, out = run(os.path.join(HERE, 'xp3_tool.py'), 'unpack', pxp3, vdir, expect=0)
        assert open(os.path.join(vdir, 'fix.txt.scn'), 'rb').read() == open(outf, 'rb').read()
        dxp3 = os.path.join(tmp, 'data.xp3')
        _, out = run(os.path.join(HERE, 'xp3_tool.py'), 'pack', pdir, dxp3,
                     '--template', pxp3, expect=0)
        vdir2 = os.path.join(tmp, 'vfy2')
        run(os.path.join(HERE, 'xp3_tool.py'), 'unpack', dxp3, vdir2, expect=0)
        assert open(os.path.join(vdir2, 'fix.txt.scn'), 'rb').read() == open(outf, 'rb').read()
        _, out = run(os.path.join(HERE, 'xp3_tool.py'), 'probe', dxp3, expect=0)
        assert '文件数' in out and 'info 布局' in out, out
        print('[10] xp3 打包/探测/解包回环 OK')

        print('=== kirikiri-translation 技能全链路自测通过 ===')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

if __name__ == '__main__':
    main()
