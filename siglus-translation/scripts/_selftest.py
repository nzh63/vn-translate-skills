# -*- coding: utf-8 -*-
# 技能全链路自测：不依赖任何游戏文件，用内置写入器合成最小 .ss 与 Scene.pck 夹具，
# 走完 解包→模板→初译→行宽门禁→校对合成→改稿优先→重建→内容级验证→回封→回读 全流程。
# 改动本目录任何脚本后必须重跑本文件，退出码 0 才算通过。
# 用法: python _selftest.py
import sys, os, struct, subprocess, tempfile, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from crypt import TBL_SCENE, xor_table, xor_key16, lzss_store, xor_u16_words

KEY = bytes(range(0x10))          # 夹具用任意密钥（ExtraKeyUse=1）

# ================= 迷你 .ss 写入器 =================

def build_ss(strings, tail=b'SCENEBIN\x00' * 8, layout='strings_first'):
    """合成 .ss 夹具。layout:
    'strings_first' —— [头132][表][字符串区][尾块]（銀色、遥か 实测布局）
    'strings_last'  —— [头132][表][字节码][字符串区]（test-other 样例实测布局）
    两者偶数位头字段都指向尾块/字节码内部，用于检验平移逻辑。"""
    blobs = []
    for i, s in enumerate(strings):
        w = xor_u16_words([ord(c) for c in s], i)
        blobs.append(struct.pack(f'<{len(w)}H', *w))
    tbl_size = len(strings) * 8
    hdr = [0] * 32
    hdr[2] = 132                            # 表偏移
    hdr[3] = len(strings)                   # 条目数
    hdr[5] = len(strings)

    if layout == 'strings_first':
        str_off = 132 + tbl_size
        tail_off = str_off + sum(len(b) for b in blobs)
        body = b''.join(blobs)
        hdr[0], hdr[1], hdr[4] = tail_off, len(tail), str_off
        hdr[6], hdr[7] = tail_off + 16, 21
        hdr[8], hdr[9] = tail_off + 48, 1000
        out = bytearray(struct.pack('<I', 132)) + b'\x00' * 128
        cur = 0
        for b in blobs:
            out += struct.pack('<II', cur // 2, len(b) // 2)   # 词单位
            cur += len(b)
        out += body + tail
    elif layout == 'strings_last':
        code = b'BYTECODE\x00' * 24
        code_off = 132 + tbl_size
        str_off = code_off + len(code)
        body = code + b''.join(blobs)
        hdr[0], hdr[1] = code_off, len(code)
        hdr[4] = str_off
        hdr[6], hdr[7] = code_off + 16, 21
        hdr[8], hdr[9] = code_off + 48, 1000
        out = bytearray(struct.pack('<I', 132)) + b'\x00' * 128
        cur = 0
        for b in blobs:
            out += struct.pack('<II', cur // 2, len(b) // 2)
            cur += len(b)
        out += code + b''.join(blobs)
    else:
        raise ValueError(layout)
    struct.pack_into('<32I', out, 4, *hdr)
    return bytes(out)

# ================= 迷你 Scene.pck 写入器 =================

def build_pck(names, ss_list):
    blobs = []
    for ss in ss_list:
        comp = lzss_store(ss)
        if KEY:
            comp = xor_key16(comp, KEY)
        blobs.append(xor_table(comp, TBL_SCENE))
    body = bytearray()
    offs = {}
    cur = 92
    for k in ['varInfo', 'varNameIdx', 'varNames', 'cmdInfo', 'cmdNameIdx',
              'cmdNames']:
        offs[k] = (cur, 0)                      # 空区段
    name_idx = b''.join(struct.pack('<II', sum(len(n) for n in names[:i]), len(n))
                        for i, n in enumerate(names))
    names_blob = b''.join(n.encode('utf-16-le') for n in names)
    info_blob = b''.join(struct.pack('<II',
                          sum(len(b) for b in blobs[:i]), len(blobs[i]))
                         for i in range(len(blobs)))
    data_blob = b''.join(blobs)
    counts = dict(scnNameIdx=len(names), scnNames=len(names),
                  scnInfo=len(names), scnData=len(names))
    for payload_name, payload in [('scnNameIdx', name_idx), ('scnNames', names_blob),
                                  ('scnInfo', info_blob), ('scnData', data_blob)]:
        offs[payload_name] = (cur, counts[payload_name])
        body += payload
        cur = 92 + len(body)
    vals = []
    for k in ['varInfo', 'varNameIdx', 'varNames', 'cmdInfo', 'cmdNameIdx',
              'cmdNames', 'scnNameIdx', 'scnNames', 'scnInfo', 'scnData']:
        o, c = offs[k]
        vals += [o, c]
    hdr = struct.pack('<I', 92) + struct.pack('<22I', *vals, 1, 0x566)
    assert len(hdr) == 92
    return hdr + bytes(body)

# ================= 用例 =================

def run(*args, expect=None):
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    r = subprocess.run([sys.executable, *args], capture_output=True,
                       text=True, encoding='utf-8', errors='replace',
                       env=env, cwd=HERE)
    out = (r.stdout or '') + (r.stderr or '')
    if expect is not None and r.returncode != expect:
        raise AssertionError(f'exit={r.returncode} (expect {expect})\n命令: {args}\n输出:\n{out}')
    return r.returncode, out


def main():
    tmp = tempfile.mkdtemp(prefix='siglus_skill_test_')
    try:
        key_hex = KEY.hex().upper()
        kf = os.path.join(tmp, 'key.txt')
        open(kf, 'w').write(key_hex + '\n')

        strings_a = [
            '',                                  # 空槽
            'CMD_INIT',                          # ASCII 结构串
            'それは──良く晴れた朝のことだった。',      # 旁白
            '雛多',                              # 说话人候选
            '「てすとのセリフです。」',                 # 对白
            '％Ｂ',                              # 控制标记（不得进模板）
            '二行目のながーいせりふ、じどうかいぎょうされるのでそのまま。',  # 长行
        ]
        strings_b = ['', 'CMD_LOAD', 'ふたつめのシーン。', '真冬', '「わんこ！」']
        ss_a = build_ss(strings_a)
        ss_b = build_ss(strings_b)
        srcdir = os.path.join(tmp, 'src')
        os.makedirs(srcdir)
        open(os.path.join(srcdir, 'a.ss'), 'wb').write(ss_a)
        open(os.path.join(srcdir, 'b.ss'), 'wb').write(ss_b)
        pck = os.path.join(tmp, 'Scene.pck')
        open(pck, 'wb').write(build_pck(['a', 'b'], [ss_a, ss_b]))

        # [1] probe / unpack 回读一致
        _, out = run(f'{HERE}/pck_tool.py', 'probe', pck, '--key-file', kf, expect=0)
        assert '密钥验证 OK' in out, out
        udir = os.path.join(tmp, 'unpacked')
        run(f'{HERE}/pck_tool.py', 'unpack', pck, udir, '--key-file', kf, expect=0)
        assert open(f'{udir}/a.ss','rb').read() == ss_a
        assert open(f'{udir}/manifest.tsv').read().count('\n') == 3  # 注释+2 行
        print('[1] probe/解包回读一致 OK')

        # [2] 模板：说话人推断、控制标记排除、防覆盖
        ws = os.path.join(tmp, 'ws')
        _, out = run(f'{HERE}/make_template.py', udir, ws, expect=0)
        tpl = open(f'{ws}/tpl/a.tsv', encoding='utf-8').read()
        assert '2\t\tそれは' in tpl, tpl
        assert '3\t\t雛多' in tpl and '4\t雛多\t「てすと' in tpl, '说话人未挂到对白'
        assert '％Ｂ' not in tpl and 'CMD_INIT' not in tpl, '控制标记/ASCII 混入模板'
        open(f'{ws}/work/a.tsv', 'a', encoding='utf-8').write('# 进度标记\n')
        _, out = run(f'{HERE}/make_template.py', udir, ws, expect=0)
        assert '跳过(已存在)' in out and '# 进度标记' in open(f'{ws}/work/a.tsv', encoding='utf-8').read()
        print('[2] 模板生成/说话人推断/防覆盖 OK')

        # [3] 初译 + 行宽门禁
        rows = [l.split('\t') for l in open(f'{ws}/work/a.tsv', encoding='utf-8').read().splitlines() if l and not l.startswith('#')]
        cn = {'それは──良く晴れた朝のことだった。': '那是一个天气晴朗的早晨。',
              '雛多': '雏多',
              '「てすとのセリフです。」': '「这是测试台词。」'}
        with open(f'{ws}/work/a.tsv', 'w', encoding='utf-8', newline='\n') as f:
            for p in rows:
                f.write('\t'.join(p[:3]) + '\t' + cn.get(p[2], '') + '\n')
        run(f'{HERE}/validate_text.py', f'{ws}/work/a.tsv', '--width=30', '--lines=1', expect=0)
        bad = os.path.join(tmp, 'bad.tsv')
        open(bad, 'w', encoding='utf-8', newline='\n').write(
            '9\t\tあ\t「' + '超' * 35 + '」\n')
        rc, out = run(f'{HERE}/validate_text.py', bad, '--width=30', '--lines=1')
        assert rc == 1 and '超宽' in out, out
        print('[3] 初译填写与行宽门禁 OK')

        # [4] 校对合成 + 改稿优先 + edit_proof + check_punct
        proof = f'{ws}/proof/a.tsv'
        run(f'{HERE}/make_proof.py', f'{udir}/a.ss', f'{ws}/work/a.tsv', proof, expect=0)
        prev = f'{ws}/proof/a.preview.txt'
        read = f'{ws}/proof/a.read.txt'
        assert os.path.exists(prev) and os.path.exists(read)
        plines = open(proof, encoding='utf-8').read().splitlines()
        with open(proof, 'w', encoding='utf-8', newline='\n') as f:
            for l in plines:
                if l.startswith('#'):
                    f.write(l + '\n'); continue
                p = l.split('\t')
                if len(p) >= 6 and p[2] == '雛多':
                    p[4], p[5] = '小雏', '更贴称呼'
                f.write('\t'.join(p) + '\n')
        run(f'{HERE}/edit_proof.py', proof, '4', '「这是校对后的台词。」', '润色', expect=0)
        run(f'{HERE}/check_punct.py', proof, expect=0)
        outf = os.path.join(tmp, 'out', 'a.ss')
        os.makedirs(os.path.dirname(outf))
        run(f'{HERE}/ss_tool.py', 'rebuild', f'{udir}/a.ss', outf,
            f'{ws}/work/a.tsv', proof, expect=0)
        rc, out = run(f'{HERE}/ss_tool.py', 'verify', f'{udir}/a.ss', outf,
                      f'{ws}/work/a.tsv', proof, expect=0)
        assert 'MATCH' in out, out
        got = {l.split('\t',1)[0]: l.split('\t',1)[1]
               for l in run(f'{HERE}/ss_tool.py', 'strings', outf)[1].splitlines()}
        assert got['3'] == '小雏', f'改稿未优先生效: {got["3"]!r}'
        assert got['4'] == '「这是校对后的台词。」'
        assert got['2'] == '那是一个天气晴朗的早晨。'
        assert got['1'] == 'CMD_INIT' and got['5'] == '％Ｂ', '结构串/控制串被改动'
        print('[4] 校对合成/改稿优先/edit_proof/check_punct/重建验证 OK')

        # [5] 负例：真实换行致命、原文不符拒绝、字面 \n 拒绝
        broken = os.path.join(tmp, 'broken.tsv')
        open(broken, 'w', encoding='utf-8', newline='\n').write('坏行不含整数索引\n')
        rc, out = run(f'{HERE}/ss_tool.py', 'rebuild', f'{udir}/a.ss',
                      os.path.join(tmp, 'x.ss'), broken)
        assert rc != 0 and '首列不是整数' in out, out
        mismatch = os.path.join(tmp, 'mm.tsv')
        open(mismatch, 'w', encoding='utf-8', newline='\n').write('2\t\t别的原文\t译文\n')
        rc, out = run(f'{HERE}/ss_tool.py', 'rebuild', f'{udir}/a.ss',
                      os.path.join(tmp, 'y.ss'), mismatch)
        assert rc != 0 and '原文不符' in out, out
        litnl = os.path.join(tmp, 'nl.tsv')
        open(litnl, 'w', encoding='utf-8', newline='\n').write(
            '2\t\tそれは──良く晴れた朝のことだった。\t第一行\\n第二行\n')
        rc, out = run(f'{HERE}/ss_tool.py', 'rebuild', f'{udir}/a.ss',
                      os.path.join(tmp, 'z.ss'), litnl)
        assert rc != 0 and '\\n 记号' in out, f'字面 \\n 未被拒绝:\n{out}'
        print('[5] 坏 TSV 致命拦截（换行拆行/原文不符/字面\\n）OK')

        # [5b] 变体布局：字符串区在文件末尾（字节码在表后）——外科替换须两种布局通吃
        ss_vb = build_ss(strings_a, layout='strings_last')
        vb = os.path.join(tmp, 'vb.ss')
        open(vb, 'wb').write(ss_vb)
        vbe = os.path.join(tmp, 'vb_empty.tsv')
        open(vbe, 'w', encoding='utf-8').write('# empty\n')
        run(f'{HERE}/ss_tool.py', 'rebuild', vb, os.path.join(tmp, 'vb_id.ss'),
            vbe, expect=0)
        assert open(os.path.join(tmp, 'vb_id.ss'), 'rb').read() == ss_vb, \
            '变体B 恒等重建应逐字节一致'
        with open(os.path.join(tmp, 'vb.tsv'), 'w', encoding='utf-8', newline='\n') as f:
            f.write(f'2\t\t{strings_a[2]}\t{"Ｅ" * 8}\n')
            f.write(f'6\t\t{strings_a[6]}\t{"长一点的译文，和原文长度不同。"}\n')
        run(f'{HERE}/ss_tool.py', 'rebuild', vb, os.path.join(tmp, 'vb_new.ss'),
            os.path.join(tmp, 'vb.tsv'), expect=0)
        rc, out = run(f'{HERE}/ss_tool.py', 'verify', vb,
                      os.path.join(tmp, 'vb_new.ss'), os.path.join(tmp, 'vb.tsv'),
                      expect=0)
        assert 'MATCH' in out, out
        from ss_tool import Ss as _Ss
        gvb = _Ss(open(os.path.join(tmp, 'vb_new.ss'), 'rb').read()).strings
        assert gvb[2] == 'Ｅ' * 8 and gvb[1] == 'CMD_INIT'
        # 字节码必须原位未动（变体B: 字节码紧跟表后，位置与大小不变）
        from crypt import TBL_SCENE as _T  # noqa 占位防误删
        nb_raw = open(os.path.join(tmp, 'vb_new.ss'), 'rb').read()
        code_off = struct.unpack_from('<32I', nb_raw, 4)[0]
        assert nb_raw[code_off:code_off + len(b'BYTECODE\x00' * 24)] == b'BYTECODE\x00' * 24
        print('[5b] 变体布局（字符串在文件尾）恒等/改稿重建 OK')

        # [6] 结构终检 / 术语检查 / 可读文本导出
        run(f'{HERE}/check_structure.py', f'{ws}/work/a.tsv', f'--tpl={ws}/tpl', expect=0)
        terms = os.path.join(tmp, 'terms.tsv')
        open(terms, 'w', encoding='utf-8', newline='\n').write('雛多\t雏多\t絊多\n')
        rc, out = run(f'{HERE}/check_terms.py', terms, f'{ws}/proof/a.tsv')
        assert rc == 0, out
        txt = os.path.join(tmp, 'text')
        run(f'{HERE}/extract_text.py', outf, txt, expect=0)
        t = open(os.path.join(txt, 'a.txt'), encoding='utf-8').read()
        assert '◆---- name: a' in t, t
        flat = t.replace('\n', '')
        assert '那是一个天气晴朗的早晨。' in flat, t
        assert '小雏「这是校对后的台词。」' in flat, t
        assert '％Ｂ' not in flat, '控制标记不应出现在阅读稿'
        print('[6] 结构终检/术语检查/可读文本导出 OK')

        # [7] 回封 + 回读内容级比对（b 场景未动 → 数据块逐字节复用）
        outdir = os.path.join(tmp, 'out')
        shutil.copy(outf, f'{udir}/a.ss')
        packed = os.path.join(tmp, 'Scene_new.pck')
        _, out = run(f'{HERE}/pck_tool.py', 'pack', pck, packed, udir,
                     '--key-file', kf, expect=0)
        assert '复用 1，重压缩 1' in out, out
        vdir = os.path.join(tmp, 'vfy')
        run(f'{HERE}/pck_tool.py', 'unpack', packed, vdir, '--key-file', kf, expect=0)
        va = open(f'{vdir}/a.ss', 'rb').read()
        from ss_tool import Ss
        g = Ss(va).strings
        assert g[3] == '小雏' and g[4] == '「这是校对后的台词。」', g[:7]
        assert open(f'{vdir}/b.ss','rb').read() == ss_b, '未改动场景应逐字节一致'
        # 缺槽位拒绝封包
        badman = os.path.join(tmp, 'badman')
        os.makedirs(badman)
        open(f'{badman}/manifest.tsv','w',encoding='utf-8').write('0\ta\ta.ss\n')
        rc, out = run(f'{HERE}/pck_tool.py', 'pack', pck, os.path.join(tmp,'bad.pck'),
                      badman, '--key-file', kf)
        assert rc != 0 and '槽位不完整' in out, out
        print('[7] 回封/回读内容级比对/缺槽拒绝 OK')

        # [8] find_key：合成"密钥打散存储"的伪引擎二进制 → 反汇编重组恢复
        try:
            import capstone  # noqa: F401
        except ImportError:
            print('[8] 跳过（未安装 capstone）')
        else:
            import random
            from crypt import TBL_SCENE
            rnd = random.Random(7)
            fake = bytearray(rnd.randrange(256) for _ in range(0x600))
            fake[0x100:0x100 + len(TBL_SCENE)] = TBL_SCENE      # 固定表 @VA 0x100
            addrs = []
            for i in range(16):                                  # 散装 16 字节
                addr = 0x250 + i * 3
                fake[addr] = KEY[i]
                addrs.append(addr)
            code = bytearray()
            order = list(range(16))
            rnd.shuffle(order)
            for i in order:                                      # 取字节→入栈槽
                code += b'\xa0' + struct.pack('<I', addrs[i])   # mov al,[imm32]
                if rnd.random() < 0.4:
                    code += b'\x31\xc9'                          # 干扰指令 xor ecx,ecx
                code += b'\x88\x45' + bytes([(-0x20 + i) & 0xff])  # mov [ebp-0x20+i],al
            site = 0x500                                         # 解密2 循环头
            fake[site:site] = b''
            loop = b'\x8a\x04\x0d' + struct.pack('<I', 0x100)   # mov al,[ecx+表VA]
            fake[site:site + len(loop)] = loop
            code_at = 0x400
            fake[code_at:code_at + len(code)] = code
            fexe = os.path.join(tmp, 'fake_engine.bin')
            open(fexe, 'wb').write(bytes(fake))
            rc, out = run(f'{HERE}/find_key.py', pck, fexe, expect=0)
            assert KEY.hex().upper() in out, out
            kout = os.path.join(tmp, 'rec_key.txt')
            run(f'{HERE}/find_key.py', pck, fexe, '--key-out', kout, expect=0)
            body = [l for l in open(kout).read().splitlines()
                    if l.strip() and not l.startswith('#')]
            assert body[0].upper() == KEY.hex().upper(), body
            print('[8] find_key 反汇编恢复散装密钥 OK')

        # [9] 多文件并发（--jobs=2）与串行的产物必须逐字节一致
        ws_ser, ws_par = os.path.join(tmp, 'ws_ser'), os.path.join(tmp, 'ws_par')
        run(f'{HERE}/make_template.py', srcdir, ws_ser, '--jobs=1', expect=0)
        run(f'{HERE}/make_template.py', srcdir, ws_par, '--jobs=2', expect=0)
        for nm in ('tpl/a.tsv', 'tpl/b.tsv', 'work/a.tsv', 'work/b.tsv'):
            assert open(f'{ws_ser}/{nm}', 'rb').read() == \
                   open(f'{ws_par}/{nm}', 'rb').read(), f'make_template {nm} 并发不一致'
        tx_s, tx_p = os.path.join(tmp, 'tx_s'), os.path.join(tmp, 'tx_p')
        run(f'{HERE}/extract_text.py', srcdir, tx_s, '--jobs=1', expect=0)
        run(f'{HERE}/extract_text.py', srcdir, tx_p, '--jobs=2', expect=0)
        for nm in ('a.txt', 'b.txt'):
            assert open(f'{tx_s}/{nm}', 'rb').read() == \
                   open(f'{tx_p}/{nm}', 'rb').read(), f'extract_text {nm} 并发不一致'
        run(f'{HERE}/validate_text.py', f'{ws}/work/a.tsv', f'{ws}/work/b.tsv',
            '--width=30', '--lines=1', '--jobs=2', expect=0)
        run(f'{HERE}/check_structure.py', f'{ws}/work/a.tsv', f'{ws}/work/b.tsv',
            f'--tpl={ws}/tpl', '--jobs=2', expect=0)
        run(f'{HERE}/check_terms.py', terms, f'{ws}/proof/a.tsv',
            f'{ws}/work/b.tsv', '--jobs=2', expect=0)
        run(f'{HERE}/check_punct.py', f'{ws}/proof/a.tsv', f'{ws}/work/b.tsv',
            '--jobs=2', expect=0)
        run(f'{HERE}/check_quotes_mirror.py', f'{ws}/proof/a.tsv',
            f'{ws}/work/b.tsv', '--jobs=2', expect=0)
        # merge_proof 目录模式：两个文件对 → 走进程池，与串行基线比对
        run(f'{HERE}/make_proof.py', f'{udir}/b.ss', f'{ws}/work/b.tsv',
            f'{ws}/proof/b.tsv', expect=0)
        mw = os.path.join(tmp, 'cw_work')
        mp2 = os.path.join(tmp, 'cw_proof')
        mw2 = os.path.join(tmp, 'cw_work2')
        os.makedirs(mw)
        os.makedirs(mp2)
        for nm in ('a.tsv', 'b.tsv'):
            shutil.copy(f'{ws}/work/{nm}', f'{mw}/{nm}')
            shutil.copy(f'{ws}/proof/{nm}', f'{mp2}/{nm}')
        shutil.copytree(mw, mw2)
        run(f'{HERE}/merge_proof.py', mw2, mp2, '--jobs=1', expect=0)
        _, out_mg = run(f'{HERE}/merge_proof.py', mw, mp2, '--jobs=2', expect=0)
        assert '合并改稿' in out_mg, out_mg
        for nm in ('a.tsv', 'b.tsv'):
            assert open(f'{mw}/{nm}', 'rb').read() == \
                   open(f'{mw2}/{nm}', 'rb').read(), f'merge_proof {nm} 并发不一致'
        # 解包/回封并发：产物与串行逐字节一致
        udir_s = os.path.join(tmp, 'unpacked_s')
        udir_j = os.path.join(tmp, 'unpacked_j')
        run(f'{HERE}/pck_tool.py', 'unpack', pck, udir_s, '--key-file', kf, expect=0)
        run(f'{HERE}/pck_tool.py', 'unpack', pck, udir_j, '--key-file', kf,
            '--jobs', '2', expect=0)
        for nm in ('a.ss', 'b.ss', 'manifest.tsv'):
            assert open(f'{udir_s}/{nm}', 'rb').read() == \
                   open(f'{udir_j}/{nm}', 'rb').read(), f'unpack {nm} 并发不一致'
        packed_j = os.path.join(tmp, 'Scene_new_j.pck')
        run(f'{HERE}/pck_tool.py', 'pack', pck, packed_j, udir,
            '--key-file', kf, '--jobs', '2', expect=0)
        assert open(packed, 'rb').read() == open(packed_j, 'rb').read(), \
            'pack 并发产物不一致'
        print('[9] 多文件并发（--jobs=2）与串行逐字节一致 OK')

        print('=== siglus-translation 技能全链路自测通过 ===')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    main()
