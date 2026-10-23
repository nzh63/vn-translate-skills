# -*- coding: utf-8 -*-
# 自测：改动本目录脚本后必跑。全部用合成样本，不依赖具体游戏。
# 用法: python _selftest.py   （全绿退出码 0）
import os, sys, tempfile, shutil, subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ks_tool
from ks_tool import KsParse, TAG_RE, load_ks

FAIL = []


def check(name, cond, detail=''):
    if cond:
        print(f'  ok  {name}')
    else:
        print(f'  FAIL {name} {detail}')
        FAIL.append(name)


SAMPLE = """[iscript]
var s = "这行[不是]标签";  // iscript 内不解析
[endscript]
;注释行 [p]
[tb_start_text mode=4 ]
#【ユリ】
「1行目[r]
2行目」[p][vs]
[_tb_end_text]
[tb_start_text mode=4 ]
#
旁白一行[p][vs]
[_tb_end_text]
[tb_start_text mode=1 ]
#【アリス】
「ruby[ruby text=サ]差分」[p]
[_tb_end_text]
""".replace('\r\n', '\n')


def test_ks_parse():
    print('== KsParse ==')
    ks = KsParse(SAMPLE)
    kinds = [(i, k) for _no, i, _sp, _t, k in ks.rows]
    check('可译行数量', len(ks.rows) == 6, kinds)
    check('iscript/注释/裸#不进表', all(
        '[不是]' not in t for *_x, t, _k in [r for r in ks.rows]))
    check('名字行', ks.rows[0][4] == 'name' and ks.rows[0][3] == '【ユリ】')
    check('名字行说话人列为空', ks.rows[0][2] == '')
    check('文本行说话人继承', ks.rows[1][2] == '【ユリ】')
    check('旁白无说话人', ks.rows[3][2] == '')
    check('多行对白按物理行拆条', ks.rows[1][3].strip().endswith('[r]')
          and ks.rows[2][3].strip().endswith('[p][vs]'))
    check('ruby 标签保留', '[ruby text=サ]' in ks.rows[5][3])


def test_tag_regex():
    print('== TAG_RE ==')
    check('普通标签', TAG_RE.findall('あ[p][vs]い') == ['[p]', '[vs]'])
    check('带参数标签', TAG_RE.findall('[ruby text=サ]') == ['[ruby text=サ]'])
    check('非标签方括号不误吞', TAG_RE.sub('', '文字[3]更多').strip() == '文字[3]更多')


def run(*args):
    r = subprocess.run([sys.executable, os.path.join(HERE, 'ks_tool.py'), *args],
                       capture_output=True, text=True)
    return r


def test_roundtrip(tmp):
    print('== rebuild/verify 回环 ==')
    scn = os.path.join(tmp, 'scn'); os.makedirs(scn, exist_ok=True)
    tpl = os.path.join(tmp, 'tpl'); os.makedirs(tpl)
    work = os.path.join(tmp, 'work'); os.makedirs(work)
    out = os.path.join(tmp, 'out'); os.makedirs(out)
    open(os.path.join(scn, 'a.ks'), 'w', encoding='utf-8', newline='\r\n').write(
        SAMPLE)  # 刻意用 CRLF 写入
    r = run('make_template', scn, os.path.join(tmp, 'ws'))
    check('make_template 成功', r.returncode == 0, r.stderr)
    wt = os.path.join(tmp, 'ws', 'work', 'a.tsv')
    # 填译文
    rows = open(wt, encoding='utf-8').read().split('\n')
    out_rows = []
    for line in rows:
        if not line.strip():
            continue
        cols = line.split('\t')
        orig = cols[2]
        if cols[0] == '0':
            trans = '屯屯屯'
        elif '[ruby' in orig:
            trans = orig.replace('差分', '中文')
        else:
            trans = orig
        out_rows.append('\t'.join(cols[:3] + [trans]))
    open(wt, 'w', encoding='utf-8', newline='\n').write('\n'.join(out_rows) + '\n')
    r = run('rebuild', os.path.join(scn, 'a.ks'), os.path.join(out, 'a.ks'), wt)
    check('rebuild 成功', r.returncode == 0, r.stderr)
    r = run('verify', os.path.join(scn, 'a.ks'), os.path.join(out, 'a.ks'), wt)
    check('verify MATCH', r.returncode == 0 and 'MATCH' in r.stdout, r.stdout + r.stderr)
    # CRLF 被归一
    out_text = open(os.path.join(out, 'a.ks'), 'rb').read().decode('utf-8')
    check('输出无 CR', '\r' not in out_text)
    check('结构行原样保留', '[tb_start_text mode=4 ]' in out_text and '[_tb_end_text]' in out_text)
    check('名字行回写带 #', '#屯屯屯' in out_text)
    check('ruby 保留且换文本', 'ruby[ruby text=サ]中文' in out_text)
    # 坏译文拒绝
    rows = out_rows
    bad = None
    for i, line in enumerate(rows):
        cols = line.split('\t')
        if '[p][vs]' in cols[3]:
            cols[3] = cols[3].replace('[p][vs]', '')
            rows[i] = '\t'.join(cols)
            bad = cols[0]
            break
    open(wt, 'w', encoding='utf-8', newline='\n').write('\n'.join(rows) + '\n')
    r = run('rebuild', os.path.join(scn, 'a.ks'), os.path.join(out, 'a.ks'), wt)
    check('标签缺失被拒', r.returncode != 0 and '标签序列不一致' in (r.stdout + r.stderr))
    # 字面 \n 拒绝
    rows = open(wt, encoding='utf-8').read().split('\n')
    for i, line in enumerate(rows):
        cols = line.split('\t')
        if cols[0] == bad:
            cols[3] = '译文' + chr(92) + 'n换行'
            rows[i] = '\t'.join(cols)
    open(wt, 'w', encoding='utf-8', newline='\n').write('\n'.join(rows) + '\n')
    r = run('rebuild', os.path.join(scn, 'a.ks'), os.path.join(out, 'a.ks'), wt)
    check('字面 \\n 被拒', r.returncode != 0 and '换行' in (r.stdout + r.stderr))
    # make_proof 兼容
    r = subprocess.run([sys.executable, os.path.join(HERE, 'make_proof.py'),
                        os.path.join(scn, 'a.ks'), wt,
                        os.path.join(tmp, 'a_proof.tsv')],
                       capture_output=True, text=True)
    check('make_proof 可用（引擎无关）', r.returncode == 0, r.stderr)


def test_asar(tmp):
    print('== asar_tool 回环 ==')
    src = os.path.join(tmp, 'pkg')
    os.makedirs(os.path.join(src, 'data', 'sub'))
    os.makedirs(os.path.join(src, 'empty'))
    files = {
        'index.html': '<html>hi</html>',
        'data/a.txt': 'A' * 100000,
        'data/sub/b.bin': bytes(range(256)) * 777,
        'data/empty_file': '',
    }
    for rel, data in files.items():
        with open(os.path.join(src, *rel.split('/')), 'wb') as w:
            w.write(data if isinstance(data, bytes) else data.encode('utf-8'))
    a1 = os.path.join(tmp, 'p1.asar')
    r = subprocess.run([sys.executable, os.path.join(HERE, 'asar_tool.py'), 'pack', src, a1],
                       capture_output=True, text=True)
    check('pack 成功', r.returncode == 0, r.stderr)
    ext = os.path.join(tmp, 'ext')
    r = subprocess.run([sys.executable, os.path.join(HERE, 'asar_tool.py'), 'extract', a1, ext],
                       capture_output=True, text=True)
    check('extract 成功且哈希校验过', r.returncode == 0, r.stderr)
    for rel, data in files.items():
        with open(os.path.join(ext, *rel.split('/')), 'rb') as f:
            got = f.read()
        check(f'内容一致 {rel}', got == (data if isinstance(data, bytes) else data.encode('utf-8')))
    r = subprocess.run([sys.executable, os.path.join(HERE, 'asar_tool.py'), 'verify', a1, ext],
                       capture_output=True, text=True)
    check('verify MATCH', r.returncode == 0 and 'MATCH' in r.stdout, r.stdout + r.stderr)
    # 带 template 回封（改一个文件）
    with open(os.path.join(ext, 'data', 'a.txt'), 'w') as w:
        w.write('B' * 50)
    a2 = os.path.join(tmp, 'p2.asar')
    r = subprocess.run([sys.executable, os.path.join(HERE, 'asar_tool.py'), 'pack', ext, a2,
                        '--template', a1], capture_output=True, text=True)
    check('带模板回封成功', r.returncode == 0, r.stderr)
    r = subprocess.run([sys.executable, os.path.join(HERE, 'asar_tool.py'), 'probe', a2],
                       capture_output=True, text=True)
    check('probe 正常', r.returncode == 0 and 'OK' in r.stdout, r.stdout)
    sys.path.insert(0, HERE)
    from asar_tool import read_header, iter_files, read_file_data
    import hashlib
    h1, o1, _ = read_header(a1)
    h2, o2, _ = read_header(a2)
    f1 = dict(iter_files(h1)); f2 = dict(iter_files(h2))
    diff = [p for p in f1
            if hashlib.sha256(read_file_data(a1, o1, f1[p])).digest()
            != hashlib.sha256(read_file_data(a2, o2, f2[p])).digest()]
    check('仅改动文件不同', diff == ['data/a.txt'], diff)
    os.remove(os.path.join(ext, 'data', 'sub', 'b.bin'))
    a3 = os.path.join(tmp, 'p3.asar')
    r = subprocess.run([sys.executable, os.path.join(HERE, 'asar_tool.py'), 'pack', ext, a3,
                        '--template', a1], capture_output=True, text=True)
    check('缺文件拒绝回封', r.returncode != 0 and '缺少' in (r.stdout + r.stderr))


def test_voice_namefix(tmp):
    print('== voice_namefix ==')
    import voice_namefix as vnf
    mp = os.path.join(tmp, 'map.tsv')
    open(mp, 'w', encoding='utf-8', newline='\n').write(
        '\t'.join(['【ユリ】', '【尤莉】']) + '\n'
        + '\t'.join(['【アリス】', '【爱丽丝】']) + '\n')
    ks = os.path.join(tmp, 'v.ks')
    body = ('[brainlabo_voconfig type=".ogg" number="1" buf="1" name="【ユリ】" storage="vo/lily" ]\r\n'
            '[voconfig name="&mp.name" vostorage="x" ]\r\n'
            '[jump storage="scene1.ks"]\r\n')
    open(ks, 'wb').write(body.encode('utf-8'))
    m = vnf.read_map(mp)
    check('改 1 处', vnf.fix_file(ks, m) == 1)
    raw = open(ks, 'rb').read()
    check('CRLF 保留', b'\r\n' in raw)
    check('注册名已改', b'name="' + '【尤莉】'.encode('utf-8') in raw)
    check('宏参数未动', b'name="&mp.name"' in raw)
    check('其他行未动', b'[jump storage="scene1.ks"]' in raw)
    check('幂等（二次 0 处）', vnf.fix_file(ks, m) == 0)


def test_quote_parity(tmp):
    print('== check_punct 块级引号净差 + fix_quotes ==')
    scn = os.path.join(tmp, 'qp'); os.makedirs(scn, exist_ok=True)
    ks_text = (
        '[tb_start_text mode=4 ]\n'
        '#【ユリ】\n'
        '「もしも、もしも、ね。[r]\n'
        '世界が終わる時になっても[r]\n'
        '互いを永遠に愛することを誓いますか？」[p][vs]\n'
        '[_tb_end_text]\n'
        '[tb_start_text mode=4 ]\n'
        '#\n'
        '『悪魔の証明』だ。[p][vs]\n'
        '[_tb_end_text]\n')
    open(os.path.join(scn, 'a.ks'), 'w', encoding='utf-8', newline='\n').write(ks_text)
    rows = [
        (0, '', '【ユリ】', '【尤莉】'),
        (1, '【ユリ】', '「もしも、もしも、ね。[r]', '「如果、如果，我说。[r]'),
        (2, '【ユリ】', '世界が終わる時になっても[r]', '「就算到了世界终结之时[r]'),
        (3, '【ユリ】', '互いを永遠に愛することを誓いますか？」[p][vs]',
         '「也要立下誓言，永远爱着彼此吗？」[p][vs]'),
        (4, '', '『悪魔の証明』だ。[p][vs]', '『魔鬼证明』。断言『没有』是不被允许的。[p][vs]'),
    ]
    wt = os.path.join(tmp, 'a.tsv')
    open(wt, 'w', encoding='utf-8', newline='\n').write(
        '\n'.join('\t'.join(str(x) for x in r) for r in rows) + '\n')
    r = subprocess.run([sys.executable, os.path.join(HERE, 'check_punct.py'), wt,
                        '--src=' + scn], capture_output=True, text=True)
    check('净差门禁拦到凭空加「', r.returncode != 0 and '凭空增引号' in r.stdout, r.stdout)
    r = subprocess.run([sys.executable, os.path.join(HERE, 'fix_quotes.py'),
                        '--src=' + scn, wt], capture_output=True, text=True)
    check('fix_quotes 自动修', r.returncode == 0 and '自动修复' in r.stdout, r.stdout)
    txt = open(wt, encoding='utf-8').read()
    check('续行行首「已删', '\t就算到了世界终结之时[r]' in txt)
    check('消息收尾保留', '爱着彼此吗？」[p][vs]' in txt)
    r = subprocess.run([sys.executable, os.path.join(HERE, 'check_punct.py'), wt,
                        '--src=' + scn], capture_output=True, text=True)
    check('修复后净差归零', r.returncode == 0, r.stdout)
    # 合法样例不得误报
    okrows = [
        (0, '', '彼女は「こんにちは」と言った。[p]', '她对我说「你好」。[p]'),
        (1, '', '"きょうだい"が増える。[r]', '『兄弟姐妹』会变多。[r]'),
    ]
    open(os.path.join(scn, 'b.ks'), 'w', encoding='utf-8', newline='\n').write(
        '[tb_start_text mode=4 ]\n#\n彼女は「こんにちは」と言った。[p]\n'
        '[_tb_end_text]\n[tb_start_text mode=4 ]\n#\n"きょうだい"が増える。[r]\n'
        '[_tb_end_text]\n')
    wt2 = os.path.join(tmp, 'b.tsv')
    open(wt2, 'w', encoding='utf-8', newline='\n').write(
        '\n'.join('\t'.join(str(x) for x in r) for r in okrows) + '\n')
    r = subprocess.run([sys.executable, os.path.join(HERE, 'check_punct.py'), wt2,
                        '--src=' + scn], capture_output=True, text=True)
    check('合法成对/转换不误报', r.returncode == 0, r.stdout)


def main():
    test_ks_parse()
    test_tag_regex()
    tmp = tempfile.mkdtemp(prefix='tyrano_selftest_')
    try:
        test_roundtrip(tmp)
        test_asar(tmp)
        test_voice_namefix(tmp)
        test_quote_parity(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    if FAIL:
        print(f'\n{len(FAIL)} 项失败: {FAIL}')
        raise SystemExit(1)
    print('\n全部通过')


if __name__ == '__main__':
    main()
