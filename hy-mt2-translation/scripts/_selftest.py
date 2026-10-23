# -*- coding: utf-8 -*-
# 自检：只依赖标准库 + mt_common，不需要 GPU / torch / transformers。改脚本后必跑。
# 用法: python _selftest.py
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mt_common as C

FAIL = []


def eq(name, got, want):
    if got != want:
        FAIL.append(f'{name}: got {got!r} want {want!r}')


def ok(name, cond, info=''):
    if not cond:
        FAIL.append(f'{name}: 断言失败 {info}')


# 1. 结构保护 / 还原
m, tags = C.protect('あ[r]い' + C.NL_LITERAL + 'う')
eq('protect.masked', m, 'あ⟨0⟩い⟨1⟩う')
eq('protect.tags', tags, ['[r]', C.NL_LITERAL])
eq('restore', C.restore(m, tags), 'あ[r]い' + C.NL_LITERAL + 'う')
eq('tags_of', C.tags_of(C.restore(m, tags)), tags)
eq('stripped', C.stripped('[r]あい[p]'), 'あい')
eq('protect.无记号', C.protect('ただの文')[1], [])

# 模型可能把占位符写成别的括号 —— 仍要能还原
eq('restore.宽容括号', C.restore('あ<0>い〈1〉う', ['[r]', '[p]']), 'あ[r]い[p]う')
# 越界编号原样保留（不猜，交给校验判失败）
eq('restore.越界', C.restore('あ⟨9⟩', ['[r]']), 'あ⟨9⟩')

# 2. 分隔符
eq('pick_delim', C.pick_delim(['a', 'b']), '‖')
eq('pick_delim.冲突', C.pick_delim(['a‖b']), '❙')
eq('split', C.split_output('甲‖乙‖丙', '‖', 3), ['甲', '乙', '丙'])
eq('split.数量不符', C.split_output('甲‖乙', '‖', 3), None)
eq('split.单条', C.split_output('甲', None, 1), ['甲'])

# 3. prompt
p1 = C.build_prompt('こんにちは', delim=None)
ok('prompt.无分隔符指令', '分隔符' not in p1, p1)
ok('prompt.目标语', '简体中文' in p1, p1)
p2 = C.build_prompt('あ‖い', delim='‖', placeholders=True,
                    terms=[('サクラ', '樱')], style='日常口语化', context=['前文'])
ok('prompt.分隔符', '‖' in p2 and '分隔符' in p2, p2)
ok('prompt.术语', 'サクラ翻译成樱' in p2, p2)
ok('prompt.风格', '日常口语化' in p2, p2)
ok('prompt.背景', '【背景信息】' in p2 and '前文' in p2, p2)
ok('prompt.占位', '⟨数字⟩' in p2, p2)

# 4. TSV 读写：只改译文列，其余列/行序/注释行原样
tmp = tempfile.mkdtemp()
src = os.path.join(tmp, '01.tsv')
raw = ('# 01.txt 标题\n'
       '0\t【ユリ】\t「おはよう」[r]\n'
       '1\t\t[iscript]x[endscript]\n'
       '2\t\tシステム\n')
open(src, 'w', encoding='utf-8', newline='\n').write(raw)

wf = C.WorkFile(src)
eq('tsv.ncol', wf.ncol, 3)
eq('tsv.rows', len(wf.rows), 3)
eq('tsv.append_mode', wf.append_mode, True)
eq('tsv.default_col', wf.default_col(), None)
wf.tcol = None
li, idx, cols = wf.rows[0]
wf.lines[li] = '\t'.join(cols + ['「早上好」[r]'])
wf.save(os.path.join(tmp, 'out.tsv'))
out = open(os.path.join(tmp, 'out.tsv'), encoding='utf-8').read()
eq('tsv.注释保留', out.split('\n')[0], '# 01.txt 标题')
eq('tsv.追加列', out.split('\n')[1], '0\t【ユリ】\t「おはよう」[r]\t「早上好」[r]')
eq('tsv.其余行不变', out.split('\n')[2:4], raw.split('\n')[2:4])

# 4 列骨架：改第 4 列
skel = os.path.join(tmp, 'w.tsv')
open(skel, 'w', encoding='utf-8', newline='\n').write('0\t\tあ\t\n1\t\tい\t已有\n')
w2 = C.WorkFile(skel)
eq('tsv.4col', w2.default_col(), 3)
eq('tsv.待译', [c[3].strip() for (_, _, c) in w2.rows], ['', '已有'])

# 5. 术语表
tp = os.path.join(tmp, 'terms.tsv')
open(tp, 'w', encoding='utf-8', newline='\n').write(
    '# 术语\nサクラ\t樱/H\t櫻\nラーメン\t拉面\n\n坏行\n')
terms = C.load_terms(tp)
eq('terms.条数', len(terms), 2)
eq('terms.多候选取首', terms[0], ('サクラ', '樱'))
eq('terms.过滤', C.filter_terms(terms, ['ラーメンを食べる'], 40), [('ラーメン', '拉面')])
eq('terms.未命中', C.filter_terms(terms, ['无关文本'], 40), [])

# 6. 清洗
eq('sanitize', C.sanitize(' a\tb\nc '), 'a b c')

# 7. 端到端：假模型跑通 prepare → 批量 → 占位还原 → 落盘（不需要 GPU）
import translate_tsv as T


class _FakeRunner:
    def __init__(self, args):
        pass

    def generate(self, prompts, max_new):
        return ['‖'.join('【译】' + s for s in p.split('\n\n')[-1].split('‖'))
                for p in prompts]


_real = T.Runner
T.Runner = _FakeRunner
try:
    d2 = os.path.join(tmp, 'e2e')
    os.makedirs(d2, exist_ok=True)
    s2 = os.path.join(d2, 't.tsv')
    # 3 列模板 → --out-dir 生成 4 列（未译行也必须补齐成 4 列）
    open(s2, 'w', encoding='utf-8', newline='\n').write(
        '# 注释\n0\t\tあ[r]い' + C.NL_LITERAL + 'う\n1\t\tかき\n')
    o2 = os.path.join(d2, 'work')
    _argv = sys.argv
    sys.argv = ['x', s2, '--out-dir', o2, '--batch', '2',
                '--failed', os.path.join(d2, 'f.tsv')]
    T.main()
    got = open(os.path.join(o2, 't.tsv'), encoding='utf-8').read().split('\n')
    eq('e2e.注释保留', got[0], '# 注释')
    eq('e2e.标签与\\n还原', got[1],
       '0\t\tあ[r]い' + C.NL_LITERAL + 'う\t【译】あ[r]い' + C.NL_LITERAL + 'う')
    eq('e2e.第二行', got[2], '1\t\tかき\t【译】かき')
    ok('e2e.全部4列', all(len(l.split('\t')) == 4 for l in got[1:] if l), got[1:])

    # 4 列骨架原地重译（--force）
    sys.argv = ['x', os.path.join(o2, 't.tsv'), '--force',
                '--failed', os.path.join(d2, 'f.tsv')]
    T.main()
    sys.argv = _argv
    got2 = open(os.path.join(o2, 't.tsv'), encoding='utf-8').read().split('\n')
    eq('e2e.原地重译', got2[2], '1\t\tかき\t【译】かき')
    eq('e2e.原地保结构', got2[1].split('\t')[:3], ['0', '', 'あ[r]い' + C.NL_LITERAL + 'う'])
finally:
    T.Runner = _real

# 8. int4 g128 量化原语（torch 可用时必跑：uint32 直序打包/往返/旧 scheme 拒载；int8 回退冒烟）
try:
    import json as _json
    import torch
    from int8_quant import (quantize_int4_g128, dequantize_int4_g128,
                            quantize_per_row_int8, dequantize_per_row_int8)
    torch.manual_seed(0)
    # 直序打包（uint32）：word0=0x21 → 列0 = 低4位 =1、列1 =2；K=128 → (1,16)
    p1 = torch.zeros(1, 16, dtype=torch.uint32)
    p1[0, 0] = 0x21
    d1 = dequantize_int4_g128(p1, torch.ones(1, 1),
                              torch.zeros(1, 1, dtype=torch.uint8))
    eq('i4.直序低列', (float(d1[0, 0]), float(d1[0, 1]), float(d1[0, 2])),
       (1.0, 2.0, 0.0))
    # 形状 / dtype / 打包往返
    w = torch.randn(64, 1024) * 0.02
    packed, sc, zp = quantize_int4_g128(w)
    ok('i4.packed形状', packed.shape == (64, 128) and packed.dtype == torch.uint32,
       (packed.shape, packed.dtype))
    ok('i4.scale形状', sc.shape == (64, 8) and sc.dtype == torch.float16, sc.shape)
    ok('i4.zp形状', zp.shape == (64, 8) and zp.dtype == torch.uint8, zp.shape)
    ok('i4.zp取值域', int(zp.max()) <= 15, int(zp.max()))
    wd = dequantize_int4_g128(packed, sc, zp)
    ok('i4.dequant形状', tuple(wd.shape) == tuple(w.shape), wd.shape)
    rel4 = float((wd.float() - w).norm() / w.norm())
    ok('i4往返精度', rel4 < 0.3, f'rel={rel4:.4f}')
    # uint32 视图 = 4 字节小端直序（K%8 行对齐）
    ok('i4.小端视图字节', packed.contiguous().view(torch.uint8).numel()
       == packed.numel() * 4)

    # scheme v2 字面 + 旧 scheme（v1）拒载：伪造 v1 index → load_into 必须拒绝
    import int8_cache
    _td = tempfile.mkdtemp()
    _old = os.environ.get('HY_MT2_CACHE_DIR')
    os.environ['HY_MT2_CACHE_DIR'] = _td

    class _Stub:
        device, pin, quant = 'cpu', False, 'int4'
    pp = int8_cache.paths('t/x', _td, quant='int4')
    os.makedirs(pp['root'], exist_ok=True)
    for _k in ('q4', 'scale', 'zp'):
        open(pp[_k], 'wb').close()
    _idx = dict(version=int8_cache.CACHE_VERSION, scheme='int4-g128-v1',
                model='t/x', dtype='float16', n_entries=0, fp8_bytes=0,
                scale_elems=0, zp_elems=0, params=0, entries={})
    with open(pp['index'], 'w', encoding='utf-8') as fh:
        _json.dump(_idx, fh)
    ok('i4scheme=v2字面', int8_cache.scheme_of('int4') == 'int4-g128-v2',
       int8_cache.scheme_of('int4'))
    ok('i4旧scheme拒载', int8_cache.load_into(_Stub(), 't/x', None, _td) is False)
    if _old is None:
        os.environ.pop('HY_MT2_CACHE_DIR', None)
    else:
        os.environ['HY_MT2_CACHE_DIR'] = _old

    # int8 回退冒烟（原语保留完好）
    q8, s8 = quantize_per_row_int8(w)
    ok('i8形状dtype', q8.shape == w.shape and q8.dtype == torch.int8
       and s8.shape == (64,), (q8.shape, q8.dtype, s8.shape))
    w8 = dequantize_per_row_int8(q8, s8)
    rel8 = float((w8 - w).norm() / w.norm())
    ok('i8往返精度', rel8 < 0.05, f'rel={rel8:.4f}')
except ImportError:
    print('(skip) 环境无 torch，跳过 int4/int8 量化自检')

print('FAIL' if FAIL else 'ALL PASS')
for f in FAIL:
    print(' -', f)
sys.exit(1 if FAIL else 0)
