# -*- coding: utf-8 -*-
# Scene.pck 容器工具：探测 / 解包 / 回封。
# Scene.pck 是"场景数据库"而非散文件归档：头部列出变量表/命令表/场景名/
# 场景信息等区段；每个场景数据块为 decrypt2(+decrypt1 游戏密钥)+LZSS 的 .ss 脚本。
#
# 用法:
#   python pck_tool.py probe  <Scene.pck> [--key HEX | --key-file F]
#   python pck_tool.py unpack <Scene.pck> <outdir> [--key HEX | --key-file F] [--jobs N]
#   python pck_tool.py pack   <orig.pck> <out.pck> <ssdir> [--key ...] [--level N] [--store] [--jobs N]
#
# unpack 会写出 manifest.tsv（序号 \t 场景名 \t 文件名）；pack 依赖它把文件对回场景槽位。
# pack 只重写 scnInfo 表与 scnData 区：内容未变的场景直接复用原数据块（逐字节相同），
# 变了的场景重压缩；其余区段零改动。游戏密钥属项目工作区，不写入本技能。
import sys, os, struct, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crypt import (TBL_SCENE, parse_key, load_key, xor_key16, xor_table,
                   lzss_decompress, lzss_store, compress_blob, decompress_blob)
from _trans_common import run_jobs

SECTIONS = ['varInfo', 'varNameIdx', 'varNames', 'cmdInfo', 'cmdNameIdx',
            'cmdNames', 'scnNameIdx', 'scnNames', 'scnInfo', 'scnData']


def parse_header(d):
    """→ dict(header_len, extra_key_use, src_header_len, sec={名:(off,count)})"""
    (hlen,) = struct.unpack_from('<I', d, 0)
    vals = struct.unpack_from('<22I', d, 4)
    sec = {SECTIONS[i // 2]: (vals[i], vals[i + 1]) for i in range(0, 20, 2)}
    return dict(header_len=hlen, extra_key_use=vals[20],
                src_header_len=vals[21], sec=sec)


def read_scene_names(d, hdr):
    d0, c0 = hdr['sec']['scnNameIdx']
    d1, _ = hdr['sec']['scnNames']
    out = []
    for i in range(c0):
        off, ln = struct.unpack_from('<II', d, d0 + i * 8)
        out.append(d[d1 + off * 2:d1 + (off + ln) * 2].decode('utf-16-le'))
    return out


def read_scene_blobs(d, hdr):
    """→ [(offset,len)] 原始（仍加密的）数据块描述"""
    d0, cnt = hdr['sec']['scnInfo']
    return [struct.unpack_from('<II', d, d0 + i * 8) for i in range(cnt)]


def scene_data_base(hdr):
    return hdr['sec']['scnData'][0]


def _nth_scene_blob(d, h, i):
    so, sl = read_scene_blobs(d, h)[i]
    base = scene_data_base(h)
    return d[base + so:base + so + sl]


def decode_scene(blob, key):
    return decompress_blob(blob, TBL_SCENE, key)


def encode_scene(ss_bytes, key, level, store):
    if store:
        comp = lzss_store(ss_bytes)
        if key:
            comp = xor_key16(comp, key)
        return xor_table(comp, TBL_SCENE)
    return compress_blob(ss_bytes, TBL_SCENE, key, level)


def _decode_job(t):
    """unpack/pack 共用的场景块解码 worker：异常就地转成文字，由主进程按槽位序汇报。"""
    i, blob, key = t
    try:
        return i, decode_scene(blob, key), ''
    except ValueError as e:
        return i, None, str(e)


def _encode_job(t):
    """pack 的重压缩 worker：明文 .ss → 加密压缩块。"""
    ss, key, level, store = t
    return encode_scene(ss, key, level, store)


def get_key(args):
    if args.key_file:
        return load_key(args.key_file)
    if args.key:
        return parse_key(args.key)
    return b''


def sanitize_name(name, used):
    """保留可读字符（含日文），只替换文件系统非法字符；冲突时加序号。"""
    bad = '<>:"/\\|?*'
    s = ''.join('_' if (ord(c) < 32 or c in bad) else c for c in name)
    s = s.rstrip(' .') or 'scene'
    base = s
    n = 1
    while s in used:
        s = f'{base}__{n}'
        n += 1
    used.add(s)
    return s


def cmd_probe(a):
    d = open(a.pck, 'rb').read()
    h = parse_header(d)
    print(f'头部长 {h["header_len"]} 字节 | ExtraKeyUse={h["extra_key_use"]}'
          f'{"（需游戏密钥）" if h["extra_key_use"] else ""}'
          f' | SourceHeaderLength={h["src_header_len"]:#x}')
    for k in SECTIONS:
        off, cnt = h['sec'][k]
        print(f'  {k:12s} off={off:#09x} count={cnt}')
    names = read_scene_names(d, h)
    print(f'场景数 {len(names)}，示例: {names[:3]}')
    key = get_key(a)
    if h['extra_key_use'] == 0:
        note = '（ExtraKeyUse=0，此包不需要游戏密钥）' if not key else ''
        try:
            blob = _nth_scene_blob(d, h, 0)
            ss = decode_scene(blob, b'')
            print(f'无密钥层验证 OK：场景[0] {len(blob)}B → {len(ss)}B 明文{note}')
        except ValueError as e:
            raise SystemExit(f'验证失败（ExtraKeyUse=0 却解不开）：{e}')
        return
    if not key:
        print('该包 ExtraKeyUse=1：需要游戏密钥（--key/--key-file），'
              '恢复方法见 references/key-recovery.md')
        return
    so, sl = read_scene_blobs(d, h)[0]
    blob = d[scene_data_base(h) + so:scene_data_base(h) + so + sl]
    try:
        ss = decode_scene(blob, key)
        print(f'密钥验证 OK：场景[0] {sl}B → {len(ss)}B 明文')
    except ValueError as e:
        raise SystemExit(f'密钥验证失败：{e}\n检查 --key 是否为本游戏的 16 字节密钥')


def cmd_unpack(a):
    d = open(a.pck, 'rb').read()
    h = parse_header(d)
    key = get_key(a)
    if h['extra_key_use'] and not key:
        raise SystemExit('该包 ExtraKeyUse=1，必须提供游戏密钥（--key / --key-file）')
    os.makedirs(a.out, exist_ok=True)
    names = read_scene_names(d, h)
    blobs = read_scene_blobs(d, h)
    base = scene_data_base(h)
    used, fns, items = set(), [], []
    for i, ((so, sl), nm) in enumerate(zip(blobs, names)):
        fns.append(sanitize_name(nm, used) + '.ss')   # 命名去重有先后依赖，必须按槽位序
        items.append((i, d[base + so:base + so + sl], key))
    rows, ok, bad = [], 0, 0
    for i, ss, err in run_jobs(_decode_job, items, a.jobs):
        if ss is None:
            bad += 1
            print(f'FAIL [{i}] {names[i]}: {err}')
        else:
            ok += 1
            open(os.path.join(a.out, fns[i]), 'wb').write(ss)
        rows.append((i, names[i], fns[i]))
    with open(os.path.join(a.out, 'manifest.tsv'), 'w', encoding='utf-8', newline='\n') as f:
        f.write('# 序号\t场景名\t文件名\n')
        for r in rows:
            f.write('\t'.join(map(str, r)) + '\n')
    print(f'解包 {ok}/{len(blobs)} 个场景 → {a.out}（失败 {bad}）')
    if bad:
        sys.exit(1)


def cmd_pack(a):
    d = open(a.pck, 'rb').read()
    h = parse_header(d)
    key = get_key(a)
    if h['extra_key_use'] and not key:
        raise SystemExit('该包 ExtraKeyUse=1，必须提供游戏密钥（--key / --key-file）')
    man = os.path.join(a.ssdir, 'manifest.tsv')
    if not os.path.exists(man):
        raise SystemExit(f'缺少 {man} —— 请使用 unpack 的产物目录，槽位对齐靠它保证')
    slots = []
    for ln, line in enumerate(open(man, encoding='utf-8'), 1):
        line = line.rstrip('\n')
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        p = line.split('\t')
        if len(p) != 3 or not p[0].isdigit():
            raise SystemExit(f'{man}:{ln}: 行格式应为 序号\\t场景名\\t文件名')
        slots.append((int(p[0]), p[2]))
    _, cnt = h['sec']['scnInfo']
    if sorted(i for i, _ in slots) != list(range(cnt)):
        raise SystemExit(f'manifest 场景槽位不完整（{len(slots)}/{cnt}），拒绝封包')

    # 原始场景明文缓存（解码并行）：内容未变的槽位直接复用原数据块
    rblobs = read_scene_blobs(d, h)
    base0 = scene_data_base(h)
    orig_ss = {}
    for i, plain, _err in run_jobs(
            _decode_job,
            [(i, d[base0 + so:base0 + so + sl], key)
             for i, (so, sl) in enumerate(rblobs)], a.jobs):
        if plain is not None:
            orig_ss[i] = plain

    new_blobs = [None] * len(slots)
    enc_items, enc_at = [], []
    reused = recoded = 0
    for k, (i, fn) in enumerate(slots):
        path = os.path.join(a.ssdir, fn)
        if not os.path.exists(path):
            raise SystemExit(f'缺文件: {path}（槽位 {i}）——解包失败的槽必须先补齐')
        ss = open(path, 'rb').read()
        if i in orig_ss and orig_ss[i] == ss:
            so, sl = rblobs[i]
            new_blobs[k] = d[scene_data_base(h) + so:scene_data_base(h) + so + sl]
            reused += 1
        else:
            enc_items.append((ss, key, a.level, a.store))   # 重压缩并行（CPU 大头）
            enc_at.append(k)
            recoded += 1
    for k, blob in zip(enc_at, run_jobs(_encode_job, enc_items, a.jobs)):
        new_blobs[k] = blob

    # 重排 scnInfo（顺序紧凑化）+ 替换 scnData 至 EOF
    out = bytearray(d)
    d0, cnt2 = h['sec']['scnInfo']
    cur = 0
    for i, blob in enumerate(new_blobs):
        struct.pack_into('<II', out, d0 + i * 8, cur, len(blob))
        cur += len(blob)
    base = scene_data_base(h)
    if base < d0 + cnt2 * 8:
        raise SystemExit('scnData 区与 scnInfo 表重叠？非预期布局，拒绝封包')
    out = out[:base] + b''.join(new_blobs)
    open(a.out, 'wb').write(bytes(out))
    print(f'封包 {len(new_blobs)} 槽（复用 {reused}，重压缩 {recoded}）'
          f' → {a.out} ({len(out)} 字节)')
    print('提示: 上真机前先对产物跑一遍 unpack 回读比对。')


def main():
    p = argparse.ArgumentParser(add_help=False,
                                description='Scene.pck 探测/解包/回封（SiglusEngine）')
    p.add_argument('cmd', choices=['probe', 'unpack', 'pack'])
    p.add_argument('pck', nargs='?')
    p.add_argument('out', nargs='?', help='unpack: 输出目录 / pack: 输出 .pck 路径')
    p.add_argument('ssdir', nargs='?', help='pack: unpack 产物目录')
    p.add_argument('--key', help='16 字节游戏密钥（十六进制 32 字符）')
    p.add_argument('--key-file', help='从文件读密钥（内容同上格式）')
    p.add_argument('--level', type=int, default=17, help='LZSS 匹配长上限 2..17（默认 17）')
    p.add_argument('--store', action='store_true', help='伪压缩（全字面量，慢机兜底用）')
    p.add_argument('--jobs', type=int, default=0,
                   help='unpack/pack 的并发进程数（0=自动按 CPU 数，1=串行）')
    a = p.parse_args()
    if a.cmd == 'probe':
        if not a.pck:
            raise SystemExit(__doc__)
        cmd_probe(a)
    elif a.cmd == 'unpack':
        if not a.pck or not a.out:
            raise SystemExit(__doc__)
        cmd_unpack(a)
    elif a.cmd == 'pack':
        if not a.pck or not a.out or not a.ssdir:
            raise SystemExit(__doc__)
        cmd_pack(a)


if __name__ == '__main__':
    main()
