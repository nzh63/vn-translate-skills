# -*- coding: utf-8 -*-
# PSB/.scn 工具：结构转储 / 字符串表查看 / 译文回写 / 内容级验证
# 适用: KAGParserEx 等编译产物 ".scn"（"mdf\0"+u32+zlib+PSB v3，或裸 PSB）
#
# 用法:
#   python psb_tool.py json    <in.scn> [out.json]           树结构 → JSON（研究/定位用）
#   python psb_tool.py strings <in.scn>                      列出字符串表 (索引\t内容)
#   python psb_tool.py rebuild <orig.scn> <out.scn> <tsv...> [--no-backlog]
#                                                            按译文 TSV 重建（原文件只读）
#   python psb_tool.py verify  <orig.scn> <new.scn> <tsv...> [--no-backlog]
#                                                            预期 vs 实际逐串比对（封包门禁）
#
# 回写原理: 只改字符串内容，表序与数量不变 → PSB 树区字节零改动；
#           显示版(含换行记号)译好后自动派生其去记号的履历/backlog 副本(--no-backlog 关闭)。
import sys, os, json, zlib, struct

def rle(d, p, n):
    v = 0
    for i in range(n): v |= d[p+i] << (8*i)
    return v

def pick_width(v):
    """PSB 定宽数组元素宽度自描述标记: 0x0d..0x10 = 1..4 字节。
    注意 0x0f 是 **3 字节** 不是 4 —— 原版超 64KB 字符串区用的就是它，写错必乱。"""
    if v <= 0xFF:        return 0x0d, 1
    if v <= 0xFFFF:      return 0x0e, 2
    if v <= 0xFF_FFFF:   return 0x0f, 3
    return 0x10, 4

class StrRef:
    __slots__ = ('idx',)
    def __init__(self, idx): self.idx = idx

class Psb:
    """PSB v3 解析。字典键在独立 Names 表中（前缀编码），算法同 FreeMote LoadNames()。"""
    def __init__(self, raw, keep_strref=False):
        assert raw[:4] == b'PSB\x00', 'not PSB'
        (self.ver, _h1, _h2, self.off_strings, self.off_strdata,
         self.off_chunks, self.off_chunkdata, self.off_entries,
         self.entries_count, self.treelen) = struct.unpack_from('<10I', raw, 4)
        self.raw = raw
        self.keep_strref = keep_strref
        self.strings = self._load_strings()
        r = [raw, 0x2c]
        charset     = self._arr(r)
        names_data  = self._arr(r)
        name_idx    = self._arr(r)
        self.names = []
        for i in range(len(name_idx)):
            out, ch = bytearray(), names_data[name_idx[i]]
            while ch != 0:
                code = names_data[ch]
                out.append((ch - charset[code]) & 0xff)
                ch = code
            self.names.append(bytes(reversed(out)).decode('utf-8', 'replace'))
        vals = []
        while r[1] < self.off_strings:
            start = r[1]
            vals.append(self._val(r))
            assert r[1] > start, 'no progress'
        self.value = vals

    def _hdr(self, r):
        d, p = r[0], r[1]
        m = d[p]; n = m - 0x0c
        assert 1 <= n <= 8, f'bad count-width marker {m:#x} at {p:#x}'
        cnt = rle(d, p+1, n); p += 1 + n
        wm = d[p]; w = wm - 0x0c
        assert 1 <= w <= 8, f'bad elem-width marker {wm:#x} at {p:#x}'
        arr = [rle(d, p+1+i*w, w) for i in range(cnt)]
        r[1] = p + 1 + cnt*w
        return arr

    def _arr(self, r): return self._hdr(r)

    def _load_strings(self):
        raw, base = self.raw, self.off_strdata
        r = [raw, self.off_strings]
        offs = self._hdr(r)
        out = []
        for o in offs:
            e = raw.find(b'\x00', base + o)
            out.append(raw[base+o:e].decode('utf-8', 'replace'))
        return out

    def _val(self, r):
        d = r[0]
        t = d[r[1]]; r[1] += 1
        if t in (0x00, 0x01): return None
        if t == 0x02: return False
        if t == 0x03: return True
        if t == 0x04: return 0
        if 0x05 <= t <= 0x0c:
            n = t - 0x04
            v = rle(d, r[1], n); r[1] += n
            if n == 8 and v >= 1 << 63: v -= 1 << 64
            return v
        if 0x0d <= t <= 0x14:
            r[1] -= 1
            return self._hdr(r)
        if 0x15 <= t <= 0x18:
            n = t - 0x14
            idx = rle(d, r[1], n); r[1] += n
            if self.keep_strref: return StrRef(idx)
            return self.strings[idx] if idx < len(self.strings) else f'<str{idx}>'
        if 0x19 <= t <= 0x1c:
            n = t - 0x18
            v = rle(d, r[1], n); r[1] += n
            return {'__res__': v}
        if t == 0x1d: return 0.0
        if t == 0x1e: v = struct.unpack_from('<f', d, r[1])[0]; r[1] += 4; return v
        if t == 0x1f: v = struct.unpack_from('<d', d, r[1])[0]; r[1] += 8; return v
        if t == 0x20:  # List
            offs = self._hdr(r); base = r[1]; end = base; items = []
            for o in offs:
                r[1] = base + o
                items.append(self._val(r)); end = max(end, r[1])
            r[1] = end
            return items
        if t == 0x21:  # Objects
            names = self._hdr(r); offs = self._hdr(r)
            base = r[1]; end = base; out = {}
            for i, k in enumerate(names):
                r[1] = base + (offs[i] if i < len(offs) else 0)
                key = self.names[k] if k < len(self.names) else f'#{k}'
                out[key] = self._val(r); end = max(end, r[1])
            r[1] = end
            return out
        raise ValueError(f'unknown type {t:#x} at {r[1]-1:#x}')

def load_scn(path, keep_strref=False):
    d = open(path, 'rb').read()
    if d[:4] == b'mdf\x00':
        raw = zlib.decompress(d[8:])
    elif d[:4] == b'PSB\x00':
        raw = d
    else:
        raise SystemExit(f'{path}: 无法识别的容器（既非 mdf 也非 PSB）')
    return Psb(raw, keep_strref=keep_strref)

# ---------------- 字符串表读写（二进制层） ----------------

def read_string_table(raw):
    """→ dict(strings, offsets, off_strings, off_strdata, off_chunks)"""
    (_ver, _h1, _h2, off_strings, off_strdata,
     off_chunks, *_rest) = struct.unpack_from('<10I', raw, 4)
    p = off_strings
    cnt_w = raw[p] - 0x0c
    cnt = int.from_bytes(raw[p+1:p+1+cnt_w], 'little')
    elem_w = raw[p+1+cnt_w] - 0x0c
    arr0 = p + 2 + cnt_w
    offs = [int.from_bytes(raw[arr0+i*elem_w:arr0+(i+1)*elem_w], 'little') for i in range(cnt)]
    strings = []
    for o in offs:
        e = raw.find(b'\x00', off_strdata + o)
        strings.append(raw[off_strdata+o:e].decode('utf-8'))
    return dict(strings=strings, offsets=offs, off_strings=off_strings,
                off_strdata=off_strdata, off_chunks=off_chunks)

def apply_translation(strings, trans, derive_backlog=True):
    """把 {idx:(jp,cn)} 应用到字符串表副本；返回 (新表, 错误列表, 派生条数)。
    jp 非空则校验原文一致；显示版含 NL 记号的条目自动派生去记号履历副本。"""
    from _trans_common import NL
    s = list(strings)
    errs = []
    for idx in sorted(trans):
        jp, cn = trans[idx]
        if not (0 <= idx < len(s)):
            errs.append(f'index {idx} 超出范围'); continue
        if jp is not None and jp != '' and s[idx] != jp:
            errs.append(f'index {idx} 原文不符: 表内={s[idx][:20]!r} != TSV={jp[:20]!r}'); continue
        if '\n' in cn or '\r' in cn or '\n' in (jp or ''):
            errs.append(f'index {idx} 含真实换行符（应写字面 \\n 记号）'); continue
        s[idx] = cn
    derived = 0
    if derive_backlog and not errs:
        explicit = set(trans)
        c2i = {}
        for i, x in enumerate(strings):
            c2i.setdefault(x, i)
        for idx, (jp, cn) in trans.items():
            if jp is not None and NL in jp:
                partner = c2i.get(jp.replace(NL, ''))
                # 全局去重表中，无记号副本即同一原文的履历版
                if partner is not None and partner not in explicit and NL not in s[partner]:
                    s[partner] = cn.replace(NL, '')
                    derived += 1
    return s, errs, derived

def relayout(raw, new_strings):
    """重建字符串偏移数组与数据区，更新头部偏移；树区与其余部分字节不动。"""
    st = read_string_table(raw)
    data = bytearray(b'\x00')          # 约定: offset[0]=0 指向空串
    offs = [0]
    for x in new_strings[1:]:
        offs.append(len(data))
        data += x.encode('utf-8') + b'\x00'
    cm, cb = pick_width(len(offs))
    em, eb = pick_width(max(offs))
    arr = bytearray([cm]) + len(offs).to_bytes(cb, 'little') + bytes([em])
    for o in offs:
        arr += o.to_bytes(eb, 'little')
    new_strdata = st['off_strings'] + len(arr)
    new_chunks = new_strdata + len(data)
    out = bytearray(raw[:st['off_strings']]) + arr + data + raw[st['off_chunks']:]
    struct.pack_into('<I', out, 0x14, new_strdata)
    struct.pack_into('<I', out, 0x18, new_chunks)
    struct.pack_into('<I', out, 0x1c, new_chunks + 3)
    struct.pack_into('<I', out, 0x20, new_chunks + 6)
    return bytes(out)

def wrap_like(orig_bytes, new_raw):
    """按原文件容器形态输出：mdf 头保持，裸 PSB 保持"""
    if orig_bytes[:4] == b'mdf\x00':
        return b'mdf\x00' + struct.pack('<I', len(new_raw)) + zlib.compress(new_raw, 9)
    return new_raw

def expected_table(orig_path, tsv_paths, derive_backlog=True):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _trans_common import load_map
    raw = open(orig_path, 'rb').read()
    raw = zlib.decompress(raw[8:]) if raw[:4] == b'mdf\x00' else raw
    strings = read_string_table(raw)['strings']
    trans = load_map(tsv_paths)
    s, errs, derived = apply_translation(strings, trans, derive_backlog)
    return s, errs, derived

def cmd_rebuild(orig, out, tsvs, backlog=True):
    ob = open(orig, 'rb').read()
    raw = zlib.decompress(ob[8:]) if ob[:4] == b'mdf\x00' else ob
    strings = read_string_table(raw)['strings']
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _trans_common import load_map
    trans = load_map(tsvs)
    new, errs, derived = apply_translation(strings, trans, backlog)
    if errs:
        raise SystemExit('TSV 校验失败（未写出任何文件）:\n  ' + '\n  '.join(errs))
    new_raw = relayout(raw, new)
    open(out, 'wb').write(wrap_like(ob, new_raw))
    print(f'{os.path.basename(out)}: 应用 {len(trans)} 条, 派生履历 {derived} 条 '
          f'(psb {len(raw)} -> {len(new_raw)} 字节)')

def cmd_verify(orig, new, tsvs, backlog=True):
    exp, errs, _ = expected_table(orig, tsvs, backlog)
    if errs:
        print('TSV 本身有错:'); [print('  ', e) for e in errs]; sys.exit(1)
    nb = open(new, 'rb').read()
    nraw = zlib.decompress(nb[8:]) if nb[:4] == b'mdf\x00' else nb
    got = read_string_table(nraw)['strings']
    bad = 0
    if len(got) != len(exp):
        print(f'FAIL: 字符串数量 {len(got)} != 预期 {len(exp)}'); sys.exit(1)
    diffs = [i for i, (a, b) in enumerate(zip(exp, got)) if a != b]
    # 结构性检查: ASCII 结构串必须零改动、树必须可完整解析
    ob = open(orig, 'rb').read()
    ostr = read_string_table(zlib.decompress(ob[8:]) if ob[:4] == b'mdf\x00' else ob)['strings']
    struct_bad = [i for i, (a, b) in enumerate(zip(ostr, got)) if a.isascii() and a and a != b]
    tree_end_ok = True
    try:
        Psb(nraw)
    except Exception as e:
        tree_end_ok = False
        print(f'FAIL: 新文件 PSB 树解析失败: {e}')
    if diffs:
        bad = len(diffs)
        print(f'FAIL: {bad} 个字符串与预期不符，示例:')
        for i in diffs[:5]:
            print(f'  [{i}] 期望={exp[i][:30]!r} 实际={got[i][:30]!r}')
    if struct_bad:
        bad += len(struct_bad)
        print(f'FAIL: {len(struct_bad)} 个 ASCII 结构串被改动，示例: {struct_bad[:5]}')
    if bad or not tree_end_ok:
        sys.exit(1)
    print(f'MATCH: {len(got)} 个字符串全部符合预期；结构串零改动；树区完整')

def main():
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    cmd = a[0]
    backlog = '--no-backlog' not in a
    args = [x for x in a[1:] if x != '--no-backlog']
    if cmd == 'json':
        p = load_scn(args[0])
        out = args[1] if len(args) > 1 else args[0] + '.json'

        def plain(o):
            if isinstance(o, StrRef): return f'<str{o.idx}>'
            if isinstance(o, list): return [plain(x) for x in o]
            if isinstance(o, dict): return {k: plain(v) for k, v in o.items()}
            return o
        json.dump(plain(p.value), open(out, 'w', encoding='utf-8'),
                  ensure_ascii=False, indent=1)
        print(f'strings: {len(p.strings)}, top-level values: {len(p.value)} -> {out}')
    elif cmd == 'strings':
        for i, s in enumerate(load_scn(args[0]).strings):
            print(f'{i}\t{s}')
    elif cmd == 'rebuild':
        cmd_rebuild(args[0], args[1], args[2:], backlog)
    elif cmd == 'verify':
        cmd_verify(args[0], args[1], args[2:], backlog)
    else:
        raise SystemExit(__doc__)

if __name__ == '__main__':
    main()
