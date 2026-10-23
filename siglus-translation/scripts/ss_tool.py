# -*- coding: utf-8 -*-
# .ss 场景脚本工具：结构查看 / 字符串表导出 / 译文回写 / 内容级验证
#
# 用法:
#   python ss_tool.py info    <in.ss>                     头部各区段一览（排查用）
#   python ss_tool.py strings <in.ss> [--all]             列出字符串表 (索引\t内容)
#   python ss_tool.py rebuild <orig.ss> <out.ss> <tsv...> 按 TSV 重建（原文件只读）
#   python ss_tool.py verify  <orig.ss> <new.ss> <tsv...> 预期 vs 实际逐串比对（封包门禁）
#
# .ss 布局: [u32 头长][32×u32 头字段][字符串表 count×(u32 词偏移,u32 词长)]
#           [字符串数据区][字节码尾块]。头字段[0]=字节码偏移、[4]=字符串区偏移、
#           [2]=表偏移、[3]=[5]=条目数；偶数位 ≥6 的字段指向字节码内部/之后的区段。
# 字符串 = UTF-16 码元流，逐码元与 (28807*索引)%65536 异或（按索引自派生，无密钥）。
#
# 回写原则: 只改字符串内容；条目数不变 → 字节码零改动。所有串长度都不变时直接
# 原位覆盖（产物与原文件逐字节同构）；有长度变化才重排布局并平移相关头字段。
import sys, os, struct, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crypt import xor_u16_words


class Ss:
    def __init__(self, raw):
        self.raw = raw
        (self.hlen,) = struct.unpack_from('<I', raw, 0)
        self.hdr = list(struct.unpack_from('<32I', raw, 4))
        self.tbl_off, self.count = self.hdr[2], self.hdr[3]
        self.str_off = self.hdr[4]
        self.entries = [struct.unpack_from('<II', raw, self.tbl_off + i * 8)
                        for i in range(self.count)]
        # 字符串明文（str）；长度为 0 的空槽保留为 ''
        self.strings = []
        for off, ln in self.entries:
            if ln == 0:
                self.strings.append('')
                continue
            words = struct.unpack_from(f'<{ln}H', raw, self.str_off + off * 2)
            self.strings.append(''.join(map(chr, xor_u16_words(words, len(self.strings)))))

    def words_of(self, idx, text):
        """文本 → 该索引下的混淆码元流。非 BMP 字符拒绝（引擎按 u16 存储）。"""
        out = []
        for c in text:
            o = ord(c)
            if o > 0xFFFF:
                raise ValueError(f'[{idx}] 含 BMP 之外字符 {c!r}（U+{o:X}）——引擎无法存储')
            out.append(o)
        return xor_u16_words(out, idx)


def apply_translation(strings, trans):
    """把 {idx:(jp,cn)} 应用到字符串表副本。jp 非空则校验原文一致。
    Siglus 无换行记号机制：译文含真实换行一律拒绝；字面 \\n 会原样显示也拒绝。"""
    from _trans_common import load_map
    s = list(strings)
    errs = []
    for idx in sorted(trans):
        jp, cn = trans[idx]
        if not (0 <= idx < len(s)):
            errs.append(f'index {idx} 超出范围'); continue
        if jp and s[idx] != jp:
            errs.append(f'index {idx} 原文不符: 表内={s[idx][:20]!r} != TSV={jp[:20]!r}')
            continue
        if '\n' in cn or '\r' in cn:
            errs.append(f'index {idx} 译文含真实换行符'); continue
        if '\\n' in cn:
            errs.append(f'index {idx} 译文含字面 \\n 记号 —— Siglus 消息窗自动折行，'
                        f'不存在换行记号，请删掉它')
            continue
        s[idx] = cn
    return s, errs


def _str_region(ss):
    """定位字符串数据区 [hdr[4], 下一个被引用的区段边界 | EOF)。
    实测存在两种布局：字符串紧随表后（尾块在后）、字符串在文件末尾（字节码在表后）。
    本函数不假设顺序，按头字段寻址。只有**偶数位**字段被视为偏移
    （奇数位是长度/计数，数值撞进区间会造成误判）。"""
    start = ss.str_off
    bigger = [v for i, v in enumerate(ss.hdr) if i % 2 == 0 and v > start]
    end = min(bigger) if bigger else len(ss.raw)
    if any(start < v < end for i, v in enumerate(ss.hdr) if i % 2 == 0):
        raise SystemExit('有头字段指向字符串区内部？非预期布局，拒绝重建')
    need = sum(ln * 2 for _, ln in ss.entries)
    if end - start < need:
        raise SystemExit(
            f'字符串区判定异常：[{start:#x},{end:#x}) 容不下全部串数据({need}B)。'
            f'请用 ss_tool.py info 检查头字段，并向技能维护者反馈该样本')
    return start, end


def rebuild_bytes(ss, new_strings):
    """→ (bytes, 是否原位覆盖)。长度全不变时在原文件上原位改写字符串区；
    有长度变化时把整个字符串区【原位】替换为新串流，其后所有内容整体平移——
    两种布局（字符串在表后 / 在文件尾）都保持原有次序，仅长度差导致平移。"""
    new_blobs = [ss.words_of(i, t) for i, t in enumerate(new_strings)]
    lens_same = all(len(w) == ln for w, (_, ln) in zip(new_blobs, ss.entries))
    str_start, str_end = _str_region(ss)
    if lens_same:
        out = bytearray(ss.raw)
        for (off, ln), w in zip(ss.entries, new_blobs):
            struct.pack_into(f'<{ln}H', out, ss.str_off + off * 2, *w)
        return bytes(out), True

    # 表内偏移以"词"(2 字节)为单位，紧凑化重排
    data = bytearray()
    offs = []
    for w in new_blobs:
        offs.append(len(data) // 2)
        data += struct.pack(f'<{len(w)}H', *w)
    shift = len(data) - (str_end - str_start)
    hdr = list(ss.hdr)
    for i, v in enumerate(hdr):
        if i % 2 == 1 or i in (2, 4):      # 长度/计数与字符串区起点不动
            continue
        if v >= str_end:                   # 压在边界上的区段（如尾块起点）也要平移
            hdr[i] = v + shift
    out = bytearray(ss.raw)
    # ① 重写字符串表（新偏移/新长度）；② 切片赋值换入新串流——
    #    等长与否都成立：不等长时 bytearray 自动把其后内容整体平移 shift
    table = []
    for o, w in zip(offs, new_blobs):
        table += (o, len(w))
    struct.pack_into(f'<{ss.count * 2}I', out, ss.tbl_off, *table)
    out[str_start:str_end] = data
    struct.pack_into('<32I', out, 4, *hdr)
    return bytes(out), False


def expected_table(orig_path, tsv_paths):
    from _trans_common import load_map
    ss = Ss(open(orig_path, 'rb').read())
    trans = load_map(tsv_paths)
    s, errs = apply_translation(ss.strings, trans)
    return ss, s, errs


def cmd_rebuild(orig, out, tsvs):
    ob = open(orig, 'rb').read()
    ss = Ss(ob)
    from _trans_common import load_map
    trans = load_map(tsvs)
    new, errs = apply_translation(ss.strings, trans)
    if errs:
        raise SystemExit('TSV 校验失败（未写出任何文件）:\n  ' + '\n  '.join(errs))
    blob, inplace = rebuild_bytes(Ss(ob), new)
    open(out, 'wb').write(blob)
    tag = '原位覆盖' if inplace else '重排布局'
    print(f'{os.path.basename(out)}: 应用 {len(trans)} 条 ({tag}, '
          f'{len(ob)} -> {len(blob)} 字节)')


def _nonstring_parts(raw, ss):
    """布局无关的结构校验视图：(表后~字符串区前, 字符串区后) 两段字节。
    rebuild 只许改 字符串表本身 与 字符串数据区，这两段之外必须逐字节一致。"""
    tbl_end = ss.tbl_off + ss.count * 8
    s, e = _str_region(ss)
    return raw[tbl_end:s], raw[e:]


def cmd_verify(orig, out_new, tsvs):
    exp_ss, exp, errs = expected_table(orig, tsvs)
    if errs:
        print('TSV 本身有错:')
        for e in errs:
            print('  ', e)
        sys.exit(1)
    nb = open(out_new, 'rb').read()
    got_ss = Ss(nb)
    got = got_ss.strings
    bad = 0
    if len(got) != len(exp):
        print(f'FAIL: 字符串数量 {len(got)} != 预期 {len(exp)}')
        sys.exit(1)
    diffs = [i for i, (x, y) in enumerate(zip(exp, got)) if x != y]
    # 结构性检查: ASCII 结构串（标签/标识符/命令名）必须零改动；非字符串区字节码必须一致
    ostr = Ss(open(orig, 'rb').read()).strings
    struct_bad = [i for i, (x, y) in enumerate(zip(ostr, got))
                  if x.isascii() and x and x != y]
    old_mid, old_after = _nonstring_parts(open(orig, 'rb').read(), exp_ss)
    new_mid, new_after = _nonstring_parts(nb, got_ss)
    tail_ok = old_mid == new_mid and old_after == new_after
    if not tail_ok:
        print('FAIL: 字节码尾块被改动（rebuild 只许动字符串）')
        bad += 1
    if diffs:
        bad += len(diffs)
        print(f'FAIL: {len(diffs)} 个字符串与预期不符，示例:')
        for i in diffs[:5]:
            print(f'  [{i}] 期望={exp[i][:30]!r} 实际={got[i][:30]!r}')
    if struct_bad:
        bad += len(struct_bad)
        print(f'FAIL: {len(struct_bad)} 个 ASCII 结构串被改动，示例: {struct_bad[:5]}')
    if bad:
        sys.exit(1)
    print(f'MATCH: {len(got)} 个字符串全部符合预期；结构串零改动；字节码零改动')


def cmd_info(path):
    ss = Ss(open(path, 'rb').read())
    print(f'头长 {ss.hlen} | 条目 {ss.count} | 表@{ss.hdr[2]:#x} '
          f'| 字符串区@{ss.hdr[4]:#x} | 尾块@{ss.hdr[0]:#x} | 文件 {len(ss.raw)}B')
    nz = [(i, hex(v)) for i, v in enumerate(ss.hdr) if v]
    print('头字段:', ' '.join(f'[{i}]={v}' for i, v in nz))


def cmd_strings(path, show_all):
    ss = Ss(open(path, 'rb').read())
    for i, s in enumerate(ss.strings):
        if show_all or s:
            print(f'{i}\t{s}')


def main():
    a = sys.argv[1:]
    cmd = a[0] if a else ''
    if cmd == 'info' and len(a) == 2:
        cmd_info(a[1])
    elif cmd == 'strings' and len(a) >= 2:
        cmd_strings(a[-1], '--all' in a)
    elif cmd == 'rebuild' and len(a) >= 4:
        cmd_rebuild(a[1], a[2], a[3:])
    elif cmd == 'verify' and len(a) >= 4:
        cmd_verify(a[1], a[2], a[3:])
    else:
        raise SystemExit(__doc__)


if __name__ == '__main__':
    main()
