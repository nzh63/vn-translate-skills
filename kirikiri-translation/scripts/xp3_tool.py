# -*- coding: utf-8 -*-
# XP3 通用解包/封包工具（标准 KiriKiri 格式 + 常见魔数混淆变体的自动探测）
# 仅标准库。
#
# 用法:
#   python xp3_tool.py probe <archive.xp3>                     探测格式（头部/索引/info 布局）
#   python xp3_tool.py unpack <archive.xp3> <outdir> [--no-verify]
#   python xp3_tool.py pack   <srcdir> <out.xp3> [--patch] [--template ref.xp3]
#
# pack 两种模式:
#   默认(整包重建): 相对路径、zlib 压缩(不划算则 stored)、带 time 块
#   --patch(补丁):  文件平铺为根目录裸名(强制, 引擎按裸名查询)、全部 stored 不压缩、
#                   File 子块为 [info][segm][adlr] 无 time —— 与实测可挂载的成品补丁同构
# --template 指定参考封包(通常是原版 data.xp3): 复制其前 32 字节头部并匹配
#   其索引布局(info 风格 / 索引是否有 pad 字段)。未指定则写标准 XP3。
#
# ⚠ 引擎行为因游戏而异，首次适配务必先用小规模伪翻译补丁做冒烟验证（见 references/workflow.md）。
import sys, os, zlib, struct, time

def u32(b, o): return struct.unpack_from('<I', b, o)[0]
def u64(b, o): return struct.unpack_from('<Q', b, o)[0]
def q(v): return struct.pack('<Q', v)   # 打包 u64（子块长度等）

STD_MAGIC = b'XP3\x0d\x0a\x20\x0a\x1a\x8b\x67\x01'   # 标准 XP3 头 11 字节

# ---------------- 索引与条目解析 ----------------

def _parse_index_tail(tail):
    """tail = 从索引起点到文件尾。返回 (skip, table_bytes)。兼容压缩/裸索引、带 pad 变体。"""
    if tail[:1] == b'\x02':                       # 未压缩索引: [02][u32 len][table...]
        return 5, tail[5:5 + u32(tail, 1)]
    if tail[:1] != b'\x01':
        raise ValueError(f'索引切换字节异常: {tail[:1]!r}')
    for skip in range(9, 40):                     # 9=标准[01]+u32+u32；17=[01]+u32+pad4+u32+pad4
        try:
            table = zlib.decompress(tail[skip:])
            if table[:4] == b'File':
                return skip, table
        except zlib.error:
            continue
    raise ValueError('zlib 解压索引失败且表头不像 File 条目')

def _parse_info(b):
    """File 内 info 子块 → (orig_size, comp_size, name, style)。自动识别两种常见布局:
    std:     [u32 flags][u64 orig][u64 comp][u32 namelen]name   (头 24B)
    compact: [u32 x4/u32 orig/u32 comp/u32 x0][u16 namelen]name (头 22B, 见于魔数混淆定制版)
    允许名字后有 0~2 字节尾差（个别写入器多补一个 NUL 终止符）。"""
    if len(b) >= 24:
        nl = u32(b, 20)
        if 0 <= len(b) - 24 - nl * 2 <= 2:
            return u64(b, 4), u64(b, 12), b[24:24+nl*2].decode('utf-16-le'), 'std'
    if len(b) >= 22:
        nl = struct.unpack_from('<H', b, 20)[0]
        if 0 <= len(b) - 22 - nl * 2 <= 2:
            return u32(b, 4), u32(b, 12), b[22:22+nl*2].decode('utf-16-le'), 'compact'
    raise ValueError('无法识别的 info 布局')

def _chunks(buf):
    out, p = {}, 0
    while p + 12 <= len(buf):
        name = buf[p:p+4]
        size = u64(buf, p+4)
        out[name] = buf[p+12:p+12+size]
        p += 12 + size
    return out

def probe_index(path, idx_off=None):
    """→ dict(hdr32, idx_off, skip, table, info_style)。
    idx_off=None 时在候选"索引偏移字段"位置 {19(标准), 0x20(加宽头变体)} 自动探测：
    读出 u64 索引偏移 → 跳到该处解析索引。也可直接传入确切的索引起点。"""
    fsize = os.path.getsize(path)
    with open(path, 'rb') as f:
        hdr32 = f.read(32)
        cands = [idx_off] if idx_off is not None else [19, 0x20]
        last = None
        for off in cands:
            try:
                if idx_off is not None:
                    pos = idx_off                      # 调用者直接给了索引起点
                else:
                    if off + 8 > fsize:
                        continue
                    f.seek(off)
                    pos = struct.unpack('<Q', f.read(8))[0]
                if not (0 < pos < fsize - 20):
                    raise ValueError(f'索引偏移越界: {pos:#x}')
                f.seek(pos)
                skip, table = _parse_index_tail(f.read())
            except ValueError as e:
                last = e
                continue
            # 取第一个 File 的 info 探测 info 风格
            style = '?'
            p = 0
            while p + 12 <= len(table):
                if table[p:p+4] == b'File':
                    ch = _chunks(table[p+12:p+12+u64(table, p+4)])
                    if b'info' in ch:
                        _o, _c, _n, style = _parse_info(ch[b'info'])
                        break
                p += 12 + u64(table, p+4)
            return dict(hdr32=hdr32, idx_off=off, skip=skip, table=table, info_style=style)
        raise SystemExit(f'索引探测失败（候选偏移 {cands}）: {last}')

def entries_of(table):
    """→ [{name, segm(flag,off,raw,packed), adler}]"""
    out, p = [], 0
    while p + 12 <= len(table):
        assert table[p:p+4] == b'File', f'表项 4CC 异常: {table[p:p+4]!r}'
        content = table[p+12:p+12+u64(table, p+4)]
        ch = _chunks(content)
        segm, info, adlr = ch[b'segm'], ch[b'info'], ch[b'adlr']
        o, c, name, _st = _parse_info(info)
        out.append(dict(name=name,
                        flag=u32(segm, 0), off=u64(segm, 4),
                        raw=u64(segm, 12), packed=u64(segm, 20),
                        adler=u32(adlr, 0)))
        p += 12 + u64(table, p+4)
    return out

def cmd_probe(path):
    r = probe_index(path)
    es = entries_of(r['table'])
    print(f'{path}:')
    print(f'  头部前32B : {r["hdr32"].hex()}')
    print(f'  标准头?   : {r["hdr32"][:11] == STD_MAGIC}  索引偏移字段 @ {hex(r["idx_off"])}'
          f'  索引 zlib 前跳过 {r["skip"]} 字节（9=标准, 17=带 pad 定制）')
    print(f'  info 布局 : {r["info_style"]}')
    print(f'  文件数    : {len(es)};  压缩标志分布: '
          f'{ {v: sum(1 for e in es if e["flag"]==v) for v in {e["flag"] for e in es}} }')

def cmd_unpack(path, outdir, verify=True, idx_off=None):
    r = probe_index(path, idx_off)
    es = entries_of(r['table'])
    ok = fail = 0
    with open(path, 'rb') as f:
        for i, e in enumerate(es):
            fname = e['name'].replace(chr(92), '/')
            if '..' in fname or fname.startswith('/'):
                print(f'  跳过可疑路径 [{i}] {fname!r}')
                fail += 1
                continue
            dest = os.path.join(outdir, fname)
            os.makedirs(os.path.dirname(dest) or '.', exist_ok=True)
            f.seek(e['off'])
            d = f.read(e['packed'])
            data = zlib.decompress(d) if e['flag'] else d
            if verify and (zlib.adler32(data) & 0xffffffff) != e['adler']:
                print(f'  FAIL [{i}] {fname}: adler32 不符')
                fail += 1
                continue
            open(dest, 'wb').write(data)
            ok += 1
    print(f'extracted ok={ok} fail={fail} -> {outdir}')
    sys.exit(1 if fail else 0)

# ---------------- 打包 ----------------

def collect(root, flat=False):
    files = []
    for dp, _, ns in os.walk(root):
        for n in sorted(ns):
            full = os.path.join(dp, n)
            rel = os.path.relpath(full, root).replace(os.sep, '/')
            files.append((rel, full))
    files.sort()
    if flat:
        bad = [rel for rel, _ in files if '/' in rel]
        if bad:
            raise SystemExit('patch 模式要求文件平铺在根目录（裸名），发现子路径: ' + ', '.join(bad[:5]))
    return files

def cmd_pack(srcdir, out, patch=False, template=None):
    files = collect(srcdir, flat=patch)
    tpl_traits = None
    if template and os.path.exists(template):
        t = probe_index(template)
        tpl_traits = dict(hdr32=t['hdr32'], padded=(t['skip'] == 17),
                          info_style=t['info_style'], idx_field=t['idx_off'])
        print(f'模板: 索引偏移@{hex(tpl_traits["idx_field"])}, '
              f'索引{"带pad" if tpl_traits["padded"] else "标准"}, info={tpl_traits["info_style"]}')
    if tpl_traits:
        hdr = bytearray(tpl_traits['hdr32'])
        idx_field = tpl_traits['idx_field']
        padded, info_style = tpl_traits['padded'], tpl_traits['info_style']
    else:
        # 标准 XP3 头：魔数 11B + 版本 u64@11 + 索引偏移 u64@19（占位）
        hdr = bytearray(STD_MAGIC + struct.pack('<Q', 0x17) + struct.pack('<Q', 0))
        idx_field = 19
        padded, info_style = False, 'std'
    # 头部总长以"索引偏移字段 + u64"为准（加宽头变体是 32+8=40；标准是 19+8=27）
    if idx_field + 8 > len(hdr):
        hdr += bytearray(idx_field + 8 - len(hdr))
    hdr_size = len(hdr)

    body = bytearray()
    ents = []
    for rel, full in files:
        content = open(full, 'rb').read()
        name = rel.encode('utf-16-le')
        off = hdr_size + len(body)
        if patch:
            blob, flag, raw_sz, pk_sz = content, 0, len(content), len(content)
            if info_style == 'std':
                # 标准 info: [u32 flags][u64 orig][u64 comp][u32 namelen][name]
                info = struct.pack('<IQQI', 0, raw_sz, pk_sz, len(rel)) + name
            else:
                info = struct.pack('<IIIII', 0, raw_sz, 0, pk_sz, 0) \
                       + struct.pack('<H', len(rel)) + name
            segm = struct.pack('<IQQQ', flag, off, raw_sz, pk_sz)
            entry = (b'info' + u64len(info) + info
                     + b'segm' + u64len(segm) + segm
                     + b'adlr' + q(4) + struct.pack('<I', zlib.adler32(content) & 0xffffffff))
        else:
            if len(content) <= 16 * 1024 * 1024:
                comp = zlib.compress(content, 9)
                flag = 1 if len(comp) < len(content) else 0
            else:
                comp, flag = content, 0
            blob = comp if flag else content
            if info_style == 'std':
                info = struct.pack('<IQQI', 0, len(content), len(blob), len(rel)) + name
            else:
                info = (struct.pack('<III', 0, len(content), 0) + struct.pack('<II', len(blob), 0)
                        + struct.pack('<H', len(rel)) + name)
            segm = struct.pack('<IQQQ', flag, off, len(content), len(blob))
            entry = (b'time' + q(8) + struct.pack('<Q', int(time.time()))
                     + b'adlr' + q(4) + struct.pack('<I', zlib.adler32(content) & 0xffffffff)
                     + b'segm' + u64len(segm) + segm
                     + b'info' + u64len(info) + info)
        body += blob
        ents.append(b'File' + u64len(entry) + entry)

    table = b''.join(ents)
    ztbl = zlib.compress(table, 9)
    idx_off = hdr_size + len(body)
    index = (b'\x01' + struct.pack('<I', len(ztbl)) + (b'\0\0\0\0' if padded else b'')
             + struct.pack('<I', len(table)) + (b'\0\0\0\0' if padded else b'') + ztbl)
    if idx_field + 8 > len(hdr):                     # 兜底（正常已在上方扩展）
        hdr += bytearray(idx_field + 8 - len(hdr))
    struct.pack_into('<Q', hdr, idx_field, idx_off)
    open(out, 'wb').write(bytes(hdr) + bytes(body) + index)
    mode = 'patch' if patch else 'data'
    print(f'{out}: {len(files)} 个文件, {os.path.getsize(out):,} 字节 [{mode}'
          + (', 模板匹配' if tpl_traits else ', 标准XP3') + ']')

def u64len(b): return struct.pack('<Q', len(b))

def main():
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    cmd = a[0]
    noverify = '--no-verify' in a
    patch = '--patch' in a
    rest = [x for x in a[1:] if not x.startswith('--')]
    tpl = None
    if '--template' in a:
        tpl = a[a.index('--template') + 1]
        rest = [x for x in rest if x != tpl]
    if cmd == 'probe' and len(rest) == 1:
        cmd_probe(rest[0])
    elif cmd == 'unpack' and 2 <= len(rest) <= 3:
        cmd_unpack(rest[0], rest[1], verify=not noverify,
                   idx_off=int(rest[2], 0) if len(rest) == 3 else None)
    elif cmd == 'pack' and len(rest) == 2:
        cmd_pack(rest[0], rest[1], patch=patch, template=tpl)
    else:
        raise SystemExit(__doc__)

if __name__ == '__main__':
    main()
