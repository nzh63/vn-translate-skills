# -*- coding: utf-8 -*-
# asar 封包工具（NW.js/Electron 封装的 TyranoScript/TyranoBuilder 游戏主封包）
# 仅用标准库。
#
# 用法:
#   python asar_tool.py probe   <app.asar>                     # 头部体检（条目数/数据偏移/结构）
#   python asar_tool.py list    <app.asar> [前缀过滤]           # 文件清单（路径\t大小）
#   python asar_tool.py extract <app.asar> <outdir> [前缀过滤]  # 解包（全部或仅某前缀）
#   python asar_tool.py pack    <indir> <new.asar> [--template <app.asar>]
#                              # 回封：有 --template 时按原包文件序排列、结构对齐；
#                              # 原包有而目录里没有的文件 → 致命；目录里新增文件 → 追加到尾部并警告
#   python asar_tool.py verify  <app.asar> <outdir>            # 逐文件 sha256 比对（解包/回封回环用）
#
# 格式要点（详见 references/asar-format.md）:
#   [0:4]=4  [4:8]=header_size  [8:12]=pickle载荷长  [12:16]=json长度  [16:]=json(+padding)
#   文件数据区起点 = 8 + header_size；offset 字段是相对数据区起点的字符串形式十进制。
import struct, json, hashlib, os, sys, io

BLOCK = 4194304  # electron asar integrity 的分块大小


def read_header(asar_path):
    with open(asar_path, 'rb') as f:
        head = f.read(16)
        if len(head) < 16:
            raise SystemExit(f'{asar_path}: 文件过小，不是 asar')
        a, b, c, d = struct.unpack('<4I', head)
        if a != 4:
            raise SystemExit(f'{asar_path}: 头部魔数异常（首4字节={a}，标准 asar 应为 4）')
        if c != d + (-(d) % 4):
            # [8:12] 应为 json 长度 + 对齐填充，不强校验，只提示
            pass
        js = f.read(d)
        if len(js) != d:
            raise SystemExit(f'{asar_path}: 头部 JSON 声明长度 {d} 超出文件')
        try:
            hdr = json.loads(js.decode('utf-8'))
        except Exception as e:
            raise SystemExit(f'{asar_path}: 头部 JSON 解析失败: {e}')
        data_off = 8 + b
        size = os.path.getsize(asar_path)
        if data_off > size:
            raise SystemExit(f'{asar_path}: 数据区起点 {data_off} 超出文件大小 {size}')
        return hdr, data_off, size


def _walk_files(node, path=''):
    for name, n in node.get('files', {}).items():
        p = path + '/' + name if path else name
        if 'files' in n:
            yield from _walk_files(n, p)
        else:
            yield p, n


def iter_files(hdr):
    return _walk_files(hdr)


def read_file_data(asar_path, data_off, node):
    with open(asar_path, 'rb') as f:
        f.seek(data_off + int(node['offset']))
        return f.read(node['size'])


def _sha_and_blocks(data):
    blocks = []
    for i in range(0, len(data), BLOCK):
        blocks.append(hashlib.sha256(data[i:i + BLOCK]).hexdigest())
    return hashlib.sha256(data).hexdigest(), blocks


def cmd_probe(args):
    hdr, data_off, size = read_header(args[0])
    files = list(iter_files(hdr))
    total = sum(n.get('size', 0) for _, n in files)
    unpacked = [p for p, n in files if n.get('unpacked')]
    print(f'文件: {args[0]}')
    print(f'包大小: {size:,}  头部数据区起点: {data_off:,}')
    print(f'条目: {len(files):,} 个文件, 共 {total:,} 字节')
    top = ', '.join(sorted(hdr.get('files', {}).keys()))
    print(f'顶层: {top}')
    if unpacked:
        print(f'警告: {len(unpacked)} 个 unpacked 条目（不在包体内，位于 <name>.unpacked/ 目录）:')
        for p in unpacked[:10]:
            print(f'  {p}')
    # 末尾文件是否越界
    bad = [p for p, n in files if data_off + int(n['offset']) + n.get('size', 0) > size]
    if bad:
        print(f'警告: {len(bad)} 个文件数据越界: {bad[:5]}')
    else:
        print('文件数据区完整性: OK（全部条目均在包内）')


def cmd_list(args):
    hdr, data_off, size = read_header(args[0])
    prefix = args[1] if len(args) > 1 else ''
    for p, n in iter_files(hdr):
        if not prefix or p.startswith(prefix):
            print(f'{n.get("size", 0):>12}\t{p}')


def cmd_extract(args):
    asar_path, outdir = args[0], args[1]
    prefix = args[2] if len(args) > 2 else ''
    hdr, data_off, size = read_header(asar_path)
    targets = [(p, n) for p, n in iter_files(hdr) if not prefix or p.startswith(prefix)]
    nfail = 0
    for i, (p, n) in enumerate(targets, 1):
        if n.get('unpacked'):
            print(f'跳过 unpacked 条目: {p}')
            continue
        dst = os.path.join(outdir, *p.split('/'))
        os.makedirs(os.path.dirname(dst) or '.', exist_ok=True)
        data = read_file_data(asar_path, data_off, n)
        if n.get('integrity', {}).get('algorithm') == 'SHA256':
            h, _ = _sha_and_blocks(data)
            if h != n['integrity']['hash']:
                print(f'校验失败: {p}')
                nfail += 1
                continue
        with open(dst, 'wb') as w:
            w.write(data)
        if i % 200 == 0 or i == len(targets):
            print(f'[{i}/{len(targets)}] {p}')
    if nfail:
        raise SystemExit(f'解包完成，{nfail} 个文件哈希校验失败')
    print(f'完成: {len(targets)} 个文件 → {outdir}')


def _build_header_json(indir, ordered_paths, data_off_hint=0):
    """ordered_paths: '/'分隔、相对 indir 的文件列表（已定序）。offset 先占位，返回 (json, [占位符])"""
    root = {}
    for p in ordered_paths:
        parts = p.split('/')
        node = root
        for seg in parts[:-1]:
            node = node.setdefault('files', {}).setdefault(seg, {'files': {}})
        node.setdefault('files', {})[parts[-1]] = {'__file__': p}
    # 深度优先转成 asar 头；offset 需要真实值，因此这里只建树，实际序列化在 pack 里做
    return root


def _write_asar(out_path, indir, ordered_paths):
    """两遍法：第一遍算每个文件 size（并顺带取 integrity），第二遍写。
    返回头 JSON 字符串。"""
    infos = []  # (path, size, integrity)
    for p in ordered_paths:
        fp = os.path.join(indir, *p.split('/'))
        h, blocks = None, None
        size = os.path.getsize(fp)
        with open(fp, 'rb') as f:
            hasher = hashlib.sha256()
            blocks = []
            while True:
                chunk = f.read(BLOCK)
                if not chunk:
                    break
                hasher.update(chunk)
                blocks.append(hashlib.sha256(chunk).hexdigest())
            h = hasher.hexdigest()
        infos.append((p, size, {'algorithm': 'SHA256', 'hash': h,
                                'blockSize': BLOCK, 'blocks': blocks}))

    # offset 相对数据区起点，与头部长度无关，一遍即可
    offsets = {}
    off = 0
    for p, size, _integ in infos:
        offsets[p] = str(off)
        off += size
    root = {'files': {}}
    for p, size, integ in infos:
        parts = p.split('/')
        node = root
        for seg in parts[:-1]:
            node = node['files'].setdefault(seg, {'files': {}})
        node['files'][parts[-1]] = {'size': size, 'offset': offsets[p],
                                    'integrity': integ}
    json_str = json.dumps(root, ensure_ascii=False, separators=(',', ':'))

    with open(out_path, 'wb') as w:
        jb = json_str.encode('utf-8')
        pad = (4 - len(jb) % 4) % 4
        payload_len = 4 + len(jb) + pad
        header_size = payload_len + 4
        w.write(struct.pack('<4I', 4, header_size, payload_len, len(jb)))
        w.write(jb)
        w.write(b'\x00' * pad)
        assert w.tell() == 8 + header_size
        for p, size, integ in infos:
            fp = os.path.join(indir, *p.split('/'))
            with open(fp, 'rb') as f:
                while True:
                    chunk = f.read(1 << 20)
                    if not chunk:
                        break
                    w.write(chunk)
    return json_str


def cmd_pack(args):
    if len(args) < 2:
        raise SystemExit(__doc__)
    indir, out_path = args[0], args[1]
    template = None
    if '--template' in args:
        template = args[args.index('--template') + 1]

    dir_files = set()
    for root, _dirs, fnames in os.walk(indir):
        for fn in fnames:
            fp = os.path.join(root, fn)
            rel = os.path.relpath(fp, indir).replace(os.sep, '/')
            dir_files.add(rel)

    if template:
        thdr, _off, _sz = read_header(template)
        tpaths = [p for p, _ in iter_files(thdr)]
        missing = [p for p in tpaths if p not in dir_files]
        if missing:
            raise SystemExit(f'回封目录缺少 {len(missing)} 个原包文件（不允许删减，除非逐个确认）:\n  '
                             + '\n  '.join(missing[:20]))
        ordered = tpaths + sorted(dir_files - set(tpaths))
        extra = sorted(dir_files - set(tpaths))
        if extra:
            print(f'警告: {len(extra)} 个新增文件将追加到包尾: {extra[:10]}')
    else:
        ordered = sorted(dir_files)

    total = sum(os.path.getsize(os.path.join(indir, *p.split('/'))) for p in ordered)
    print(f'回封 {len(ordered):,} 个文件, 共 {total:,} 字节 → {out_path}')
    _write_asar(out_path, indir, ordered)
    print(f'完成: {out_path} ({os.path.getsize(out_path):,} 字节)')


def cmd_verify(args):
    asar_path, outdir = args[0], args[1]
    hdr, data_off, size = read_header(asar_path)
    nfail, nskip = 0, 0
    entries = list(iter_files(hdr))
    for i, (p, n) in enumerate(entries, 1):
        dst = os.path.join(outdir, *p.split('/'))
        if not os.path.exists(dst):
            print(f'缺失: {p}')
            nfail += 1
            continue
        data = read_file_data(asar_path, data_off, n)
        with open(dst, 'rb') as f:
            disk = f.read()
        if len(disk) != len(data):
            print(f'大小不一致: {p} 包内 {len(data)} vs 磁盘 {len(disk)}')
            nfail += 1
            continue
        if hashlib.sha256(disk).hexdigest() != hashlib.sha256(data).hexdigest():
            print(f'内容不一致: {p}')
            nfail += 1
    extra = []
    for root, _dirs, fnames in os.walk(outdir):
        for fn in fnames:
            fp = os.path.join(root, fn)
            rel = os.path.relpath(fp, outdir).replace(os.sep, '/')
            if rel not in {p for p, _ in entries}:
                extra.append(rel)
    if extra:
        print(f'注意: 目录里有 {len(extra)} 个包外多余文件（不参与比对）')
    if nfail:
        raise SystemExit(f'verify 失败: {nfail} 处不一致')
    print(f'MATCH: {len(entries)} 个文件全部一致（asr vs {outdir}）')


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__ if '__doc__' in globals() else '用法见文件头')
    cmd, args = sys.argv[1], sys.argv[2:]
    {'probe': cmd_probe, 'list': cmd_list, 'extract': cmd_extract,
     'pack': cmd_pack, 'verify': cmd_verify}[cmd](args)


if __name__ == '__main__':
    main()
