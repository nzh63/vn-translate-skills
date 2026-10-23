# -*- coding: utf-8 -*-
# 游戏密钥恢复：从引擎二进制（SiglusEngine.exe 等）里找出 16 字节场景密钥。
#
# 用法:
#   python find_key.py <Scene.pck> <候选二进制...> [--key-out <path>] [--verbose]
# 例:
#   python find_key.py Scene.pck SiglusEngine.exe --key-out workspace/key.txt
#
# 原理（两层策略，自动依次尝试）:
#   A 明文滑窗（零依赖）: 场景块明文前 4 字节 == u32(块长)，故
#       key[0:4] = cipher[0:4] ^ u32(块长) ^ 固定表[0:4]
#     拿它到候选文件里滑窗找连续 16 字节密钥（老引擎常直接存明文，可命中）。
#   B 反汇编重组（需 capstone，pip install capstone）: 新引擎把密钥**打散**成
#     单字节藏在数据区，运行时逐字节装入栈缓冲再参与异或。做法:
#       1) 字节搜索定位 256 字节固定表 → 表 VA；
#       2) 找引用该表的解密循环头（mov al, [reg + 表VA]）；
#       3) 向前线性反汇编，收集 "mov al,[imm32]" 与 "mov byte [ebp+d8],al" 配对；
#       4) 按槽位顺序重组 16 字节候选，用场景块全量验证。
#   验证标准: 解掉固定表层后 压缩长==块长 且完整 LZSS 解压成功（多场景复核）。
#   若两式皆失败: 见 references/key-recovery.md §4 的动态路线（skf.exe 等）。
import sys, os, struct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crypt import TBL_SCENE, xor_table, xor_key16, lzss_decompress


def load_blobs(pck_path, want=3):
    """→ [(blob, ...)] 原始（仍加密的）场景数据块"""
    d = open(pck_path, 'rb').read()
    vals = struct.unpack_from('<22I', d, 4)
    info_off, cnt = vals[16], vals[17]
    data_off = vals[18]
    out = []
    for i in range(min(want, cnt)):
        so, sl = struct.unpack_from('<II', d, info_off + i * 8)
        out.append(d[data_off + so:data_off + so + sl])
    return out


def verify(key, blobs):
    """密钥能否解开所有测试块：压缩长==块长 且完整解压"""
    if len(key) != 16:
        return False
    for blob in blobs:
        dec = xor_key16(xor_table(blob, TBL_SCENE), key)
        cs, ds = struct.unpack_from('<II', dec, 0)
        if cs != len(dec):
            return False
        try:
            if len(lzss_decompress(dec[8:], ds)) != ds:
                return False
        except ValueError:
            return False
    return True


def expected4(blob):
    """密文推出 key[0:4]: cipher ^ u32(块长) ^ 表头4字节"""
    plain = struct.pack('<I', len(blob))
    return bytes(a ^ b ^ c for a, b, c in zip(blob[:4], plain, TBL_SCENE[:4]))


def scan_plain(data, e4, blobs):
    """模式 A：滑窗找连续存储的密钥 → [(偏移, key)]"""
    hits, pos = [], 0
    while True:
        i = data.find(e4, pos)
        if i < 0 or i + 16 > len(data):
            break
        cand = data[i:i + 16]
        if verify(cand, blobs):
            hits.append((i, cand))
        pos = i + 1
    return hits


# ---------------- 模式 B：反汇编重组 ----------------

def _va_mapper(path, data):
    """VA ↔ 文件偏移。PE 用节表映射；非 PE 文件按恒等映射处理。"""
    try:
        import pefile
        pe = pefile.PE(path, fast_load=True)
        base = pe.OPTIONAL_HEADER.ImageBase

        def off2va(off):
            return base + pe.get_rva_from_offset(off)

        def va2off(va):
            return pe.get_offset_from_rva(va - base)

        return off2va, va2off
    except Exception:
        return (lambda off: off), (lambda va: va)


def scan_disasm(path, data, blobs, verbose=False):
    """模式 B：定位固定表→解密循环→回溯散装取字节指令→重组候选密钥"""
    ti = data.find(TBL_SCENE)
    if ti < 0:
        print('  [B] 未找到 256 字节固定表（引擎变体不同？），跳过反汇编模式')
        return []
    off2va, va2off = _va_mapper(path, data)
    tbl_va = off2va(ti)
    if verbose:
        print(f'  [B] 固定表 @文件{ti:#x} VA={tbl_va:#x}')

    import capstone
    md = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_32)

    # 循环头: mov al, byte ptr [reg + 表VA]。编码 8A /r，SIB 后跟 imm32==表VA。
    pat32 = struct.pack('<I', tbl_va)
    sites = []
    pos = 0
    while True:
        j = data.find(pat32, pos)
        if j < 0:
            break
        # mov r8,[reg+disp32] 的编码: 8A modrm [SIB] imm32 —— imm 前第 2 或 3 字节为 8A
        for k in (2, 3):
            head = j - k
            if head >= 0 and data[head] == 0x8A:
                sites.append(head)
                break
        pos = j + 1
    if verbose:
        print(f'  [B] 解密循环头 {len(sites)} 处: {[hex(s) for s in sites[:4]]}')

    results = []
    for site in sites:
        # 函数起点启发: 向前找 push ebp; mov ebp,esp
        fs = data.rfind(b'\x55\x8b\xec', max(0, site - 0x4000), site)
        if fs < 0 or site - fs > 0x1000:
            fs = max(0, site - 0x800)

        def collect(start):
            pairs = {}
            last_imm = None
            gap = 99
            for ins in md.disasm(data[start:site], start):
                m = ins.mnemonic, ins.op_str
                if m[0] == 'mov' and m[1].startswith('al, byte ptr ['):
                    try:
                        last_imm = int(m[1][len('al, byte ptr ['):-1], 16)
                        gap = 0
                    except ValueError:
                        last_imm, gap = None, 99
                    continue
                if (m[0] == 'mov' and m[1].startswith('byte ptr [ebp') and
                        m[1].endswith('], al')):
                    try:
                        raw = m[1][m[1].index('ebp') + 3:-len('], al')]
                        d8 = int(raw.replace(' ', ''), 16)
                    except ValueError:
                        d8 = None
                    if last_imm is not None and d8 is not None and gap <= 6:
                        pairs[d8] = last_imm
                    last_imm, gap = None, 99
                    continue
                gap += 1
            return pairs

        pairs = collect(fs)
        if len(pairs) < 16:
            # 线性反汇编可能中途失步（数据区混在前面）：分块重叠解码取并集
            merged = dict(pairs)
            step, size = 16, 48
            for s in range(fs, site - 1, step):
                merged.update(collect(s))
            pairs = merged
        if len(pairs) < 16:
            if verbose:
                print(f'  [B] @{site:#x}: 仅配对 {len(pairs)} 对，不足 16')
            continue
        lo = min(pairs)
        # 尝试所有连续 16 槽窗口（容错个别错配）
        bases = sorted(pairs)
        starts = [d for d in bases]
        tried = set()
        for start in starts:
            win = tuple(range(start, start + 16))
            if win in tried or any(d not in pairs for d in win):
                continue
            tried.add(win)
            cand = bytes(exe_byte(data, va2off(pairs[d])) for d in win)
            if verify(cand, blobs):
                results.append((site, start, cand))
    return results


def exe_byte(data, off):
    return data[off] if 0 <= off < len(data) else 0


def main():
    a = sys.argv[1:]
    key_out = None
    if '--key-out' in a:
        i = a.index('--key-out')
        key_out = a[i + 1]
        del a[i:i + 2]
    verbose = '--verbose' in a
    a = [x for x in a if x != '--verbose']
    if len(a) < 2:
        raise SystemExit(__doc__)
    pck, cands = a[0], a[1:]
    blobs = load_blobs(pck)
    e4 = expected4(blobs[0])
    print(f'密文推导 key[0:4] = {e4.hex().upper()}'
          f'（校验: 对真密钥此值必成立）')

    found = []
    for path in cands:
        data = open(path, 'rb').read()
        name = os.path.basename(path)
        hits = scan_plain(data, e4, blobs)
        for off, k in hits:
            print(f'[A] {name}@{off:#x}: 连续明文密钥 {k.hex().upper()} ✔')
            found.append((f'{name}@{off:#x}', k))
        try:
            import capstone  # noqa
        except ImportError:
            print('[B] 需要 capstone（python -m pip install capstone），已跳过')
            continue
        for site, start, k in scan_disasm(path, data, blobs, verbose):
            print(f'[B] {name}: 循环@{site:#x} 栈槽[{start:#x}] '
                  f'重组密钥 {k.hex().upper()} ✔')
            found.append((f'{name}@{site:#x}', k))

    if not found:
        print('未找到密钥。下一步见 references/key-recovery.md §4（同族 key / 动态工具）。')
        sys.exit(1)
    uniq = {k: src for src, k in found}
    for k, src in uniq.items():
        print(f'✔ 密钥: {k.hex().upper()}   （来自 {src}）')
    if key_out:
        os.makedirs(os.path.dirname(os.path.abspath(key_out)), exist_ok=True)
        with open(key_out, 'w', encoding='utf-8', newline='\n') as f:
            f.write('# 由 find_key.py 恢复\n')
            f.write(next(iter(uniq)).hex().upper() + '\n')
        print(f'已写入 {key_out}')


if __name__ == '__main__':
    main()
