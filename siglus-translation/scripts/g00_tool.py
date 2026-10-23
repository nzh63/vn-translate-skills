"""
Siglus g00 容器解码/诊断/图集重生成工具
=========================================

g00 是 Siglus 引擎的图像/数据容器，类型 2（cut data）用于普通 sprite/CG 切图
和预渲染位数字形图集（如 _mj_SE20_NNNN.g00 = 手机聊天 20px 字体图集）。

## 容器结构（type 2）

  byte  type = 2
  u16   width / u16 height
  u32   cut_cnt
  G00_CUT_DATABASE[cut_cnt]  (6 ints/entry = 24B, PSD 还原元数据, 引擎忽略)
  然后 LZSS 压缩的 cut 表（1 字节对齐偏移, 引擎侧扫基址偏移）:
    [u32 arc_size][u32 org_size][LZSS data]
    LZSS: 12-bit 索引 + 4-bit 长度, BREAK_EVEN=1, LSB-first
  解压后 = cut 表:
    u32  cut_cnt
    (cut_cnt × u32 offset, u32 size)  偏移相对解压缓冲起点
    cut 块:
      G00_CUT_HEADER(116B, MSVC对齐) + G00_CHIP_HEADER(92B) + 原始 RGBA8888

## 手机聊天位数字形图集（銀色、遥か 已实测验证）

  文件: g00/_mj_SE20_0000..0025.g00 (26 个)
  映射: file = code // 2601 ;  chip = code % 2601   (2601 = 51×51 网格)
  字形: 16×20 RGBA（RGB=0 黑, alpha=笔画覆盖; 引擎绘制时用 alpha 出白字）
  空 cut(116B) = 该码无字形 → 游戏渲染空白

  → 简体独有字（值晚雏酱她等）在日文原图集中无字形 → 聊天空白。
    用 CJK 字体重生成图集即可修复（见 regen 子命令）。

## 引擎加载 g00 的扩展名回退顺序
  g00 → jpg → png → bmp → dds （找不到弹错误框; 汉化可用 .ggg 扩展名交换, 见 rendering-fonts.md 附录 B）
"""
import struct
import sys
import os
import io
import time
import shutil
try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = ImageDraw = ImageFont = None

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
import crypt as C  # noqa: E402  (lzss)

CUT_HDR_SIZE = 116
CHIP_HDR_SIZE = 92
EMPTY_CUT_SIZE = 116

# ---------- 解码 ----------
def find_lzss(file_data, expected_cut_cnt):
    base = 7 + 24 * expected_cut_cnt
    for start in range(base - 4, base + 8):
        if start + 8 > len(file_data):
            break
        a, o = struct.unpack_from('<II', file_data, start)
        if not (100 < a < 50_000_000 and 50_000 < o < 100_000_000):
            continue
        try:
            raw = C.lzss_decompress(file_data[start + 8: start + 8 + a], o)
        except Exception:
            continue
        if len(raw) == o and struct.unpack_from('<I', raw, 0)[0] == expected_cut_cnt:
            return raw, start
    return None, None

def decode_g00(file_data):
    if not file_data or file_data[0] != 2:
        return None
    w, h, cut_cnt = struct.unpack_from('<HHI', file_data, 1)
    raw, lzss_off = find_lzss(file_data, cut_cnt)
    if raw is None:
        return None
    return w, h, cut_cnt, raw, lzss_off

def get_cut_header(raw, cut_no):
    off, size = struct.unpack_from('<II', raw, 4 + cut_no * 8)
    return raw[off:off + size], size

def decode_chip(cut_block):
    if len(cut_block) < CUT_HDR_SIZE:
        return None
    type_ = cut_block[0]
    count = struct.unpack_from('<H', cut_block, 2)[0]
    (x, y, disp_xl, disp_yl, xc, yc, cut_xl, cut_yl) = struct.unpack_from('<8i', cut_block, 4)
    ch = cut_block[CUT_HDR_SIZE: CUT_HDR_SIZE + CHIP_HDR_SIZE]
    if len(ch) < CHIP_HDR_SIZE:
        return None
    cx, cy, ctype, cxl, cyl = struct.unpack_from('<HHBHH', ch, 0)
    rgba_off = CUT_HDR_SIZE + CHIP_HDR_SIZE
    rgba = cut_block[rgba_off: rgba_off + cxl * cyl * 4]
    return dict(type=type_, count=count, disp=(disp_xl, disp_yl), cell=(cut_xl, cut_yl),
                chip_pos=(cx, cy), chip_type=ctype, chip_size=(cxl, cyl), rgba=bytes(rgba))

def chip_rgba(raw, idx):
    blk, size = get_cut_header(raw, idx)
    if size <= EMPTY_CUT_SIZE:
        return None
    for ch_off in (116, 120, 128, 208):
        if ch_off + CHIP_HDR_SIZE > size:
            continue
        cxl, cyl = struct.unpack_from('<HH', blk, ch_off + 6)
        if 8 <= cxl <= 32 and 8 <= cyl <= 32 and ch_off + CHIP_HDR_SIZE + cxl * cyl * 4 <= size:
            return (cxl, cyl, blk[ch_off + CHIP_HDR_SIZE: ch_off + CHIP_HDR_SIZE + cxl * cyl * 4])
    return None

# ---------- 位数字形图集重生成 ----------
# 映射公式（銀色、遥か 实测 28/28 验证）: file = code//2601, chip = code%2601

def _render_glyph(char, font, cell_w=16, cell_h=20, em_top=-2):
    """渲染单个字符为 cell_w×cell_h RGBA（RGB=0 黑 + alpha）。

    对齐方式：按字体的 em box 定位（anchor='la'），保证所有字符垂直一致。
    校准（与 dat/font01.ttf 字号15 匹配）：em_top=-2 → 墨迹中心 ≈ 原版图集的 10.4，
    墨高 ≈ 13px（原版 12-14），一 的横画居中而非贴底。
    罕见超宽字回退为缩放+居中。无字形返回 None。
    """
    try:
        tmp = Image.new('L', (64, 64), 0)
        ImageDraw.Draw(tmp).text((0, 0), char, font=font, fill=255, anchor='la')
        bb = tmp.getbbox()
        if bb is None or bb[2] - bb[0] <= 0 or bb[3] - bb[1] <= 0:
            return None
        gw, gh = bb[2] - bb[0], bb[3] - bb[1]
        rgba = bytearray(cell_w * cell_h * 4)
        if gw > cell_w - 1 or gh > cell_h - 2:
            s = min((cell_w - 1) / gw, (cell_h - 2) / gh, 1.0)
            g = tmp.crop(bb).resize((max(1, int(gw * s)), max(1, int(gh * s))), Image.BILINEAR)
            canvas = Image.new('L', (cell_w, cell_h), 0)
            canvas.paste(g, ((cell_w - g.width) // 2, (cell_h - g.height) // 2))
        else:
            canvas = Image.new('L', (cell_w, cell_h), 0)
            x = cell_w / 2.0 - (bb[0] + bb[2]) / 2.0  # 墨迹水平居中
            ImageDraw.Draw(canvas).text((x, em_top), char, font=font, fill=255, anchor='la')
        a = canvas.tobytes()
        for i, v in enumerate(a):
            rgba[i*4+3] = v
        return bytes(rgba)
    except Exception:
        return None

CUT_HDR_16X20 = (b'\x01\x00\x01\x00' + struct.pack('<8i', 0, 0, 16, 20, 0, 0, 20, 20) + b'\x00' * 80)
CHIP_HDR_16X20 = (b'\x00\x00\x00\x00\x01\x00\x10\x00\x14\x00\x00\x00' + b'\x00' * 80)
EMPTY_CUT = (b'\x01\x00\x00\x00' + b'\xff' * 8 + b'\x01\x00\x00\x00\x01\x00\x00\x00' + b'\x00' * 96)

def build_moji_file(codes_to_chips, file_no, keep_header=None):
    cut_cnt = 2601
    table = bytearray()
    table += struct.pack('<I', cut_cnt)
    blobs = []
    for chip in range(cut_cnt):
        rgba = codes_to_chips.get(file_no * 2601 + chip)
        blobs.append((CUT_HDR_16X20 + CHIP_HDR_16X20 + rgba) if rgba else EMPTY_CUT)
    pos = 4 + cut_cnt * 8
    for blk in blobs:
        table += struct.pack('<II', pos, len(blk))
        pos += len(blk)
    for blk in blobs:
        table += blk
    comp = C.lzss_compress(bytes(table), level=17)
    if keep_header:
        return keep_header + comp
    hdr = b'\x02' + struct.pack('<HHI', 1020, 1020, cut_cnt)
    hdr += b'\x00' * (24 * cut_cnt + 2)
    return hdr + comp

def regen_moji_dir(g00_dir, font_path, files=None, backup_suffix='.orig'):
    """批量重生成图集为 _mj_SE20_*.chs（汉化共存模式）。

    原版 g00/_mj_SE20_*.g00 保留不动（引擎扩展名交换后：先查 .chs 再退 .g00）。
    汉化版输出为 .chs（g00 type-2 内容，引擎按 g00 解析 pct_type=1）。
    backup_suffix 保留仅为兼容旧调用；共存模式下不需要备份（原 .g00 天然保留）。
    """
    font = ImageFont.truetype(font_path, 15)
    for fn in (files if files is not None else range(26)):
        src = os.path.join(g00_dir, f'_mj_SE20_{fn:04d}.g00')
        dst = os.path.join(g00_dir, f'_mj_SE20_{fn:04d}.chs')
        if not os.path.exists(src):
            print(f'[{fn}] 原 .g00 缺失, 跳过' + ('（已有 .chs，保留）' if os.path.exists(dst) else ''))
            continue
        info = decode_g00(open(src, 'rb').read())
        if info is None:
            print(f'[{fn}] 跳过（解析失败）'); continue
        w, h, cut_cnt, raw, lzss_off = info
        keep_header = open(src, 'rb').read()[:lzss_off]
        codes = {}
        t0 = time.time()
        for chip in range(cut_cnt):
            code = fn * 2601 + chip
            if 0x20 <= code <= 0xFFFF and not (0xD800 <= code <= 0xDFFF):
                rgba = _render_glyph(chr(code), font)
                if rgba:
                    codes[code] = rgba
        newdata = build_moji_file(codes, fn, keep_header=keep_header)
        open(dst, 'wb').write(newdata)
        print(f'[{fn}] 渲染 {len(codes)} 字形 → {os.path.basename(dst)} ({len(newdata)}B, {time.time()-t0:.1f}s)')

# ---------- 其它子命令 ----------
def dump_g00(path, out_dir, only_nonempty=True, scale=1):
    with open(path, 'rb') as f:
        data = f.read()
    info = decode_g00(data)
    if info is None:
        print(f'[!] 解析失败: {path}', file=sys.stderr); return 1
    w, h, cut_cnt, raw, _ = info
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for i in range(cut_cnt):
        got = chip_rgba(raw, i)
        if got is None:
            continue
        cxl, cyl, rgba = got
        im = Image.frombytes('RGBA', (cxl, cyl), rgba)
        if scale != 1:
            im = im.resize((cxl * scale, cyl * scale), Image.NEAREST)
        im.save(os.path.join(out_dir, f'chip_{i:04d}_{cxl}x{cyl}.png'))
        saved += 1
    print(f'{path}: 导出 {saved} 个 chip 到 {out_dir}')
    return 0

def diff_g00(path_old, path_new):
    with open(path_old, 'rb') as f:
        d1 = decode_g00(f.read())
    with open(path_new, 'rb') as f:
        d2 = decode_g00(f.read())
    if not d1 or not d2 or d1[2] != d2[2]:
        print('无法比较'); return 1
    cnt = d1[2]
    diffs = []
    for i in range(cnt):
        b1, s1 = get_cut_header(d1[3], i)
        b2, s2 = get_cut_header(d2[3], i)
        if s1 != s2 or b1 != b2:
            diffs.append((i, s1, s2))
    print(f'差异 chip: {len(diffs)}/{cnt}')
    for i, s1, s2 in diffs[:30]:
        print(f'  chip {i}: {s1}B -> {s2}B')
    return 0

def probe_chip_layout(path, chip_indices):
    with open(path, 'rb') as f:
        info = decode_g00(f.read())
    if not info:
        return
    _, _, _, raw, _ = info
    for i in chip_indices:
        blk, size = get_cut_header(raw, i)
        chip = decode_chip(blk) if size > EMPTY_CUT_SIZE else None
        if chip:
            print(f'  chip {i:4}: size={size:5} disp={chip["disp"]} chip={chip["chip_size"]}')
        else:
            print(f'  chip {i:4}: EMPTY ({size}B)')

def main():
    import argparse
    ap = argparse.ArgumentParser(description='Siglus g00 type-2 解码/诊断/图集重生成')
    sub = ap.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('decode', help='打印头部信息'); p.add_argument('file')
    p = sub.add_parser('dump', help='导出 chip 为 PNG'); p.add_argument('file'); p.add_argument('-o', '--out', default='./g00_dump'); p.add_argument('--scale', type=int, default=4)
    p = sub.add_parser('diff', help='比较两个 g00'); p.add_argument('old_file'); p.add_argument('new_file')
    p = sub.add_parser('probe', help='检查指定 chip'); p.add_argument('file'); p.add_argument('chips', nargs='+', type=int)
    p = sub.add_parser('regen', help='重生成 _mj_SE20 图集为 .chs（共存模式，原 .g00 保留）')
    p.add_argument('g00_dir', help='g00 目录')
    p.add_argument('font', help='CJK 字体文件路径（如 dat/font01.ttf）')
    p.add_argument('-f', '--files', help='要重生成的文件号, 逗号分隔(默认全部 0-25)')
    p.add_argument('--no-backup', action='store_true')
    args = ap.parse_args()
    if args.cmd == 'decode':
        with open(args.file, 'rb') as f:
            info = decode_g00(f.read())
        if not info: print('解析失败'); return 1
        w, h, cut_cnt, raw, lo = info
        nonempty = sum(1 for i in range(cut_cnt) if struct.unpack_from('<I', raw, 4+i*8+4)[0] > EMPTY_CUT_SIZE)
        print(f'type2 w={w} h={h} cut_cnt={cut_cnt} lzss_off={hex(lo)} 非空chip={nonempty}')
    elif args.cmd == 'dump':
        return dump_g00(args.file, args.out, scale=args.scale)
    elif args.cmd == 'diff':
        return diff_g00(args.old_file, args.new_file)
    elif args.cmd == 'probe':
        probe_chip_layout(args.file, args.chips)
    elif args.cmd == 'regen':
        files = [int(x) for x in args.files.split(',')] if args.files else None
        regen_moji_dir(args.g00_dir, args.font, files=files, backup_suffix=None if args.no_backup else '.orig')
    return 0

if __name__ == '__main__':
    sys.exit(main())
