# -*- coding: utf-8 -*-
# 制作 Tyrano 官方补丁包 .tpatch（标准 zip，相对路径覆盖/新增文件）
# 引擎启动时检测 <exe目录>/<projectID>.tpatch：整包解 asar → zip 覆盖 → 重封 asar → 删 tpatch。
# projectID 见 data/system/Config.tjs 的 ;projectID=（本仓库样本为 yurialice）。
#
# 用法: python make_tpatch.py <payload目录> <输出.tpatch>
#   payload 目录内的文件按相对路径进入 zip 根，例如:
#     payload/data/scenario/scene1.ks  →  zip 内 data/scenario/scene1.ks
import os, sys, zipfile


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    payload, out = sys.argv[1], sys.argv[2]
    n = 0
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as z:
        for root, _dirs, files in os.walk(payload):
            for fn in sorted(files):
                fp = os.path.join(root, fn)
                arc = os.path.relpath(fp, payload).replace(os.sep, '/')
                z.write(fp, arc)
                n += 1
    print(f'{out}: {n} 个文件, {os.path.getsize(out):,} 字节')


if __name__ == '__main__':
    main()
