# -*- coding: utf-8 -*-
"""量化权重的磁盘缓存：量化一次，之后直接 mmap 读回。

为什么需要
----------
量化 7B 要 **36s**（要读 13 GiB 权重 + 逐组/逐行求 scale），而且过程中 RSS
会冲到 ~9 GiB（safetensors 页面缓存），是本机物理内存最紧张的环节。既然结果
完全确定，就没必要每次启动重算。

两种量化、两套缓存（互不覆盖；根目录默认 ext4 `$HOME/.cache/hy-mt2-quant`，
env `HY_MT2_CACHE_DIR` 可覆盖 —— 放 /mnt/c 的 9p 上 mmap 缺页极慢）:
------------------------------
    int4-g128-v2（默认）：<root>/int4-g128-v2/<key>.q4.bin  packed uint32 直序（~3.25 GiB）
                                        <key>.scale.bin   每组 fp16 scale（~104 MiB）
                                        <key>.zp.bin      每组 uint8 zero-point（~52 MiB）
                                        <key>.index.json  元数据
    （v2 = uint32 直序打包；小端字节流与 v1 的 uint8 打包逐位相同，但 scheme
      升版防混读——load 见到 v1 scheme 一律拒绝回退重建。）
    per-row-int8-v1（回退）：<root>/<key>.fp8.bin    int8 权重原始字节（~6.50 GiB；
                             文件名沿用历史后缀 fp8，线格式不变）
                             <key>.scale.bin   per-row fp16 scale（~2.7 MB）
                             <key>.index.json  元数据：偏移 / 形状 / kind / revision / scheme

为什么拆成多个 bin
------------------
`torch.from_file` 只能**从文件头开始**映射，无法指定偏移；而量化字节与 fp16 的
元素宽度不同，塞进同一文件就没法对两者都做零拷贝映射。拆开后：

  - packed/int8 用 `from_file` **真正 mmap**：数 GiB 不必一次读进 RSS，
    按层缺页调入、OS 可随时回收 —— 正好解掉「量化副本常驻把物理内存榨干」；
  - scale/zp 只有 ~150 MB，直接读进内存，无所谓。

缓存失效条件
------------
`version` / `scheme` / `model` / `revision` 任一不符，或文件长度对不上，
就视为未命中并自动回退到重新量化（`--rebuild-cache` 删当前量化模式的文件）。
"""
import json
import os

import numpy as np
import torch

CACHE_VERSION = 1
SCHEME = 'per-row-int8-v1'          # int8 回退（沿用历史文件布局）
SCHEME_INT4 = 'int4-g128-v2'        # int4 g128 非对称（uint32 直序打包）


def cache_root(here=None):
    """量化缓存根：env HY_MT2_CACHE_DIR 可覆盖，默认 ext4 的 ~/.cache/hy-mt2-quant。

    （旧版曾放 skill 内 scripts/cache/——那是 /mnt/c 的 9p，mmap 缺页 ~170-230MB/s，
    已整体迁走；here 参数仅为兼容旧调用保留。）
    """
    return (os.environ.get('HY_MT2_CACHE_DIR')
            or os.path.expanduser('~/.cache/hy-mt2-quant'))


def scheme_of(quant):
    return SCHEME_INT4 if quant == 'int4' else SCHEME


def key_of(model_id):
    return model_id.replace('/', '__').replace(':', '_')


def paths(model_id, here, quant='int8'):
    root = cache_root(here)
    k = key_of(model_id)
    if quant == 'int4':
        root = os.path.join(root, SCHEME_INT4)   # 新格式独立子目录
        return dict(root=root,
                    q4=os.path.join(root, k + '.q4.bin'),
                    scale=os.path.join(root, k + '.scale.bin'),
                    zp=os.path.join(root, k + '.zp.bin'),
                    index=os.path.join(root, k + '.index.json'))
    return dict(root=root,
                fp8=os.path.join(root, k + '.fp8.bin'),
                scale=os.path.join(root, k + '.scale.bin'),
                index=os.path.join(root, k + '.index.json'))


def revision_of(model_dir):
    """快照目录名即 commit hash。"""
    return os.path.basename(os.path.normpath(model_dir))


def describe(model_id, here, revision=None, quant='int8'):
    """返回缓存状态摘要，供 CLI 打印。"""
    p = paths(model_id, here, quant=quant)
    if not os.path.exists(p['index']):
        return None
    try:
        with open(p['index'], encoding='utf-8') as fh:
            idx = json.load(fh)
    except Exception:
        return None
    if idx.get('scheme') != scheme_of(quant):
        return None
    if revision and idx.get('revision') and idx['revision'] != revision:
        return dict(stale=True, revision=idx.get('revision'), **idx)
    return idx


# ------------------------------------------------------------------ 保存

def save(store, model_id, revision, here, verbose=True):
    """把 store 里的量化权重与 scale(/zp) 落盘。

    用 numpy memmap 直接写入目标文件，避免构造巨型中间 bytes：
    单次最大额外内存 = 单个最大 Linear。
    """
    quant = getattr(store, 'quant', 'int8')
    p = paths(model_id, here, quant=quant)
    os.makedirs(p['root'], exist_ok=True)

    q_total = 0     # packed/int8 字节数（uint32 元素 ×4，必须按字节记账）
    sc_total = 0    # scale 元素数
    zp_total = 0    # zp 元素数（仅 int4）
    for e in store.weights.values():
        if e.get('kind') == 'fused':
            for qk, sk, zk in (('qg', 'sg', 'zg'), ('qu', 'su', 'zu')):
                q_total += e[qk].numel() * e[qk].element_size()
                sc_total += e[sk].numel()
                if zk in e and e[zk] is not None:
                    zp_total += e[zk].numel()
        else:
            q_total += e['q'].numel() * e['q'].element_size()
            sc_total += e['s'].numel()
            if e.get('z') is not None:
                zp_total += e['z'].numel()

    data_key = 'q4' if quant == 'int4' else 'fp8'
    tmp_data = p[data_key] + '.tmp'
    tmp_sc = p['scale'] + '.tmp'
    dst = np.memmap(tmp_data, dtype=np.uint8, mode='w+', shape=(max(1, q_total),))
    sc_buf = torch.empty(max(1, sc_total), dtype=torch.float16)
    zp_buf = torch.empty(max(1, zp_total), dtype=torch.uint8) if quant == 'int4' else None

    entries = {}
    off = 0
    sc_off = 0
    zp_off = 0

    def _write_q(t):
        """写一段 packed/int8 字节，返回起始偏移。"""
        nonlocal off
        t = t.detach().contiguous().reshape(-1)
        if t.dtype != torch.uint8:
            t = t.view(torch.uint8)
        n = t.numel()
        dst[off:off + n] = t.cpu().numpy()
        start = off
        off += n
        return start

    def _write_sc(t):
        nonlocal sc_off
        n = t.numel()
        sc_buf[sc_off:sc_off + n] = t.detach().to(torch.float16).cpu().reshape(-1)
        start = sc_off
        sc_off += n
        return start

    def _write_zp(t):
        nonlocal zp_off
        n = t.numel()
        zp_buf[zp_off:zp_off + n] = t.detach().to(torch.uint8).cpu().reshape(-1)
        start = zp_off
        zp_off += n
        return start

    for name, e in store.weights.items():
        if e.get('kind') == 'fused':
            parts = []
            for sub, qk, sk, zk in (('g', 'qg', 'sg', 'zg'),
                                    ('u', 'qu', 'su', 'zu')):
                q, s = e[qk], e[sk]
                # shape=逻辑形状（账目用）；qshape=量化后形状（view 用，
                # int4 是 uint32 packed (rows, K/8)，int8 时两者相同）
                logical = (list(q.shape) if quant != 'int4'
                           else [q.shape[0], q.shape[1] * 8])
                part = dict(sub=sub, shape=logical, qshape=list(q.shape),
                            fp8_off=_write_q(q),
                            fp8_len=q.numel() * q.element_size(),
                            sc_off=_write_sc(s), sc_len=s.numel())
                if quant == 'int4':
                    z = e[zk]
                    part['zp_off'] = _write_zp(z)
                    part['zp_len'] = z.numel()
                parts.append(part)
            entries[name] = dict(kind='fused', shape=list(e['shape']), parts=parts)
        else:
            q, s = e['q'], e['s']
            ent = dict(kind='plain', shape=list(e['shape']),
                       fp8_off=_write_q(q),
                       fp8_len=q.numel() * q.element_size(),
                       sc_off=_write_sc(s), sc_len=s.numel())
            if quant == 'int4':
                ent['qshape'] = list(q.shape)
                z = e['z']
                ent['zp_off'] = _write_zp(z)
                ent['zp_len'] = z.numel()
            entries[name] = ent

    dst.flush()
    del dst                       # 关闭 memmap
    sc_buf[:sc_off].numpy().tofile(tmp_sc)
    del sc_buf
    if zp_buf is not None:
        zp_buf[:zp_off].numpy().tofile(p['zp'] + '.tmp')
        del zp_buf

    os.replace(tmp_data, p[data_key])
    os.replace(tmp_sc, p['scale'])
    if quant == 'int4':
        os.replace(p['zp'] + '.tmp', p['zp'])

    index = dict(version=CACHE_VERSION, scheme=scheme_of(quant), model=model_id,
                 revision=revision, dtype='float16',
                 n_entries=len(entries), fp8_bytes=q_total, scale_elems=sc_off,
                 params=store.stats.get('params', 0), entries=entries)
    if quant == 'int4':
        index['zp_elems'] = zp_off
    tmp_idx = p['index'] + '.tmp'
    with open(tmp_idx, 'w', encoding='utf-8') as fh:
        json.dump(index, fh)
    os.replace(tmp_idx, p['index'])

    if verbose:
        if quant == 'int4':
            print(f'[cache] 已存 {q_total/2**30:.2f} GiB int4 packed + '
                  f'{sc_off*2/2**20:.1f} MiB scale + {zp_off/2**20:.1f} MiB zp '
                  f'→ {p["root"]}', flush=True)
        else:
            print(f'[cache] 已存 {q_total/2**30:.2f} GiB int8 权重 + '
                  f'{sc_off*2/2**20:.1f} MiB scale → {p["root"]}', flush=True)
    return p


# ------------------------------------------------------------------ 载入

def load_into(store, model_id, revision, here, verbose=True):
    """把缓存读进 store。命中返回 True，未返回 False。

    量化大块是 `torch.from_file` 的 mmap 视图 —— **零拷贝**，
    启动几乎瞬时，且数 GiB 不占 RSS（按层缺页调入）。
    """
    quant = getattr(store, 'quant', 'int8')
    p = paths(model_id, here, quant=quant)
    data_key = 'q4' if quant == 'int4' else 'fp8'
    need = [p[data_key], p['scale'], p['index']]
    if quant == 'int4':
        need.append(p['zp'])
    for f in need:
        if not os.path.exists(f):
            return False
    try:
        with open(p['index'], encoding='utf-8') as fh:
            idx = json.load(fh)
    except Exception:
        return False

    if (idx.get('version') != CACHE_VERSION or idx.get('scheme') != scheme_of(quant)
            or idx.get('model') != model_id):
        return False
    if revision and idx.get('revision') and idx['revision'] != revision:
        return False
    if os.path.getsize(p[data_key]) != idx['fp8_bytes']:
        return False
    if os.path.getsize(p['scale']) != idx['scale_elems'] * 2:
        return False
    if quant == 'int4':
        if idx.get('zp_elems') is None or os.path.getsize(p['zp']) != idx['zp_elems']:
            return False

    data_total = idx['fp8_bytes']
    if store.device == 'cpu':
        big = torch.from_file(p[data_key], shared=True, size=data_total,
                              dtype=torch.uint8)
    else:
        big = torch.from_file(p[data_key], shared=True, size=data_total,
                              dtype=torch.uint8).to(store.device)
    sc = torch.from_file(p['scale'], shared=True, size=idx['scale_elems'],
                         dtype=torch.float16)
    if store.device != 'cpu':
        sc = sc.to(store.device)
    zp_big = None
    if quant == 'int4':
        zp_big = torch.from_file(p['zp'], shared=True, size=idx['zp_elems'],
                                 dtype=torch.uint8)
        if store.device != 'cpu':
            zp_big = zp_big.to(store.device)

    weights = {}
    for name, ent in idx['entries'].items():
        if ent['kind'] == 'fused':
            d = dict(kind='fused', shape=tuple(ent['shape']))
            for part in ent['parts']:
                o, n = part['fp8_off'], part['fp8_len']    # 字节偏移/长度
                # int8 历史格式 shape=逻辑形状（按 int8 重解释）；
                # int4 v2 是 uint32 直序 packed（qshape=(rows, K/8)）。
                qs = part.get('qshape') or part['shape']
                q = big[o:o + n]
                if quant == 'int4':
                    q = q.view(torch.uint32).view(qs)     # o/n 均为 4 的倍数
                else:
                    q = q.view(torch.int8).view(qs)
                so, sn = part['sc_off'], part['sc_len']
                d['q' + part['sub']] = q
                d['s' + part['sub']] = sc[so:so + sn]
                if quant == 'int4':
                    # scale/zp 是 (rows, K/128) 组表，扁平切片必须 view 回二维
                    rows_, k_ = part['shape']
                    g_ = k_ // 128
                    d['s' + part['sub']] = sc[so:so + sn].view(rows_, g_)
                    zo, zn = part['zp_off'], part['zp_len']
                    d['z' + part['sub']] = zp_big[zo:zo + zn].view(rows_, g_)
            weights[name] = d
        else:
            o, n = ent['fp8_off'], ent['fp8_len']          # 字节偏移/长度
            qs = ent.get('qshape') or ent['shape']
            q = big[o:o + n]
            if quant == 'int4':
                q = q.view(torch.uint32).view(qs)
            else:
                q = q.view(torch.int8).view(qs)
            so, sn = ent['sc_off'], ent['sc_len']
            ent_d = dict(kind='plain', shape=tuple(ent['shape']),
                         q=q, s=sc[so:so + sn])
            if quant == 'int4':
                m_, k_ = ent['shape']
                g_ = k_ // 128
                ent_d['s'] = sc[so:so + sn].view(m_, g_)
                ent_d['z'] = zp_big[ent['zp_off']:
                                    ent['zp_off'] + ent['zp_len']].view(m_, g_)
            weights[name] = ent_d

    store.weights = weights
    store._big = big
    store._big_off = data_total
    store.cached = True
    # 还原统计，让后续打印/预算计算有据可依
    n_params = 0
    fp16_bytes = 0
    for name, d in weights.items():
        n = d['shape'][0] * d['shape'][1]
        n_params += n
        fp16_bytes += n * 2
    extra = idx['scale_elems'] * 2 + (idx.get('zp_elems') or 0)
    store.stats = dict(n=len(weights), params=n_params,
                       fp8_bytes=data_total + extra,
                       fp16_bytes=fp16_bytes, skipped=0)

    if verbose:
        tag = 'int4 packed' if quant == 'int4' else 'int8'
        zp_txt = (f' + zp {idx.get("zp_elems", 0)/2**20:.1f} MiB'
                  if quant == 'int4' else '')
        print(f'[cache] 命中 {idx.get("revision", "")[:12]}：'
              f'{len(weights)} 个 Linear / {n_params/1e9:.3f}B 参数，'
              f'{tag} {data_total/2**30:.2f} GiB + scale '
              f'{idx["scale_elems"]*2/2**20:.1f} MiB{zp_txt}'
              f'（mmap 零拷贝，未读入 RSS）', flush=True)
    return True


def remove(model_id, here, quant='int8', verbose=True):
    p = paths(model_id, here, quant=quant)
    n = 0
    for k in p:
        if k in ('root', 'index'):
            continue
        if os.path.exists(p[k]):
            os.remove(p[k])
            n += 1
    if os.path.exists(p['index']):
        os.remove(p['index'])
        n += 1
    if verbose and n:
        print(f'[cache] 已删除 {n} 个缓存文件（{scheme_of(quant)}）', flush=True)
    return n
