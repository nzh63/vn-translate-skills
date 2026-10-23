# -*- coding: utf-8 -*-
# 本地 Hy-MT2 批量翻译译文 TSV。
#
# 用法:
#   python translate_tsv.py work/*.tsv [选项]                  # 原地补空（4 列骨架 / 6 列对照稿）
#   python translate_tsv.py tpl/*.tsv --out-dir work [选项]    # 3 列模板 → 生成 4 列译文
#   python translate_tsv.py work/01.tsv --dry-run --limit 3    # 只打 prompt，不加载模型
#
# 只写译文列：索引/说话人/原文列、行序、注释行原样保留（check_structure 硬门禁）。
# 行内标签 [r]/[p] 与字面 \n 记号为结构，先占位保护、译后还原并校验序列一致；
# 不一致的条自动降级为逐条重译，仍失败则记进 --failed 文件（译文留空，不写脏数据）。
import os, re, sys, time, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mt_common as C

# 本机 HF 直连不通：未显式设置时默认走镜像。权重不预下载，
# transformers 首次 from_pretrained 会自动拉取到 HF 缓存（~/.cache/huggingface/hub）。
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')

DENSE_GEN = dict(temperature=0.7, top_p=0.6, top_k=20, repetition_penalty=1.05)
MOE_GEN = dict(temperature=0.7, top_p=1.0, top_k=-1, repetition_penalty=1.0)


# ---------------- 参数 ----------------

def build_parser():
    p = argparse.ArgumentParser(add_help=True, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('inputs', nargs='+', help='译文/模板 TSV 文件或 glob')
    p.add_argument('--model', default='tencent/Hy-MT2-7B',
                   help='HF repo id 或本地目录（默认 tencent/Hy-MT2-7B）')
    p.add_argument('--out-dir', help='3 列模板输入时必填：输出目录（生成 4 列）')
    p.add_argument('--col', type=int, help='强制指定译文列下标（0 起；默认 3 列=追加，其余=3）')
    p.add_argument('--batch', type=int, default=8, help='一次请求合并翻译多少条（默认 8）')
    p.add_argument('--micro-batch', type=int, default=4, help='同时在 GPU 上跑多少"批"（默认 4）')
    p.add_argument('--context', type=int, default=0, help='带前 N 条原文作背景（默认 0；2~4 更连贯但更慢）')
    p.add_argument('--target-lang', default='简体中文')
    p.add_argument('--style', default='', help='风格指令，如 "日常口语化"')
    p.add_argument('--extra', action='append', default=[], help='追加到指令里的额外要求（可重复）')
    p.add_argument('--terms', help='术语表 terms.tsv（原文\\t规范译文[\\t禁用…]，与 check_terms.py 同格式）')
    p.add_argument('--max-terms', type=int, default=40, help='单批最多带多少条术语（默认 40）')
    p.add_argument('--skip-regex', action='append', default=[],
                   help='原文命中则跳过不译（可重复）。例：--skip-regex "^【[^】]*】$" 跳过名字行')
    p.add_argument('--force', action='store_true', help='已译文也重译（默认只补空）')
    p.add_argument('--limit', type=int, default=0, help='只处理前 N 条（冒烟用）')
    p.add_argument('--max-new-tokens', type=int, default=0, help='0=按输入长度自动估算')
    p.add_argument('--ratio', type=float, default=1.4, help='自动估算系数：max_new = len(src)*ratio+64')
    p.add_argument('--greedy', action='store_true', help='贪婪解码（确定性、分隔符更稳，但译文偏平）')
    p.add_argument('--quant', choices=['int4', 'int8'], default='int4',
                   help='权重量化模式（默认 int4 g128 非对称；int8=per-row 回退）')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--save-every', type=int, default=20, help='每 N 批落盘一次（断点续跑）')
    p.add_argument('--failed', default='_mt_failed.tsv', help='失败条目记录文件')
    p.add_argument('--dry-run', action='store_true', help='只打印 prompt，不加载模型')
    p.add_argument('--verbose', action='store_true')
    return p


# ---------------- 数据装配 ----------------

def expand(patterns):
    out, seen = [], set()
    for pat in patterns:
        hits = sorted(glob.glob(pat)) if any(c in pat for c in '*?[') else [pat]
        if not hits:
            raise SystemExit(f'找不到输入: {pat}')
        for h in hits:
            rp = os.path.realpath(h)
            if rp not in seen:
                seen.add(rp)
                out.append(h)
    return out


def prepare(files, out_dir, col_override):
    """载入输入并确定落盘目标 / 译文列。

    3 列模板：目标 = --out-dir/<同名>，全部行先补齐成 4 列（未译留空，
    保证列数一致，否则 check_structure 判失败、续跑也读不回来）；
    若目标已存在则直接载入它续跑（幂等）。
    """
    works = []
    for path in files:
        wf = C.WorkFile(path)
        tcol = col_override if col_override is not None else wf.default_col()
        if tcol is None:
            if not out_dir:
                raise SystemExit(f'{path} 是 3 列模板，必须给 --out-dir')
            dst = os.path.join(out_dir, os.path.basename(wf.path))
            if os.path.exists(dst):
                w2 = C.WorkFile(dst)
                if w2.ncol != 4:
                    raise SystemExit(f'{dst}: 期望 4 列，实际 {w2.ncol} 列')
                print(f'续跑已有译文: {dst}')
                wf.lines, wf.rows, wf.ncol = w2.lines, w2.rows, w2.ncol
            else:
                for pos, (li, idx, cols) in enumerate(wf.rows):
                    cols = cols + ['']
                    wf.rows[pos] = (li, idx, cols)
                    wf.lines[li] = '\t'.join(cols)
                wf.ncol += 1
            wf.tcol = 3
            wf.dest = dst
        else:
            if tcol >= wf.ncol:
                raise SystemExit(f'{path}: --col {tcol} 越界（本文件 {wf.ncol} 列）')
            wf.tcol = tcol
            wf.dest = wf.path
        works.append(wf)
    return works


def collect_jobs(works, force, skip_res, limit):
    """job = (file_idx, row_pos, src)。只收"译文列为空且原文有可译内容"的条。"""
    comps = [re.compile(r) for r in skip_res]
    jobs = []
    for fi, wf in enumerate(works):
        for pos, (_li, _idx, cols) in enumerate(wf.rows):
            src = wf.source_of(cols)
            if not C.stripped(src):
                continue                                   # 纯标签/空行，无需翻译
            if not force and cols[wf.tcol].strip():
                continue                                   # 已有译文，断点续跑
            if any(r.search(src) for r in comps):
                continue
            jobs.append((fi, pos, src))
            if limit and len(jobs) >= limit:
                return jobs
    return jobs


def apply_result(works, job, cn):
    fi, pos, _src = job
    wf = works[fi]
    li, _idx, cols = wf.rows[pos]
    cols[wf.tcol] = cn
    wf.rows[pos] = (li, _idx, cols)
    wf.lines[li] = '\t'.join(cols)


def save_all(works):
    for wf in works:
        wf.save(wf.dest)


# ---------------- 翻译核心 ----------------

class Runner:
    def __init__(self, args):
        import torch
        from transformers import AutoConfig, AutoTokenizer
        # 量化管线的导入链（连带 triton）必须晚于 CUDA 初始化
        torch.cuda.init()
        import offload_int8 as O
        self.torch = torch
        torch.manual_seed(args.seed)
        if not torch.cuda.is_available():
            raise SystemExit('torch 看不到 GPU（ROCm 下 torch.cuda.is_available() 应为 True），'
                             '先跑 env_check.py 体检')
        self.device = 'cuda'
        arch = ''
        try:
            arch = torch.cuda.get_device_properties(0).gcnArchName
        except Exception:
            pass
        print(f'[env] {torch.cuda.get_device_name(0)} | {arch or "?"} | '
              f'{getattr(args, "quant", "int4")} 分层常驻管线 '
              f'| vram={torch.cuda.get_device_properties(0).total_memory/2**30:.1f}GiB',
              flush=True)

        self.tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        self.tok.padding_side = 'left'

        # 与 _run_wsl_fused.sh / int8_pipe.py 同一套量化管线（默认 int4，--quant 可切）：
        # meta 结构建模 + 磁盘缓存 + 分层常驻 + 单块异步预取 + triton 融合 GEMV。
        cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        props = torch.cuda.get_device_properties(0)
        nso = argparse.Namespace(
            model=args.model, pin=False, no_cache=False, rebuild_cache=False,
            reserve_mib=1000, pin_layers=0, fp8_resident=True, no_pin=False,
            pin_mlp=False, interleave=True, no_fused_gemm=False, no_prefetch=False,
            prefetch_slots=3, issue_at='auto', inline_copy=False,
            quant=getattr(args, 'quant', 'int4'))
        self.model, mode = O.load_meta_fp8(nso, cfg, self.tok, props)
        print(f'[model] {mode}', flush=True)

        gen = dict(MOE_GEN if 'A3B' in args.model.upper() else DENSE_GEN)
        if args.greedy:
            gen = dict(do_sample=False, num_beams=1)
        else:
            gen['do_sample'] = True      # 光给 temperature/top_p 不生效，必须显式开
        self.gen = gen
        self.pad_id = self.tok.pad_token_id
        self.args = args

    def prompt_lens(self, prompts):
        """分桶用：每条 prompt 的 token 长度（不过 chat 模板——模板增量近似常数，
        长度序几乎不变，足够分桶；比字长代理准，覆盖中日混排比例差）。"""
        return [len(self.tok(p, add_special_tokens=False)['input_ids'])
                for p in prompts]

    def generate(self, prompts, max_new):
        torch = self.torch
        # 必须用 chat 模板把 user 消息包成模型训练时的格式，否则模型不会进入
        # "翻译"模式，而是自由续写（表现为回吐原文、给出多个候选、中英日混杂）。
        texts = [self.tok.apply_chat_template(
            [{'role': 'user', 'content': p}],
            tokenize=False, add_generation_prompt=True) for p in prompts]
        enc = self.tok(texts, return_tensors='pt', padding=True).to(self.device)
        with torch.inference_mode():
            out = self.model.generate(**enc, max_new_tokens=max_new,
                                      pad_token_id=self.pad_id, **self.gen)
        gen = out[:, enc['input_ids'].shape[1]:]
        return self.tok.batch_decode(gen, skip_special_tokens=True)


LABEL_RE = re.compile(r'^\s*(翻译结果|译文|翻译|Translation)\s*[:：]\s*')


def clean_output(s):
    s = C.sanitize(s)
    s = LABEL_RE.sub('', s)
    return s.strip()


def run_batch_group(runner, works, groups, terms_all, args):
    """groups: [(jobs, masked_srcs, tags_list, delim, context_lines)]

    每条源文本单独成一条 prompt；**组内按长度分桶**再逐桶 generate：
    同桶 max_len≈min_len，砍 left-padding 浪费（prefill 成本 ∝ batch×max_len）。

    续跑不变量：分桶只发生在**本 chunk 内**，chunk = 外层按原序连续切块
    （pending 顺序的后缀）。真实中断只可能停在 flush 边界（未 flush 的结果
    整体丢失→整块回到 pending），因此 resume 看到的 chunk 序列与一次跑完
    完全相同 → 每条的桶成员/输入逐一致 → 最终文件逐字节一致。
    写盘始终按原行号写（wf.lines[li]），输出行序 = 输入行序。

    旧方案（整 chunk 一次 generate、按最长 pad）已废弃——同上不再做
    分隔符拼接，保持"每条独立 prompt"的批量推理。

    返回 (结果 dict job->cn, 失败 jobs)。"""
    res, failed = {}, []
    items = []                                   # (job, prompt, tags) 保持原顺序
    for jobs, srcs, tags_list, delim, ctx in groups:
        for job, src, tags in zip(jobs, srcs, tags_list):
            use_terms = C.filter_terms(terms_all, [src], args.max_terms)
            items.append((job, C.build_prompt(src, target_lang=args.target_lang,
                                              style=args.style,
                                              terms=use_terms, extra=args.extra,
                                              context=ctx,
                                              delim=None, placeholders=bool(tags)),
                          tags))
        if args.verbose and srcs:
            print('--- PROMPT (chunk head) ---\n' + items[-len(srcs)][1]
                  + '\n--------------', flush=True)

    if not items:
        return res, failed

    # 分桶：优先真实 token 长度（Runner.prompt_lens），无则用 prompt 字长代理
    lens_fn = getattr(runner, 'prompt_lens', None)
    if callable(lens_fn):
        keys = lens_fn([it[1] for it in items])
    else:
        keys = [len(it[1]) for it in items]
    order = sorted(range(len(items)), key=lambda i: keys[i])   # 稳定排序

    for s in range(0, len(order), args.batch):
        win = [items[i] for i in order[s:s + args.batch]]
        prompts = [it[1] for it in win]
        n = max(len(p) for p in prompts)
        max_new = args.max_new_tokens or min(4096, max(64, int(n * args.ratio) + 64))
        outs = runner.generate(prompts, max_new)
        for (job, _p, tags), raw in zip(win, outs):
            out = clean_output(raw)
            cn = C.restore(out, tags)
            if tags and C.tags_of(cn) != tags:
                failed.append(job)          # 标签序列变了 → 结构破坏，留空
                continue
            if C.PH_PAT.search(cn):
                failed.append(job)          # 占位符没还原干净
                continue
            if not cn.strip():
                failed.append(job)
                continue
            res[job] = cn
    return res, failed


def build_group(works, jobs, args, cache):
    """把一批 job 组装成 (masked_srcs, tags_list, delim, context)。"""
    srcs, tags_list = [], []
    for job in jobs:
        masked, tags = C.protect(job[2])
        srcs.append(masked)
        tags_list.append(tags)
    delim = C.pick_delim(srcs) if len(srcs) > 1 else None
    ctx = []
    if args.context > 0:
        fi, pos, _ = jobs[0]
        wf = works[fi]
        allsrc = cache.setdefault(fi, [wf.source_of(c) for (_, _, c) in wf.rows])
        for i in range(max(0, pos - args.context), pos):
            t = C.stripped(allsrc[i])
            if t:
                ctx.append(t)
    return jobs, srcs, tags_list, delim, ctx


# ---------------- main ----------------

def main():
    args = build_parser().parse_args()
    files = expand(args.inputs)
    if args.dry_run and not args.out_dir:
        import tempfile
        args.out_dir = tempfile.mkdtemp(prefix='hymt2-dry-')   # dry-run 不落盘，只为过参数校验
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
    works = prepare(files, args.out_dir, args.col)
    jobs = collect_jobs(works, args.force, args.skip_regex, args.limit)
    total_pending = len(jobs)
    if not jobs:
        print('没有待译条目（都已译或全被跳过）。')
        return

    print(f'待译 {total_pending} 条 / {len(works)} 个文件')

    if args.dry_run:
        cache = {}
        for i in range(0, min(len(jobs), args.batch * 2), args.batch):
            g = build_group(works, jobs[i:i + args.batch], args, cache)
            jobs_b, srcs, tags_list, _delim, ctx = g
            for src, tags in zip(srcs, tags_list):
                terms = C.filter_terms(C.load_terms(args.terms), [src], args.max_terms)
                print('=' * 60)
                print(C.build_prompt(src, target_lang=args.target_lang, style=args.style,
                                     terms=terms, extra=args.extra, context=ctx,
                                     delim=None, placeholders=bool(tags)))
        return

    runner = Runner(args)
    terms_all = C.load_terms(args.terms)
    results, applied, failed_all = {}, [], []
    cache = {}

    def flush():
        for job, cn in results.items():
            apply_result(works, job, cn)
            applied.append(job)
        results.clear()

    t0 = time.time()
    done = 0

    batches = [jobs[i:i + args.batch] for i in range(0, len(jobs), args.batch)]
    for bi in range(0, len(batches), max(1, args.micro_batch)):
        chunk = batches[bi:bi + max(1, args.micro_batch)]
        groups = [build_group(works, b, args, cache) for b in chunk]
        res, failed = run_batch_group(runner, works, groups, terms_all, args)
        results.update(res)
        failed_all.extend(failed)
        done += sum(len(b) for b in chunk)
        print(f'\r[{done}/{total_pending}] {time.time()-t0:.0f}s', end='', flush=True)
        if (bi // max(1, args.micro_batch)) % args.save_every == args.save_every - 1:
            flush()
            save_all(works)
    print()

    # 逐条重译（分隔符/标签校验没过的）
    retry = list(dict.fromkeys(failed_all))
    if retry:
        print(f'逐条重译 {len(retry)} 条…', flush=True)
        still = []
        for i in range(0, len(retry), max(1, args.micro_batch)):
            groups = [build_group(works, [j], args, cache) for j in retry[i:i + max(1, args.micro_batch)]]
            res, failed = run_batch_group(runner, works, groups, terms_all, args)
            results.update(res)
            still.extend(failed)
        if still:
            with open(args.failed, 'w', encoding='utf-8', newline='\n') as f:
                for job in still:
                    f.write(f'{os.path.basename(works[job[0]].path)}\t{job[1]}\t{job[2]}\n')
            print(f'仍有 {len(still)} 条失败（译文留空）→ {args.failed}')

    flush()
    save_all(works)

    n_ok = len(applied)
    print(f'完成 {n_ok}/{total_pending} 条，用时 {time.time()-t0:.0f}s'
          f'（{n_ok/max(1e-9,(time.time()-t0))*60:.0f} 条/分）')
    for fi, wf in enumerate(works):
        n_new = sum(1 for j in applied if j[0] == fi)
        n_tr = sum(1 for (_, _, c) in wf.rows if c[wf.tcol].strip())
        print(f'  {wf.dest}: 本次 +{n_new}，累计 {n_tr}/{len(wf.rows)} 已译')


if __name__ == '__main__':
    main()
