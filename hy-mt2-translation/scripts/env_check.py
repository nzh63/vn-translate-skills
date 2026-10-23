# -*- coding: utf-8 -*-
# 环境体检：torch/ROCm 是否可用、显存、dtype 建议，并跑一条真实翻译冒烟。
# 在 venv 的 python 下运行:  python env_check.py [--model <repo或本地目录>]
import os, sys, argparse

# 本机 HF 直连不通：未显式设置时默认走镜像；权重由 transformers 首次使用时自动下载。
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=os.environ.get('HYMT2_MODEL', 'tencent/Hy-MT2-7B'))
    args = ap.parse_args()

    import torch
    print(f'python      : {sys.version.split()[0]}  ({sys.executable})')
    print(f'torch       : {torch.__version__}')
    print(f'cuda avail  : {torch.cuda.is_available()}')
    if not torch.cuda.is_available():
        print('!! ROCm 没起来。按 references/rocm-setup.md 排障（HSA_OVERRIDE_GFX_VERSION / '
              'rocm-sdk 版本 / PYTORCH_HIP_ALLOC_CONF）')
        return 1
    p = torch.cuda.get_device_properties(0)
    arch = getattr(p, 'gcnArchName', '?')
    print(f'device      : {torch.cuda.get_device_name(0)} | {arch} | '
          f'{p.total_memory/2**30:.1f}GiB')
    x = torch.zeros(1, device='cuda')
    print(f'tensor ok   : {x.device}')

    print('--- 加载模型冒烟 ---')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    dtype = torch.float16 if arch.startswith('gfx103') else torch.bfloat16
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype,
                                                     attn_implementation='sdpa',
                                                     trust_remote_code=True)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype,
                                                     attn_implementation='sdpa',
                                                     trust_remote_code=True)
    model = model.to('cuda').eval()

    def tr(text):
        prompt = (f'请将以下文本准确翻译为简体中文，注意只需要输出翻译后的结果，'
                  f'不要额外解释。\n\n{text}')
        enc = tok.apply_chat_template([{'role': 'user', 'content': prompt}],
                                      add_generation_prompt=True, return_tensors='pt').to('cuda')
        with torch.inference_mode():
            out = model.generate(**enc, max_new_tokens=128, temperature=0.7, top_p=0.6,
                                 top_k=20, repetition_penalty=1.05)
        return tok.decode(out[0][enc['input_ids'].shape[-1]:], skip_special_tokens=True).strip()

    for s in ['「今日はいい天気ですね。」と彼女は言った。',
              '[r]やめて！[p]そんなこと……']:
        print(f'  {s!r}\n   → {tr(s)!r}')
    used = torch.cuda.max_memory_allocated() / 2**30
    print(f'峰值显存    : {used:.2f}GiB / {p.total_memory/2**30:.1f}GiB')
    print('OK')
    return 0


if __name__ == '__main__':
    sys.exit(main())
