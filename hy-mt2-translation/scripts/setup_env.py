# -*- coding: utf-8 -*-
# 建/修本地 Hy-MT2 的 WSL2 运行环境（uv + ROCm torch + triton）。幂等：重复跑只补缺。
# 不预下权重：transformers 首次 from_pretrained 自动拉到 HF 缓存。
#
# 用法（在 WSL 内执行）:
#   python setup_env.py                        # 默认：~/.local/share/hy-mt2（ext4）
#   python setup_env.py --root ~/hy-mt2-env    # 自建独立环境（按 env/pyproject.toml 配方）
#
# 配方是仓库里的真实文件 env/pyproject.toml（torch/triton/rocm 精确钉版本）。
# 首次搭建的完整前置（librocdxg fork、五个 HSA 环境变量、python3.12-dev）见
# references/rocm-setup.md。
import os, sys, shutil, subprocess, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.dirname(HERE)
# 环境放 ext4：/mnt/c 的 9p 顺序读仅 ~170-230 MB/s，import torch/transformers
# 与包内 .py 逐文件读取都被拖慢（实测 import transformers 47s → 迁后秒级）。
DEFAULT_ROOT = os.path.expanduser('~/.local/share/hy-mt2')
TPL = os.path.join(SKILL, 'env', 'pyproject.toml')


def run(cmd, **kw):
    print('$ ' + ' '.join(cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default=DEFAULT_ROOT,
                    help='环境目录（含 pyproject.toml / .venv）。'
                         '默认 = ~/.local/share/hy-mt2（ext4）')
    args = ap.parse_args()

    if not shutil.which('uv'):
        raise SystemExit('没找到 uv。先装：\n'
                         '  curl -LsSf https://astral.sh/uv/install.sh | sh')

    # uv 缓存/新建 venv 都留在 ext4（PATH 里若无 ~/.local/bin 请用登录 shell）
    os.environ.setdefault('UV_CACHE_DIR', os.path.expanduser('~/.cache/uv'))

    os.makedirs(args.root, exist_ok=True)
    pp = os.path.join(args.root, 'pyproject.toml')
    if not os.path.exists(pp):
        shutil.copyfile(TPL, pp)
        print(f'[env] 写入 {pp}（来自 env/pyproject.toml）')
    else:
        print(f'[env] {pp} 已存在，沿用（不覆盖）')

    venv = os.path.join(args.root, '.venv')
    py = os.path.join(venv, 'bin', 'python')
    if not os.path.exists(py):
        run(['uv', 'venv', venv, '--python', '3.12'])
    run(['uv', 'sync', '--project', args.root])

    print(f'''
[env] 完成。
  环境: {args.root}
  python: {py}
  —— 不预下载权重。权重在 HF 缓存（默认 $HOME/.cache/huggingface，ext4；
     本地已有全量缓存 + HF_HUB_OFFLINE=1，不会联网）。

前置提醒（缺了跑不动，详见 references/rocm-setup.md）：
  - librocdxg（RDNA2 用社区 fork）+ ~/.bashrc 里五个 HSA_* / LIBROCDXG_* 变量
  - sudo apt install python3.12-dev   （triton 现场编译要 Python.h）

下一步（均在 WSL 内）:
  "{py}" "{os.path.join(HERE, 'env_check.py')}" --model tencent/Hy-MT2-7B
  "{py}" "{os.path.join(HERE, 'translate_tsv.py')}" tpl/01.tsv --dry-run --limit 4
  bash "{os.path.join(HERE, '_run_wsl_fused.sh')}"     # int4 快线
''')


if __name__ == '__main__':
    main()
