# -*- coding: utf-8 -*-
# hy-mt2 批量翻译共享库。仅用标准库 —— 没 GPU / 没装 torch 也能跑自检。
#
# 职责：
#   1) 译文 TSV 读写：只改译文列，其余列、行序、注释行原样保留
#      （三个引擎技能的 check_structure 都是"前缀列必须与模板逐字节一致"的硬门禁）
#   2) 行内标签 [r]/[p]… 与字面 \n 换行记号的占位保护与还原
#   3) 多条合并成一批的分隔符策略 + 输出切分 + 失败判定
#   4) Hy-MT2 官方指令模板拼装（默认 / 术语干预 / 风格 / 分隔符 / 背景信息）
import os, re

# 字面 \n 记号：反斜杠 + n 两个字符，与引擎内部存储一致。
# 任何环节都不允许出现真实换行符（0x0A）。
NL_LITERAL = '\\n'

# 需要保护的结构记号：行内标签（以字母/下划线开头、不跨行的 [xxx]）+ 字面 \n
PROTECT_RE = re.compile(r'\[[A-Za-z_][^\[\]]*\]|' + re.escape(NL_LITERAL))

# 占位符样式：⟨0⟩ ⟨1⟩ …（训练分布内的"不可翻译占位"写法，见官方 Structured Data 指令）
PH_PAT = re.compile(r'[⟨〈<]\s*(\d+)\s*[⟩〉>]')
PH_FMT = '⟨{}⟩'

# 批分隔符候选：出现冲突时顺延。‖ 是官方 Delimiters 指令里的推荐分隔符。
DELIM_CANDIDATES = ['‖', '❙', '⦀', '␟', '⁞']


# ---------------- 1. 结构保护 / 还原 ----------------

def protect(text):
    """把行内标签与字面 \\n 换成占位符 → (遮蔽文本, 记号列表)。

    例：'あ[r]い\\nう' → ('あ⟨0⟩い⟨1⟩う', ['[r]', '\\\\n'])
    """
    tags = []

    def _rep(m):
        tags.append(m.group(0))
        return PH_FMT.format(len(tags) - 1)

    return PROTECT_RE.sub(_rep, text), tags


def restore(text, tags):
    """把占位符换回记号。越界编号原样留下（交给校验去判失败），绝不猜。"""
    def _rep(m):
        i = int(m.group(1))
        return tags[i] if 0 <= i < len(tags) else m.group(0)
    return PH_PAT.sub(_rep, text)


def tags_of(text):
    """提取结构记号序列（与 protect 同一正则）——校验只用序列相等，宽容换行位置。"""
    return PROTECT_RE.findall(text)


def stripped(text):
    """去掉结构记号后的"可见文本"；为空说明这行没有可译内容。"""
    return PROTECT_RE.sub('', text).strip()


def sanitize(text):
    """译文落地前清洗：真实换行/制表符 → 空格（TSV 单元格内不允许），去首尾空白。"""
    return re.sub(r'[\r\n\t]+', ' ', text).strip()


# ---------------- 2. 批量分隔符 ----------------

def pick_delim(sources):
    """选一个在所有待译文本里都没出现过的分隔符。"""
    blob = '\n'.join(sources)
    for d in DELIM_CANDIDATES:
        if d not in blob:
            return d
    raise SystemExit('所有候选分隔符都在原文中出现，请减小 --batch 或改 DELIM_CANDIDATES')


def split_output(out, delim, n):
    """按分隔符切回 n 条。数量不符返回 None（该批判失败，走逐条重试）。"""
    if delim is None:
        parts = [out]
    else:
        parts = out.split(delim)
    if len(parts) != n:
        return None
    return [p.strip() for p in parts]


# ---------------- 3. prompt 构造 ----------------

def build_prompt(src, target_lang='简体中文', style='', terms=(), extra=(),
                 context=(), delim=None, placeholders=False):
    """拼 Hy-MT2 官方指令。模型无默认 system prompt，全部塞进一条 user 消息。

    delim        批量分隔符；None = 单条翻译，不加分隔符指令
    placeholders 原文含 ⟨数字⟩ 占位符（有保护记号时为 True）
    """
    parts = []
    if context:
        parts.append('【背景信息】')
        parts.extend(context)
        parts.append('')
        parts.append('【待翻译文本】')
    if terms:
        parts.append('参考下面的翻译：')
        for a, b in terms:
            parts.append(f'{a}翻译成{b}')
        parts.append('')

    head = f'请将以下文本准确翻译为{target_lang}，注意只需要输出翻译后的结果，不要额外解释。'
    if style:
        head += f'注意翻译的风格要严格符合【**{style}**】。'
    if delim:
        head += (f'你必须在译文中保留等量的分隔符「{delim}」，绝对不可遗漏、转义或翻译该符号，'
                 f'并注意分隔符的位置。')
    if placeholders:
        head += ('原文中的 ⟨数字⟩ 是占位标记，必须在译文中原样保留，数量、内容与顺序完全一致，'
                 '不得翻译、删除或改写。')
    for e in extra:
        head += e
    parts.append(head)
    parts.append('')
    parts.append(src)
    return '\n'.join(parts)


# ---------------- 4. TSV 读写 ----------------

class WorkFile:
    """一个译文 TSV。lines 是原始行（不含换行符），rows 是数据行下标列表。

    cols 规则（与三个引擎技能一致）：
      3 列 = 模板 tpl（索引\\t说话人\\t原文）          → 追加第 4 列译文
      4 列 = work 骨架（…\\t译文）                    → 改第 4 列
      ≥6 列 = 对照稿（…\\t初译\\t校对\\t备注）         → 改"初译"列(下标3)
    """

    def __init__(self, path):
        self.path = path
        with open(path, 'rb') as f:
            raw = f.read()
        if raw.startswith(b'\xef\xbb\xbf'):
            raise SystemExit(f'{path}: 含 UTF-8 BOM，先转成无 BOM')
        if b'\r' in raw:
            raise SystemExit(f'{path}: 含 CR，先用引擎技能的 rebuild/unix 化修好')
        self.lines = raw.decode('utf-8').split('\n')
        if self.lines and self.lines[-1] == '':
            self.lines.pop()          # 末尾空行由 writer 补回
        self.rows = []                # [(行下标, 索引, cols[list])]
        self.ncol = None
        self.tcol = None              # 译文列下标（None = 追加）
        self.dest = path              # 落盘目标（3 列模板时指向 --out-dir）
        for i, line in enumerate(self.lines):
            if not line.strip() or line.lstrip().startswith('#'):
                continue
            cols = line.split('\t')
            try:
                idx = int(cols[0])
            except ValueError:
                raise SystemExit(f'{path}:{i+1}: 数据行首列不是整数 —— 文件可能被真实换行拆坏')
            if self.ncol is None:
                self.ncol = len(cols)
            elif len(cols) != self.ncol:
                raise SystemExit(f'{path}:{i+1}: 列数 {len(cols)} != 本文件其他行 {self.ncol}')
            self.rows.append((i, idx, cols))

    @property
    def append_mode(self):
        return self.ncol == 3

    def default_col(self):
        """3 列返回 None（追加）；否则返回初译列下标。"""
        return None if self.ncol == 3 else 3

    def source_of(self, cols):
        """日文原文列：4 列及以上固定第 3 列（下标 2）。"""
        return cols[2] if len(cols) >= 3 else cols[-1]

    def save(self, out_path=None):
        """原子写回：临时文件 + os.replace。"""
        target = out_path or self.path
        tmp = target + '.tmp'
        with open(tmp, 'w', encoding='utf-8', newline='\n') as f:
            f.write('\n'.join(self.lines))
            f.write('\n')
        os.replace(tmp, target)


def load_terms(path):
    """术语表：原文片段 \\t 规范译文 [\\t 禁用形式…]，与 check_terms.py 同格式。
    只取前两列（规范译文取 / 分隔的第一个候选）。"""
    out = []
    if not path:
        return out
    for ln, line in enumerate(open(path, encoding='utf-8'), 1):
        line = line.rstrip('\n')
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        cols = line.split('\t')
        if len(cols) < 2 or not cols[0].strip() or not cols[1].strip():
            print(f'警告: {path}:{ln} 术语格式异常，已跳过', file=__import__('sys').stderr)
            continue
        dst = cols[1].split('/')[0].strip()
        if dst:
            out.append((cols[0].strip(), dst))
    return out


def filter_terms(terms, sources, limit=40):
    """只留命中当前批次原文的术语（省 token，也避免无关术语干扰译文）。"""
    blob = '\n'.join(sources)
    hit = [t for t in terms if t[0] in blob]
    return hit[:limit]
