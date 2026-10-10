"""
文本清洗：把原始语料过滤成可训练的干净文本。

## 为什么清洗是必须的，不能跳过

    从网上抓的语料含有大量垃圾：导航栏、版权声明、重复模板、
    乱码、超短碎片、纯符号行。这些内容会：
      - 浪费 token 预算（2B tokens 里有 20% 是垃圾，就等于白跑 1 天）
      - 让模型学会输出垃圾（重复退化、输出网页模板）
      - 拉高 loss 的噪声，让训练曲线难以判断

## 清洗策略的选择

    语料清洗有很多重型手段（MinHash 跨文档去重、质量分类器打分、
    perplexity 过滤）。本项目**只做轻量清洗**，原因是：

    1. 我们用的语料（Fineweb-Edu 4_5 档、Wikipedia）**本身已经过上游清洗**，
       再做重型处理收益递减
    2. MinHash 跨文档去重需要几十 GB 内存和大量 CPU 时间，
       在"5 天训练"的时间预算下不划算
    3. 保持流程可理解、可调试，比追求极致的语料质量更重要

    真正必做的是**长度过滤**和**乱码过滤**，这两项性价比最高。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


# ===========================================================================
# 清洗配置
# ===========================================================================
@dataclass
class CleanConfig:
    """清洗规则的参数。

    各阈值都是经验值，来自常见中文语料的特点。
    """

    min_chars: int = 200
    """文档最少字符数。

    太短的文档（标题、残句）信息量低，且会让训练序列大量出现文档边界。
    200 字符约等于 2-3 句中文，是个合理的下限。
    """

    max_chars: int = 100_000
    """文档最多字符数。

    超长文档（几 MB 的拼接网页）往往是爬虫错误导致的重复内容，
    而且会占用大量内存。截断而不是丢弃。
    """

    min_cjk_ratio: float = 0.30
    """汉字占非空白字符的最低比例。

    用来过滤"名义上是中文语料、实际以英文/代码/符号为主"的文档。
    设 0.30 比较宽松 —— 技术类文章常中英混排，不能卡太严。
    """

    max_symbol_ratio: float = 0.30
    """符号（非字母数字汉字）占比上限。过滤纯符号行、分隔线、乱码。"""

    max_digit_ratio: float = 0.50
    """数字占比上限。过滤表格、纯数字列表这类低信息量内容。"""

    max_line_repeat_ratio: float = 0.30
    """重复行占比上限。过滤模板页（导航栏、页脚反复出现）。"""

    remove_urls: bool = True
    """是否移除 URL。URL 本身价值低，且会浪费大量 token。"""

    remove_emails: bool = True

    normalize_whitespace: bool = True
    """是否规范化空白字符（把连续空行/空格压成一个）。"""

    min_line_len: int = 2
    """单行最少字符数，低于此的行会被删掉（过滤孤立标点）。"""


# ===========================================================================
# 预编译正则（编译一次复用，比每次调用重新编译快很多）
# ===========================================================================
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_EMAIL = re.compile(r"\S+@\S+\.\S+")
# 连续 3 个以上相同字符，通常是分隔线（=====）或噪声
_RE_REPEAT_CHAR = re.compile(r"(.)\1{3,}")
_RE_MULTI_NEWLINE = re.compile(r"\n\s*\n\s*\n+")
_RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")
# 控制字符（保留 \n \t）
_RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def is_cjk(ch: str) -> bool:
    """判断是否为 CJK 汉字。

    覆盖基本汉字区（U+4E00-U+9FFF）和扩展 A 区（U+3400-U+4DBF）。
    扩展 A 区包含一些生僻字，实际语料里少见但出现了也不该算作"非汉字"。
    """
    return "一" <= ch <= "鿿" or "㐀" <= ch <= "䶿"


def clean_text(text: str, cfg: CleanConfig | None = None) -> str | None:
    """清洗单篇文档。

    返回清洗后的文本；如果文档被判定为低质量，返回 None 表示丢弃。

    处理顺序是刻意安排的：先做廉价的字符串替换，再做需要遍历全文的统计，
    最后才做需要逐行处理的重复度检测。这样大部分垃圾在早期就被廉价地滤掉了。
    """
    cfg = cfg or CleanConfig()

    if not text or not isinstance(text, str):
        return None

    # ---- 1. Unicode 规范化 ----
    # NFKC 会把全角字母数字转半角、合并兼容字符。
    # 这能让"１２００"和"1200"变成同一个表示，减少词表的无谓分裂。
    # 注意：NFKC 会改变部分字符（如 ① → 1），对中文正文影响很小，可以接受。
    text = unicodedata.normalize("NFKC", text)

    # ---- 2. 移除控制字符 ----
    text = _RE_CONTROL.sub("", text)

    # ---- 3. 移除 URL 和邮箱 ----
    if cfg.remove_urls:
        text = _RE_URL.sub("", text)
    if cfg.remove_emails:
        text = _RE_EMAIL.sub("", text)

    # ---- 4. 规范化空白 ----
    if cfg.normalize_whitespace:
        text = _RE_MULTI_NEWLINE.sub("\n\n", text)
        text = _RE_MULTI_SPACE.sub(" ", text)

    # ---- 5. 逐行过滤 ----
    lines = []
    for line in text.split("\n"):
        line = line.strip()
        # 丢掉过短的行（孤立标点、单字导航项）
        if len(line) < cfg.min_line_len:
            continue
        # 丢掉同一字符重复过多的行（"=========="、"--------"）
        stripped = _RE_REPEAT_CHAR.sub("", line)
        if len(stripped) < cfg.min_line_len:
            continue
        lines.append(line)

    if not lines:
        return None
    text = "\n".join(lines)

    # ---- 6. 长度过滤 ----
    n_chars = len(text)
    if n_chars < cfg.min_chars:
        return None
    if n_chars > cfg.max_chars:
        # 超长文档截断而非丢弃 —— 内容本身可能是好的，只是被拼接得过长
        text = text[: cfg.max_chars]

    # ---- 7. 字符构成统计 ----
    # 这是最耗时的一步（要遍历全文），所以放在长度过滤之后，
    # 让短文档先被廉价地淘汰掉。
    n_cjk = 0
    n_digit = 0
    n_symbol = 0
    n_visible = 0

    for ch in text:
        if ch.isspace():
            continue
        n_visible += 1
        if is_cjk(ch):
            n_cjk += 1
        elif ch.isdigit():
            n_digit += 1
        elif not ch.isalnum():
            n_symbol += 1

    if n_visible == 0:
        return None

    cjk_ratio = n_cjk / n_visible
    digit_ratio = n_digit / n_visible
    symbol_ratio = n_symbol / n_visible

    if cjk_ratio < cfg.min_cjk_ratio:
        return None
    if digit_ratio > cfg.max_digit_ratio:
        return None
    if symbol_ratio > cfg.max_symbol_ratio:
        return None

    # ---- 8. 重复行检测 ----
    # 模板页（导航栏/页脚/免责声明反复出现）的典型特征是大量重复行。
    # 这一步放在最后，因为它是 O(行数) 的字符串哈希操作，相对昂贵。
    all_lines = text.split("\n")
    if len(all_lines) >= 5:
        unique_lines = len(set(all_lines))
        repeat_ratio = 1 - unique_lines / len(all_lines)
        if repeat_ratio > cfg.max_line_repeat_ratio:
            return None

    return text


def clean_documents(
    docs,
    cfg: CleanConfig | None = None,
    stats: dict | None = None,
):
    """批量清洗文档（生成器版本）。

    参数
    ----
    docs : Iterable[str]
        原始文档。
    stats : dict, optional
        传入一个 dict，会就地更新各项统计（用于输出清洗报告）。

    产出
    ----
    str
        清洗后通过筛选的文档。

    **用生成器而不是返回 list**：语料可能有几千万篇，
    收集成 list 会爆内存。生成器让内存占用恒定。
    """
    cfg = cfg or CleanConfig()

    if stats is not None:
        stats.setdefault("total", 0)
        stats.setdefault("kept", 0)
        stats.setdefault("chars_in", 0)
        stats.setdefault("chars_out", 0)

    for doc in docs:
        if stats is not None:
            stats["total"] += 1
            stats["chars_in"] += len(doc) if isinstance(doc, str) else 0

        cleaned = clean_text(doc, cfg)

        if cleaned is not None:
            if stats is not None:
                stats["kept"] += 1
                stats["chars_out"] += len(cleaned)
            yield cleaned


def format_clean_stats(stats: dict) -> str:
    """把清洗统计格式化成可读报告。

    这个报告很重要：如果保留率异常低（比如 < 30%），说明
    过滤规则太严或者语料本身有问题，需要调整而不是硬跑。
    """
    total = stats.get("total", 0)
    kept = stats.get("kept", 0)
    if total == 0:
        return "  （无数据）"

    keep_rate = kept / total * 100
    chars_in = stats.get("chars_in", 0)
    chars_out = stats.get("chars_out", 0)
    char_keep = chars_out / chars_in * 100 if chars_in else 0

    lines = [
        f"    文档数    : {total:,} → {kept:,}  (保留 {keep_rate:.1f}%)",
        f"    字符数    : {chars_in:,} → {chars_out:,}  (保留 {char_keep:.1f}%)",
    ]

    # 给出健康度判断，避免"清洗完才发现保留率极低"这种事后发现
    if keep_rate < 30:
        lines.append(f"    ⚠ 保留率偏低（{keep_rate:.1f}%），建议检查过滤阈值是否过严")
    elif keep_rate > 95:
        lines.append(f"    ⚠ 保留率偏高（{keep_rate:.1f}%），过滤可能没起作用")

    return "\n".join(lines)


class Deduplicator:
    """基于内容哈希的精确去重。

    ## 为什么只做精确去重，不做 MinHash 近似去重

        精确去重（完全相同的内容）实现简单、内存开销可控、
        命中率高（网页转载会产生大量完全相同的副本）。

        MinHash/LSH 的近似去重能找到"几乎相同"的文档，
        但需要：
          - 几十 GB 内存存签名矩阵（本机只有 32GB）
          - 大量 CPU 时间做 Jaccard 估计
        在 5 天训练预算下不划算，而且我们选的语料已经过上游清洗。

    ## 内存控制

        只保存文档的 64 位哈希（8 字节），而不是文档本身。
        1000 万篇文档约占 80MB 内存，完全可以接受。
        理论上会有哈希碰撞，但 64 位空间下碰撞概率极低
        （1000 万篇时约 2.7e-6），可以忽略。
    """

    def __init__(self, max_items: int = 20_000_000) -> None:
        self._seen: set[int] = set()
        self.max_items = max_items
        self.duplicates = 0

    def is_duplicate(self, text: str) -> bool:
        """检查是否重复。重复返回 True，否则记录并返回 False。"""
        # 用 Python 内置 hash 不够稳定（有随机化），改用 hashlib 保证跨进程一致
        import hashlib

        # 只对前 2000 字符做哈希：长文档的差异通常在前部就能体现，
        # 而且这样能显著加速（不用遍历全文）
        h = int.from_bytes(
            hashlib.blake2b(text[:2000].encode("utf-8", errors="ignore"), digest_size=8).digest(),
            "big",
        )

        if h in self._seen:
            self.duplicates += 1
            return True

        # 内存保护：超过上限就不再记录（宁可漏掉一些重复，也不要 OOM）
        if len(self._seen) < self.max_items:
            self._seen.add(h)
        return False

    def __len__(self) -> int:
        return len(self._seen)
