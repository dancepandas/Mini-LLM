"""
分词器：训练字节级 BPE，管理特殊 token。

## 为什么选字节级 BPE（ByteLevel BPE）

    备选方案是字符级或词级分词。字节级 BPE 的优势：
      - **永远不会出现 UNK**。任何字符（包括 emoji、生僻字、乱码）都能被拆成
        UTF-8 字节序列表示。词级分词遇到未登录词只能吐 UNK，
        而中文的未登录词问题很严重。
      - 词表大小可控。纯字节只有 256 个基础符号，靠 merge 学习常见的字节组合。
      - 中文没有空格分词，词级方案需要额外做分词，而分词本身就会引入错误。

    代价：一个汉字在 UTF-8 下占 3 字节，所以压缩率不如大词表方案。
    业界（Qwen、DeepSeek）用 15 万词表把中文压到 ~0.6 token/字，
    我们用 32768 词表，预计在 1.3~1.6 token/字左右。
    这个差距是刻意的：0.3B 模型用 15 万词表的话，embedding 就占 1.5 亿参数，
    接近总参数的一半，对小模型不划算。

## 特殊 token 的格式选择：ChatML

    采用 Qwen / OpenAI 的 ChatML 约定：

        <|im_start|>system
        你是一个助手<|im_end|>
        <|im_start|>user
        你好<|im_end|>
        <|im_start|>assistant
        你好！有什么可以帮你的？<|im_end|>

    为什么不用 `<|system|>` `<|user|>` `<|assistant|>` 这类独立角色 token：
      - ChatML 只需要 2 个控制 token（im_start/im_end），更省词表
      - 角色名作为普通文本出现，模型更容易学会"角色是可变的内容"这一抽象
      - 是经过大规模验证的成熟格式，后续要接 HF 生态也方便

## 最关键的一点：特殊 token 必须在预训练前就定好

    特殊 token 一旦确定就不能改，因为改了就要重新预训练。原因：
      - 模型需要在**预训练阶段**就见到这些 token，才能学到它们的语义
        （比如"im_end 之后应该停下"）。
      - 如果预训练完才加，模型对这些新 token 的 embedding 是随机初始化的，
        必须要额外的训练才能让它们有意义，而且很难学好。

    所以本模块的设计是：**先定义特殊 token → 训练分词器 → 预训练 →
    SFT（复用同一套 token）→ 推理（还是同一套）**，全流程一致。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from tokenizers.processors import TemplateProcessing


# ===========================================================================
# 特殊 token 定义
# ===========================================================================
class SpecialTokens:
    """集中管理所有特殊 token 的字符串和 id。

    为什么要集中定义而不是散落在各处：
        训练分词器、SFT 构造样本、推理时判断停止，三处都要用这些 id。
        任何一处用了硬编码的数字（比如写死 128000）都是隐患 ——
        一旦改词表就会静默出错（不报错，但语义全乱）。
        集中定义 + 从分词器实际查询，能杜绝这个问题。
    """

    # 控制 token
    PAD = "<|pad|>"              # 填充。批次内长度不齐时补齐用。
    ENDOFTEXT = "<|endoftext|>"  # 文档分隔符。预训练时把多篇文档拼成一条长序列时用。
    IM_START = "<|im_start|>"    # 一条消息（message）开始
    IM_END = "<|im_end|>"        # 一条消息结束

    # 顺序很重要：前面的先分配 id。把最常用的放前面便于调试时辨认。
    ALL: list[str] = [ENDOFTEXT, IM_START, IM_END, PAD]

    # 角色名。它们**不是**特殊 token，而是普通文本，
    # 放在 im_start 之后表示"这条消息来自谁"。
    ROLE_SYSTEM = "system"
    ROLE_USER = "user"
    ROLE_ASSISTANT = "assistant"

    @staticmethod
    def extra_ids() -> list[str]:
        """返回**额外保留**但不在词表里的 id 占位（本项目暂不需要）。

        保留这个接口是为了以后要加诸如 <|fim_prefix|>（代码补全用）之类的 token 时
        有个明确的扩展点。
        """
        return []


# ===========================================================================
# 训练分词器
# ===========================================================================
def train_tokenizer(
    texts: Iterable[str],
    vocab_size: int = 32768,
    min_frequency: int = 2,
    save_path: str | Path | None = None,
    show_progress: bool = True,
) -> Tokenizer:
    """在给定文本上训练字节级 BPE 分词器。

    参数
    ----
    texts : Iterable[str]
        **必须是迭代器/生成器，不能是 list。**
        训练语料动辄几个 GB，先全部读进内存会直接爆掉 32GB 内存。
        传生成器可以边读边训。
    vocab_size : int
        目标词表大小（含特殊 token）。
        本项目用 32768：既能让中文有合理压缩率，又不至于让 embedding
        占比过高（32768 × 1024 = 33.6M 参数，占总量的 10.6%）。
    min_frequency : int
        一个字节对至少出现多少次才会被合并。设 2 可以过滤掉偶然的噪声组合。
    save_path : str | Path, optional
        训练完保存到哪里。

    返回
    ----
    Tokenizer
        训练好的分词器。
    """
    # BPE 模型。unk_token=None 表示"不可能出现未知词" ——
    # 字节级 BPE 的每个字节都在词表里，任何输入都能编码，所以不需要 unk。
    tokenizer = Tokenizer(models.BPE(unk_token=None))

    # 预分词：把文本先切成字节级符号。
    # add_prefix_space=False 对中文很重要 —— 中文没有词间空格，
    # 加前缀空格会凭空插入空格，污染语料。
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)

    # 解码器：把 token 还原成文本时，需要把字节级符号拼回 UTF-8 字节串
    tokenizer.decoder = decoders.ByteLevel()

    # 训练器
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SpecialTokens.ALL,
        # initial_alphabet 把 256 个字节符号预置进词表。
        # 不做这一步的话，训练语料里没出现过的字节（比如某些 emoji）就没有对应 token，
        # 编码时会产生 UNK —— 而我们刚刚才说字节级 BPE 不该有 UNK。
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=show_progress,
    )

    tokenizer.train_from_iterator(texts, trainer=trainer)

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(save_path))
        print(f"[分词器] 已保存到 {save_path}")

    return tokenizer


def setup_post_processor(tokenizer: Tokenizer) -> Tokenizer:
    """配置后处理：在序列末尾自动追加 <|endoftext|>。

    仅用于**预训练数据**的编码。SFT 数据不能用它 ——
    SFT 的每条样本有自己精确的 token 结构，自动追加会破坏格式。

    这里先定义成独立函数而不是塞进 train_tokenizer，
    是因为"要不要自动追加"取决于用途，属于调用方的选择。
    """
    eot = tokenizer.token_to_id(SpecialTokens.ENDOFTEXT)
    if eot is None:
        raise ValueError(f"分词器里找不到 {SpecialTokens.ENDOFTEXT}，请先用 SpecialTokens.ALL 训练")

    tokenizer.post_processor = TemplateProcessing(
        single=f"$A {SpecialTokens.ENDOFTEXT}",
        special_tokens=[(SpecialTokens.ENDOFTEXT, eot)],
    )
    return tokenizer


# ===========================================================================
# 加载与查询
# ===========================================================================
def load_tokenizer(path: str | Path) -> Tokenizer:
    """从文件加载分词器。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"分词器文件不存在: {path.resolve()}\n"
            f"请先运行 python scripts/01_train_tokenizer.py 训练分词器。"
        )
    return Tokenizer.from_file(str(path))


def get_special_id(tokenizer: Tokenizer, token: str) -> int:
    """查询特殊 token 的 id，查不到就明确报错。

    绝不在调用处硬编码 id 数字 —— 那样改词表时会静默出错，
    表现为模型行为诡异但不报错，非常难排查。
    """
    tid = tokenizer.token_to_id(token)
    if tid is None:
        raise ValueError(
            f"分词器里没有特殊 token {token!r}。"
            f"当前词表包含的特殊 token: "
            f"{[t for t in SpecialTokens.ALL if tokenizer.token_to_id(t) is not None]}"
        )
    return tid


def special_token_ids(tokenizer: Tokenizer) -> dict[str, int]:
    """一次性取回所有特殊 token 的 id，方便训练循环使用。"""
    return {t: get_special_id(tokenizer, t) for t in SpecialTokens.ALL}


# ===========================================================================
# 分析与验证
# ===========================================================================
def compression_report(tokenizer: Tokenizer, samples: list[str]) -> dict:
    """测算分词器在真实中文样本上的压缩率。

    这个报告是决定"2B tokens 需要下载多少原始文本"的依据。

    为什么必须实测而不是估算：
        "1 个汉字约等于 1.5 个 token" 这种经验法则很不准 ——
        实际比值受标点、数字、英文、代码、空格的占比影响极大。
        差 20% 就意味着多下载或少下载 1GB 原始文本。

    参数
    ----
    samples : list[str]
        有代表性的文本样本。应该覆盖不同来源（百科、新闻、对话），
        否则测出来的比值会有系统性偏差。

    返回
    ----
    dict，包含各项统计。
    """
    total_chars = 0
    total_bytes = 0
    total_tokens = 0
    total_cjk = 0
    per_sample: list[dict] = []

    for text in samples:
        if not text:
            continue
        n_tok = len(tokenizer.encode(text).ids)
        n_chars = len(text)
        n_bytes = len(text.encode("utf-8"))
        # 统计 CJK 汉字数量（Unicode 范围 U+4E00-U+9FFF 是基本汉字区）
        n_cjk = sum(1 for ch in text if "一" <= ch <= "鿿")

        total_tokens += n_tok
        total_chars += n_chars
        total_bytes += n_bytes
        total_cjk += n_cjk
        per_sample.append({"chars": n_chars, "tokens": n_tok, "cjk": n_cjk})

    return {
        "n_samples": len(per_sample),
        "total_chars": total_chars,
        "total_cjk_chars": total_cjk,
        "total_bytes": total_bytes,
        "total_tokens": total_tokens,
        # tokens / 字符：整体压缩率
        "tokens_per_char": total_tokens / total_chars if total_chars else 0.0,
        # tokens / 汉字：判断中文压缩能力的核心指标
        "tokens_per_cjk": total_tokens / total_cjk if total_cjk else 0.0,
        # 字符 / token：越大说明压缩越好
        "chars_per_token": total_chars / total_tokens if total_tokens else 0.0,
    }


def verify_roundtrip(tokenizer: Tokenizer, texts: list[str]) -> bool:
    """验证解码(编码(x)) == x，确保分词器是无损的。

    字节级 BPE 理论上应该完美往返。这个检查能抓出：
        - 特殊 token 配置错误（被当成普通文本处理了）
        - 预分词器把某些字符吃掉了

    这是数据管线的最后一道保险 —— 有损的分词器会让模型学到错误的对应关系，
    而且这种错误在 loss 曲线上完全看不出来。

    ## 两个必须注意的点（踩过坑）

    1. **必须显式传 `add_special_tokens=False`。**
       如果分词器装了后处理器（预训练用来在末尾追加 <|endoftext|>），
       默认的 encode 会自动加上它，导致解码结果比原文多一个 <|endoftext|>，
       看起来像"往返有损"，实际是测试方法错了。

    2. **解码也要 `skip_special_tokens=False`。**
       否则会把原文里本来就有的特殊 token 文本吃掉。
       两边都关掉特殊 token 处理，才能纯粹地检验字节级的无损性。
    """
    all_ok = True
    for text in texts:
        # 关键：关闭后处理器，只测字节级的编解码
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        decoded = tokenizer.decode(ids, skip_special_tokens=False)
        if decoded != text:
            all_ok = False
            print(f"  [往返失败] 原文: {text[:60]!r}")
            print(f"             解码: {decoded[:60]!r}")
    return all_ok


def verify_special_tokens(tokenizer: Tokenizer) -> bool:
    """验证特殊 token 能被正确识别为**单个** token，而不是被拆成普通字符。

    这是最容易出错的地方：如果特殊 token 没有正确注册，
    "<|im_end|>" 会被拆成 '<' '|' 'i' 'm' ... 等一堆字符 token。
    模型依然能训练，loss 也会正常下降，但**对话格式彻底失效** ——
    模型永远学不会"看到 im_end 就停下"。这种错误极难从 loss 上发现，
    所以必须显式验证。
    """
    all_ok = True
    for token in SpecialTokens.ALL:
        tid = tokenizer.token_to_id(token)
        if tid is None:
            print(f"  [缺失] 特殊 token 不在词表里: {token}")
            all_ok = False
            continue

        # 关键检查：编码该 token 应该只产生 1 个 id
        encoded = tokenizer.encode(token, add_special_tokens=False).ids
        if len(encoded) != 1 or encoded[0] != tid:
            print(f"  [错误] {token} 被拆成了 {len(encoded)} 个 token: {encoded[:8]}")
            print(f"         正确行为应该是编码成单个 id {tid}")
            all_ok = False

    return all_ok
