"""
阶段 1：训练分词器。

这个脚本要产出整个项目最重要的基础资产 —— `tokenizer/tokenizer.json`。

**为什么这一步必须在预训练之前，而且只能做一次：**

    特殊 token（`<|im_start|>` 等）必须在预训练阶段就存在于词表中，
    模型才能学到它们的语义（比如"看到 im_end 就该停下"）。
    如果预训练完才加，这些 token 的 embedding 是随机初始化的，
    需要额外的训练才能让它们变得有意义，而且效果通常不好。

    所以本脚本一旦定下词表，后续 SFT / RL / 推理全部复用同一个文件，
    中途不能改。改了就要重新预训练（5 天）。

用法：
    python scripts/01_train_tokenizer.py                    # 用默认采样量
    python scripts/01_train_tokenizer.py --n-docs 40000     # 用更多样本
    python scripts/01_train_tokenizer.py --offline-samples data/raw/samples.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ⚠️ 先 import minilm：它会设置 PYTORCH_CUDA_ALLOC_CONF（必须在 torch 之前）
import minilm  # noqa: E402,F401

from minilm.tokenizer import (  # noqa: E402
    SpecialTokens,
    compression_report,
    load_tokenizer,
    setup_post_processor,
    train_tokenizer,
    verify_roundtrip,
    verify_special_tokens,
)

DEFAULT_OUT = PROJECT_ROOT / "tokenizer" / "tokenizer.json"


def main() -> int:
    parser = argparse.ArgumentParser(description="阶段 1：训练 BPE 分词器")
    parser.add_argument("--vocab-size", type=int, default=32768,
                        help="词表大小。32768 时 embedding 占 33.6M 参数（总量的 10.6%%）")
    parser.add_argument("--n-docs", type=int, default=20000,
                        help="每个数据集采样多少条文档用于训练分词器")
    parser.add_argument("--min-frequency", type=int, default=2,
                        help="字节对的最小合并频率")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="输出路径")
    parser.add_argument("--offline-samples", default=None,
                        help="用本地文本文件训练（每行一条），跳过下载。用于网络不通时")
    parser.add_argument("--proxy", default="http://127.0.0.1:7897",
                        help="下载用的代理地址")
    parser.add_argument("--skip-verify", action="store_true", help="跳过验证步骤")
    args = parser.parse_args()

    print("=" * 72)
    print("  阶段 1：训练分词器")
    print("=" * 72)

    # ---------------------------------------------------------------
    # 第 1 步：准备训练语料
    # ---------------------------------------------------------------
    if args.offline_samples:
        print(f"\n[1/4] 从本地文件读取样本: {args.offline_samples}")
        path = Path(args.offline_samples)
        if not path.exists():
            print(f"  [错误] 文件不存在: {path}")
            return 1
        with path.open("r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if len(line.strip()) > 50]
        print(f"  读到 {len(texts)} 条文本")
    else:
        print(f"\n[1/4] 从 HuggingFace 采样训练语料")
        # 延迟导入：download 模块会设置代理环境变量，需要在此处才生效
        from minilm.data.download import (
            PRETRAIN_DATASETS,
            NetworkConfig,
            check_connectivity,
            sample_texts,
        )

        net = NetworkConfig(proxy=args.proxy, cache_dir=str(PROJECT_ROOT / "data" / "hf_cache"))
        net.apply()  # 必须在 import datasets 之前
        print(f"  网络配置: {net.describe()}\n")

        if not check_connectivity(net):
            print("\n[中止] 网络不通。可以先用 --offline-samples 指定本地文本文件。")
            return 1

        print()
        # 从各数据源按配比采样，保证分词器对不同来源的文本都有好的压缩率。
        # 采样量比预训练实际用量小得多是刻意为之：
        # BPE 只需要统计字节对频率，几百 MB 就能收敛。
        texts = sample_texts(
            PRETRAIN_DATASETS,
            net,
            n_docs_per_spec=args.n_docs,
        )

    if not texts:
        print("\n[中止] 没有采到任何文本。")
        return 1

    total_chars = sum(len(t) for t in texts)
    print(f"\n  采样统计：{len(texts):,} 条文本，共 {total_chars:,} 字符 "
          f"(约 {total_chars/1e6:.1f}M 字符)")

    if total_chars < 10_000_000:
        print(f"  [警告] 样本量偏少（< 10M 字符），分词器质量可能不够好。")
        print(f"         建议加大 --n-docs。")

    # ---------------------------------------------------------------
    # 第 2 步：训练分词器
    # ---------------------------------------------------------------
    print(f"\n[2/4] 训练字节级 BPE（vocab_size={args.vocab_size}）")
    print(f"  特殊 token: {' '.join(SpecialTokens.ALL)}")

    tokenizer = train_tokenizer(
        texts=iter(texts),  # 传迭代器，train_from_iterator 会按需消费
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        save_path=args.out,
    )

    actual_vocab = tokenizer.get_vocab_size()
    print(f"  训练完成，实际词表大小: {actual_vocab:,}")
    if actual_vocab < args.vocab_size:
        print(f"  [说明] 比目标小 {args.vocab_size - actual_vocab:,}，"
              f"说明语料里的高频字节对已经合并完了，属正常现象。")

    # ---------------------------------------------------------------
    # 第 3 步：配置后处理器（仅供预训练数据编码使用）
    # ---------------------------------------------------------------
    print(f"\n[3/4] 配置后处理器")
    # 注意：这里配置的是"编码时自动在末尾追加 <|endoftext|>"，
    # 用于预训练语料的文档分隔。SFT 数据不能用带这个后处理器的编码路径，
    # 因为 SFT 的每条样本有自己精确的 token 结构。
    # 实际使用时会重新 load 一份干净的 tokenizer 给 SFT 用。
    setup_post_processor(tokenizer)
    print(f"  预训练编码时会在末尾自动追加 {SpecialTokens.ENDOFTEXT}")
    # 重新保存一次，把后处理器也存进去
    tokenizer.save(args.out)

    # ---------------------------------------------------------------
    # 第 4 步：验证
    # ---------------------------------------------------------------
    if args.skip_verify:
        print("\n[4/4] 已跳过验证")
        return 0

    print(f"\n[4/4] 验证")
    # 用**重新加载**的文件做验证，而不是内存里的对象 ——
    # 确保存盘再读回来的分词器行为和训练时一致（能抓到序列化丢失配置这类问题）
    reloaded = load_tokenizer(args.out)

    print("\n  (a) 特殊 token 检查：")
    if verify_special_tokens(reloaded):
        print(f"      全部 {len(SpecialTokens.ALL)} 个特殊 token 都能正确编码成单个 id ✓")
    else:
        print(f"      [失败] 有特殊 token 配置不正确，会导致对话格式失效！")
        return 1

    print("\n  (b) 无损往返检查：")
    # 挑一些有代表性的难样本：中文、中英混排、emoji、生僻字、代码
    test_texts = [
        "你好，世界！",
        "这是一个测试句子，包含中文标点：，。！？；：\"\"''",
        "Mixed 中英文 text with numbers 12345 and symbols @#$%",
        "emoji 测试 🎉🚀🌟 和特殊符号 ∑∫√∞",
        "生僻字：龘靐齉爩鱻麤龗",
        "def fibonacci(n):\n    return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
        "机器学习的核心是让模型从数据中学习规律。",
    ]
    if verify_roundtrip(reloaded, test_texts):
        print(f"      {len(test_texts)} 条测试文本全部无损往返 ✓")
        print(f"      说明任何输入（含 emoji、生僻字、代码）都不会被破坏")
    else:
        print(f"      [失败] 往返有损，分词器会破坏训练数据！")
        return 1

    print("\n  (c) 压缩率测算：")
    # 这个数字直接决定"2B tokens 需要多少原始文本"。
    # 必须实测 —— "1 汉字 ≈ 1.5 token" 这类经验法则误差可达 20%，
    # 意味着多下或少下 1GB 数据。
    report = compression_report(reloaded, texts[:2000])
    print(f"      样本数          : {report['n_samples']:,}")
    print(f"      总字符          : {report['total_chars']:,}")
    print(f"      其中汉字        : {report['total_cjk_chars']:,}")
    print(f"      总 token        : {report['total_tokens']:,}")
    print(f"      {'-'*46}")
    print(f"      tokens / 字符   : {report['tokens_per_char']:.3f}")
    print(f"      tokens / 汉字   : {report['tokens_per_cjk']:.3f}   <-- 中文压缩能力")
    print(f"      字符 / token    : {report['chars_per_token']:.3f}")

    # 据此反推 2B tokens 需要多少原始文本
    target_tokens = 2_000_000_000
    tpc = report["tokens_per_char"]
    if tpc > 0:
        needed_chars = target_tokens / tpc
        needed_gb = needed_chars * 3 / 1e9  # 汉字 UTF-8 约 3 字节
        print(f"\n      反推：要产出 2B tokens 的训练数据，")
        print(f"            约需 {needed_chars/1e9:.2f}B 字符，"
              f"对应原始文本约 {needed_gb:.1f} GB")

    print("\n" + "=" * 72)
    print(f"  阶段 1 完成 —— 分词器已保存到 {args.out}")
    print("=" * 72)
    print("  下一步：python scripts/02_prepare_data.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
