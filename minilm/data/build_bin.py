"""
把清洗后的文本分词，写成扁平的 uint16 二进制文件（memmap）。

## 为什么要预分词成二进制

    训练时如果在线分词，每个 step 都要跑一遍 BPE。BPE 是 CPU 密集的 Rust/Python 逻辑，
    在 4618 tok/s 的训练吞吐下会变成严重瓶颈 —— GPU 会一直空等 CPU，
    实际吞吐可能掉到几百 tok/s。

    预分词后，训练时只需要对一个大数组做一次内存切片，开销可以忽略。
    这是 nanoGPT 等项目的标准做法。

## 为什么用 uint16

    词表大小 32768 < 65536，所以 uint16（2 字节）足够表示所有 token id。
    相比 int32/int64 能省一半到四分之三的磁盘和内存带宽：
        2B tokens × 2 字节 = 4GB   （uint16）
        2B tokens × 4 字节 = 8GB   （uint32）
        2B tokens × 8 字节 = 16GB  （int64）

    本项目的 ModelConfig.__post_init__ 会检查 vocab_size <= 65536 来保证这个前提。

## 文件格式

    输出是两个裸数组文件（没有头部，就是纯粹的 token id 序列）：
        data/pretrain/train.bin   训练集
        data/pretrain/val.bin     验证集
        data/pretrain/meta.json   元信息（dtype、token 数、来源统计等）

    没有文件头是刻意的：可以直接用 np.memmap 打开并按偏移量随机访问，
    不需要解析任何元数据。训练时随机裁剪一个 seq_len 的窗口就是一次 O(1) 的切片。
    元信息单独放 meta.json，避免污染数据文件。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from tqdm import tqdm

from .clean import CleanConfig, Deduplicator, clean_text, format_clean_stats


# ===========================================================================
# 分词器封装
# ===========================================================================
class TokenizeWriter:
    """边分词边写入二进制文件的写入器。

    ## 设计要点

    1. **内存恒定**：token 先在内存缓冲区累积，攒够一定量就刷到磁盘。
       不会把所有 token 都留在内存里（2B tokens 是 4GB，加上 Python 开销会更多）。

    2. **预分配 + 截断**：按估算的总量预分配文件空间可以减少文件系统的碎片，
       但语料量难以精确预估，所以更简单可靠的做法是顺序追加。

    3. **原子性**：写临时文件，完成后重命名。中途失败不会留下半成品，
       避免"以为数据准备好了，其实只有一半"这种问题。
    """

    def __init__(
        self,
        path: Path,
        dtype: np.dtype = np.uint16,
        buffer_tokens: int = 8_000_000,
    ) -> None:
        self.path = Path(path)
        self.dtype = dtype
        self.buffer_tokens = buffer_tokens

        self._buffer: list[np.ndarray] = []
        self._buffered = 0
        self.total_tokens = 0

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        self._fh = self._tmp.open("wb")

    def add_tokens(self, ids: list[int]) -> None:
        """加入一批 token id。"""
        if not ids:
            return
        arr = np.asarray(ids, dtype=self.dtype)
        self._buffer.append(arr)
        self._buffered += len(arr)
        self.total_tokens += len(arr)

        if self._buffered >= self.buffer_tokens:
            self.flush()

    def flush(self) -> None:
        """把缓冲区刷到磁盘。"""
        if not self._buffer:
            return
        # 先拼成一个大数组再一次性写，比逐个小数组写快得多
        merged = np.concatenate(self._buffer)
        self._fh.write(merged.tobytes())
        self._buffer.clear()
        self._buffered = 0

    def close(self) -> Path:
        """收尾：刷盘、关闭、原子重命名。"""
        self.flush()
        self._fh.close()
        # 原子替换：Windows 和 Linux 上 os.replace 都是原子的
        import os

        os.replace(self._tmp, self.path)
        return self.path


# ===========================================================================
# 主构建流程
# ===========================================================================
@dataclass
class BuildConfig:
    """数据构建配置。"""

    vocab_path: str = "tokenizer/tokenizer.json"
    out_dir: str = "data/pretrain"

    target_tokens: int = 2_000_000_000
    """训练集目标 token 数。

    这是 tokenize **之后**的数量，不是字符数。
    因为分词器压缩率事先不知道，所以是按 token 数控制，
    达到目标就停止读取语料。
    """

    val_tokens: int = 2_000_000
    """验证集 token 数。取总量的 0.1%。

    验证集从语料里**先切出来**，且不参与训练。
    只取 0.1% 是因为我们主要看 loss 是否过拟合，
    不需要很大的验证集来获得低方差的估计。
    """

    batch_docs: int = 1000
    """一次处理多少篇文档。

    tokenizers 的 encode_batch 在批量调用时能利用多核并行（内部是 Rust + rayon）。
    批量太小会浪费并行度，太大会占内存。
    """

    dedup: bool = True
    clean: bool = True

    clean_cfg: CleanConfig = field(default_factory=CleanConfig)

    log_every: int = 50
    """每处理多少批输出一次进度。"""


def build_bin_from_records(
    records: Iterator[dict[str, Any]],
    text_field: str,
    cfg: BuildConfig,
    tokenizer,
    source_name: str = "unknown",
    token_budget: int | None = None,
) -> dict[str, Any]:
    """从一个记录流构建二进制数据。

    参数
    ----
    records : Iterator[dict]
        原始记录流（来自 download.py 或 modelscope.py）。
    text_field : str
        正文所在的字段名。
    token_budget : int, optional
        本次最多产出多少 token。用于控制各个数据源的混合比例。
        None 表示用 cfg.target_tokens。

    返回
    ----
    dict
        本次构建的统计信息。
    """
    budget = token_budget if token_budget is not None else cfg.target_tokens

    dedup = Deduplicator() if cfg.dedup else None
    stats: dict[str, Any] = {"source": source_name, "raw_docs": 0, "kept_docs": 0, "dup_docs": 0}

    writer = TokenizeWriter(Path(cfg.out_dir) / f"{source_name}.bin")

    batch: list[str] = []
    t0 = time.time()
    n_batch = 0
    stopped = False

    def flush_batch():
        """把积攒的文档批量分词并写入。"""
        nonlocal batch
        if not batch:
            return
        # encode_batch 会并行处理，比逐条 encode 快数倍
        encodings = tokenizer.encode_batch(batch, add_special_tokens=False)
        for enc in encodings:
            writer.add_tokens(enc.ids)
        batch = []

    pbar = tqdm(total=budget, desc=f"  {source_name}", unit="tok", unit_scale=True)

    for record in records:
        # 预算用完就停 —— 这是控制混合比例和多源总量不超标的关键
        if writer.total_tokens >= budget:
            stopped = True
            break

        stats["raw_docs"] += 1

        # ---- 抽取正文 ----
        text = record.get(text_field)
        if not isinstance(text, str) or not text:
            continue
        # 有些数据集正文是列表
        if not text.strip():
            continue

        # ---- 清洗 ----
        if cfg.clean:
            text = clean_text(text, cfg.clean_cfg)
            if text is None:
                continue

        # ---- 去重 ----
        if dedup is not None and dedup.is_duplicate(text):
            stats["dup_docs"] += 1
            continue

        stats["kept_docs"] += 1
        batch.append(text)

        if len(batch) >= cfg.batch_docs:
            flush_batch()
            n_batch += 1
            pbar.update(writer.total_tokens - pbar.n)
            if n_batch % cfg.log_every == 0:
                elapsed = time.time() - t0
                rate = writer.total_tokens / elapsed if elapsed > 0 else 0
                pbar.set_postfix({
                    "tok/s": f"{rate:.0f}",
                    "保留": f"{stats['kept_docs']/max(1,stats['raw_docs'])*100:.0f}%",
                })

    # 收尾
    flush_batch()
    pbar.update(writer.total_tokens - pbar.n)
    pbar.close()
    writer.close()

    stats["tokens"] = writer.total_tokens
    stats["dup_docs"] = dedup.duplicates if dedup else 0
    stats["stopped_at_budget"] = stopped
    stats["elapsed_sec"] = time.time() - t0
    stats["output"] = str(writer.path)

    return stats
