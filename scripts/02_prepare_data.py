"""
阶段 2：数据准备 —— 下载 → 清洗 → 分词 → 二进制 memmap。

分两个独立阶段，可以分开跑、也可以一起跑：

    --stage download   只下载原始语料到 data/raw/<name>/
    --stage tokenize   只做清洗+分词，产出 data/pretrain/{train,val}.bin
    --stage all        两步都做

## 为什么要分阶段

    下载是网络瓶颈（1.5 MB/s 单连接），耗时以小时计；
    分词是 CPU 瓶颈，也慢。两者分开后：
      - 下载可以放后台跑，中断了重跑会自动跳过已下载的文件
      - 分词可以在下载完成后独立重跑，不影响已下载的数据
      - 出问题时能快速定位是哪一步的问题

## 断点续传

    下载阶段靠"文件已存在且非空则跳过"实现。中断后重跑不会重复下载。
    分词阶段从头开始（因为要保证 token 序列的确定性），
    但这一步比下载快得多。

用法：
    python scripts/02_prepare_data.py --stage download
    python scripts/02_prepare_data.py --stage tokenize
    python scripts/02_prepare_data.py --stage all
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import minilm  # noqa: E402,F401  (必须先于 torch 导入，以设置显存分配器)

from minilm.utils import human_int, write_json  # noqa: E402


def load_data_config(path: Path) -> dict:
    """读数据配置 YAML。"""
    import yaml

    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ===========================================================================
# 阶段 A：下载
# ===========================================================================
def stage_download(cfg: dict, only: list[str] | None = None) -> dict:
    """从魔塔并发下载所有数据源。"""
    from minilm.data.modelscope import (
        check_speed,
        download_files_parallel,
        list_files,
        match_files,
    )

    raw_dir = PROJECT_ROOT / cfg["raw_dir"]
    workers = cfg.get("download_workers", 8)
    max_bytes = cfg.get("max_download_bytes_per_source", 8_000_000_000)

    print("\n" + "=" * 72)
    print("  阶段 A：下载原始语料")
    print("=" * 72)
    print(f"  并发线程: {workers}")
    print(f"  保存到  : {raw_dir}")

    summary: dict[str, dict] = {}
    t_all = time.time()

    for src in cfg["sources"]:
        name = src["name"]
        if only and name not in only:
            continue

        # 新配置用 dir + include_files 描述"本地已有文件"，
        # 这类源不需要从这里下载（数据由单独的下载脚本拉取）。
        # 没有 repo_id 就跳过，而不是崩掉。
        if not src.get("repo_id"):
            print(f"\n[跳过] {name}: 没有 repo_id，数据应已预先放好")
            continue

        print(f"\n{'─'*72}")
        print(f"[{name}] {src['repo_id']}")
        print(f"  模式: {src['file_pattern']}   目标: {human_int(src['target_tokens'])} tokens")
        print(f"{'─'*72}")

        dest = raw_dir / name
        dest.mkdir(parents=True, exist_ok=True)

        # 列文件树
        try:
            files = list_files(src["repo_id"], verbose=True)
        except Exception as e:  # noqa: BLE001
            print(f"  [失败] 列文件树出错: {type(e).__name__}: {str(e)[:150]}")
            summary[name] = {"status": "failed", "error": str(e)[:200]}
            continue

        matched = match_files(files, src["file_pattern"])
        if not matched:
            print(f"  [失败] 没有文件匹配 {src['file_pattern']!r}")
            summary[name] = {"status": "failed", "error": "no files matched"}
            continue

        total_bytes = sum(f.get("Size", 0) or 0 for f in matched)
        print(f"  匹配到 {len(matched)} 个文件, {total_bytes/1e9:.2f} GB")

        # 按大小排序：优先下小文件，这样能尽快拿到可用数据开始后续步骤。
        # 大文件放后面，即使下不完也不影响前期工作。
        matched.sort(key=lambda f: f.get("Size", 0) or 0)

        # 估算需要下载多少字节。
        # 经验值：中文语料约 1.25 token/字符，UTF-8 汉字 3 字节，
        # 但 parquet/jsonl 里还有 JSON 结构和元数据的开销。
        # 这里按 "1 token ≈ 4 字节原始文件" 粗略估算，宁可多下一点。
        need_bytes = min(int(src["target_tokens"] * 4), max_bytes, total_bytes)
        # 多下 30% 作为清洗损耗的缓冲
        need_bytes = int(need_bytes * 1.3)
        print(f"  计划下载约 {need_bytes/1e9:.2f} GB（含清洗损耗缓冲）")

        # 逐个累加文件，直到凑够需要的字节数
        selected = []
        acc = 0
        for f in matched:
            selected.append(f["Path"])
            acc += f.get("Size", 0) or 0
            if acc >= need_bytes:
                break
        print(f"  选中 {len(selected)} 个文件, 合计 {acc/1e9:.2f} GB")

        t0 = time.time()
        try:
            paths = download_files_parallel(
                src["repo_id"],
                selected,
                dest,
                workers=workers,
                verbose=True,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  [失败] 下载出错: {type(e).__name__}: {str(e)[:150]}")
            summary[name] = {"status": "failed", "error": str(e)[:200]}
            continue

        dt = time.time() - t0
        got = sum(p.stat().st_size for p in paths)
        mbps = got / 1e6 / dt if dt > 0 else 0
        print(f"\n  完成: {len(paths)} 个文件, {got/1e9:.2f} GB, 耗时 {dt/60:.1f} 分钟 "
              f"({mbps:.2f} MB/s 聚合)")

        summary[name] = {
            "status": "ok",
            "files": len(paths),
            "bytes": got,
            "seconds": round(dt, 1),
            "mbps": round(mbps, 2),
        }

    print(f"\n{'='*72}")
    print(f"  下载阶段完成，总耗时 {(time.time()-t_all)/60:.1f} 分钟")
    print(f"{'='*72}")
    for name, s in summary.items():
        if s["status"] == "ok":
            print(f"  {name:16s} {s['files']:5d} 文件  {s['bytes']/1e9:6.2f} GB  {s['mbps']:5.2f} MB/s")
        else:
            print(f"  {name:16s} [失败] {s.get('error','')[:80]}")

    return summary


# ===========================================================================
# 阶段 B：清洗 + 分词 + 打包
# ===========================================================================
def iter_source_documents(src: dict, raw_dir: Path):
    """从已下载的原始文件里迭代出正文文本。

    这里只处理**本地已下载**的文件，不再走网络 ——
    下载（阶段 A）和分词（阶段 B）彻底解耦，各自可以独立重跑。

    支持配置两种定位方式：
        dir + include_files : 在 raw_dir/<dir>/ 下只挑指定的文件
                              （minimind 把预训练/SFT/RL 数据都放在一个目录里，
                                必须显式挑，否则会把 SFT 数据当成预训练语料）
        name                : 兼容旧配置，用 raw_dir/<name>/ 下的全部文件
    """
    from minilm.data.modelscope import iter_file_records

    sub = src.get("dir") or src["name"]
    dest = raw_dir / sub
    if not dest.exists():
        print(f"    [警告] 目录不存在: {dest}")
        return

    # 决定要读哪些文件
    if src.get("include_files"):
        files = [dest / f for f in src["include_files"] if (dest / f).exists()]
        missing = [f for f in src["include_files"] if not (dest / f).exists()]
        if missing:
            print(f"    [警告] 以下文件还没下载完成，跳过: {missing}")
    else:
        files = [f for f in dest.glob("*") if f.is_file()]

    # 排除下载产生的临时文件
    files = [f for f in files if not f.name.endswith((".part", ".assembling"))]

    if not files:
        print(f"    [警告] 没有可读文件")
        return

    field = src["text_field"]
    for f in files:
        print(f"    读取 {f.name} ({f.stat().st_size/1e6:.0f} MB)")
        try:
            for rec in iter_file_records(f):
                text = rec.get(field)
                if isinstance(text, str) and text.strip():
                    yield text
                elif isinstance(text, list):
                    # 少数数据集把正文放在列表里
                    joined = "\n".join(str(x) for x in text if isinstance(x, str))
                    if joined.strip():
                        yield joined
        except Exception as e:  # noqa: BLE001
            print(f"    [跳过] {f.name}: {type(e).__name__}: {str(e)[:80]}")


def stage_tokenize(cfg: dict, tokenizer_path: Path, only: list[str] | None = None) -> dict:
    """清洗 + 分词 + 写出二进制文件。"""
    from minilm.data.build_bin import BuildConfig, TokenizeWriter
    from minilm.data.clean import CleanConfig, Deduplicator, clean_text, format_clean_stats
    from minilm.tokenizer import load_tokenizer
    from tqdm import tqdm

    print("\n" + "=" * 72)
    print("  阶段 B：清洗 + 分词 + 打包")
    print("=" * 72)

    if not tokenizer_path.exists():
        print(f"  [失败] 分词器不存在: {tokenizer_path}")
        print(f"         请先运行 python scripts/01_train_tokenizer.py")
        return {"status": "failed"}

    tokenizer = load_tokenizer(tokenizer_path)
    print(f"  分词器: {tokenizer_path}  (词表 {tokenizer.get_vocab_size():,})")

    out_dir = PROJECT_ROOT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = PROJECT_ROOT / cfg["raw_dir"]

    target = cfg["target_tokens"]
    val_tokens = cfg["val_tokens"]

    # 验证集单独开一个 writer，从每个源的开头切一小段出来。
    # 为什么要先切验证集：要保证验证集和训练集不重叠。
    # 简单做法是每个源的前 N 条进验证集，其余的进训练集。
    train_writer = TokenizeWriter(out_dir / "train.bin", buffer_tokens=8_000_000)
    val_writer = TokenizeWriter(out_dir / "val.bin", buffer_tokens=2_000_000)

    # 验证集从每个源取一小部分，保证验证集的分布和训练集一致。
    # 如果只用第一个源做验证，验证 loss 就不能代表整体训练情况。
    n_sources = len([s for s in cfg["sources"] if not only or s["name"] in only])
    val_per_source = val_tokens // max(1, n_sources)

    clean_cfg = CleanConfig()
    dedup = Deduplicator()
    source_stats: list[dict] = []
    t_all = time.time()

    for src in cfg["sources"]:
        name = src["name"]
        if only and name not in only:
            continue

        quota = src["target_tokens"]
        print(f"\n{'─'*72}")
        print(f"[{name}] 配额 {human_int(quota)} tokens")
        print(f"{'─'*72}")

        stats = {"total": 0, "kept": 0, "chars_in": 0, "chars_out": 0}
        n_tokens = 0
        n_val_tokens = 0
        t0 = time.time()

        batch: list[str] = []
        BATCH = 500
        # 文档顺序：前 val_per_source 个 token 进验证集，之后进训练集
        phase = "val"

        docs = iter_source_documents(src, raw_dir)

        pbar = tqdm(total=quota, desc=f"  分词", unit="tok", unit_scale=True)

        def flush():
            """把积攒的文档批量分词并写入。

            注意 `nonlocal` 声明：下面用到了 `n_tokens += ...` 这类自增赋值。
            Python 的规则是"函数内只要有赋值，该名字就是局部的" ——
            所以不加 nonlocal 的话，n_tokens/n_val_tokens 会被当成 flush 的局部变量，
            在条件判断里读取时会直接抛 UnboundLocalError（还没赋值就先读了）。
            """
            nonlocal batch, n_tokens, n_val_tokens
            if not batch:
                return
            encs = tokenizer.encode_batch(batch, add_special_tokens=False)
            for enc in encs:
                ids = enc.ids
                # 前面的 val_per_source 个 token 进验证集，之后进训练集。
                # 不需要显式切换状态 —— 验证集填满后条件自然为假，就都进训练集了。
                if phase == "val" and n_val_tokens < val_per_source:
                    val_writer.add_tokens(ids)
                    n_val_tokens += len(ids)
                else:
                    train_writer.add_tokens(ids)
                    n_tokens += len(ids)
            batch = []

        for text in docs:
            if n_tokens >= quota or train_writer.total_tokens >= target:
                break

            stats["total"] += 1
            stats["chars_in"] += len(text)

            cleaned = clean_text(text, clean_cfg)
            if cleaned is None:
                continue

            if dedup.is_duplicate(cleaned):
                continue

            stats["kept"] += 1
            stats["chars_out"] += len(cleaned)
            batch.append(cleaned)

            if len(batch) >= BATCH:
                flush()
                pbar.update(n_tokens + n_val_tokens - pbar.n)

        flush()
        pbar.update(n_tokens + n_val_tokens - pbar.n)
        pbar.close()

        dt = time.time() - t0
        rate = (n_tokens + n_val_tokens) / dt if dt > 0 else 0
        print(f"  产出 train {human_int(n_tokens)} + val {human_int(n_val_tokens)} tokens, "
              f"耗时 {dt/60:.1f} 分钟 ({rate:.0f} tok/s)")
        print(format_clean_stats(stats))

        source_stats.append({
            "name": name,
            # repo_id 是可选的：新配置用 dir + include_files 定位本地文件，
            # 不再需要 repo_id。这里用 .get() 兼容两种配置写法。
            "repo_id": src.get("repo_id", src.get("dir", name)),
            "train_tokens": n_tokens,
            "val_tokens": n_val_tokens,
            "license": src.get("license", ""),
            "clean_stats": stats,
        })

    train_writer.close()
    val_writer.close()

    # ---- 验证 token id 范围 ----
    # uint16 只能表示 0~65535。如果分词器词表超过这个范围，
    # 写文件时会静默溢出（numpy 会截断），导致训练数据损坏。
    # 这里显式检查，避免"训了 5 天才发现数据是坏的"。
    vocab_size = tokenizer.get_vocab_size()
    if vocab_size > 65536:
        print(f"  [警告] 词表大小 {vocab_size} 超过 uint16 上限 65536，"
              f"token id 会被截断！请改用 uint32 或缩小词表。")

    meta = {
        "tokenizer": str(tokenizer_path),
        "vocab_size": vocab_size,
        "dtype": "uint16",
        "train_tokens": train_writer.total_tokens,
        "val_tokens": val_writer.total_tokens,
        "sources": source_stats,
        "elapsed_sec": round(time.time() - t_all, 1),
    }
    write_json(out_dir / "meta.json", meta)

    print(f"\n{'='*72}")
    print(f"  分词阶段完成")
    print(f"{'='*72}")
    print(f"  train.bin : {human_int(train_writer.total_tokens)} tokens "
          f"({train_writer.total_tokens*2/1e9:.2f} GB)")
    print(f"  val.bin   : {human_int(val_writer.total_tokens)} tokens "
          f"({val_writer.total_tokens*2/1e6:.1f} MB)")
    print(f"  总耗时    : {(time.time()-t_all)/60:.1f} 分钟")
    print(f"  输出目录  : {out_dir}")

    return meta


# ===========================================================================
# 主流程
# ===========================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="阶段 2：数据准备")
    parser.add_argument("--stage", choices=["download", "tokenize", "all"], default="all")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "data.yaml"))
    parser.add_argument("--tokenizer", default=str(PROJECT_ROOT / "tokenizer" / "tokenizer.json"))
    parser.add_argument("--only", nargs="*", default=None,
                        help="只处理指定的数据源（按 configs/data.yaml 里的 name）")
    args = parser.parse_args()

    print("=" * 72)
    print("  Mini_LM 阶段 2：数据准备")
    print("=" * 72)

    cfg = load_data_config(Path(args.config))

    if args.stage in ("download", "all"):
        stage_download(cfg, only=args.only)

    if args.stage in ("tokenize", "all"):
        stage_tokenize(cfg, Path(args.tokenizer), only=args.only)

    print("\n" + "=" * 72)
    print("  阶段 2 完成")
    print("=" * 72)
    print("  下一步：python scripts/03_pretrain.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
