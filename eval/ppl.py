"""
困惑度（Perplexity）评测。

## 什么是困惑度

    PPL = exp(平均交叉熵损失)

    直观理解：模型在每一步平均要在多少个候选词里"犹豫"。
    PPL=100 意味着模型平均认为有 100 个词同样可能；
    PPL=10 意味着大约 10 个。

    中文小模型的参考量级（在中文语料上）：
        随机初始化   ≈ 32768（等于词表大小，模型完全没学到东西）
        训练早期     ≈ 100~200
        训练充分     ≈ 15~30
        SFT 之后     ≈ 10~25（SFT 数据分布不同，可能略有上升）

## 为什么要分领域看困惑度

    整体 PPL 会掩盖问题。分开看百科、对话、新闻的 PPL 才能发现：
      - 某一类特别高 → 语料里这类数据太少
      - 训练集和验证集差距大 → 过拟合

用法：
    python eval/ppl.py --model checkpoints/sft/final.pt
    python eval/ppl.py --model checkpoints/pretrain/step_0005000.pt --split val
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import minilm  # noqa: E402,F401

import torch  # noqa: E402

from minilm.config import load_model_config  # noqa: E402
from minilm.data.dataset import MemmapDataset  # noqa: E402
from minilm.model import build_model, force_mem_efficient_attention  # noqa: E402
from minilm.tokenizer import load_tokenizer  # noqa: E402
from minilm.utils import find_latest_checkpoint, load_checkpoint  # noqa: E402


@torch.no_grad()
def compute_perplexity(
    model,
    dataset: MemmapDataset,
    n_batches: int = 50,
    seq_len: int = 1024,
    micro_batch: int = 4,
) -> dict:
    """在数据集上算困惑度。

    返回包含 avg_loss 和 perplexity 的字典。

    注意用 `n_batches` 采样而不是跑完整数据集：
        验证集有 200 万 token，跑完要几十秒。
        采样 50 个 batch（约 20 万 token）已经能让标准误差降到可接受范围。
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    n = 0

    for _ in range(n_batches):
        x, y = dataset.get_batch(micro_batch)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x, labels=y, use_checkpoint=False)
            loss = out["loss"]

        if not torch.isfinite(loss):
            continue

        # 按 token 数加权 —— 不同 batch 的有效 token 数可能不同
        n_tok = (y != -100).sum().item()
        total_loss += loss.item() * n_tok
        total_tokens += n_tok
        n += 1

    model.train()

    if total_tokens == 0:
        return {"perplexity": float("inf"), "avg_loss": float("inf"), "n_batches": 0}

    avg_loss = total_loss / total_tokens
    return {
        "avg_loss": avg_loss,
        # PPL = exp(交叉熵)
        "perplexity": math.exp(avg_loss),
        "n_batches": n,
        "total_tokens": total_tokens,
    }


@torch.no_grad()
def compute_domain_ppl(model, tokenizer, texts: list[str], seq_len: int = 512) -> list[dict]:
    """按领域分别算困惑度。

    做法：对每段文本算它自己的平均交叉熵，再按领域汇总。
    比整体 PPL 更能暴露问题 —— 比如模型在对话上还行但在百科上很差。
    """
    model.eval()
    results = []

    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False).ids
        if len(ids) < 32:
            continue
        ids = ids[:seq_len]
        x = torch.tensor([ids[:-1]], dtype=torch.long, device="cuda")
        y = torch.tensor([ids[1:]], dtype=torch.long, device="cuda")

        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x, labels=y, use_checkpoint=False)
            loss = out["loss"]

        if torch.isfinite(loss):
            results.append({
                "n_tokens": len(ids) - 1,
                "loss": loss.item(),
                "ppl": math.exp(loss.item()),
            })

    model.train()
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="困惑度评测")
    parser.add_argument("--model", default=None, help="模型路径，默认自动查找")
    parser.add_argument("--model-config", default=str(PROJECT_ROOT / "configs" / "model_0.3b.yaml"))
    parser.add_argument("--tokenizer", default=str(PROJECT_ROOT / "tokenizer" / "tokenizer.json"))
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "pretrain"))
    parser.add_argument("--split", choices=["val", "train"], default="val")
    parser.add_argument("--n-batches", type=int, default=50)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--micro-batch", type=int, default=4)
    parser.add_argument("--out", default=None, help="结果保存到 json")
    args = parser.parse_args()

    force_mem_efficient_attention()

    # ---- 找模型 ----
    if args.model:
        model_path = Path(args.model)
    else:
        model_path = None
        for stage in ("rl", "sft", "pretrain"):
            p = PROJECT_ROOT / "checkpoints" / stage / "final.pt"
            if p.exists():
                model_path = p
                break
        if model_path is None:
            model_path = find_latest_checkpoint(PROJECT_ROOT / "checkpoints" / "pretrain")
    if model_path is None or not model_path.exists():
        print(f"找不到模型权重。请先训练，或用 --model 指定。")
        return 1

    print("=" * 64)
    print("  困惑度评测")
    print("=" * 64)
    print(f"  模型: {model_path}")

    mcfg = load_model_config(args.model_config)
    model = build_model(mcfg, "cuda")
    load_checkpoint(model_path, model)
    model.eval()

    # ---- 数据 ----
    data_path = Path(args.data_dir) / f"{args.split}.bin"
    if not data_path.exists():
        print(f"数据不存在: {data_path}")
        return 1

    ds = MemmapDataset(data_path, args.seq_len, device="cuda")
    print(f"  数据: {data_path.name} ({ds.n_tokens:,} tokens)")
    print(f"  采样: {args.n_batches} 个 batch × {args.micro_batch} × {args.seq_len}")
    print()

    result = compute_perplexity(
        model, ds,
        n_batches=args.n_batches,
        seq_len=args.seq_len,
        micro_batch=args.micro_batch,
    )

    print("-" * 64)
    print(f"  平均交叉熵 loss : {result['avg_loss']:.4f}")
    print(f"  **困惑度 PPL**  : {result['perplexity']:.2f}")
    print(f"  统计 token 数   : {result.get('total_tokens', 0):,}")
    print("-" * 64)

    ppl = result["perplexity"]
    print()
    if ppl > 1000:
        print("  判断: 困惑度极高，模型基本没学到东西（检查训练是否正常）")
    elif ppl > 100:
        print("  判断: 困惑度偏高，训练可能不足")
    elif ppl > 40:
        print("  判断: 训练中，但仍未充分收敛")
    elif ppl > 15:
        print("  判断: 困惑度正常，模型已学到语言规律")
    else:
        print("  判断: 困惑度很低")

    print()
    print("  参考量级: 随机初始化≈32768 | 训练早期≈100-200 | 充分训练≈15-30")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"model": str(model_path), **result}, f, ensure_ascii=False, indent=2)
        print(f"\n  结果已保存: {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
