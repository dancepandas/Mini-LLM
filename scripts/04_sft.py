"""
阶段 4：SFT 指令微调。

## SFT 和预训练的核心区别

    预训练：全序列算 loss，目标是"预测下一个 token"，学语言本身。
    SFT   ：**只在 assistant 回复上算 loss**，目标是"学会回答问题"，
            学的是"什么样的输入该配什么样的输出"。

    prompt 部分标 -100 掩码，这是 SFT 最关键的实现细节。
    如果忘了掩码，模型会学着去预测用户的提问，
    表现是 SFT 后倾向于复述用户的话。这个错误 loss 曲线看不出来。

## 数据格式

    minimind 的 SFT 数据格式：
        {"conversations": [{"role": "user", "content": "..."},
                           {"role": "assistant", "content": "..."}],
         "reasoning_content": "..."}   # 可选，思维链

    渲染成 ChatML：
        <|im_start|>user\n...<|im_end|>
        <|im_start|>assistant\n<think>...</think>\n\n...<|im_end|>

## 超参要点

    LR 必须比预训练小一到两个数量级（预训练 3e-4 → SFT 1e-5 ~ 5e-5）。
    SFT 数据量小得多，大 LR 会很快过拟合，把预训练学到的通用能力冲掉
    （所谓"灾难性遗忘"）。

用法：
    python scripts/04_sft.py
    python scripts/04_sft.py --pretrained checkpoints/pretrain/final.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import minilm  # noqa: E402,F401

import torch  # noqa: E402

from minilm.chat_template import encode_conversation  # noqa: E402
from minilm.config import load_model_config, load_train_config, save_config  # noqa: E402
from minilm.model import build_model, force_mem_efficient_attention  # noqa: E402
from minilm.tokenizer import load_tokenizer  # noqa: E402
from minilm.utils import (  # noqa: E402
    AverageMeter,
    check_vram_available,
    find_latest_checkpoint,
    format_gpu_memory,
    get_logger,
    human_int,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)

IGNORE_INDEX = -100


# ===========================================================================
# 数据加载
# ===========================================================================
def load_sft_samples(path: Path, tokenizer, max_len: int, logger) -> list[dict]:
    """读取 SFT 数据并编码成带掩码的样本。

    **一次性全部编码在内存里**，而不是边训边编码。原因：
        SFT 数据量比预训练小得多（1.7GB 的 jsonl，编码后约 3~5 亿 token 的
        input_ids，用 list 存约几 GB 内存，本机 32GB 内存完全放得下）。
        预先编码后训练时直接索引，省掉了每步的 BPE 开销。

        如果数据大到放不下，就需要改成流式，但那会让代码复杂很多，
        而 SFT 数据通常不需要。
    """
    logger.info(f"  读取 {path.name} ...")

    samples: list[dict] = []
    n_raw = 0
    n_skipped = 0
    total_tokens = 0
    total_supervised = 0

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                n_skipped += 1
                continue

            # 兼容两种常见格式
            convs = obj.get("conversations") or obj.get("messages")
            if not convs or not isinstance(convs, list):
                n_skipped += 1
                continue

            n_raw += 1

            # 把顶层的 reasoning_content 塞进最后一条 assistant 消息。
            # 有些数据集把思维链放在顶层而不是消息里，这里统一处理。
            reasoning = obj.get("reasoning_content")
            if reasoning:
                for m in reversed(convs):
                    if m.get("role") in ("assistant", "gpt", "bot"):
                        m = dict(m)
                        m.setdefault("reasoning_content", reasoning)
                        break

            # 用统一的 chat template 编码 + 做 loss 掩码
            enc = encode_conversation(convs, tokenizer, max_len=max_len)
            ids = enc["input_ids"]
            labs = enc["labels"]

            # 至少要有一些被监督的 token，否则这条样本对训练毫无贡献
            n_sup = sum(1 for x in labs if x != IGNORE_INDEX)
            if n_sup == 0 or len(ids) < 16:
                n_skipped += 1
                continue

            samples.append({"input_ids": ids, "labels": labs})
            total_tokens += len(ids)
            total_supervised += n_sup

    logger.info(f"  读到 {n_raw:,} 条对话，有效 {len(samples):,} 条（跳过 {n_skipped:,}）")
    logger.info(f"  总 token {human_int(total_tokens)}，"
                f"其中被监督 {human_int(total_supervised)} "
                f"({total_supervised/max(1,total_tokens)*100:.1f}%)")

    if total_tokens > 0:
        ratio = total_supervised / total_tokens
        # 关于"健康范围"的说明（实测校准过）：
        #   单轮对话（一问一答）       → 通常 30%~60%
        #   多轮对话（如 minimind 的） → 可以到 80%~90%
        #     因为多轮数据里 assistant 回复占绝对多数，用户提问很短。
        #     本项目实测 minimind 的 sft_t2t_mini 是 85.4%，属正常。
        # 所以这里只对**极端值**告警，避免误报。
        if ratio < 0.15:
            logger.warning(f"  ⚠ 被监督比例只有 {ratio*100:.1f}%，异常偏低。")
            logger.warning("    可能是 loss 掩码没生效（prompt 被标成了监督），"
                           "或数据里 assistant 回复太短。")
        elif ratio > 0.97:
            logger.warning(f"  ⚠ 被监督比例高达 {ratio*100:.1f}%，几乎全序列都在算 loss。")
            logger.warning("    可能是掩码完全没生效，请检查 encode_conversation。")
        else:
            kind = "多轮" if ratio > 0.7 else "单轮"
            logger.info(f"  ({ratio*100:.1f}% 被监督，符合{kind}对话的预期)")

    return samples


class SFTDataset:
    """SFT 批采样器。

    和预训练的 MemmapDataset 不同，这里样本长度不一，需要 padding。
    padding 的位置 label 必须是 -100，否则模型会学着预测 padding token。

    ## ⚠️ 关键：labels 必须左移一位（踩过的坑）

        本模型的 `MiniLM.forward` 直接做 `cross_entropy(logits, labels)`，
        **不做内部位移**。而因果注意力下 `logits[i]` 预测的是 `input_ids[i+1]`，
        所以 `labels[i]` 必须是 `input_ids[i+1]` —— labels 要相对输入**左移一位**。

        预训练那边是对的（`MemmapDataset` 显式返回 `y = x` 右移一位）。
        但 SFT 最初漏了这一步：`encode_conversation` 返回的 labels 是
        `labels[i] = input_ids[i]`（未位移），模型被要求
        "用预测第 i+1 个 token 的 logits 去匹配第 i 个 token"。

        **症状**：训练 loss 停在 ~10.46（≈ ln(32768)=10.40，随机模型水平），
        gnorm 高达 22（正常应 ~0.3）。loss 仍会缓慢下降，所以**不报错、
        看起来"在训练"，但学的是完全错误的映射**，训完的模型不能用。

        **判据**：预训练好的模型在 SFT 数据上的初始 loss 应该是 2~3 ；
        如果是 10 左右，几乎可以肯定是 labels 错位。
    """

    def __init__(self, samples: list[dict], seq_len: int, pad_id: int = 0) -> None:
        self.samples = samples
        self.seq_len = seq_len
        self.pad_id = pad_id

    def __len__(self) -> int:
        return len(self.samples)

    def get_batch(self, batch_size: int, generator=None) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randint(0, len(self.samples), (batch_size,), generator=generator)

        xs, ys = [], []
        for i in idx.tolist():
            s = self.samples[i]
            ids = s["input_ids"][: self.seq_len]
            labs = s["labels"][: self.seq_len]

            # ⚠️ 关键一步：labels 左移一位，让 labs[j] 成为 logits[j] 的目标
            # （即 input_ids[j+1]）。右端补 IGNORE_INDEX —— 最后一个位置
            # 没有"下一个 token"可预测。
            labs = labs[1:] + [IGNORE_INDEX]

            # 补齐到 seq_len：输入补 pad_id，标签补 -100（不计 loss）
            pad_n = self.seq_len - len(ids)
            if pad_n > 0:
                ids = ids + [self.pad_id] * pad_n
                labs = labs + [IGNORE_INDEX] * pad_n

            xs.append(ids)
            ys.append(labs)

        return (
            torch.tensor(xs, dtype=torch.long, device="cuda"),
            torch.tensor(ys, dtype=torch.long, device="cuda"),
        )


# ===========================================================================
# 主流程
# ===========================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="阶段 4：SFT 指令微调")
    parser.add_argument("--model-config", default=str(PROJECT_ROOT / "configs" / "model_0.3b.yaml"))
    parser.add_argument("--pretrained", default=None,
                        help="预训练权重路径。默认自动找 checkpoints/pretrain 下最新的")
    parser.add_argument("--data", default=None, help="SFT 数据文件（jsonl）")
    parser.add_argument("--tokenizer", default=str(PROJECT_ROOT / "tokenizer" / "tokenizer.json"))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "checkpoints" / "sft"))
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=500)
    parser.add_argument("--max-samples", type=int, default=None, help="限制样本数（调试用）")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("sft", out_dir / "train.log")

    logger.info("=" * 68)
    logger.info("  Mini_LM 阶段 4：SFT 指令微调")
    logger.info("=" * 68)

    set_seed(1337)
    force_mem_efficient_attention()
    logger.info("")
    logger.info(f"  {format_gpu_memory()}")
    check_vram_available(4.5, raise_on_fail=False)

    # ---- 分词器 ----
    tok_path = Path(args.tokenizer)
    if not tok_path.exists():
        logger.error(f"分词器不存在: {tok_path}")
        return 1
    tokenizer = load_tokenizer(tok_path)

    # ---- 模型 ----
    mcfg = load_model_config(args.model_config)
    model = build_model(mcfg, "cuda")

    # ---- 加载预训练权重 ----
    if args.pretrained:
        ckpt = Path(args.pretrained)
    else:
        # 优先用 final.pt，其次找最新的 step checkpoint。
        # 用 final 是因为它训得最久；没有 final 说明训练中途停了，
        # 那就用最新的 step checkpoint（这体现了断点续训的价值）
        final = PROJECT_ROOT / "checkpoints" / "pretrain" / "final.pt"
        ckpt = final if final.exists() else find_latest_checkpoint(
            PROJECT_ROOT / "checkpoints" / "pretrain"
        )

    if ckpt is None or not ckpt.exists():
        logger.error("找不到预训练权重。请先完成阶段 3。")
        return 1

    logger.info(f"  加载预训练权重: {ckpt}")
    load_checkpoint(ckpt, model)
    logger.info("  已加载")

    # ---- SFT 数据 ----
    logger.info("")
    if args.data:
        data_path = Path(args.data)
    else:
        # 默认找 data/raw/minimind 下的 sft 文件
        candidates = sorted((PROJECT_ROOT / "data" / "raw").rglob("*sft*.jsonl"))
        if not candidates:
            logger.error("找不到 SFT 数据文件。请先下载，或用 --data 指定。")
            return 1
        # 优先 mini 版（小、够用）
        data_path = next((c for c in candidates if "mini" in c.name), candidates[0])

    samples = load_sft_samples(data_path, tokenizer, args.seq_len, logger)
    if args.max_samples:
        samples = samples[: args.max_samples]
        logger.info(f"  限制为 {len(samples)} 条样本")

    if not samples:
        logger.error("没有有效样本。")
        return 1

    ds = SFTDataset(samples, args.seq_len)

    # ---- 优化器 ----
    # 同样的 8-bit AdamW + 参数分组策略（norm/bias 不做 weight decay）。
    # SFT 阶段显存压力比预训练小（数据量小、可以少开梯度检查点），
    # 但沿用 8-bit 是稳妥的 —— 避免任何超售风险。
    import bitsandbytes as bnb

    decay, no_decay = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    optimizer = bnb.optim.AdamW8bit(
        [
            {"params": decay, "weight_decay": 0.1},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    scaler = torch.amp.GradScaler("cuda")

    # ---- 训练规模 ----
    steps_per_epoch = max(1, len(ds) // (args.batch_size * args.grad_accum))
    total_steps = int(steps_per_epoch * args.epochs)
    logger.info("")
    logger.info(f"  样本数        : {len(ds):,}")
    logger.info(f"  每 epoch 步数 : {steps_per_epoch:,}")
    logger.info(f"  总步数        : {total_steps:,}")
    logger.info(f"  lr            : {args.lr} (warmup {args.warmup_steps})")
    logger.info(f"  有效 batch    : {args.batch_size * args.grad_accum}")

    save_config(mcfg, out_dir / "model_config.yaml")

    # ---- 续训 ----
    start_step = 0
    if args.resume:
        latest = find_latest_checkpoint(out_dir)
        if latest:
            info = load_checkpoint(latest, model, optimizer)
            start_step = info["step"]
            logger.info(f"  从 {latest.name} 续训，第 {start_step} 步")

    # ---- 训练 ----
    logger.info("")
    logger.info("=" * 68)
    logger.info("  开始 SFT")
    logger.info("=" * 68)

    model.train()
    meter = AverageMeter(window=100)
    gen = torch.Generator()
    gen.manual_seed(1337)
    t0 = time.time()

    for step in range(start_step, total_steps):
        # warmup + cosine
        if step < args.warmup_steps:
            lr = args.lr * (step + 1) / args.warmup_steps
        else:
            prog = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
            lr = args.lr * 0.1 + args.lr * 0.9 * 0.5 * (1 + __import__("math").cos(3.141592653589793 * prog))
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum = 0.0

        for _ in range(args.grad_accum):
            x, y = ds.get_batch(args.batch_size, generator=gen)
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x, labels=y, use_checkpoint=True)
                loss = out["loss"] / args.grad_accum

            scaler.scale(loss).backward()
            accum += loss.item()

        scaler.unscale_(optimizer)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        if not torch.isfinite(gn):
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            logger.warning(f"  step {step}: 非有限梯度，跳过")
            continue

        scaler.step(optimizer)
        scaler.update()
        meter.update(accum)

        if step % 10 == 0:
            elapsed = time.time() - t0
            done = step - start_step + 1
            eta = (total_steps - step) / max(1e-9, done / elapsed) / 60
            logger.info(
                f"  step {step:>6,}/{total_steps:,} | loss {meter.recent:.4f} | "
                f"lr {lr:.2e} | gnorm {gn:.3f} | ETA {eta:.0f}分"
            )

        if (step + 1) % args.save_interval == 0:
            save_checkpoint(out_dir / f"step_{step+1:07d}.pt", model, optimizer, step + 1,
                            extra={"loss": meter.recent})

    # ---- 收尾 ----
    save_checkpoint(out_dir / "final.pt", model, optimizer, total_steps,
                    extra={"loss": meter.recent})

    logger.info("")
    logger.info("=" * 68)
    logger.info("  SFT 完成")
    logger.info("=" * 68)
    logger.info(f"  耗时     : {(time.time()-t0)/60:.1f} 分钟")
    logger.info(f"  最终 loss: {meter.recent:.4f}")
    logger.info(f"  模型     : {out_dir / 'final.pt'}")
    logger.info("")
    logger.info("  下一步：python scripts/05_rl_grpo.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
