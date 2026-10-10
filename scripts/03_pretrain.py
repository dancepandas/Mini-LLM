"""
阶段 3：预训练主循环。

这是整个项目最核心的训练脚本。设计要点：

## 断点续训是必需品，不是加分项

    训练要连续跑约 5 天。桌面机跑 5 天，断电、系统更新重启、误关窗口
    都是**大概率事件**。没有断点续训，一次意外就要从头再来 —— 代价是 5 天。
    所以这里把 resume 做成一等公民：默认自动找最新的 checkpoint 接着跑。

## 显存安全是硬约束

    实测数据（RTX 2060 6GB）：
        micro_batch=4 → reserved 4.90GB，4618 tok/s   ✅
        micro_batch=6 → reserved 6.40GB，2846 tok/s   ❌ 超售
    一旦显存超售，Windows 会静默换页，速度掉 3~16 倍，而且**不报错**。
    所以启动时要主动检查空闲显存，训练中也要周期性监控。

## 为什么要记录这么多指标

    loss 只是表象。真正能看出问题的是：
      - grad_norm 突然暴涨 → 即将发散，要看 lr 或数据
      - 吞吐突然下降 → 显存碎片化或超售，需要重启
      - val loss 先降后升 → 过拟合
    这些在 5 天的长跑里都是真实会遇到的，没有监控就只能盲跑。

用法：
    python scripts/03_pretrain.py                    # 从头开始
    python scripts/03_pretrain.py --resume           # 自动续训
    python scripts/03_pretrain.py --max-steps 5000   # 覆盖总步数
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import minilm  # noqa: E402,F401  (必须先于 torch：设置显存分配器)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from minilm.config import load_model_config, load_train_config, save_config  # noqa: E402
from minilm.model import build_model, force_mem_efficient_attention  # noqa: E402
from minilm.utils import (  # noqa: E402
    AverageMeter,
    ThroughputTracker,
    check_vram_available,
    find_latest_checkpoint,
    format_gpu_memory,
    get_logger,
    gpu_memory,
    human_int,
    load_checkpoint,
    rotate_checkpoints,
    save_checkpoint,
    set_seed,
)


# ===========================================================================
# 优化器
# ===========================================================================
def build_optimizer(model: torch.nn.Module, tcfg, logger):
    """构建优化器。

    **默认必须是 8-bit AdamW**，原因很实际：
        普通 fp32 AdamW 的优化器状态（m 和 v 各一份 fp32）要 2.36GB，
        总显存涨到 6.7GB 超过物理显存 → 触发 Windows 显存超售 → 训练慢 16 倍
        （实测 305 tok/s vs 4618 tok/s）。
        8-bit 版本只要 0.59GB，是**避免超售的必需项**，不是可选优化。

    参数分组：对 weight decay 做区分处理。
        只对 2 维以上的权重矩阵（Linear/Embedding 的 weight）施加 weight decay，
        对 RMSNorm 的 1 维缩放系数**不施加** ——
        对归一化层的缩放系数做衰减会把它们推向 0，直接损害模型表达力。
        这是 GPT 以来的标准做法。
    """
    decay_params = []
    no_decay_params = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # 1 维参数是 norm 的缩放系数和 bias，不做 weight decay
        if p.dim() >= 2:
            decay_params.append(p)
        else:
            no_decay_params.append(p)

    groups = [
        {"params": decay_params, "weight_decay": tcfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    n_decay = sum(p.numel() for p in decay_params)
    n_nodecay = sum(p.numel() for p in no_decay_params)
    logger.info(f"  参数分组: {human_int(n_decay)} 做 weight decay, "
                f"{human_int(n_nodecay)} 不做（norm/bias）")

    if tcfg.optimizer == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as e:
            raise RuntimeError(
                "配置要求 adamw8bit 但 bitsandbytes 未安装。\n"
                "安装：pip install bitsandbytes\n"
                "注意：改用普通 fp32 AdamW 会多占 1.9GB 显存，很可能触发超售。"
            ) from e
        logger.info("  优化器: AdamW 8-bit (bitsandbytes)")
        return bnb.optim.AdamW8bit(groups, lr=tcfg.lr, betas=tcfg.betas)

    if tcfg.optimizer == "adamw":
        logger.info("  优化器: AdamW fp32（显存占用大，仅用于对照）")
        return torch.optim.AdamW(groups, lr=tcfg.lr, betas=tcfg.betas, foreach=True)

    raise ValueError(f"未知优化器: {tcfg.optimizer}")


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    """设置学习率。因为用了参数分组，要遍历所有 group。"""
    for g in optimizer.param_groups:
        g["lr"] = lr


# ===========================================================================
# 验证
# ===========================================================================
@torch.no_grad()
def evaluate(model, dataset, tcfg, n_batches: int = 20) -> float:
    """在验证集上算平均 loss。

    参数
    ----
    n_batches : int
        采样多少个批次。取 20 个而不是整个验证集，是为了控制验证耗时 ——
        每个 batch 是 4×1024=4096 tokens，20 个约 8 万 tokens，
        对 2M tokens 的验证集来说采样误差已经足够小。
    """
    model.eval()
    losses = []
    for _ in range(n_batches):
        x, y = dataset.get_batch(tcfg.micro_batch_size)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(tcfg.dtype == "fp16")):
            out = model(x, labels=y, use_checkpoint=False)
        losses.append(out["loss"].item())
    model.train()
    return sum(losses) / len(losses)


# ===========================================================================
# 主训练循环
# ===========================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="阶段 3：预训练")
    parser.add_argument("--model-config", default=str(PROJECT_ROOT / "configs" / "model_0.3b.yaml"))
    parser.add_argument("--train-config", default=str(PROJECT_ROOT / "configs" / "pretrain.yaml"))
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "pretrain"))
    parser.add_argument("--resume", action="store_true", help="从最新 checkpoint 续训")
    parser.add_argument("--resume-from", default=None, help="指定 checkpoint 路径续训")
    parser.add_argument("--max-steps", type=int, default=None, help="覆盖配置里的总步数")
    parser.add_argument("--micro-batch", type=int, default=None, help="覆盖 micro batch")
    parser.add_argument("--no-save", action="store_true", help="不保存 checkpoint（调试用）")
    args = parser.parse_args()

    # ---- 配置 ----
    mcfg = load_model_config(args.model_config)
    tcfg = load_train_config(args.train_config)
    if args.max_steps is not None:
        tcfg.max_steps = args.max_steps
    if args.micro_batch is not None:
        tcfg.micro_batch_size = args.micro_batch

    out_dir = PROJECT_ROOT / tcfg.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("pretrain", out_dir / "train.log")

    logger.info("=" * 68)
    logger.info("  Mini_LM 阶段 3：预训练")
    logger.info("=" * 68)
    logger.info(f"  out_dir : {out_dir}")
    logger.info("")
    logger.info(f"  micro_batch      : {tcfg.micro_batch_size}")
    logger.info(f"  grad_accum       : {tcfg.grad_accum_steps}")
    logger.info(f"  seq_len          : {tcfg.seq_len}")
    logger.info(f"  有效 batch       : {tcfg.tokens_per_step:,} tokens/step")
    logger.info(f"  总步数           : {tcfg.max_steps:,}")
    logger.info(f"  总 token         : {human_int(tcfg.total_tokens)}")
    logger.info(f"  lr               : {tcfg.lr} (warmup {tcfg.warmup_steps} 步, cosine 到 {tcfg.lr*tcfg.min_lr_ratio:.1e})")
    logger.info(f"  dtype            : {tcfg.dtype}")
    logger.info(f"  梯度检查点       : {tcfg.gradient_checkpointing}")

    set_seed(tcfg.seed)
    force_mem_efficient_attention()

    # ---- 显存检查（超售是最隐蔽的性能陷阱）----
    free = gpu_memory()["free"]
    logger.info("")
    logger.info(f"  {format_gpu_memory()}")
    check_vram_available(4.6, raise_on_fail=False)

    # ---- 模型 ----
    logger.info("")
    model = build_model(mcfg, "cuda")

    # ---- 数据 ----
    from minilm.data.dataset import MemmapDataset, load_meta

    train_path = Path(args.data_dir) / "train.bin"
    val_path = Path(args.data_dir) / "val.bin"
    if not train_path.exists():
        logger.error(f"训练数据不存在: {train_path}")
        logger.error("请先运行: python scripts/02_prepare_data.py --stage tokenize")
        return 1

    train_ds = MemmapDataset(train_path, tcfg.seq_len, device="cuda")
    val_ds = MemmapDataset(val_path, tcfg.seq_len, device="cuda") if val_path.exists() else None

    meta = load_meta(args.data_dir)
    logger.info("")
    logger.info(f"  训练集: {human_int(train_ds.n_tokens)} tokens")
    if val_ds:
        logger.info(f"  验证集: {human_int(val_ds.n_tokens)} tokens")
    if meta.get("sources"):
        for s in meta["sources"]:
            logger.info(f"    - {s['name']}: {human_int(s['train_tokens'])} tokens")

    # 检查 token 数是否够跑完计划
    needed = tcfg.total_tokens
    if train_ds.n_tokens < needed:
        epochs = needed / train_ds.n_tokens
        logger.warning(
            f"  ⚠ 数据量({human_int(train_ds.n_tokens)}) 少于计划的 {human_int(needed)} tokens，"
            f"相当于要跑 {epochs:.2f} 个 epoch。"
        )
        logger.warning(f"    本次最多能跑 {train_ds.n_tokens // tcfg.tokens_per_step} 步。")
        tcfg.max_steps = min(tcfg.max_steps, train_ds.n_tokens // tcfg.tokens_per_step)
        logger.warning(f"    已将 max_steps 调整为 {tcfg.max_steps}")

    # 保存本次实际使用的配置，方便日后回溯
    save_config(mcfg, out_dir / "model_config.yaml")
    save_config(tcfg, out_dir / "train_config.yaml")

    # ---- 优化器 / 混合精度 ----
    logger.info("")
    optimizer = build_optimizer(model, tcfg, logger)

    scaler = None
    if tcfg.dtype == "fp16":
        # fp16 的动态范围窄（最小正规数约 6e-5），小梯度会直接下溢成 0，
        # 导致这些参数永远不更新。GradScaler 先把 loss 放大再反传，
        # 把梯度推进 fp16 可表示的范围内，更新前再缩放回来。
        # {sm_75} 没有 bf16 硬件加速，所以必须走 fp16 + GradScaler 这条路。
        scaler = torch.amp.GradScaler("cuda")
        logger.info("  混合精度: fp16 + GradScaler")

    # ---- 续训 ----
    start_step = 0
    ckpt_path = Path(args.resume_from) if args.resume_from else (
        find_latest_checkpoint(out_dir) if args.resume else None
    )
    if ckpt_path and ckpt_path.exists():
        logger.info("")
        logger.info(f"  从 checkpoint 续训: {ckpt_path.name}")
        info = load_checkpoint(ckpt_path, model, optimizer)
        start_step = info["step"]
        # GradScaler 的状态也要恢复，否则 loss scale 会从初始值重新爬，
        # 造成续训后最初几百步的梯度实际偏小，loss 会有一个小台阶
        extra = info.get("extra", {})
        if scaler is not None and extra.get("scaler") is not None:
            scaler.load_state_dict(extra["scaler"])
        logger.info(f"  从第 {start_step:,} 步继续")

    # ---- TensorBoard ----
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(out_dir / "tb"))
        logger.info(f"  TensorBoard: tensorboard --logdir {out_dir / 'tb'}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"  TensorBoard 不可用（不影响训练）: {e}")

    # ---- 训练循环 ----
    logger.info("")
    logger.info("=" * 68)
    logger.info("  开始训练")
    logger.info("=" * 68)

    model.train()
    loss_meter = AverageMeter(window=100)
    grad_meter = AverageMeter(window=100)
    tracker = ThroughputTracker()
    t_start = time.time()

    # 用独立的随机数生成器，让数据顺序只取决于 step 数。
    # 这样断点续训时能从正确的数据位置继续，而不是重新洗一遍。
    gen = torch.Generator()
    gen.manual_seed(tcfg.seed)

    # 断点续训时要把生成器推进到对应位置，保证数据顺序一致。
    # 这里的做法是简单地跳过 start_step 次采样 —— 开销很小（只是几个 randint）。
    if start_step > 0:
        logger.info(f"  推进数据采样器到第 {start_step} 步...")
        for _ in range(start_step):
            for _ in range(tcfg.grad_accum_steps):
                _ = torch.randint(0, 10_000_000, (tcfg.micro_batch_size,), generator=gen)

    nan_skips = 0

    for step in range(start_step, tcfg.max_steps):
        # 学习率按余弦调度
        lr = tcfg.cosine_lr(step)
        set_lr(optimizer, lr)

        # ---- 梯度累积 ----
        # 用 micro_batch 跑多次前向反向，累积梯度后再更新一次参数。
        # 这样等效 batch = micro_batch * grad_accum，但显存只按 micro_batch 算。
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro in range(tcfg.grad_accum_steps):
            x, y = train_ds.get_batch(tcfg.micro_batch_size, generator=gen)

            with torch.autocast("cuda", dtype=torch.float16, enabled=(tcfg.dtype == "fp16")):
                out = model(x, labels=y, use_checkpoint=tcfg.gradient_checkpointing)
                loss = out["loss"]
                # 除以累积步数，让累积后的梯度等效于大 batch 的平均梯度。
                # 不做这个除法的话，梯度会是累加值，等效把 lr 放大了 grad_accum 倍。
                loss = loss / tcfg.grad_accum_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            accum_loss += loss.item()
            tracker.update(tcfg.micro_batch_size * tcfg.seq_len)

        # ---- 梯度裁剪 + 参数更新 ----
        if scaler is not None:
            # unscale 必须在 clip 之前：要把放大过的梯度缩放回真实值，
            # 否则裁剪阈值 1.0 是在放大后的尺度上比较，等于没裁
            scaler.unscale_(optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)

        # ---- NaN / Inf 检测 ----
        # fp16 训练偶发梯度溢出。检测到就跳过这一步，而不是让污染的参数写回去 ——
        # 一次坏更新可能让 loss 直接爆掉，而且要很久才能恢复。
        if not torch.isfinite(grad_norm):
            nan_skips += 1
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                # 让 scaler 降低 loss scale，下次不那么容易溢出
                scaler.update()
            logger.warning(f"  step {step}: 检测到非有限梯度，跳过该步 (累计 {nan_skips} 次)")
            continue

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        loss_meter.update(accum_loss)
        grad_meter.update(grad_norm.item())

        # ---- 日志 ----
        if step % tcfg.log_interval == 0 or step == start_step:
            elapsed = time.time() - t_start
            steps_done = step - start_step + 1
            tokens_done = steps_done * tcfg.tokens_per_step
            tps = tokens_done / elapsed if elapsed > 0 else 0

            remaining_steps = tcfg.max_steps - step - 1
            eta_hours = (remaining_steps * tcfg.tokens_per_step / tps / 3600) if tps > 0 else 0

            mem = gpu_memory()
            # 显存超售检测：reserved 超过物理空闲显存就说明在换页
            warn = "  [显存超售!]" if mem["reserved"] > mem["free"] + mem["reserved"] else ""

            logger.info(
                f"  step {step:>6,}/{tcfg.max_steps:,} | "
                f"loss {loss_meter.recent:.4f} | "
                f"lr {lr:.2e} | "
                f"gnorm {grad_meter.recent:.3f} | "
                f"{tps:>6.0f} tok/s | "
                f"mem {mem['reserved']:.2f}GB | "
                f"ETA {eta_hours:>5.1f}h{warn}"
            )

            if writer:
                writer.add_scalar("train/loss", loss_meter.recent, step)
                writer.add_scalar("train/lr", lr, step)
                writer.add_scalar("train/grad_norm", grad_meter.recent, step)
                writer.add_scalar("train/tokens_per_sec", tps, step)
                writer.add_scalar("train/mem_reserved_gb", mem["reserved"], step)

            # 周期性重置吞吐统计，让 ETA 反映当前速度而不是全局平均
            tracker.reset()

        # ---- 验证 ----
        if val_ds is not None and (step + 1) % tcfg.eval_interval == 0 and step > start_step:
            vloss = evaluate(model, val_ds, tcfg)
            logger.info(f"  >>> 验证 loss @ step {step+1:,}: {vloss:.4f}")
            if writer:
                writer.add_scalar("val/loss", vloss, step)
            model.train()

        # ---- 保存 checkpoint ----
        if not args.no_save and (step + 1) % tcfg.save_interval == 0:
            ckpt_file = out_dir / f"step_{step+1:07d}.pt"
            save_checkpoint(
                ckpt_file,
                model,
                optimizer,
                step + 1,
                extra={
                    "loss": loss_meter.recent,
                    "scaler": scaler.state_dict() if scaler is not None else None,
                    "config": {"model": vars(mcfg), "train": vars(tcfg)},
                },
            )
            # 滚动删除旧 checkpoint，防止磁盘被写满
            # （每个约 1.3GB，30 个就是 40GB）
            deleted = rotate_checkpoints(out_dir, tcfg.keep_last_n_checkpoints)
            msg = f"  已保存 {ckpt_file.name}"
            if deleted:
                msg += f"，清理了 {len(deleted)} 个旧 checkpoint"
            logger.info(msg)

    # ---- 收尾 ----
    logger.info("")
    logger.info("=" * 68)
    logger.info("  训练完成")
    logger.info("=" * 68)

    final = out_dir / "final.pt"
    if not args.no_save:
        save_checkpoint(final, model, optimizer, tcfg.max_steps,
                        extra={"loss": loss_meter.recent})
        logger.info(f"  最终模型: {final}")

    total_hours = (time.time() - t_start) / 3600
    logger.info(f"  总耗时  : {total_hours:.2f} 小时")
    logger.info(f"  最终loss: {loss_meter.recent:.4f}")
    if nan_skips:
        logger.info(f"  NaN 跳过: {nan_skips} 次")

    if writer:
        writer.close()

    logger.info("")
    logger.info("  下一步：python scripts/04_sft.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
