"""
阶段 0：环境自检 + 吞吐基准测试。

**这是整个项目的决策门。**

为什么要先做这一步，而不是直接开训：
    训练 2B tokens 的预计耗时完全取决于吞吐量，而吞吐量会因显存超售而在
    305 ~ 4985 tok/s 之间浮动（相差 16 倍，对应的 ETA 是 5 天 vs 76 天）。
    Windows 的显存超售是**静默的** —— 不报错，只是慢。
    所以必须在投入 5 天之前，用一个真实配置的小规模基准把真实吞吐量测出来。

    如果实测吞吐低于预期，就该**当场调整 token 预算或模型规模**，
    而不是硬跑 15 天之后才发现来不及。

用法：
    python scripts/00_setup.py                 # 完整自检 + 基准
    python scripts/00_setup.py --quick         # 跳过基准，只做环境检查
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path

# 让脚本能 import 到 minilm 包（脚本在 scripts/ 下，包在上一级）
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ⚠️ 导入顺序很重要：必须**先 import minilm，再 import torch**。
# minilm/__init__.py 会设置 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True，
# 而这个环境变量只在 CUDA 分配器首次初始化时被读取。
# 一旦 torch 先被导入并初始化了 CUDA，再设置就无效了。
# 实测这个设置能把 reserved 从 5.40GB 压到 4.90GB，从而避免显存超售。
import minilm  # noqa: E402,F401  (仅为了触发环境变量设置)

import torch  # noqa: E402

from minilm.config import ModelConfig, load_model_config, load_train_config  # noqa: E402
from minilm.model import (  # noqa: E402
    assert_mem_efficient_active,
    build_model,
    estimate_training_memory,
    force_mem_efficient_attention,
    model_summary,
)
from minilm.utils import (  # noqa: E402
    check_vram_available,
    format_gpu_memory,
    gpu_memory,
    human_int,
    set_seed,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ===========================================================================
# 第 1 步：环境检查
# ===========================================================================
REQUIRED_PACKAGES = [
    ("torch", "深度学习框架"),
    ("numpy", "数值计算"),
    ("tokenizers", "训练 BPE 分词器"),
    ("datasets", "下载 HF 数据集"),
    ("huggingface_hub", "HF 下载基础设施"),
    ("pyarrow", "读写 parquet"),
    ("bitsandbytes", "8-bit 优化器（避免显存超售的必需品）"),
    ("safetensors", "权重的安全序列化格式"),
    ("yaml", "读配置文件"),
    ("tqdm", "进度条"),
    ("tensorboard", "训练曲线可视化"),
]


def check_environment() -> bool:
    """检查 Python 环境和依赖包。"""
    print("\n" + "=" * 72)
    print("第 1 步：环境检查")
    print("=" * 72)
    print(f"  Python   : {sys.version.split()[0]}")
    print(f"  解释器   : {sys.executable}")

    all_ok = True
    missing = []
    for mod, desc in REQUIRED_PACKAGES:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            print(f"  [OK]   {mod:18s} {ver:12s} {desc}")
        except ImportError:
            print(f"  [缺失] {mod:18s} {'':12s} {desc}")
            missing.append(mod)
            all_ok = False

    if missing:
        print(f"\n缺失的包，请先安装：")
        print(f"  pip install {' '.join(missing)}")
        print(f"\n注意：本机需要走代理，安装前先设置：")
        print(f"  set http_proxy=http://127.0.0.1:7897")
        print(f"  set https_proxy=http://127.0.0.1:7897")

    return all_ok


def check_gpu() -> bool:
    """检查 GPU 与关键硬件特性。"""
    print("\n" + "=" * 72)
    print("第 2 步：GPU 检查")
    print("=" * 72)

    if not torch.cuda.is_available():
        print("  [错误] CUDA 不可用，无法训练。请检查显卡驱动和 PyTorch 的 CUDA 版本。")
        return False

    name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    m = gpu_memory()

    print(f"  设备      : {name}")
    print(f"  算力      : sm_{cap[0]}{cap[1]}")
    print(f"  总显存    : {m['total']:.2f} GB")
    print(f"  当前空闲  : {m['free']:.2f} GB   <-- 这个值决定会不会超售")
    print(f"  已被占用  : {m['used']:.2f} GB")

    # bf16 支持情况 —— 这里必须把"能跑"和"有硬件加速"区分开
    bf16_ok = torch.cuda.is_bf16_supported()
    print(f"\n  bf16 支持 : {bf16_ok}")
    if bf16_ok and cap[0] < 8:
        # sm_75 (Turing) 没有 bf16 张量核，PyTorch 会走模拟路径
        print(f"            注意：sm_{cap[0]}{cap[1]} 没有 bf16 张量核，"
              f"虽然'能跑'但走的是模拟路径，速度远不如 fp16。")
        print(f"            本项目使用 fp16 —— Turing 有 fp16 张量核，是真正的快路径。")

    print("\n  注意力后端探测：")
    try:
        assert_mem_efficient_active()
        print("            这是本项目的关键依赖（math 后端会让注意力显存涨 15 倍）")
    except RuntimeError as e:
        print(f"            [失败] {e}")
        return False

    return True


# ===========================================================================
# 第 3 步：模型构建与参数量核对
# ===========================================================================
def check_model(cfg: ModelConfig) -> tuple[bool, int]:
    """构建模型，核对参数量。"""
    print("\n" + "=" * 72)
    print("第 3 步：模型构建与参数量核对")
    print("=" * 72)

    model = build_model(cfg, "cuda")
    print()
    print(model_summary(cfg, model))

    est = cfg.num_params()
    actual = model.num_parameters()
    ok = est == actual
    if not ok:
        print(f"\n  [错误] 解析式({est:,}) 与实测({actual:,}) 不一致，相差 {actual-est:,} 个参数。")
        print(f"         说明 model.py 的结构和 config.py 的公式不同步，请检查。")

    # 前向反向烟测
    print("\n  前向 + 反向烟测：")
    model.train()
    idx = torch.randint(0, cfg.vocab_size, (2, 128), device="cuda")
    tgt = torch.randint(0, cfg.vocab_size, (2, 128), device="cuda")
    tgt[:, :10] = -100  # 验证 loss 掩码（SFT 阶段要用）
    out = model(idx, labels=tgt, use_checkpoint=True)
    loss = out["loss"]
    loss.backward()

    import math
    expected = math.log(cfg.vocab_size)
    print(f"    logits 形状 : {tuple(out['logits'].shape)}")
    print(f"    loss        : {loss.item():.4f}  (随机初始化理论值 ln({cfg.vocab_size}) = {expected:.4f})")
    print(f"    反向传播    : 成功")

    # 清理，把显存还给基准测试
    del model, out, loss, idx, tgt
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    return ok, cfg.num_params()


# ===========================================================================
# 第 4 步：吞吐基准测试 —— 决策门的核心
# ===========================================================================
def build_optimizer(model: torch.nn.Module, kind: str, lr: float, cfg=None):
    """按配置构建优化器。

    8-bit Adam 是本项目的默认选择，原因很实际：
        普通 fp32 AdamW 的优化器状态要 2.36GB（两个 fp32 的 m/v），
        总显存会涨到 6.7GB 超过物理显存 → 触发超售 → 训练慢 16 倍。
        8-bit 版本只要 0.59GB，是**避免超售的必需项**而非可选优化。
    """
    if kind == "adamw8bit":
        try:
            import bitsandbytes as bnb
        except ImportError as e:
            raise RuntimeError(
                "配置要求 adamw8bit，但 bitsandbytes 未安装。\n"
                "请安装：pip install bitsandbytes\n"
                "或改用 --optimizer adamw / adafactor"
            ) from e

        return bnb.optim.AdamW8bit(
            model.parameters(),
            lr=lr,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )

    if kind == "adafactor":
        # Adafactor 的优化器状态是分解的，显存占用极小，本可作为兜底方案。
        # 但常见的实现来自 transformers，而本项目明确不依赖 transformers
        # （transformers 会解析到 5.x 大版本，API 有变动风险）。
        # 实测 bitsandbytes 的 8-bit AdamW 在本机（Windows + py3.13）工作正常，
        # 所以暂时不需要这个兜底；真要用了再自己实现一个（约 50 行）。
        raise NotImplementedError(
            "adfactor 尚未实现。本项目默认用 adamw8bit，实测在本机可用。\n"
            "如果 bitsandbytes 出问题，可先用 --optimizer adamw 对照，"
            "但注意 fp32 AdamW 的优化器状态要多占 1.9GB，很可能触发显存超售。"
        )

    if kind == "adamw":
        # 普通 fp32 AdamW。显存占用大（优化器状态 2.36GB vs 8-bit 的 0.59GB），
        # 在 6GB 卡上很可能触发超售，仅用于对照实验。
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
            foreach=True,
        )

    raise ValueError(f"未知优化器: {kind}")


def benchmark_throughput(
    cfg: ModelConfig,
    tcfg,
    micro_batch: int,
    n_warmup: int = 2,
    n_steps: int = 8,
    use_optimizer: bool = True,
) -> dict:
    """实测一个配置下的训练吞吐。

    参数
    ----
    n_warmup : int
        预热步数。**前几步会明显偏慢**（CUDA kernel 首次编译、显存分配器还没稳定、
        cuDNN 还在选算法），必须丢弃，否则测出来的数偏低。
    n_steps : int
        正式计时的步数。取得太少会被噪声主导，取得太多则整个基准太慢。
        8~10 步是比较好的平衡。

    返回包含耗时、吞吐、显存的字典。
    """
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = build_model(cfg, "cuda")
    model.train()

    optimizer = None
    scaler = None
    if use_optimizer:
        optimizer = build_optimizer(model, tcfg.optimizer, tcfg.lr, tcfg)
        if tcfg.dtype == "fp16":
            # fp16 必须配 GradScaler：fp16 的动态范围窄（最小正规数约 6e-5），
            # 小梯度会直接下溢成 0，导致这些参数永远不更新。
            # GradScaler 先把 loss 放大再反传，把梯度推到 fp16 能表示的范围内，
            # 更新前再缩放回来。
            scaler = torch.amp.GradScaler("cuda")

    vocab = cfg.vocab_size
    seq = tcfg.seq_len

    def one_step() -> float:
        # 模拟真实训练：随机 token 的 loss 会围绕 ln(vocab) 波动，
        # 和真实数据的计算量完全一致（计算量只取决于形状，与内容无关）
        idx = torch.randint(0, vocab, (micro_batch, seq), device="cuda")
        with torch.autocast("cuda", dtype=torch.float16, enabled=(tcfg.dtype == "fp16")):
            out = model(idx, labels=idx, use_checkpoint=tcfg.gradient_checkpointing)
            loss = out["loss"]

        if scaler is not None:
            scaler.scale(loss).backward()
            if optimizer is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if optimizer is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        return loss.item()

    result: dict = {"micro_batch": micro_batch, "ok": False}

    try:
        for _ in range(n_warmup):
            one_step()
        torch.cuda.synchronize()

        t0 = time.time()
        losses = [one_step() for _ in range(n_steps)]
        torch.cuda.synchronize()
        dt = time.time() - t0

        sec_per_step = dt / n_steps
        tokens_per_step = micro_batch * seq
        tps = tokens_per_step / sec_per_step

        m = gpu_memory()
        # 用 max_allocated 判断是否超售，这里比 reserved 更贴近"模型真实需求"
        peak = torch.cuda.max_memory_allocated() / 2**30
        reserved = torch.cuda.max_memory_reserved() / 2**30

        result.update(
            ok=True,
            sec_per_step=sec_per_step,
            tokens_per_sec=tps,
            peak_allocated=peak,
            peak_reserved=reserved,
            loss=sum(losses) / len(losses),
            free_before=m["free"],
        )

    except torch.cuda.OutOfMemoryError as e:
        result["error"] = f"显存不足: {str(e)[:120]}"
    except RuntimeError as e:
        result["error"] = str(e)[:160]
    finally:
        del model, optimizer, scaler
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    return result


def run_benchmark(cfg: ModelConfig, tcfg) -> dict:
    """跑几组不同 micro_batch 的对照，找出可用配置。"""
    print("\n" + "=" * 72)
    print("第 4 步：吞吐基准测试（决策门）")
    print("=" * 72)
    print(f"  模型     : {cfg.num_params()/1e6:.1f}M 参数")
    print(f"  序列长度 : {tcfg.seq_len}")
    print(f"  精度     : {tcfg.dtype}")
    print(f"  优化器   : {tcfg.optimizer}")
    print(f"  梯度检查点: {tcfg.gradient_checkpointing}")
    print(f"  梯度累积 : {tcfg.grad_accum_steps}  (有效 batch = "
          f"{tcfg.micro_batch_size * tcfg.grad_accum_steps * tcfg.seq_len:,} tokens/step)")
    print()

    free_gb = gpu_memory()["free"]

    # 对照测几个 micro_batch，让"超售悬崖"变得可见 ——
    # 实测中 micro_batch 从 4 加到 8，吞吐会腰斩而不是翻倍，
    # 这个反直觉的现象只有实测才能发现
    candidates = sorted({2, tcfg.micro_batch_size, tcfg.micro_batch_size * 2})

    results = []
    for mb in candidates:
        if mb * tcfg.seq_len * cfg.d_model * 4 > free_gb * 2**30 * 8:
            # 粗筛：明显不可能放下的直接跳过，省时间
            continue
        print(f"  测试 micro_batch={mb} ...", end="", flush=True)
        r = benchmark_throughput(cfg, tcfg, micro_batch=mb)
        results.append(r)

        if r["ok"]:
            oversold = "  <== 显存超售！" if r["peak_reserved"] > free_gb else ""
            print(f"  {r['sec_per_step']*1000:7.0f} ms/step | "
                  f"{r['tokens_per_sec']:7.0f} tok/s | "
                  f"峰值 {r['peak_allocated']:.2f} GB "
                  f"(持有 {r['peak_reserved']:.2f} GB){oversold}")
        else:
            print(f"  失败: {r.get('error', '未知错误')}")

    ok_results = [r for r in results if r["ok"]]
    best = max(ok_results, key=lambda r: r["tokens_per_sec"]) if ok_results else None

    if best:
        print(f"\n  最佳配置: micro_batch={best['micro_batch']}, "
              f"{best['tokens_per_sec']:.0f} tok/s")

        # 用最佳吞吐推算真实的 ETA，这是整个基准测试的最终目的
        total_tokens = tcfg.tokens_per_step * tcfg.max_steps
        hours = total_tokens / best["tokens_per_sec"] / 3600
        print(f"\n  {"="*64}")
        print(f"  ETA 推算")
        print(f"  {"="*64}")
        print(f"    当前总 token 预算   : {human_int(total_tokens)}")
        print(f"    最佳实测吞吐        : {best['tokens_per_sec']:.0f} tok/s")
        print(f"    → 预计预训练耗时    : {hours:.1f} 小时  ({hours/24:.1f} 天)")
        if hours > 24 * 10:
            print(f"\n    [!] 超过 10 天，建议调整：")
            reduced = total_tokens * (24 * 6) / hours
            print(f"        把 token 预算降到约 {human_int(reduced)}，约 6 天可完成")
            print(f"        改动方式：调小 configs/pretrain.yaml 里的 max_steps")
        else:
            print(f"\n    [OK] 在可接受范围内（方案预估 4~8 天）")
    else:
        print("\n  [错误] 没有任何配置能成功运行，请检查上面的报错。")

    return {"results": results, "best": best}


# ===========================================================================
# 第 5 步：显存估算
# ===========================================================================
def show_memory_estimate(cfg: ModelConfig, tcfg) -> None:
    """打印显存占用估算表，让超售风险变得可见。"""
    print("\n" + "=" * 72)
    print("第 5 步：显存占用估算")
    print("=" * 72)

    for mb in (4, 8):
        est = estimate_training_memory(cfg, micro_batch=mb, seq_len=tcfg.seq_len)
        print(f"\n  micro_batch={mb}:")
        print(f"    权重(fp32)            {est['权重(fp32)']:6.2f} GB")
        print(f"    梯度(fp32)            {est['梯度(fp32)']:6.2f} GB")
        print(f"    优化器 8-bit          {est['优化器(8bit)']:6.2f} GB   <-- 用这个")
        print(f"    优化器 fp32(对照)     {est['优化器(fp32)']:6.2f} GB   <-- 会超售，不要用")
        print(f"    激活(开梯度检查点)     {est['激活(开检查点)']:6.2f} GB")
        print(f"    激活(关梯度检查点)     {est['激活(关检查点)']:6.2f} GB   <-- 会超售")
        print(f"    固定开销              {est['固定开销']:6.2f} GB")
        print(f"    {'-'*44}")
        print(f"    合计 (8bit + 检查点)  {est['合计_8bit_开检查点']:6.2f} GB   <-- 本项目配置")
        print(f"    合计 (fp32 + 检查点)  {est['合计_fp32_开检查点']:6.2f} GB   <-- 超售风险")

    m = gpu_memory()
    print(f"\n  当前空闲显存: {m['free']:.2f} GB")
    print(f"  说明：估算值是数量级参考，实际占用以基准测试里的'峰值'为准。")


# ===========================================================================
# 主流程
# ===========================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description="阶段 0：环境自检 + 吞吐基准")
    parser.add_argument("--model-config", default=str(PROJECT_ROOT / "configs" / "model_0.3b.yaml"))
    parser.add_argument("--train-config", default=str(PROJECT_ROOT / "configs" / "pretrain.yaml"))
    parser.add_argument("--quick", action="store_true", help="跳过基准测试，只做环境和模型检查")
    args = parser.parse_args()

    print("=" * 72)
    print("  Mini_LM 阶段 0：环境自检 + 吞吐基准")
    print("=" * 72)

    # 环境
    if not check_environment():
        print("\n[中止] 环境检查未通过。")
        return 1

    # GPU
    if not check_gpu():
        print("\n[中止] GPU 检查未通过。")
        return 1

    # 配置
    print("\n" + "=" * 72)
    print("配置加载")
    print("=" * 72)
    cfg = load_model_config(args.model_config)
    tcfg = load_train_config(args.train_config)
    print(f"  模型配置 : {args.model_config}")
    print(f"  训练配置 : {args.train_config}")
    print(f"  有效 batch: {tcfg.tokens_per_step:,} tokens/step")
    print(f"  总 token  : {human_int(tcfg.total_tokens)}")
    print(f"  总步数    : {tcfg.max_steps:,}")

    set_seed(tcfg.seed)
    force_mem_efficient_attention()

    # 模型
    model_ok, _ = check_model(cfg)
    if not model_ok:
        print("\n[中止] 模型结构与配置不一致。")
        return 1

    if args.quick:
        print("\n[--quick] 跳过基准测试。")
        return 0

    # 显存估算
    show_memory_estimate(cfg, tcfg)

    # 基准
    check_vram_available(5.0, raise_on_fail=False)
    run_benchmark(cfg, tcfg)

    print("\n" + "=" * 72)
    print("  阶段 0 完成")
    print("=" * 72)
    print("  下一步：python scripts/01_train_tokenizer.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
