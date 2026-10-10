"""
通用工具：随机种子、显存监控、日志、checkpoint 管理、格式化。

这些看起来琐碎，但本项目有两件事必须靠它们兜住：
    1. **显存监控** —— 6GB 卡上超售会让训练慢 16 倍，而且是静默的（不报错，只是慢）。
       必须在训练前和训练中主动检查，不能等它自己暴露。
    2. **checkpoint 管理** —— 预训练要连跑约 5 天，断电/重启是大概率事件。
       checkpoint 和断点续训是必需品而不是加分项。
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ===========================================================================
# 随机种子
# ===========================================================================
def set_seed(seed: int, deterministic: bool = False) -> None:
    """固定所有随机源，让实验可复现。

    参数
    ----
    seed : int
        随机种子。
    deterministic : bool
        是否强制 CUDA 使用确定性算法。
        **默认关闭**，因为确定性算法通常更慢，而且有些算子根本没有确定性实现
        （会直接报错）。只在排查"两次运行结果不一致"这类问题时才打开。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # benchmark=True 让 cuDNN 自动搜索最快的卷积算法。
        # 注意：会让每次运行的数值略有不同（算法选择可能变），这是可复现性的代价。
        torch.backends.cudnn.benchmark = True


# ===========================================================================
# 显存监控 —— 本项目最重要的工具
# ===========================================================================
def gpu_memory() -> dict[str, float]:
    """返回当前 GPU 显存状态，单位 GB。

    区分三个概念（很容易混淆，但对本项目至关重要）：

        total     : 显卡物理显存总量（6GB）
        free      : 驱动层面还剩多少没被任何进程占用
                    —— **这是判断会不会超售的依据**
        allocated : PyTorch 当前实际用着的张量内存
        reserved  : PyTorch 向驱动申请下来、自己管理的总池子
                    —— 通常大于 allocated，多出来的是缓存以备复用

    **判断会不会显存超售，要看 `reserved` 和 `free` 的关系**，
    而不是 allocated。因为 PyTorch 会持有 reserved 不放，即使张量已经释放。
    """
    if not torch.cuda.is_available():
        return {"total": 0.0, "free": 0.0, "used": 0.0, "allocated": 0.0, "reserved": 0.0}

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    gib = 2**30
    return {
        "total": total_bytes / gib,
        "free": free_bytes / gib,
        "used": (total_bytes - free_bytes) / gib,
        "allocated": torch.cuda.memory_allocated() / gib,
        "reserved": torch.cuda.memory_reserved() / gib,
        "max_allocated": torch.cuda.max_memory_allocated() / gib,
        "max_reserved": torch.cuda.max_memory_reserved() / gib,
    }


def format_gpu_memory(m: dict[str, float] | None = None) -> str:
    """把显存状态格式化成一行可读文本。"""
    m = m or gpu_memory()
    return (
        f"显存: 空闲 {m['free']:.2f} / 总 {m['total']:.2f} GB | "
        f"PyTorch 已用 {m['allocated']:.2f} GB, 持有 {m['reserved']:.2f} GB"
    )


def check_vram_available(required_gb: float, raise_on_fail: bool = False) -> bool:
    """训练前检查空闲显存是否够用。

    为什么需要这个：
        PyTorch 不会因为显存不够就报错 —— 在 Windows(WDDM) 上它会**默认允许超售**，
        把一部分数据换到内存里，然后训练速度暴跌 3~16 倍，但**一声不吭**。
        本机实测的惨痛案例：显存超售时吞吐 305 tok/s，不超售时 4786 tok/s。

        所以必须在开跑前主动检查，把"静默变慢"变成"明确告警"。

    参数
    ----
    required_gb : float
        本次训练预计需要的显存（看 `max_reserved`，不是 allocated）。
    raise_on_fail : bool
        不够时是抛异常还是只打印警告。默认只警告 ——
        因为估算总是有误差，直接中断可能误杀本来能跑的配置。

    返回
    ----
    bool
        显存是否充足。
    """
    m = gpu_memory()
    ok = m["free"] >= required_gb
    if not ok:
        msg = (
            f"\n{'!'*70}\n"
            f"显存不足警告：需要约 {required_gb:.2f} GB，但当前空闲只有 {m['free']:.2f} GB。\n"
            f"（总显存 {m['total']:.2f} GB，已被其他程序占用 {m['used']:.2f} GB）\n\n"
            f"后果：Windows 会静默启用显存超售，训练速度可能下降 3~16 倍。\n"
            f"实测对比：不超售 4786 tok/s，超售 305 tok/s。\n\n"
            f"建议：关闭浏览器、聊天软件、其他 CUDA 程序后重试。\n"
            f"{'!'*70}"
        )
        if raise_on_fail:
            raise RuntimeError(msg)
        print(msg, file=sys.stderr)
    return ok


def human_int(n: float) -> str:
    """把大数字格式化成易读形式：1234567 -> '1.23M'。"""
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n/div:.2f}{unit}"
    return f"{n:.0f}"


# ===========================================================================
# 日志
# ===========================================================================
def get_logger(name: str = "minilm", log_file: str | Path | None = None) -> logging.Logger:
    """创建一个同时输出到控制台和文件的 logger。

    写文件是为了长跑实验的可追溯性 —— 训练 5 天后想回看当时的 loss 曲线，
    光靠终端滚动缓冲区是不够的。

    注意：这里显式指定 encoding="utf-8"，否则 Windows 下默认 GBK
    写中文注释或特殊符号时会报 UnicodeEncodeError。
    """
    logger = logging.getLogger(name)
    if logger.handlers:  # 避免重复添加 handler 导致日志打印多遍
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        fmt="[%(asctime)s] %(message)s",
        datefmt="%m-%d %H:%M:%S",
    )

    # 控制台
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # 文件
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


class AverageMeter:
    """滑动平均累加器，用来统计 loss、吞吐这类逐 step 波动的指标。

    为什么要用滑动平均而不是直接看单步值：
        单步 loss 会因为数据难度不同而剧烈波动，肉眼很难判断趋势。
        看最近 N 步的平均值才能判断训练是否正常。
    """

    def __init__(self, window: int = 100) -> None:
        self.window = window
        self.reset()

    def reset(self) -> None:
        self.values: list[float] = []
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.values.append(value)
        # 只保留最近 window 个值，避免长跑时列表无限增长
        if len(self.values) > self.window:
            self.values.pop(0)
        self.total += value * n
        self.count += n

    @property
    def avg(self) -> float:
        """全局平均（从 reset 至今）。"""
        return self.total / self.count if self.count else 0.0

    @property
    def recent(self) -> float:
        """最近 window 步的滑动平均。训练监控主要看这个。"""
        return sum(self.values) / len(self.values) if self.values else 0.0


class ThroughputTracker:
    """跟踪训练吞吐（tokens/s），用来实时发现性能退化。

    这东西的实际价值：如果训练中途因为显存碎片化触发了超售，
    loss 看起来一切正常，只有吞吐会掉。有个显式的吞吐指标就能立刻发现。
    """

    def __init__(self) -> None:
        self.t0 = time.time()
        self.tokens = 0

    def update(self, n_tokens: int) -> None:
        self.tokens += n_tokens

    @property
    def tokens_per_sec(self) -> float:
        dt = time.time() - self.t0
        return self.tokens / dt if dt > 0 else 0.0

    def reset(self) -> None:
        self.t0 = time.time()
        self.tokens = 0

    def eta_hours(self, remaining_tokens: int) -> float:
        """按当前吞吐估算剩余训练时间（小时）。"""
        tps = self.tokens_per_sec
        return (remaining_tokens / tps / 3600) if tps > 0 else float("inf")


# ===========================================================================
# Checkpoint 管理
# ===========================================================================
def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    """保存 checkpoint，支持断点续训。

    保存内容：
        model       : 模型权重
        optimizer   : 优化器状态（**续训必需**，否则动量信息丢失，loss 会跳一下）
        step        : 当前步数
        extra       : 配置、loss 等附加信息

    用 `torch.save(..., _use_new_zipfile_serialization=True)` 的默认格式。
    这里没有用 safetensors，因为 safetensors 不支持保存优化器状态和嵌套结构，
    而断点续训必须要这些。

    写入采用"先写临时文件再原子重命名"的策略：
    如果写到一半断电，不会留下一个损坏的 checkpoint 覆盖掉上一个好的。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 未包裹的模型（比如 DDP 或 compile 过的）需要先取出原始模块
    model_to_save = getattr(model, "_orig_mod", model)

    payload = {
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "step": step,
        "extra": extra or {},
    }

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    # 原子替换：os.replace 在 Windows 和 Linux 上都是原子的
    os.replace(tmp, path)
    return path


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str = "cuda",
    strict: bool = True,
) -> dict[str, Any]:
    """加载 checkpoint，返回其中的 step 和 extra 信息。

    参数
    ----
    strict : bool
        是否要求 state_dict 严格匹配。默认 True ——
        如果加载的 checkpoint 和当前模型结构不一致（比如改了层数却忘了改配置），
        严格模式会立刻报错，而不是悄悄加载一半权重继续训练。
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint 不存在: {path.resolve()}")

    payload = torch.load(path, map_location=map_location, weights_only=False)

    model_to_load = getattr(model, "_orig_mod", model)
    model_to_load.load_state_dict(payload["model"], strict=strict)

    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])

    return {
        "step": payload.get("step", 0),
        "extra": payload.get("extra", {}),
    }


def find_latest_checkpoint(ckpt_dir: str | Path, pattern: str = "step_*.pt") -> Path | None:
    """在目录里找步数最大的 checkpoint，用于自动续训。

    自动找到最新的一个，这样重启训练只需要加 `--resume` 而不必手动指定路径 ——
    对于一个要跑 5 天、可能中断多次的任务，减少人工步骤很重要。
    """
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return None

    candidates: list[tuple[int, Path]] = []
    for p in ckpt_dir.glob(pattern):
        # 文件名形如 step_001500.pt，从中解析出步数
        stem = p.stem  # "step_001500"
        try:
            step = int(stem.split("_")[-1])
        except ValueError:
            continue
        candidates.append((step, p))

    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])[1]


def rotate_checkpoints(ckpt_dir: str | Path, keep_last_n: int, pattern: str = "step_*.pt") -> list[Path]:
    """只保留最近 N 个 checkpoint，删掉更旧的，防止磁盘被写满。

    2B tokens 的训练会存 30+ 个 checkpoint，每个约 1.3GB（模型+优化器状态），
    不清理会吃掉 40GB 磁盘。这个函数是长时间训练的必需品。

    返回被删除的文件列表。
    """
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return []

    candidates: list[tuple[int, Path]] = []
    for p in ckpt_dir.glob(pattern):
        try:
            step = int(p.stem.split("_")[-1])
        except ValueError:
            continue
        candidates.append((step, p))

    if len(candidates) <= keep_last_n:
        return []

    candidates.sort(key=lambda x: x[0])
    to_delete = candidates[:-keep_last_n]

    deleted = []
    for _, p in to_delete:
        p.unlink()
        deleted.append(p)
    return deleted


# ===========================================================================
# 其他小工具
# ===========================================================================
def count_lines(path: str | Path) -> int:
    """快速统计文本文件行数（二进制方式读，避免编码问题）。"""
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def write_json(path: str | Path, data: Any) -> None:
    """写 JSON，统一用 UTF-8 + 不转义中文。

    `ensure_ascii=False` 很重要：否则中文会被写成 \\uXXXX 转义，
    文件肉眼不可读，调试数据管线时会很痛苦。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path: str | Path) -> Any:
    """读 JSON 文件。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
