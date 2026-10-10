"""
配置系统：把模型结构和训练超参从代码里剥离出来，全部放进 YAML。

为什么要有这一层？
    本项目的硬件环境很紧（RTX 2060 6GB），显存超售会让训练慢 16 倍。
    因此在 6GB 卡上调参是常态而非例外 —— 如果超参写死在代码里，
    每次调整都要改代码、容易改错、也无法记录"当时用的是哪套配置"。
    把配置抽成 YAML 后，调参只改文件，且配置文件本身可以作为实验记录留存。

设计取舍：
    用 dataclass 而不是直接拿 dict 用，是为了**在启动阶段就报错**。
    比如把 n_heads 写成 16 而 d_model 写成 1024、head_dim 却是 48，
    这种错误如果在训练了 3 小时之后才因为 shape 不匹配崩掉，代价很大。
    dataclass + __post_init__ 校验能在加载配置的瞬间就拦住。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# 模型结构配置
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    """定义 LLaMA 风格 decoder-only 模型的结构超参。

    默认值对应本项目实测的 316.7M 参数配置（见 README 的"模型设计"一节）。
    """

    # ---- 词表与维度 ----
    vocab_size: int = 32768
    """词表大小。含预留的特殊 token。必须是 65536 以内，因为训练数据用 uint16 存储。"""

    d_model: int = 1024
    """隐藏层维度。Qwen2-0.5B 用 896，我们取 1024 让 head_dim 正好是 64。"""

    n_layers: int = 24
    """Transformer block 层数。"""

    n_heads: int = 16
    """Query 的注意力头数。d_model / n_heads = head_dim。"""

    n_kv_heads: int = 8
    """Key/Value 的注意力头数（GQA）。

    必须能整除 n_heads。取 n_heads 的一半即 2:1 的 GQA 比例：
      - 推理时 KV cache 直接减半，对 6GB 卡很关键
      - 训练时省注意力激活
      - 后续 RL 阶段要同时把 policy 和 reference 两个模型装进显存，
        没有 GQA 几乎不可能放下
    """

    intermediate_size: int = 2816
    """SwiGLU 前馈网络的中间层维度。

    SwiGLU 有 3 个矩阵（gate/up/down），而传统 MLP 只有 2 个。
    所以为了参数量对齐，SwiGLU 的中间维度通常取传统 4*d_model 的 2/3 左右：
        4 * 1024 * 2/3 ≈ 2731，向上取到 256 的整数倍即 2816。
    """

    # ---- 序列与位置编码 ----
    max_seq_len: int = 1024
    """训练时的最大序列长度。

    实测显存允许更长（mem_efficient 注意力下 seq=2048 的注意力激活显存几乎不变），
    但计算量随 seq 线性增长，预训练先用 1024 换取速度，SFT 阶段可再提到 2048。
    """

    rope_theta: float = 10000.0
    """RoPE 的基底频率。值越大，位置编码的波长越长、越利于长度外推。"""

    # ---- 归一化与正则 ----
    rms_norm_eps: float = 1e-5
    """RMSNorm 的数值稳定项。开方前加在均方上，防止除零。"""

    dropout: float = 0.0
    """dropout 概率。预训练 2B tokens 的规模下不需要 dropout，设 0 还能省一点开销。"""

    # ---- 初始化 ----
    initializer_range: float = 0.02
    """权重初始化的标准差。GPT-2 以来的惯例值。"""

    # ---- 其他 ----
    tie_word_embeddings: bool = True
    """是否让输出层和输入 embedding 共享权重。

    对 0.3B 这种小模型很重要：embedding 占 33.6M 参数（总量的 10%），
    共享能省下这部分，同时通常还能略微提升效果（相当于一种正则）。
    """

    def __post_init__(self) -> None:
        """加载后立即校验，把错误拦在训练开始之前。"""
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model({self.d_model}) 必须能被 n_heads({self.n_heads}) 整除，"
                f"否则无法均匀切分注意力头"
            )
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError(
                f"n_heads({self.n_heads}) 必须能被 n_kv_heads({self.n_kv_heads}) 整除，"
                f"否则 GQA 的 KV 头无法平摊到 Q 头上"
            )
        if self.vocab_size > 65536:
            raise ValueError(
                f"vocab_size({self.vocab_size}) 超过 65536，uint16 存不下训练数据。"
                f"要么缩小词表，要么把 build_bin.py 里的 dtype 改成 uint32"
            )

    @property
    def head_dim(self) -> int:
        """每个注意力头的维度。"""
        return self.d_model // self.n_heads

    @property
    def n_rep(self) -> int:
        """每个 KV 头要被多少个 Q 头共享（GQA 的重复倍数）。"""
        return self.n_heads // self.n_kv_heads

    def num_params(self, exclude_embedding: bool = False) -> int:
        """解析式计算参数量，用来在实例化模型前做交叉校验。

        这里手算一遍而不是只信模型的实测值，是为了在写错结构时能立刻发现：
        如果解析式和实测值对不上，说明实现和设计不一致。

        注意 tied embedding 时输出头不额外占参数。
        """
        # embedding 层
        emb = self.vocab_size * self.d_model
        if exclude_embedding:
            emb = 0

        # 每层的注意力：Q 全量，K/V 只有 n_kv_heads 份，O 是全量
        hd = self.head_dim
        attn = (
            self.d_model * (self.n_heads * hd)      # q_proj
            + self.d_model * (self.n_kv_heads * hd)  # k_proj
            + self.d_model * (self.n_kv_heads * hd)  # v_proj
            + (self.n_heads * hd) * self.d_model     # o_proj
        )
        # 每层的 SwiGLU：gate / up / down 三个矩阵
        mlp = 3 * self.d_model * self.intermediate_size
        # 每层两个 RMSNorm 的权重（注意力前一个、前馈前一个）
        norms = 2 * self.d_model
        # 再加上输出前的最后一个 RMSNorm。
        # 踩过的坑：一开始漏算了它，导致解析式比实测少了正好 d_model 个参数。
        # 正是"解析式 vs 实测"这个自检把问题暴露出来的。
        final_norm = self.d_model

        return emb + self.n_layers * (attn + mlp + norms) + final_norm


# ---------------------------------------------------------------------------
# 训练配置
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    """训练循环的超参。预训练 / SFT / RL 共用大部分字段。"""

    # ---- 批次与序列 ----
    micro_batch_size: int = 4
    """单次前向的样本数。

    **这个值是实测出来的硬约束，不要随意调大。**
    本机实测：micro_batch=4 时显存 reserved 4.51GB，正好放进 6GB 卡，吞吐 4786 tok/s；
    调到 8 则 reserved 5.89GB 触发 Windows WDDM 显存超售，吞吐腰斩到 2595 tok/s。
    """

    grad_accum_steps: int = 32
    """梯度累积步数。micro_batch * grad_accum * seq_len 即每次参数更新的实际 token 数。"""

    seq_len: int = 1024
    """训练序列长度。必须 <= ModelConfig.max_seq_len。"""

    # ---- 优化 ----
    lr: float = 3e-4
    min_lr_ratio: float = 0.1
    """cosine 衰减的终点学习率 = lr * min_lr_ratio。不衰减到 0 通常更稳定。"""

    warmup_steps: int = 2000
    """学习率线性预热步数。0.3B 模型取总步数的 ~13% 比较稳。"""

    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    """AdamW 的 beta。beta2 用 0.95 而非 0.999 是大模型训练的常见选择，
    能更快响应梯度的变化。"""

    grad_clip: float = 1.0
    """梯度裁剪阈值。fp16 训练必须开，防止偶发的梯度爆炸把权重打飞。"""

    optimizer: str = "adamw8bit"
    """优化器类型：adamw8bit / adamw / adafactor。

    **默认必须是 8bit**：普通 fp32 AdamW 的优化器状态要 2.54GB，
    总显存会涨到 6.7GB 超过物理显存，触发超售后训练慢 16 倍（实测 305 tok/s）。
    """

    # ---- 精度 ----
    dtype: str = "fp16"
    """训练精度。RTX 2060 是 sm_75，**bf16 没有硬件加速**，必须用 fp16 + GradScaler。

    虽然实测 bf16 的矩阵乘"能跑"，但那是 PyTorch 走模拟路径的结果，速度远不如 fp16。
    """

    # ---- 显存优化 ----
    gradient_checkpointing: bool = True
    """是否用梯度检查点（时间换显存）。

    **这是必需项不是可选项。** 实测关掉后连 micro_batch=2 都会超售
    （reserved 6.54GB），吞吐从 4985 掉到 863 tok/s。
    原理：不保存中间激活，反向传播时重新算一遍前向，用 ~33% 的额外计算换掉大量激活显存。
    """

    # ---- 训练规模 ----
    max_steps: int = 15260
    """总优化步数。2B tokens / (4 * 32 * 1024) = 15260 步。"""

    # ---- 日志与保存 ----
    log_interval: int = 10
    eval_interval: int = 1000
    save_interval: int = 500
    """checkpoint 间隔。桌面机跑 5 天，断电是大概率事件，500 步存一次是安全网。"""

    keep_last_n_checkpoints: int = 3
    """滚动保留最近 N 个 checkpoint，防止磁盘被写满。"""

    seed: int = 1337
    out_dir: str = "checkpoints/pretrain"
    run_name: str = "pretrain_0.3b"

    def __post_init__(self) -> None:
        if self.grad_accum_steps < 1:
            raise ValueError("grad_accum_steps 必须 >= 1")
        if self.dtype not in ("fp16", "bf16", "fp32"):
            raise ValueError(f"不支持的 dtype: {self.dtype}")

    @property
    def tokens_per_step(self) -> int:
        """每次参数更新实际消费的 token 数。"""
        return self.micro_batch_size * self.grad_accum_steps * self.seq_len

    @property
    def total_tokens(self) -> int:
        """整个训练计划消费的总 token 数。"""
        return self.tokens_per_step * self.max_steps

    def cosine_lr(self, step: int) -> float:
        """按 step 计算当前学习率（warmup + cosine 衰减）。

        直接写在配置类里，是因为训练循环和可视化都要用同一个公式，
        分开实现容易出现两处不一致、导致日志里的 lr 曲线是错的。

        参数
        ----
        step : int
            当前的优化步数（从 0 开始）。
        """
        # 预热阶段：从 0 线性升到 lr
        if step < self.warmup_steps:
            return self.lr * (step + 1) / self.warmup_steps

        # 预热结束：cosine 从 lr 平滑降到 lr * min_lr_ratio
        if step >= self.max_steps:
            return self.lr * self.min_lr_ratio

        progress = (step - self.warmup_steps) / max(1, self.max_steps - self.warmup_steps)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.lr * self.min_lr_ratio + (self.lr - self.lr * self.min_lr_ratio) * coeff


# ---------------------------------------------------------------------------
# 配置加载工具
# ---------------------------------------------------------------------------
def _filter_known_fields(cls: type, data: dict[str, Any]) -> dict[str, Any]:
    """从 dict 里挑出目标 dataclass 认识的字段，忽略多余的。

    这样做是为了让 YAML 可以写注释性的额外键，而不必和 dataclass 严格一一对应。
    但**未知字段会被打印出来提醒**，避免把 lr 拼成 learning_rate 这种错误静默吞掉。
    """
    known = {f.name for f in cls.__dataclass_fields__.values()}
    unknown = set(data) - known
    if unknown:
        print(f"[配置警告] {cls.__name__} 收到未知字段，已忽略: {sorted(unknown)}")
    return {k: v for k, v in data.items() if k in known}


def load_yaml(path: str | Path) -> dict[str, Any]:
    """读一个 YAML 文件成 dict。文件不存在时给出清晰报错。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path.resolve()}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是字典: {path}")
    return data


def load_model_config(path: str | Path) -> ModelConfig:
    """从 YAML 加载模型配置。"""
    return ModelConfig(**_filter_known_fields(ModelConfig, load_yaml(path)))


def load_train_config(path: str | Path) -> TrainConfig:
    """从 YAML 加载训练配置。

    特殊处理：YAML 里的 betas 会读成 list，但 dataclass 声明的是 tuple。
    torch 两者都接受，统一转成 tuple 是为了让 asdict() 的输出稳定可比。
    """
    data = _filter_known_fields(TrainConfig, load_yaml(path))
    if "betas" in data and isinstance(data["betas"], list):
        data["betas"] = tuple(data["betas"])
    return TrainConfig(**data)


def save_config(cfg: ModelConfig | TrainConfig, path: str | Path) -> None:
    """把配置写回 YAML，用于在每个 checkpoint 里留一份"当时的配置"。

    这样几周后回头看某个 checkpoint，能确切知道它是用什么配置训出来的 ——
    否则做消融实验时会完全分不清哪个模型对应哪套超参。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, allow_unicode=True, sort_keys=False)
