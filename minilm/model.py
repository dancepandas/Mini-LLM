"""
MiniLM 模型定义 —— LLaMA 风格的 decoder-only Transformer。

这个文件是整个项目的核心，也是学习价值最高的部分。它自己实现了：
    - RMSNorm 归一化
    - RoPE 旋转位置编码
    - GQA（分组查询注意力）
    - SwiGLU 前馈网络
    - 梯度检查点

**为什么不用 transformers 的 AutoModel？**
    1. 本项目的目的就是"从零走一遍"，调库就失去意义了
    2. 需要完全掌控显存：6GB 卡上每一 MB 都要精打细算，
       而 transformers 的封装会带来我们不需要的开销和不可控的默认行为
    3. transformers 即将解析到 5.x 大版本，API 有变动风险

**硬件约束（实测，写在最前面提醒）**：
    目标设备是 RTX 2060 6GB（sm_75），它：
      - 有 fp16 张量核，但**没有 bf16 张量核** → 必须用 fp16
      - 不支持 FlashAttention-2 → 必须用 PyTorch SDPA 的 mem_efficient 后端
      - mem_efficient 后端**默认不保证被选中**，不显式指定会静默回退到 math 后端，
        导致注意力显存从 145MB 暴涨到 2192MB（seq=1024）→ 直接 OOM
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as _ckpt

from .config import ModelConfig


# ===========================================================================
# 注意力后端配置 —— 本文件最容易踩坑的地方，放在最前面强制执行
# ===========================================================================
def force_mem_efficient_attention() -> None:
    """强制 SDPA 使用 mem_efficient 后端。

    这是**必须调用**的初始化步骤。实测数据（seq=1024, batch=8, fp16, 前向+反向）：

        后端            峰值显存
        math            2192 MB      <-- 会直接把 6GB 卡打爆
        mem_efficient    145 MB      <-- 相差 15 倍
        flash           不可用（本 PyTorch 构建未编译）
        cudnn           不可用（仅支持 sm80+）

    而且实测中发现默认路径会打印
        "Memory Efficient attention has been runtime disabled"
    也就是说**默认不保证选中 mem_efficient**。一旦静默回退到 math，
    注意力会实体化完整的 seq×seq 矩阵，显存从 O(seq) 变成 O(seq²)。

    这个函数把后端选择显式钉死，不留任何依赖默认行为的余地。
    """
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_flash_sdp(False)   # 本构建未编译，关掉避免混淆
    torch.backends.cuda.enable_cudnn_sdp(False)   # 仅 sm80+ 支持，sm_75 用不了
    torch.backends.cuda.enable_math_sdp(True)     # 保留作兜底，但不希望被用到


def assert_mem_efficient_active(verbose: bool = True) -> bool:
    """断言 mem_efficient 后端真的可用，而不是"配置了但实际没生效"。

    为什么需要这个：`enable_xxx_sdp(True)` 只是**允许**该后端，
    实际能否使用还取决于硬件和输入形状。配置成功但运行时不生效，
    正是本项目最容易出现的静默失败。

    返回
    ----
    bool
        真正可用的情况返回 True；否则抛异常（因为静默降级到 math 会 OOM，
        宁可现在就崩，也不要训练到一半才发现）。
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel

    # 构造一个符合 mem_efficient 要求的 4D 张量做真实探测。
    # 注意必须是 4D：(batch, heads, seq, head_dim)。
    # 传 3D 会报 "All fused kernels requires query, key and value to be 4 dimensional"，
    # 那是形状错误而不是后端不可用 —— 别把这个报错误判成"后端不支持"。
    q = torch.randn(1, 2, 64, 32, device="cuda", dtype=torch.float16)
    try:
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            F.scaled_dot_product_attention(q, q, q)
    except Exception as e:  # noqa: BLE001 - 这里就是要捕获所有异常并转成明确的错误
        raise RuntimeError(
            f"mem_efficient 注意力后端不可用！\n"
            f"原因: {type(e).__name__}: {e}\n"
            f"如果继续训练，SDPA 会回退到 math 后端，注意力显存涨 15 倍，必然 OOM。"
        ) from e

    if verbose:
        print("[注意力] mem_efficient 后端已确认可用 ✓")
    return True


# ===========================================================================
# 基础组件
# ===========================================================================
class RMSNorm(nn.Module):
    """均方根归一化（Root Mean Square Layer Normalization）。

    与 LayerNorm 的区别：不减均值、不学 bias，只做 x / sqrt(mean(x²) + eps) * weight。

    为什么用它：
        1. 比 LayerNorm 少一次均值计算和一次减法，更快
        2. 没有 bias，参数更少
        3. 对 fp16 更友好 —— LayerNorm 的减均值操作在 fp16 下容易损失精度，
           而 RMSNorm 省掉了这一步
        4. LLaMA / Qwen / DeepSeek 等主流模型都用它

    注意实现上的一个坑：计算必须在 fp32 下做，否则 fp16 下 x.pow(2) 容易溢出。
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        # 可学习的缩放系数，初始化为全 1（即初始时不做任何缩放）
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 关键：先升到 fp32 再算，避免 fp16 下平方溢出。
        # 计算完再降回原 dtype，这样不会让后续层被迫在 fp32 下跑。
        dtype = x.dtype
        x = x.float()
        # rsqrt 比 1/sqrt 快，且数值上更稳
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


class RoPE(nn.Module):
    """旋转位置编码（Rotary Position Embedding）。

    核心思想：不额外加位置向量，而是**把位置信息编码成对 q/k 向量的旋转**。
    位置 m 和 n 之间的相对关系自然体现在旋转角差 (m-n) 上，
    所以模型天然学到的是相对位置，外推能力比绝对位置编码好。

    实现要点：
        - cos/sin 表**预计算并缓存**。原始实现每次调用都重算 arange/outer/cos/sin，
          在每层每步都调用的情况下是纯粹的浪费。
        - 缓存按需扩展：先在 max_seq_len 上建表，遇到更长序列再扩容，
          这样生成阶段可以用比训练更长的序列。
    """

    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta

        # inv_freq[i] = 1 / theta^(2i/head_dim)，即每个频率分量的周期倒数。
        # 低频分量变化慢（负责长距离），高频分量变化快（负责近距离）。
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        # persistent=False：这是纯计算缓存，不需要存进 checkpoint 文件里，
        # 存了反而会让 checkpoint 变大、还可能因为设备不同而出错。
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # cos/sin 缓存**必须也注册成 buffer**，不能用普通属性。
        # 踩过的坑：普通属性不受 nn.Module.to(device) 管辖，
        # 把模型搬到 GPU 时它们会留在 CPU 上，前向时就会报
        # "Expected all tensors to be on the same device, but found at least two devices"。
        # 先注册成 None，稍后由 _build_cache 填入真实张量。
        self.register_buffer("_cos", None, persistent=False)
        self.register_buffer("_sin", None, persistent=False)

        self._built_len = 0
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        """构建 / 扩展 cos、sin 缓存表。"""
        if seq_len <= self._built_len:
            return

        # 位置索引 (seq_len,) 与频率 (head_dim/2,) 做外积，得到 (seq_len, head_dim/2)
        pos = torch.arange(seq_len, dtype=torch.float32, device=self.inv_freq.device)
        freqs = torch.outer(pos, self.inv_freq)
        # 把频率复制一份拼起来，使维度对上 head_dim：
        # [f0,f1,...,f31] -> [f0,f1,...,f31,f0,f1,...,f31]
        # 这样才能和按前后半段切分的 q/k 做逐元素相乘
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, head_dim)
        self._cos = emb.cos()
        self._sin = emb.sin()
        self._built_len = seq_len

    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        """对 q、k 施加旋转。

        参数
        ----
        q, k : torch.Tensor
            形状 (batch, heads, seq, head_dim)
        seq_len : int
            当前序列长度，用于按需扩展缓存
        """
        self._build_cache(seq_len)
        # 取前 seq_len 行并转成 q/k 的 dtype。
        # 缓存本体保持 fp32（精度更好），只在用时降精度。
        cos = self._cos[:seq_len].to(dtype=q.dtype)
        sin = self._sin[:seq_len].to(dtype=q.dtype)
        # 广播到 (1, 1, seq, head_dim)，与 (batch, heads, seq, head_dim) 对齐
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)

        return _apply_rope(q, cos, sin), _apply_rope(k, cos, sin)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """把向量的前后两半交换并给前半取负：[x1, x2] -> [-x2, x1]。

    这是二维旋转公式在拼接表示下的等价写法。
    一个二维向量 (x1, x2) 旋转角度 φ 后是 (x1·cosφ - x2·sinφ, x1·sinφ + x2·cosφ)，
    把它写成 x·cos + rotate_half(x)·sin 就能一次性处理完整个 head_dim 维向量。
    """
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """施加旋转：x_rotated = x * cos + rotate_half(x) * sin。"""
    return x * cos + _rotate_half(x) * sin


class Attention(nn.Module):
    """因果自注意力，带 GQA（分组查询注意力）。

    GQA 是什么：
        标准多头注意力里 Q/K/V 的头数相同（n_heads）。GQA 让 K/V 只用更少的头
        （n_kv_heads），每个 KV 头被多个 Q 头共享。这里 n_heads=16、n_kv_heads=8，
        即每 2 个 Q 头共享 1 组 K/V。

    为什么用 GQA：
        - **推理时 KV cache 直接减半** —— 对 6GB 显存是实打实的收益
        - 训练时 K/V 的投影参数和激活都减半
        - 后续 RL 阶段需要同时驻留 policy 和 reference 两个模型，
          没有 GQA 基本放不下
        - 效果损失很小，已经是 LLaMA-2/3、Qwen2/3、DeepSeek 的标准配置
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_rep  # 每个 KV 头被多少个 Q 头共享

        # 无 bias：省参数且训练更稳，是 LLaMA 以来的惯例
        self.q_proj = nn.Linear(cfg.d_model, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, cfg.d_model, bias=False)

        self.rope = RoPE(self.head_dim, cfg.max_seq_len, cfg.rope_theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape

        # 投影并拆成多头：(B, S, H*D) -> (B, S, H, D) -> (B, H, S, D)
        # SDPA 要求 4D 输入 (batch, heads, seq, head_dim)，
        # 传 3D 会触发 "All fused kernels requires ... 4 dimensional" 警告并退回非融合路径。
        q = self.q_proj(x).view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # 只对 q/k 加位置编码，v 不加（位置信息通过注意力权重起作用）
        q, k = self.rope(q, k, seq)

        # GQA：把 KV 头复制到和 Q 头一样的数量。
        # repeat_interleave 会实体化一份完整副本，但训练时这点显存可以忽略
        # （B=4, H=16, S=1024, D=64, fp16 约 8MB）。
        # 之所以不用 SDPA 的 enable_gqa 参数，是为了兼容性 —— 它在不同后端上支持不一。
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # is_causal=True 让 SDPA 内部生成因果掩码，
        # 比手工构造 (S, S) 的三角矩阵更省显存 —— 手工构造那份掩码本身就是 O(S²)。
        # 在 mem_efficient 后端的加持下，注意力本身的显存是 O(S) 而非 O(S²)。
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # 合并多头：(B, H, S, D) -> (B, S, H, D) -> (B, S, H*D)
        out = out.transpose(1, 2).reshape(batch, seq, self.n_heads * self.head_dim)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    """SwiGLU 前馈网络。

    结构：down(silu(gate(x)) * up(x))

    和传统 MLP 的区别：
        传统 MLP 是 up(act(x)) 两层（2 个矩阵），SwiGLU 是三层（3 个矩阵）并用
        门控机制：gate 分支决定"哪些信息通过"，up 分支提供"内容"。
        在相同参数量下效果优于 GELU MLP，是 LLaMA 以来的标准配置。

    因为多了 1 个矩阵，中间维度通常取传统 4*d_model 的 2/3 以对齐参数量：
        4 * 1024 * 2/3 ≈ 2731，向上取到 256 的整数倍 = 2816
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(cfg.d_model, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.d_model, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # silu(x) = x * sigmoid(x)，比 ReLU 更平滑，梯度性质更好
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Block(nn.Module):
    """一个 Transformer 层：注意力 + 前馈，都带残差连接。

    采用 **pre-norm** 结构（归一化放在子层之前）：
        x = x + attn(norm(x))
        x = x + mlp(norm(x))

    为什么 pre-norm 而不是原始的 post-norm：
        post-norm（x = norm(x + sublayer(x))）在深层网络中容易梯度消失，
        需要小心的 warmup 才能训起来。pre-norm 让残差路径上没有任何归一化，
        梯度可以无衰减地直通到底，训练明显更稳定 —— 对 fp16 尤其重要。

    **梯度检查点**：
        开启后，前向时不保存中间激活，反向时重新算一遍。
        用约 33% 的额外计算换取大量显存。本项目实测这是**必需项**：
        关掉后连 micro_batch=2 都会导致显存超售（reserved 6.54GB），
        吞吐从 4985 掉到 863 tok/s。
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.attn = Attention(cfg)
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x: torch.Tensor, use_checkpoint: bool = False) -> torch.Tensor:
        if use_checkpoint and self.training:
            # use_reentrant=False 是新版实现：
            #   - 支持非张量输入、支持嵌套
            #   - 与 autograd 的交互更正确（不会在反向时重复触发 hook）
            #   旧的 use_reentrant=True 有一些已知的坑，新代码应该用 False
            x = x + _ckpt(self._attn_branch, x, use_reentrant=False)
            x = x + _ckpt(self._mlp_branch, x, use_reentrant=False)
            return x
        x = x + self._attn_branch(x)
        return x + self._mlp_branch(x)

    def _attn_branch(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(self.attn_norm(x))

    def _mlp_branch(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.mlp_norm(x))


# ===========================================================================
# 完整模型
# ===========================================================================
class MiniLM(nn.Module):
    """完整的 LLaMA 风格语言模型。

    结构：
        token embedding
        → N × Block
        → 最终 RMSNorm
        → lm_head 投影到词表

    实测参数量 316.7M（24 层，d_model=1024），符合 0.3B 目标。
    """

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        # 最后的归一化。注意 LLaMA 在这里**不需要**再加一个独立的输出归一化，
        # 这一层就是输出前的归一化，别重复加。
        self.norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        if cfg.tie_word_embeddings:
            # 输入 embedding 和输出投影共享同一份权重。
            # 直接赋值是让两者指向**同一个 Parameter 对象**，而不是复制一份，
            # 这样反向传播时梯度会累加到同一个张量上 —— 这正是 tied weights 的语义。
            # nn.Module.parameters() 会自动去重，不会重复计数。
            self.lm_head.weight = self.token_emb.weight

        # 初始化必须在 tie weights 之后进行，否则会覆盖掉共享关系
        self.apply(self._init_weights)

        n_params = self.num_parameters()
        print(f"[模型] MiniLM 已构建：{n_params/1e6:.1f}M 参数 "
              f"({cfg.n_layers} 层, d_model={cfg.d_model}, "
              f"GQA {cfg.n_heads}:{cfg.n_kv_heads}, "
              f"tied={cfg.tie_word_embeddings})")

    def _init_weights(self, module: nn.Module) -> None:
        """权重初始化：正态分布 N(0, initializer_range)。

        注意 tied embedding 的权重会被初始化两次（token_emb 和 lm_head 各一次），
        但因为两者是同一个 Parameter 对象，第二次会覆盖第一次 —— 结果仍然是干净的
        一次初始化，不会出问题。

        RMSNorm 的 weight 必须保持为 1（那是缩放系数，不是权重），
        所以只对 Linear / Embedding 做初始化。
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.initializer_range)

    def num_parameters(self, exclude_embedding: bool = False) -> int:
        """统计参数量（去重，tied weights 只算一次）。

        用 named_parameters 的默认行为（remove_duplicate=True）来去重，
        这正是 tied embedding 场景下需要的。
        """
        total = 0
        for name, p in self.named_parameters():
            if exclude_embedding and "token_emb" in name:
                continue
            total += p.numel()
        return total

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
        logits_last_only: bool = False,
        use_checkpoint: bool = False,
    ) -> dict[str, torch.Tensor | None]:
        """前向传播。

        参数
        ----
        input_ids : torch.Tensor
            形状 (batch, seq) 的 token id（int64）
        labels : torch.Tensor, optional
            形状 (batch, seq) 的目标 token id。**需要用 -100 标记"不计 loss 的位置"**
            （比如 SFT 阶段用户提问的部分），这是 PyTorch CrossEntropyLoss 的约定。
        logits_last_only : bool
            只计算最后一个位置的 logits。生成阶段用这个能省掉大量无用计算 ——
            自回归生成时只有最后一个位置的输出有用，算全序列是浪费。
        use_checkpoint : bool
            是否启用梯度检查点。**由训练循环传入，而不是模型自己决定** ——
            因为"要不要用检查点"是训练策略，取决于当前显存余量，
            同一个模型在预训练和推理时应该有不同选择。

        返回
        ----
        dict，包含：
            "logits" : (batch, seq, vocab) 或 (batch, 1, vocab)
            "loss"   : 标量，未传 labels 时为 None
        """
        batch, seq = input_ids.shape
        if seq > self.cfg.max_seq_len:
            raise ValueError(
                f"输入序列长度 {seq} 超过模型上限 {self.cfg.max_seq_len}。"
                f"训练时请检查 data 的 seq_len 配置；生成时请检查生成步数。"
            )

        x = self.token_emb(input_ids)

        for block in self.blocks:
            x = block(x, use_checkpoint=use_checkpoint)

        x = self.norm(x)

        if logits_last_only:
            # 只对最后一个位置算 lm_head，大幅省计算和显存。
            # 生成场景下不需要 loss，直接返回 None。
            logits = self.lm_head(x[:, -1:, :])
            return {"logits": logits, "loss": None}

        logits = self.lm_head(x)
        loss = None
        if labels is not None:
            # 交叉熵要求把 (B,S,V) 和 (B,S) 摊平成 (B*S,V) 和 (B*S,)
            # ignore_index=-100 让标记为 -100 的位置不产生梯度
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )

        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int | None = 50,
        top_p: float | None = 0.9,
        repetition_penalty: float = 1.1,
        eos_token_id: int | None = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """自回归生成。

        参数
        ----
        input_ids : (batch, prompt_len) 的提示 token
        max_new_tokens : 最多生成多少新 token
        temperature : 温度。越低越确定，越高越发散。<=0 表示贪心解码。
        top_k : 只从概率最高的 k 个 token 里采样。None 表示不限制。
        top_p : 核采样，只从累积概率达到 p 的最小 token 集合里采样。None 表示不限制。
        repetition_penalty : 重复惩罚。>1 会降低已出现 token 的概率。
            小模型很容易陷入复读循环，这个参数实际很有用。
        eos_token_id : 遇到这个 token 就停止。
        use_cache : 是否用 KV cache。

        关于 KV cache：
            这里为了代码清晰，**每次生成都重新计算整个序列的前向**，没有实现 KV cache。
            对 0.3B 模型、seq<1024 的场景，这个开销可以接受。
            实现 KV cache 是后续优化项（GQA 已经把 cache 大小减半了，为它铺好了路）。
        """
        self.eval()
        # 确保 mem_efficient 生效，否则长序列生成会 OOM
        force_mem_efficient_attention()

        generated = input_ids
        finished = torch.zeros(input_ids.size(0), dtype=torch.bool, device=input_ids.device)

        for _ in range(max_new_tokens):
            # 序列太长时从左边截断，避免超过模型上限
            ctx = generated[:, -self.cfg.max_seq_len:]

            out = self.forward(ctx, logits_last_only=True)
            logits = out["logits"][:, -1, :].float()  # (B, vocab)，升 fp32 做采样更稳

            # ---- 重复惩罚 ----
            # 把已经出现过的 token 的 logit 压低（或抬高），降低复读概率。
            # 小模型不加重罚很容易陷入 "的的的的" 这种循环。
            if repetition_penalty and repetition_penalty != 1.0:
                for b in range(generated.size(0)):
                    for tok in torch.unique(generated[b]):
                        # 正 logit 除以惩罚，负 logit 乘以惩罚 —— 保证都是往"降低"的方向走。
                        # 如果统一除以 penalty，负 logit 反而会变大（接近 0），那就成了奖励。
                        if logits[b, tok] > 0:
                            logits[b, tok] /= repetition_penalty
                        else:
                            logits[b, tok] *= repetition_penalty

            # ---- 温度 ----
            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature

                # ---- top-k ----
                if top_k is not None and top_k > 0:
                    k = min(top_k, logits.size(-1))
                    # 取第 k 大的值作为阈值，低于它的全部设成 -inf
                    kth = torch.topk(logits, k, dim=-1).values[:, -1, None]
                    logits = logits.masked_fill(logits < kth, float("-inf"))

                # ---- top-p (核采样) ----
                if top_p is not None and 0 < top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                    probs = F.softmax(sorted_logits, dim=-1)
                    # 累积概率超过 top_p 的位置标记出来
                    cum = probs.cumsum(dim=-1)
                    # 右移一位：保留"刚刚超过 p"的那个 token，
                    # 否则累积概率恰好等于 p 时会多砍一个有效 token
                    remove = cum - probs > top_p
                    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                    # 散射回原始顺序
                    logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token], dim=1)

            # 整批都生成了 eos 就提前退出
            if eos_token_id is not None:
                finished |= next_token.squeeze(-1) == eos_token_id
                if finished.all():
                    break

        return generated


# ===========================================================================
# 辅助工具
# ===========================================================================
def build_model(cfg: ModelConfig, device: str | torch.device = "cuda") -> MiniLM:
    """构建模型并搬到指定设备上。

    统一入口，保证"强制 mem_efficient"这一步不会被漏掉 ——
    它是本项目最容易漏且后果最严重的一个初始化步骤。
    """
    force_mem_efficient_attention()
    model = MiniLM(cfg)
    return model.to(device)


def model_summary(cfg: ModelConfig, model: MiniLM | None = None) -> str:
    """生成模型结构的文字摘要，用于核对配置是否符合预期。

    会同时给出**解析式估算值**和**模型实测值**并对比 ——
    两者不一致就说明实现和设计不符，这是个很有用的自检。
    """
    lines = []
    lines.append("=" * 68)
    lines.append("MiniLM 模型摘要")
    lines.append("=" * 68)
    lines.append(f"  vocab_size        : {cfg.vocab_size:,}")
    lines.append(f"  d_model           : {cfg.d_model}")
    lines.append(f"  n_layers          : {cfg.n_layers}")
    lines.append(f"  n_heads           : {cfg.n_heads}  (head_dim = {cfg.head_dim})")
    lines.append(f"  n_kv_heads        : {cfg.n_kv_heads}  (GQA 比例 {cfg.n_heads}:{cfg.n_kv_heads})")
    lines.append(f"  intermediate_size : {cfg.intermediate_size}")
    lines.append(f"  max_seq_len       : {cfg.max_seq_len}")
    lines.append(f"  rope_theta        : {cfg.rope_theta}")
    lines.append(f"  tie_embeddings    : {cfg.tie_word_embeddings}")
    lines.append("-" * 68)

    est = cfg.num_params()
    lines.append(f"  参数量（解析式估算） : {est/1e6:.2f} M")
    if model is not None:
        actual = model.num_parameters()
        lines.append(f"  参数量（模型实测）   : {actual/1e6:.2f} M")
        if est != actual:
            # 差一点点通常是因为解析式漏算了 RMSNorm 之类的零碎参数或无 bias 的细节
            lines.append(f"  ⚠ 两者相差 {abs(est-actual):,} 个参数，请检查结构实现")
        else:
            lines.append("  ✓ 解析式与实测完全一致")

        emb = cfg.vocab_size * cfg.d_model
        lines.append("-" * 68)
        lines.append(f"  embedding 占比      : {emb/actual*100:.1f}%  ({emb/1e6:.1f}M)")
        lines.append(f"  单层参数量          : {(actual-emb)/cfg.n_layers/1e6:.2f} M")
    lines.append("=" * 68)
    return "\n".join(lines)


def estimate_training_memory(cfg: ModelConfig, micro_batch: int, seq_len: int) -> dict[str, float]:
    """估算训练时的显存占用（GB），用于在真正跑之前判断会不会超售。

    这是把实测经验固化成公式，避免每次改配置都要试错一遍。

    注意：激活部分只能粗略估算，实际值依赖 PyTorch 的分配策略和碎片情况，
    所以真实判断仍应以实测 `torch.cuda.max_memory_reserved()` 为准。
    这里给的是数量级参考。
    """
    n = cfg.num_params()

    weights_fp32 = n * 4 / 2**30          # fp32 主权重
    grads_fp32 = n * 4 / 2**30            # fp32 梯度
    opt_8bit = n * 2 / 2**30              # 8-bit Adam 的 m 和 v（各 1 字节）
    opt_fp32 = n * 8 / 2**30              # 普通 AdamW 的 m 和 v（各 4 字节）

    # 开梯度检查点后每层只存层边界处的激活：
    #   每层的激活 ≈ micro_batch * seq * d_model * 2 字节（fp16）
    #   边界数量 = n_layers + 1
    # mem_efficient 注意力让注意力部分不随 seq² 增长，所以这里不乘 seq 因子。
    act_ckpt = (cfg.n_layers + 1) * micro_batch * seq_len * cfg.d_model * 2 / 2**30
    # 不开检查点的话，每层内部的中间激活也要存，量级大得多（这里取经验系数）
    act_nockpt = act_ckpt * 6

    overhead = 0.4  # CUDA context + 分配器碎片

    return {
        "权重(fp32)": weights_fp32,
        "梯度(fp32)": grads_fp32,
        "优化器(8bit)": opt_8bit,
        "优化器(fp32)": opt_fp32,
        "激活(开检查点)": act_ckpt,
        "激活(关检查点)": act_nockpt,
        "固定开销": overhead,
        "合计_8bit_开检查点": weights_fp32 + grads_fp32 + opt_8bit + act_ckpt + overhead,
        "合计_fp32_开检查点": weights_fp32 + grads_fp32 + opt_fp32 + act_ckpt + overhead,
    }
