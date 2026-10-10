"""
训练用的数据集：从二进制 memmap 随机裁剪出训练批次。

## 核心思想

    预训练数据被展平成一个超长的 token id 数组（2B 个 uint16）。
    训练时不需要任何"数据集"概念 —— 只要在数组里随机取一个起点，
    往后读 seq_len+1 个 token 就得到一个训练样本。

    这个做法（nanoGPT 最先采用）有几个显著优点：
      - **零预处理开销**：取样本就是一次内存切片，O(1)
      - **没有 padding**：所有样本都是满长度，不像按文档切分那样需要补齐
      - **文档边界自然处理**：直接拼接不同文档，用 <|endoftext|> 分隔，
        模型自己学会边界语义
      - **可断点续训**：只需要记录 step 数，数据顺序由随机种子决定

## 关于 batch 的组织

    这里不用 torch 的 DataLoader，而是自己实现 `get_batch`。
    原因是：
      1. Windows 上 DataLoader 的多进程默认是 spawn，启动开销大，
         而我们的数据在 memmap 里，根本不需要多进程预取
      2. 训练循环需要精确控制每个 micro-batch，自己实现更直接
      3. 少一层抽象就少一个出问题的地方
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch


class MemmapDataset:
    """从 .bin 文件读取训练数据。

    参数
    ----
    data_path : str | Path
        二进制文件路径（纯 token id 序列，无文件头）。
    seq_len : int
        每个训练样本的序列长度。
    dtype : str
        存储类型。默认 uint16（词表 < 65536 时最省空间）。
    device : str
        目标设备。注意：**数据先放在 CPU，按需搬到 GPU**。
        不要把整个 4GB 的数据集拷到显存上（6GB 卡装不下，
        而且训练只需要当前这批）。
    """

    def __init__(
        self,
        data_path: str | Path,
        seq_len: int,
        dtype: str = "uint16",
        device: str = "cuda",
    ) -> None:
        self.data_path = Path(data_path)
        self.seq_len = seq_len
        self.np_dtype = np.dtype(dtype)
        self.device = device

        if not self.data_path.exists():
            raise FileNotFoundError(
                f"数据文件不存在: {self.data_path.resolve()}\n"
                f"请先运行 python scripts/02_prepare_data.py 准备数据。"
            )

        # memmap 让操作系统按需把文件页载入内存，而不是一次性读进来。
        # 2B tokens 是 4GB，直接 np.fromfile 会占用 4GB 常驻内存；
        # memmap 则让 OS 的页缓存来管理，多个 epoch 间还能复用缓存。
        self.data = np.memmap(self.data_path, dtype=self.np_dtype, mode="r")

        if len(self.data) <= seq_len:
            raise ValueError(
                f"数据太短：只有 {len(self.data):,} 个 token，"
                f"但 seq_len={seq_len}。请准备更多数据。"
            )

    def __len__(self) -> int:
        """可用作训练起点的位置数量。

        要减去 seq_len 是因为最后一个样本需要读到 data[start + seq_len]，
        即目标序列要多一个 token（用于预测"下一个 token"）。
        """
        return len(self.data) - self.seq_len - 1

    @property
    def n_tokens(self) -> int:
        """总 token 数。"""
        return len(self.data)

    def get_batch(
        self,
        batch_size: int,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """随机采样一个批次。

        返回
        ----
        (x, y) : tuple[Tensor, Tensor]
            x 形状 (batch_size, seq_len)，输入
            y 形状 (batch_size, seq_len)，目标是 x 右移一位

        为什么 y 是 x 右移一位：
            语言模型的任务是"给定前 t 个 token，预测第 t+1 个"。
            所以目标序列就是输入序列左移一位，这样每个位置都在预测下一个 token。
            一个长度为 seq_len 的窗口能产出 seq_len 个训练信号（而非只有 1 个），
            这是 Transformer 并行训练的威力所在。
        """
        # 随机选 batch_size 个起点。
        # 用 torch 的 generator 是为了让随机性和训练步数绑定，
        # 这样断点续训时能复现数据顺序。
        max_start = len(self)
        starts = torch.randint(0, max_start, (batch_size,), generator=generator).numpy()

        # 一次性把所有窗口读出来：用 numpy 的 fancy indexing
        # 这里用 np.stack 逐个取，因为每个窗口是连续的，逐个取更省内存带宽
        x = np.stack([self.data[s : s + self.seq_len] for s in starts])
        y = np.stack([self.data[s + 1 : s + 1 + self.seq_len] for s in starts])

        # 转成 int64：PyTorch 的 embedding 层要求索引是 int64，
        # uint16 不能直接喂进去。
        # 注意这里必须用 .long() 而不是 .to(torch.int64)，
        # 效果一样但意图更清晰。
        x_t = torch.from_numpy(x.astype(np.int64)).to(self.device, non_blocking=True)
        y_t = torch.from_numpy(y.astype(np.int64)).to(self.device, non_blocking=True)

        return x_t, y_t


class PackedSFTDataset:
    """SFT 数据集：把对话样本打包成定长序列。

    和预训练不同，SFT 的每个 token 有"该不该算 loss"的区别：
    用户提问的部分不该算 loss（-100 掩码），只有助手回复的部分才算。

    实现方式：
        构造 labels 时，把 prompt 部分全部填 -100，
        只保留 assistant 回复部分的真实 token id。
        PyTorch 的 CrossEntropyLoss 会自动忽略 -100 的位置。

    为什么这个细节很重要：
        如果 prompt 也参与 loss 计算，模型会学着去"预测用户的提问"，
        而不是"生成回答"。表现是 SFT 后模型倾向于复述用户的话，
        或者回答质量明显低于预期。这是 SFT 最常见的错误。
    """

    IGNORE_INDEX = -100

    def __init__(self, samples: list[dict], seq_len: int) -> None:
        """
        参数
        ----
        samples : list[dict]
            每条包含 "input_ids"（完整序列）和 "labels"（-100 掩码后的标签）。
        """
        self.seq_len = seq_len
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def get_batch(self, batch_size: int, generator=None) -> tuple[torch.Tensor, torch.Tensor]:
        """采样一个批次。

        注意这里**不做 padding**：如果样本长度不齐，需要在构造样本时就
        统一到 seq_len（截断或补齐）。补齐时 labels 的补位也要是 -100，
        否则模型会学着去预测 padding token。
        """
        idx = torch.randint(0, len(self.samples), (batch_size,), generator=generator)

        xs, ys = [], []
        for i in idx.tolist():
            s = self.samples[i]
            ids = s["input_ids"][: self.seq_len]
            labs = s["labels"][: self.seq_len]

            # 补齐到 seq_len：输入补 pad(0)，标签补 -100（不计 loss）
            pad_n = self.seq_len - len(ids)
            if pad_n > 0:
                ids = ids + [0] * pad_n
                labs = labs + [self.IGNORE_INDEX] * pad_n

            xs.append(ids)
            ys.append(labs)

        x = torch.tensor(xs, dtype=torch.long)
        y = torch.tensor(ys, dtype=torch.long)
        return x, y


def load_meta(data_dir: str | Path) -> dict:
    """读取数据集的元信息（token 数、来源统计等）。

    元信息里记录了每个数据源贡献了多少 token，
    这对复现实验和调整混合比例很有用。
    """
    path = Path(data_dir) / "meta.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
