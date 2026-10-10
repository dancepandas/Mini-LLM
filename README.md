# Mini_LM —— 从零训练一个 0.3B 中文语言模型

一个**完全从零**的中文小模型训练项目：自己实现模型结构、自己训练分词器、
自己写训练循环，完整走一遍 `tokenizer → pre-training → SFT → RL` 的流程，
最终得到一个能进行基本中文对话的 0.3B 模型。

> **这不是调库项目。** 没有用 `transformers` 的 `AutoModel`，没有用 `Trainer`，
> 模型结构、训练循环、数据管线、GRPO 强化学习全部手写。
> 目的是把每一环都亲手做一遍 —— 代码本身就是教材，每个关键决策都有中文注释说明"为什么"。

---

## 仓库里**没有**什么（clone 后必读）

本仓库**只包含代码和分词器**，不含训练产物 —— 它们总计约 31GB，
且 GitHub 单文件上限是 100MB：

| 不在仓库里 | 大小 | 怎么得到 |
|---|---:|---|
| `checkpoints/*.pt` | 16 GB | 跑 `scripts/03_pretrain.py` 等脚本重新训练（约 5 天算力） |
| `data/`（原始语料 + 分词后的 .bin） | 15 GB | 跑 `scripts/02_prepare_data.py` 重新下载和分词 |

**在仓库里的**：`tokenizer/tokenizer.json`（2.6MB）——
它是全部训练的前提，且**一旦定下就不能改**（换了要重新预训练），
所以必须进版本库。

所以完整复现的流程是：
```bash
git clone <本仓库>
pip install -r requirements.txt
python scripts/00_setup.py              # 先跑基准，确认你机器的显存和吞吐
python scripts/01_train_tokenizer.py    # 分词器（仓库里已有，这步可选）
python scripts/02_prepare_data.py --stage download && python scripts/02_prepare_data.py --stage tokenize
python scripts/03_pretrain.py           # 约 5 天
python scripts/04_sft.py --max-samples 50000 --epochs 1
python scripts/05_rl_grpo.py --steps 150 --data data/raw/rl_easy_arith.jsonl
python web/app.py                       # 网页 Demo
```

> ⚠️ **数据许可证**：训练用的 `gongjy/minimind_dataset` 是
> **CC-BY-NC-4.0（非商用）**。代码本身你可以自由使用，
> 但**用这个数据训出来的模型不能商用**，商用需要换数据源。

---

## 目录

- [快速开始](#快速开始)
- [环境与硬件约束（实测）](#环境与硬件约束实测)
- [模型设计](#模型设计)
- [显存与吞吐实测](#显存与吞吐实测)
- [数据来源](#数据来源)
- [项目结构](#项目结构)
- [六个阶段详解](#六个阶段详解)
- [排错指引](#排错指引)
- [踩过的坑（完整清单）](#踩过的坑完整清单)

---

## 快速开始

```bash
# 0) 激活环境（Windows）
conda activate vmodel

# 1) 环境自检 + 吞吐基准（必须先跑，会实测显存和速度）
python scripts/00_setup.py

# 2) 训练分词器
python scripts/01_train_tokenizer.py

# 3) 下载数据（魔塔社区，不需要代理）
python scripts/02_prepare_data.py --stage download

# 4) 清洗 + 分词 + 打包成二进制
python scripts/02_prepare_data.py --stage tokenize

# 5) 预训练（约 5 天，可断点续训）
python scripts/03_pretrain.py

# 6) SFT 指令微调（约 1 小时）
python scripts/04_sft.py

# 7) GRPO 强化学习（约 1 小时）
python scripts/05_rl_grpo.py

# 8) 对话
python scripts/06_chat.py
```

---

## 环境与硬件约束（实测）

本项目在**一台消费级机器**上开发，所有约束都是实测出来的，不是推测：

| 项 | 实测值 | 对方案的影响 |
|---|---|---|
| GPU | RTX 2060 **6GB**，算力 sm_75 | 只支持 fp16；**bf16 无硬件加速** |
| 空闲显存 | 约 4.9 GB | 必须 8-bit 优化器 + 梯度检查点 |
| 内存 / CPU | 32GB / i7-13700KF 16核 | 数据预处理够用 |
| 磁盘 | D: 剩余约 280GB | 够放 tokenized 数据（2B tokens ≈ 4GB） |
| Python | 3.13.13 | 部分包需要较新版本 |
| PyTorch | 2.7.1+cu126 | — |
| 网络 | 魔塔直连可用；HF 需代理 | 数据走魔塔 |

### 三个关键硬件事实

**1. 必须用 fp16，不能用 bf16**

Turing（sm_75）架构**没有 bf16 张量核**。实测 bf16 矩阵乘"能跑"，
但那是 PyTorch 走模拟路径的结果，速度远不如 fp16。
所以全程使用 fp16 + `GradScaler`（fp16 动态范围窄，需要缩放梯度防下溢）。

**2. 必须显式强制 mem_efficient 注意力后端**

实测 seq=1024/batch=8 的注意力前向+反向峰值显存：

| 后端 | seq=1024 | seq=2048 | 备注 |
|---|---|---|---|
| `math` | **2192 MB** | **4240 MB** | 实体化完整 seq² 矩阵，单独就能打爆 6GB 卡 |
| `mem_efficient` | **145 MB** | **145 MB** | 显存 O(seq) |

**相差 15 倍。** 而且默认路径**不保证选中** mem_efficient
（会打印 `Memory Efficient attention has been runtime disabled`），
不显式指定就会静默回退到 math → 必然 OOM。
代码见 `minilm/model.py: force_mem_efficient_attention()`。

**3. 显存超售是最隐蔽的性能陷阱**

Windows(WDDM) 在显存不足时**不会报错**，而是把数据换页到内存，
训练速度暴跌但一声不吭。实测对比：

| 配置 | 吞吐 | 说明 |
|---|---|---|
| 正常 | **4618 tok/s** | — |
| 显存超售 | **305 tok/s** | 慢 15 倍，且无任何报错 |

这就是为什么 `scripts/00_setup.py` 要先跑 —— 必须在投入 5 天前测出真实速度。

---

## 模型设计

LLaMA 风格的 decoder-only Transformer：

```yaml
vocab_size        = 32768    # 含 4 个特殊 token
d_model           = 1024
n_layers          = 24
n_heads           = 16       # head_dim = 64
n_kv_heads        = 8        # GQA 2:1
intermediate_size = 2816     # SwiGLU
max_seq_len       = 1024
rope_theta        = 10000.0
tie_word_embeddings = True
```

**实测参数量 316.72M**（解析式与模型实测完全一致）。

### 设计选择及理由

| 选择 | 理由 |
|---|---|
| **GQA** (8 KV heads) | 推理时 KV cache 减半；训练省激活。**对 RL 阶段至关重要** —— 要同时装 policy 和 reference 两个模型 |
| **SwiGLU** | 同参数量下优于 GELU MLP，现代 LLM 标配 |
| **RMSNorm + pre-norm** | 数值稳定，对 fp16 友好；残差路径无归一化，梯度可直通 |
| **RoPE** | 相对位置编码，外推能力好 |
| **tied embedding** | 省 33.6M 参数（总量的 10.6%） |
| **无 bias** | 省参数且训练更稳 |
| **fp16 + GradScaler** | sm_75 硬件限制 |

---

## 显存与吞吐实测

### ⚠️ 最重要的发现：梯度累积会导致显存碎片化

**单步基准测试的显存数字会显著低估真实训练的占用。** 这是本项目踩的最大的一个坑：

| 场景 | micro_batch=4 的峰值 reserved | 吞吐 |
|---|---:|---:|
| **基准测试**（每次前向反向后就更新参数） | 4.90 GB | 4618 tok/s |
| **真实训练**（连续 32 次前向反向才更新一次） | **6.13 GB** | 3991 tok/s |

差 **1.23 GB**，直接从"看起来没问题"变成"超售 1.1GB"。

**原因**：梯度累积时反复分配/释放中间张量，
PyTorch 的缓存分配器会因**碎片化**而不断扩张自己的池子。
基准测试每次只做一轮，看不到这个累积效应。

**教训**：性能基准必须**用真实的训练循环**跑（含梯度累积），
否则测出来的数字是乐观的假象。

### 实测的各配置对比（有效 batch 固定 131k tokens）

| 配置 | 吞吐 | 峰值 reserved | 判断 |
|---|---:|---:|---|
| `mb=1 seq=1024 accum=128` | 4001 tok/s | **4.34 GB** | ✅ **当前配置** |
| `mb=2 seq=512 accum=128` | 3999 tok/s | 4.34 GB | ✅ 但序列更短 |
| `mb=2 seq=1024 accum=64` | 4603 tok/s | 4.99 GB | ⚠️ 压线，无余量 |
| `mb=3 seq=1024 accum=44` | 4336 tok/s | 5.81 GB | ❌ 超售 |
| `mb=4 seq=1024 accum=32` | 3991 tok/s | 6.13 GB | ❌ 超售 |

**为什么选 mb=1 而不是更快的 mb=2**：mb=2 快 15%，但 4.99GB 正好卡在
4.98GB 空闲显存上，**没有任何余量** —— 用户开个浏览器就会超售。
稳定性优先于 15% 的速度。

### 训练配置

```yaml
micro_batch_size : 1        # 硬约束，见上表
grad_accum_steps : 128      # → 有效 batch 131k tokens/step
seq_len          : 1024
lr               : 3e-4     # cosine 衰减到 3e-5，warmup 2000 步
optimizer        : adamw8bit  # 必需，fp32 AdamW 会多占 1.9GB → 超售
gradient_checkpointing: true  # 必需，关掉后 batch=2 都超售
max_steps        : 13000    # 1.70B tokens
```

**预计预训练耗时：1.70e9 ÷ 4001 tok/s ≈ 118 小时 ≈ 4.9 天**

---

## 数据来源

数据来自**魔塔社区 ModelScope**（国内直连，不走代理）。

### 预训练

主料是 `gongjy/minimind_dataset`（调研自开源项目
[jingyaogong/minimind](https://github.com/jingyaogong/minimind)）：

| 文件 | 大小 | 说明 |
|---|---|---|
| `pretrain_t2t.jsonl` | 8.28 GB | 纯文本 `{"text": "..."}`，约 1.6~1.9B tokens |
| `pretrain_t2t_mini.jsonl` | 1.24 GB | 小样，用于快速验证 |

补充 `AI-ModelScope/wikipedia-cn-20230720-filtered`（520MB）
提供**长文本和事实知识** ——
minimind 的预训练数据是短问答拼接（平均 215 字符/条），
光用它训出来的模型「会聊天但知识面窄」，需要掺长文本缓解。

### ⚠️ 许可证

**`minimind_dataset` 是 CC-BY-NC-4.0（非商用）。**
学习研究用途没问题，**商用必须换数据源**。
若需商用可考虑 `swift/Chinese-Qwen3-235B-Thinking-2507-Distill-data-110k-SFT`（Apache-2.0）。

### SFT / RL

同一数据集里就有现成的：
- `sft_t2t_mini.jsonl`（1.74 GB）—— SFT
- `dpo.jsonl` / `rlaif.jsonl` / `agent_rl_math.jsonl` —— RL

---

## 项目结构

```
Mini_LM/
├── README.md               # 本文件：总纲
├── requirements.txt
├── configs/                # 所有超参配置（改配置不改代码）
│   ├── README.md
│   ├── model_0.3b.yaml     # 模型结构
│   ├── pretrain.yaml       # 预训练超参
│   └── data.yaml           # 数据源与配额
├── minilm/                 # 核心库（自己实现的部分）
│   ├── README.md
│   ├── config.py           # YAML ↔ dataclass，带启动期校验
│   ├── model.py            # LLaMA 架构 + mem_efficient 强制
│   ├── tokenizer.py        # 字节级 BPE + ChatML 特殊 token
│   ├── chat_template.py    # 对话模板（训练/推理必须一致）
│   ├── utils.py            # 显存监控、checkpoint、日志
│   └── data/
│       ├── README.md
│       ├── modelscope.py   # 魔塔下载（分块并行 + 断点续传）
│       ├── clean.py        # 文本清洗 + 精确去重
│       ├── build_bin.py    # 分词 → uint16 memmap
│       └── dataset.py      # 训练用的数据读取器
├── scripts/                # 按序号执行的入口
│   ├── README.md
│   ├── 00_setup.py         # 环境自检 + 吞吐基准【决策门】
│   ├── 01_train_tokenizer.py
│   ├── 02_prepare_data.py  # 下载 / 分词
│   ├── 03_pretrain.py      # 预训练（断点续训）
│   ├── 04_sft.py           # SFT（loss 掩码）
│   ├── 05_rl_grpo.py       # GRPO + 自实现 LoRA
│   └── 06_chat.py          # 命令行对话
├── eval/                   # 评测
│   └── README.md
├── checkpoints/            # 训练产物
└── data/                   # 数据（raw / pretrain）
```

---

## 六个阶段详解

### 阶段 0：环境自检 + 吞吐基准（`scripts/00_setup.py`）

**这是整个项目的决策门。** 在投入 5 天训练之前，必须实测：
- 参数量是否和设计一致（解析式 vs 实测交叉校验）
- mem_efficient 后端是否真的生效
- 实际吞吐是多少 → 推算真实 ETA

如果实测吞吐远低于预期，就该当场调整 token 预算或模型规模，
而不是硬跑几天才发现来不及。

### 阶段 1：训练分词器（`scripts/01_train_tokenizer.py`）

- 字节级 BPE，词表 **32768**
- **4 个特殊 token 必须在预训练前定好**：
  `<|endoftext|>` `<|im_start|>` `<|im_end|>` `<|pad|>`

**为什么特殊 token 不能后加**：模型需要在预训练阶段就见到它们，
才能学到语义（比如"看到 im_end 就该停下"）。
预训练完再加的话，它们的 embedding 是随机初始化的，很难学好。
所以词表一旦定下，全流程（预训练/SFT/RL/推理）必须复用同一个文件。

实测压缩率：**0.877 tokens/汉字，1.574 字符/token**。

### 阶段 2：数据准备（`scripts/02_prepare_data.py`）

分两个独立阶段，可分开跑：

```bash
python scripts/02_prepare_data.py --stage download   # 只下载
python scripts/02_prepare_data.py --stage tokenize   # 只分词
```

**为什么预分词成二进制**：训练时在线分词的话，BPE 会成为瓶颈
（GPU 等 CPU）。预分词后训练时只是一次内存切片，开销可忽略。

输出是纯裸数组（无文件头）：
- `data/pretrain/train.bin` — uint16，2B tokens ≈ 4GB
- `data/pretrain/val.bin`
- `data/pretrain/meta.json`

用 uint16 是因为词表 32768 < 65536，比 int32 省一半空间和带宽。

### 阶段 3：预训练（`scripts/03_pretrain.py`）

```bash
python scripts/03_pretrain.py            # 从头开始
python scripts/03_pretrain.py --resume   # 断点续训
python scripts/03_pretrain.py --max-steps 3000   # 覆盖步数
```

**断点续训是必需品**：要连跑 5 天，桌面机断电/重启是大概率事件。
每 500 步存一次，滚动保留最近 3 个（每个约 1.3GB）。

监控指标（TensorBoard + 日志）：
- `loss` —— 最基本的
- `grad_norm` —— 突然暴涨说明即将发散
- 吞吐 tok/s —— 突然下降说明显存碎片化或超售
- 显存 reserved —— 超过物理空闲显存就是超售

内置保护：
- 梯度裁剪（fp16 训练必须开）
- **NaN/Inf 检测**：检测到非有限梯度就跳过该步，不让坏更新污染参数
- 显存超售警告

### 阶段 4：SFT（`scripts/04_sft.py`）

**核心是 loss 掩码**：只在 assistant 回复上算 loss，prompt 部分标 -100。

不掩码的后果：模型会学着预测用户的提问，
表现是 SFT 后倾向于复述用户的话。**而且 loss 曲线看不出来**。

实现细节（`minilm/chat_template.py`）：
- 不硬编码 token 位置，而是 tokenize `<|im_start|>assistant\n` 得到实际的前缀 id 序列
- `<|im_end|>` 也**要算 loss** —— 否则模型学不会"什么时候该停"，
  推理时会一直说到撞上长度上限

### 阶段 5：RL —— GRPO（`scripts/05_rl_grpo.py`）

**为什么不用 PPO**：PPO 需要和策略同尺寸的 critic 网络。
6GB 显存装不下 policy + reference + critic 三个模型。

**GRPO 的洞察**：用同一道题一组采样的相对好坏估计优势，
`advantage_i = (r_i - mean) / std`，**完全不需要 critic**。

**为什么用 LoRA**：需要同时驻留 policy（训练中）和 reference（冻结），
全量微调放不下。LoRA 把可训练参数降到 1% 量级。
LoRA 是**自己实现的**（`minilm/../../scripts/05_rl_grpo.py: LoRALinear`），没依赖 peft。

**奖励设计**（RL 成败的关键）：
- 正确性奖励：抽取答案与 gold 精确匹配 → 1.0
- 格式奖励：输出规范 → 小权重（0.1）
- **格式权重必须远小于正确性**，否则模型会学会"输出格式漂亮的废话"骗分

**难度匹配的重要性**：如果同一题的一组 rollout 全对或全错，
组内标准差为 0 → 优势全为 0 → **梯度为 0，这一步完全白跑**。
脚本会统计"有效题占比"（组内有对有错），这个指标比 reward 更能反映 RL 是否起作用。

### 阶段 6：对话（`scripts/06_chat.py`）

```bash
python scripts/06_chat.py              # 交互模式
python scripts/06_chat.py --prompt "你好"   # 单次提问
```

支持流式输出、多轮对话、温度/top-p/top-k/重复惩罚调节。

**必须使用和训练时完全一致的 chat template** ——
这也是为什么模板集中在 `minilm/chat_template.py`，训练和推理从同一个函数取。

---

## 排错指引

### 训练突然变慢（最常见）

**症状**：吞吐从 4600 掉到几百 tok/s，但没有任何报错。

**原因**：显存超售。Windows 静默把显存换页到内存。

**排查**：
```bash
nvidia-smi    # 看显存占用是否接近 6GB
```
看训练日志里的 `mem` 字段，如果超过 4.9GB 就是超售了。

**解决**：
1. 关掉浏览器、聊天软件等占显存的程序（能省 1~2GB）
2. 确认 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 生效
3. 调小 micro_batch：`--micro-batch 2`

### OOM（显存不足）

按顺序降级：
1. `--micro-batch 2`（同时把 `grad_accum_steps` 翻倍保持有效 batch 不变）
2. 调小 `seq_len`：1024 → 512
3. 确认 `gradient_checkpointing: true`（关掉的话 batch=2 都超售）

### 下载失败 / 很慢

魔塔不需要代理，**必须直连**：
```bash
python scripts/02_prepare_data.py --stage download
```
代码里已经用 `ProxyHandler({})` 强制绕过代理（清环境变量不够，
urllib 会读 Windows 系统代理）。

单连接只有 1.5 MB/s，代码已实现**分块并行下载**（HTTP Range + 8 并发），
聚合能到 5~6 MB/s。如果还是很慢，检查网络本身。

### loss 变成 NaN

代码会自动跳过该步并记录。如果频繁出现：
1. 降低学习率
2. 确认梯度裁剪开启（`grad_clip: 1.0`）
3. 检查数据里有没有异常样本

### 对话效果差 / 模型答非所问

**先检查 chat template 是否一致**。训练和推理用了不同的模板是最常见的原因，
而且不会报错。确认 `scripts/06_chat.py` 和 `scripts/04_sft.py`
都用 `minilm/chat_template.py` 的函数。

如果模板没问题：
- 检查 SFT 的 loss 掩码是否生效（日志里会打印"被监督比例"，正常应在 30%~70%）
- 预训练 token 数是否够（太少的话模型语言能力不够）

### Windows 编码错误

`UnicodeEncodeError: 'gbk' codec can't encode character`
—— `minilm/__init__.py` 已自动把 stdout 重定向为 UTF-8。
如果自己写脚本遇到，记得先 `import minilm`。

---

## 踩过的坑（完整清单）

这些都是开发过程中实际遇到并修复的，记录下来避免重蹈覆辙：

| # | 问题 | 原因 | 修复 |
|---|---|---|---|
| 1 | RoPE 报"两个设备"错误 | `_cos`/`_sin` 是普通属性，`model.to(cuda)` 不会搬它们 | 注册成 `buffer` |
| 2 | 参数量差 1024 | 解析式公式漏算最后的输出 RMSNorm | 补上 `final_norm` |
| 3 | 打印 `✓` 崩溃 | Windows 控制台是 GBK，编不出该字符 | 包初始化时把 stdout 改成 UTF-8 |
| 4 | **HF 完全连不上**（urllib 却正常） | 环境里 `SSL_CERT_FILE` 指向不存在的证书文件，httpx 读它并报错 | 检测到文件不存在就删除该变量 |
| 5 | 魔塔列文件返回空 | 用了 `/oss/tree`，正确端点是 `/repo/tree` | 改正端点；返回空不报错，极易误判 |
| 6 | 魔塔分页失效 | SDK 的 `get_dataset_files` 只返回第一页（999 个） | 改用手写 HTTP 分页 |
| 7 | 下载速度只有 0.08 MB/s | 单连接被限流 | 实现 HTTP Range 分块 + 8 并发 → 5~6 MB/s |
| 8 | 分块文件位置不对 | SDK 把文件塞进深层嵌套目录 | 改成纯直连 HTTP，完全可控 |
| 9 | 往返测试误报失败 | 后处理器自动追加 `<\|endoftext\|>`，测试没关掉 | 测试时显式 `add_special_tokens=False` |
| 10 | 吞吐读数 305 tok/s（假象） | 普通 AdamW 导致显存超售 | 换 8-bit 优化器 + expandable_segments |
| 11 | **基准测试显存数字是假的** | 基准每次只做 1 轮前向反向，看不到梯度累积的碎片化效应 | 基准必须用真实训练循环；实测 mb=4 的真实占用是 6.13GB 而非 4.90GB |
| 12 | `UnboundLocalError: n_val_tokens` | 闭包内用了 `+=` 自增赋值，Python 把它当局部变量 | 加 `nonlocal` 声明 |
| 13 | `KeyError: 'repo_id'` | 改了配置结构但脚本还引用旧字段 | 用 `.get()` 兼容两种配置写法 |
| 14 | 定时监控**静默失效** | 正则写的是 `step +[0-9]+/`，但步数超 1000 后带千分位逗号（`4,510`），匹配不到任何东西，监控什么都不报 | 正则改成 `[0-9,]+` |
| 15 | tokenizer 训练语料与预训练语料不一致 | 分词器是用维基百科训的，但预训练主体是短问答数据，merge 规则偏向书面语 | 见下方说明 |
| 16 | 把 loss 的正常波动**误读成上升趋势**，差点误判训练异常 | 日志里的 loss 是**最近 100 步的滑动平均**，相邻读数高度相关，短窗口抖动看起来很像趋势 | 判断趋势必须**拉长窗口做线性回归**，不能看相邻几行 |
| 17 | **SFT 的 labels 错位一位，模型学的是错的映射** | 模型的 `forward` 不做内部位移，但 `encode_conversation` 返回的 labels 是未位移的 | `SFTDataset.get_batch` 里加 `labs = labs[1:] + [IGNORE_INDEX]` |

### 关于 #17：最危险的一类 bug

这个 bug 的症状值得单独记，因为**它不会崩溃、不会报错、loss 还在下降**：

```
修复前：step 0  loss 10.4629  gnorm 22.448
修复后：step 0  loss  4.9091  gnorm  9.799
```

**根因**：因果注意力下 `logits[i]` 预测的是 `input_ids[i+1]`。
本模型的 `MiniLM.forward` 直接做 `cross_entropy(logits, labels)`，
**不做内部位移**（不像 HuggingFace 的模型会自己 shift）。
所以调用方必须保证 `labels[i]` 就是 `input_ids[i+1]`。

- 预训练是对的：`MemmapDataset` 显式返回 `y = x` 右移一位
- SFT 漏了：`labels[i] = input_ids[i]`，于是模型被要求
  "用预测第 i+1 个 token 的 logits 去匹配第 i 个 token"

**为什么危险**：loss 从 10.46 缓慢下降，看起来"在训练"，
但学的是完全错误的映射，训完的模型根本不能用。
如果不检查**loss 的量级是否合理**，会一直训到结束才发现白跑。

**判据（重要）**：预训练好的模型，在 SFT 数据上的初始 loss 应该是
2~5 的量级。**如果是 10 左右（≈ ln(vocab_size)），几乎必然是 labels 错位。**
这是一个不需要任何额外工具、看一行日志就能判断的自检点。

同样地，预训练时如果 loss 一直停在 ln(vocab_size) 不降，
也要先怀疑数据管线里的 `y` 是否正确地右移了一位。

### 关于 #16：怎么正确判断 loss 趋势

训练到 60% 时，我观察到 loss 从 2.088 升到 2.128（+0.04），一度怀疑训练出问题
（因为 LR 正在衰减，loss 本该下降）。拉长窗口分析后发现是虚惊：

```
最近 3000 步线性回归 : -0.0746 loss / 1000 步   ← 其实在下降
标准差               :  0.0776                  ← 我看到的 0.04 波动远小于 1 个标准差
区间                 :  2.0576 ~ 2.3280
```

**教训**：日志里的 loss 是**滑动平均**（本项目是 100 步窗口），
相邻两次读数高度相关，300~800 步窗口内的正常抖动（±1 个标准差）看起来就像趋势。
**要判断趋势必须拉长窗口做回归**，并和标准差对比 ——
波动幅度小于 1 个标准差就说明不了任何问题。

日常监控看滑动平均是够的（能平滑掉单步噪声），但**判断"是否异常"时必须换方法**。

### 关于 #15：分词器应该在预训练语料上训练

本项目的分词器用了 233MB 中文维基百科训练。虽然字节级 BPE 泛化性好
（实测压缩率 0.877 tokens/汉字，无损往返通过），但严格来说**分词器的 merge 规则
应该匹配最终预训练语料的分布**，否则对某些文本类型的压缩率会次优。

**更合理的做法**：从 `data/raw/` 里把最终预训练语料（含对话、问答、网页）
按比例采样后训练分词器，而不是只用单一来源。

本项目因为时间预算没有这样做，属于已知的小瑕疵，不影响流程正确性。
如果重训，`scripts/01_train_tokenizer.py` 的 `--offline-samples` 可以指向
任何语料文件，改起来很容易。

---

## 参考

- [jingyaogong/minimind](https://github.com/jingyaogong/minimind) —— 数据来源，也是本项目流程的重要参考
- [gongjy/minimind_dataset](https://www.modelscope.cn/datasets/gongjy/minimind_dataset) —— 魔塔数据集
- [nanoGPT](https://github.com/karpathy/nanoGPT) —— 二进制 memmap 数据格式的经典做法
- GRPO 算法来自 DeepSeek-R1 论文

## 许可证

本项目代码仅供学习研究。**训练数据 `minimind_dataset` 是 CC-BY-NC-4.0（非商用）**，
训练出的模型继承该限制，不可用于商业用途。
