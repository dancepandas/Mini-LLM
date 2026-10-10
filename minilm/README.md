# `minilm/` —— 核心库

这个目录是项目的主体，包含**全部从零实现的模型与训练基础设施**。
它不依赖 `transformers` —— 模型结构、分词、对话模板、数据管线全部自己写。

> **为什么不用 `transformers`？**
> 1. 本项目的目的就是"从零走一遍"，调库就失去意义了
> 2. 需要完全掌控显存 —— 6GB 卡上每一 MB 都要精打细算，
>    而库的封装会带来我们不需要的开销和不可控的默认行为
> 3. `transformers` 即将解析到 5.x 大版本，API 有变动风险

---

## 文件一览

| 文件 | 作用 | 关键点 |
|---|---|---|
| `__init__.py` | 包初始化 + **两处必须在 import torch 前完成的环境设置** | 见下方 |
| `config.py` | 配置系统：YAML ↔ dataclass | 启动期校验，把错误拦在训练之前 |
| `model.py` | LLaMA 风格模型结构 | **强制 mem_efficient 注意力** |
| `tokenizer.py` | 字节级 BPE 训练与加载 | 特殊 token 必须预训练前定好 |
| `chat_template.py` | ChatML 对话模板 + loss 掩码 | 训练/推理必须用同一份 |
| `utils.py` | 显存监控、checkpoint、日志 | 断点续训的基础设施 |
| `data/` | 数据管线 | 见 `data/README.md` |

---

## `__init__.py` —— 导入顺序很重要

这个文件做两件事，**都必须在 `import torch` 之前生效**：

### 1. 设置显存分配器

```python
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
```

**实测效果**（micro_batch=4）：

| 分配器 | reserved | 结果 |
|---|---|---|
| 默认 | 5.40 GB | 超过 4.98GB 空闲显存 → 轻度超售，4417 tok/s |
| `expandable_segments` | 4.90 GB | 不超售 → **4618 tok/s** |

关掉超售后吞吐反而涨了 4.5%。这个环境变量只在 CUDA 分配器**首次初始化**时被读取，
之后再设就无效了。

### 2. Windows 控制台编码

Windows 默认控制台编码是 GBK，编不出 `✓` `⚠` `→` 这类字符，
一旦 `print` 到就会抛 `UnicodeEncodeError` 打断程序。
这里统一把标准输出改成 UTF-8，一次性解决。

### ⚠️ 因此：所有脚本必须先 `import minilm` 再 `import torch`

```python
import sys
sys.path.insert(0, str(PROJECT_ROOT))

import minilm          # ← 必须在前

import torch           # ← 必须在后
```

顺序反了的话，`PYTORCH_CUDA_ALLOC_CONF` 设置就晚了，会退回到默认分配器 → 显存超售。

---

## `config.py` —— 配置系统

把超参从代码里剥离到 YAML。**用 dataclass 而不是裸 dict 是为了在启动时就报错**：
比如把 `n_heads` 写成 16 而 `d_model` 写成 1024、`head_dim` 却是 48，
这种错误如果在训练 3 小时后才因为 shape 不匹配崩掉，代价很大。

`__post_init__` 里做的校验：
- `d_model` 必须能被 `n_heads` 整除
- `n_heads` 必须能被 `n_kv_heads` 整除（GQA 的前提）
- `vocab_size` 必须 ≤ 65536（否则 uint16 存不下训练数据）

`num_params()` 用解析式算参数量，**用于和模型实测值交叉校验**。
开发时就靠这个发现了"漏算输出层 RMSNorm"的 bug（差正好 1024 个参数）。

---

## `model.py` —— 模型结构

### 组件

| 类 | 作用 |
|---|---|
| `RMSNorm` | 均方根归一化。计算必须在 fp32 下做，否则 fp16 会溢出 |
| `RoPE` | 旋转位置编码。cos/sin 表**预计算并缓存**，且注册为 buffer |
| `Attention` | 因果自注意力 + GQA |
| `SwiGLU` | 门控前馈网络（3 个矩阵，中间维度取 4d 的 2/3） |
| `Block` | 一层 Transformer（pre-norm + 残差 + 梯度检查点） |
| `MiniLM` | 完整模型 |

### 三个必须注意的实现细节

**1. `force_mem_efficient_attention()` 必须调用**

不加这个，SDPA 会静默回退到 math 后端，
注意力显存从 145MB 涨到 2192MB（seq=1024）→ 直接 OOM。

**2. RoPE 的 cos/sin 必须是 buffer，不能是普通属性**

普通属性不受 `nn.Module.to(device)` 管辖，
把模型搬到 GPU 时它们会留在 CPU 上，前向时报"两个设备"错误。

**3. SDPA 要求 4D 输入**

`(batch, heads, seq, head_dim)`。传 3D 会触发
`All fused kernels requires query, key and value to be 4 dimensional`
并退回非融合路径 —— 性能大幅下降。

### 辅助函数

- `model_summary()` —— 打印结构摘要 + 参数量交叉校验
- `estimate_training_memory()` —— 估算显存占用，跑之前判断会不会超售

---

## `tokenizer.py` —— 分词器

字节级 BPE（ByteLevel），词表 **32768**。

**为什么是字节级**：永远不会出现 UNK。任何字符（emoji、生僻字、乱码）
都能被拆成 UTF-8 字节表示。词级分词遇到未登录词只能吐 UNK，
而中文的未登录词问题很严重。

**特殊 token（ChatML 格式，4 个）**：

| token | 用途 |
|---|---|
| `<\|endoftext\|>` | 文档分隔（预训练拼接长序列时用） |
| `<\|im_start\|>` | 一条消息开始 |
| `<\|im_end\|>` | 一条消息结束，也是生成停止信号 |
| `<\|pad\|>` | 填充 |

**为什么必须在预训练前定好**：模型需要在预训练阶段就见到这些 token，
才能学到它们的语义。预训练完才加的话，它们的 embedding 是随机初始化的，
需要额外训练才能学好 —— 而且效果通常不好。

因此**词表一旦定下，全流程必须复用同一个 `tokenizer.json`**。

关键函数：
- `train_tokenizer()` —— 训练（接受迭代器，不把语料全读进内存）
- `verify_special_tokens()` —— 验证特殊 token 能编码成**单个** id
  （被拆成多个字符 token 的话，对话格式会彻底失效，但 loss 曲线看不出来）
- `verify_roundtrip()` —— 验证无损。注意要传 `add_special_tokens=False`
- `compression_report()` —— 实测压缩率，用来反推需要多少原始语料

---

## `chat_template.py` —— 对话模板

**这是 SFT 和推理之间最容易出错的地方。**

### ChatML 格式

```
<|im_start|>system
你是一个有用的AI助手。<|im_end|>
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
你好！有什么可以帮你的吗？<|im_end|>
```

### `encode_conversation()` —— loss 掩码

SFT 的核心：**只在 assistant 回复上算 loss**。

| 部分 | input_ids | labels |
|---|---|---|
| system / user 消息 | ✅ 作为输入 | `-100`（不学） |
| `<\|im_start\|>assistant\n` 前缀 | ✅ 作为输入 | `-100`（不学） |
| assistant 正文 | ✅ 作为输入 | **token 本身**（要学） |
| `<\|im_end\|>` | ✅ 作为输入 | **token 本身**（要学！） |

两个容易错的点：

1. **不掩码 prompt 的后果**：模型会学着预测用户的提问，
   SFT 后倾向于复述用户的话。**loss 曲线完全看不出来。**

2. **`<|im_end|>` 必须算 loss**：这是模型学会"什么时候该停下"的唯一途径。
   标成 -100 的话，模型永远学不会停止，推理时会一直说到撞上长度上限。

### 实现技巧

不硬编码 token 位置，而是 tokenize `<|im_start|>assistant\n` 得到实际的 id 序列。
因为这段文本可能被编码成多个 token，具体几个取决于分词器的 merge 规则，
硬编码必然出错。

---

## `utils.py` —— 工具

### 显存监控（本项目最重要的工具）

`gpu_memory()` 区分几个容易混淆的概念：

| 字段 | 含义 |
|---|---|
| `free` | 驱动层面还有多少没被占用 —— **判断会不会超售的依据** |
| `allocated` | PyTorch 当前实际用着的张量内存 |
| `reserved` | PyTorch 向驱动申请下来、自己管理的总池子 |

**判断超售要看 `reserved` 和 `free` 的关系**，不是 `allocated`。
因为 PyTorch 会持有 reserved 不放。

`check_vram_available()` 在训练前主动检查，把"静默变慢"变成"明确告警"。

### Checkpoint

- `save_checkpoint()` —— **先写临时文件再原子重命名**，
  写到一半断电不会留下损坏文件覆盖上一个好的
- `find_latest_checkpoint()` —— 按步数找最新的，用于自动续训
- `rotate_checkpoints()` —— 滚动保留最近 N 个（每个约 1.3GB，不清理会写满磁盘）

### 其他

`AverageMeter`（滑动平均，用于看 loss 趋势）、
`ThroughputTracker`（吞吐 + ETA 估算）、
`get_logger`（同时输出到控制台和文件）。
