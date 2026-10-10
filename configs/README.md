# `configs/` —— 超参配置

所有可调参数都放在这里。**改配置不改代码** —— 这在 6GB 显存这种紧约束下尤其重要，
因为调参是常态而非例外，写死在代码里既容易改错，也无法记录"当时用的是哪套配置"。

配置文件本身也会被复制到每个 checkpoint 目录下（`save_config()`），
这样几周后回头看某个 checkpoint，能确切知道它是用什么配置训出来的。

---

## 文件一览

| 文件 | 作用 |
|---|---|
| `model_0.3b.yaml` | 模型结构（层数、维度、注意力头数等） |
| `pretrain.yaml` | 预训练超参（batch、lr、步数、保存间隔等） |
| `data.yaml` | 数据源、配额、下载并发数 |

---

## `model_0.3b.yaml`

```yaml
vocab_size        : 32768    # 含 4 个特殊 token；必须 ≤ 65536（uint16 限制）
d_model           : 1024
n_layers          : 24
n_heads           : 16       # head_dim = d_model / n_heads = 64
n_kv_heads        : 8        # GQA 2:1
intermediate_size : 2816     # SwiGLU，≈ 4*d_model*2/3 取到 256 的倍数
max_seq_len       : 1024
rope_theta        : 10000.0
rms_norm_eps      : 1.0e-5
dropout           : 0.0
tie_word_embeddings: true
```

**实测参数量 316.72M。** `minilm/config.py` 里有解析式公式，
`scripts/00_setup.py` 会把它和模型实测值对比 —— 两者必须相等，
不等说明实现和设计不同步。

### 改动时的约束（`__post_init__` 会校验）

| 参数 | 约束 | 不满足的后果 |
|---|---|---|
| `d_model` | 必须能被 `n_heads` 整除 | 无法均匀切分注意力头 |
| `n_heads` | 必须能被 `n_kv_heads` 整除 | GQA 的 KV 头无法平摊到 Q 头 |
| `vocab_size` | ≤ 65536 | uint16 存不下 token id，写入时会静默截断 |

### 几个设计的取舍

**`n_kv_heads = 8`（GQA 2:1）**
推理时 KV cache 直接减半，训练时省激活。
对 RL 阶段至关重要 —— 要同时装 policy 和 reference 两个模型，没有 GQA 放不下。

**`vocab_size = 32768`**
权衡点：词表越大中文压缩越好，但 embedding 参数占比越高。
32768 时 embedding 占 33.6M = 总量的 10.6%。
（minimind 用 6400 词表，embedding 只占 5%，但中文压缩率更差。）

**`max_seq_len = 1024`**
实测 mem_efficient 注意力下，seq 从 1024 提到 2048 显存几乎不涨
（同为 145MB，因为显存是 O(seq) 不是 O(seq²)）。
但计算量随 seq 线性增长，预训练先用 1024 换速度。

**`tie_word_embeddings = true`**
省 33.6M 参数（10.6%），对小模型很值。

---

## `pretrain.yaml`

```yaml
micro_batch_size : 4         # 【硬约束】调到 6 就显存超售
grad_accum_steps : 32        # → 有效 batch 131k tokens/step
seq_len          : 1024
lr               : 3.0e-4
min_lr_ratio     : 0.1       # cosine 衰减到 3e-5
warmup_steps     : 2000
weight_decay     : 0.1
betas            : [0.9, 0.95]
grad_clip        : 1.0
optimizer        : adamw8bit # 【必需】fp32 AdamW 会多占 1.9GB → 超售
dtype            : fp16      # 【必需】sm_75 无 bf16 硬件加速
gradient_checkpointing: true # 【必需】关掉后 batch=2 都超售
max_steps        : 15260
save_interval    : 500
keep_last_n_checkpoints: 3
```

### 实测依据（改这些值之前请先读）

| 配置 | 显存 reserved | 吞吐 | 判断 |
|---|---:|---:|---|
| micro_batch=2 | 3.91 GB | 4035 tok/s | ✅ |
| **micro_batch=4** | **4.90 GB** | **4618 tok/s** | ✅ **当前** |
| micro_batch=6 | 6.40 GB | 2846 tok/s | ❌ 超售 |
| micro_batch=8 | 7.79 GB | 2356 tok/s | ❌ 严重超售 |
| micro_batch=2，关梯度检查点 | 6.54 GB | 863 tok/s | ❌ 超售 |
| micro_batch=4，关梯度检查点 | 10.09 GB | 357 tok/s | ❌ 严重超售 |

> ⚠️ micro_batch=4 能到 4.90GB 的前提是
> `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 生效
> （不开是 5.40GB，会轻度超售）。
> 该环境变量由 `minilm/__init__.py` 自动设置，要求脚本先 `import minilm`。

### 为什么 optimizer 必须是 adamw8bit

普通 fp32 AdamW 的优化器状态（m 和 v 各一份 fp32）要 2.36GB，
总显存涨到 6.7GB 超过物理显存 → 触发 Windows 显存超售 →
**训练慢 16 倍（实测 305 tok/s vs 4618 tok/s），而且不报错**。

8-bit 版本只要 0.59GB。这是**避免超售的必需项**，不是可选优化。

### 调整 token 预算

改 `max_steps` 即可：
```
步数 = 目标 tokens / (micro_batch × grad_accum × seq_len)
     = 目标 tokens / 131072
```

| 目标 | 步数 | 预计耗时 |
|---|---:|---:|
| 1B tokens | 7,630 | 2.5 天 |
| 1.5B tokens | 11,444 | 3.8 天 |
| **2B tokens** | **15,260** | **5.0 天** |

---

## `data.yaml`

数据源、配额、下载参数。详见 `minilm/data/README.md`。

```yaml
target_tokens : 2000000000   # 训练集目标
val_tokens    : 2000000      # 验证集（从语料开头切，不参与训练）
download_workers: 8          # 并发下载线程
sources:
  - name: minimind_pretrain  # 主料，约 1.8B tokens
  - name: wikipedia          # 补长文本，约 0.2B tokens
```

**各源配额用 token 数而不是文件大小**，因为分词器压缩率事先不知道，
按 token 数控制才能精确达到目标。`build_bin` 在达到配额后停止读取该源。

### 调整某个源的占比

改对应的 `target_tokens`。总和应该等于 `target_tokens`。

### 添加新数据源

```yaml
- name: my_source
  dir: my_source              # data/raw/ 下的子目录
  include_files: [data.jsonl] # 只读指定文件
  text_field: text            # 正文所在的字段名
  target_tokens: 100000000
  license: "..."
```

**`include_files` 是必要的**：minimind 把预训练/SFT/RL 数据都放在同一个目录里，
不显式指定的话会把 SFT 数据当成预训练语料读进去。
