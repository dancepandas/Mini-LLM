# `scripts/` —— 按序号执行的入口脚本

六个阶段，按编号顺序执行。每个脚本都可以独立重跑。

```
00_setup.py            环境自检 + 吞吐基准     【决策门，必须最先跑】
01_train_tokenizer.py  训练 BPE 分词器
02_prepare_data.py     下载数据 / 清洗分词打包
03_pretrain.py         预训练（约 5 天，可断点续训）
04_sft.py              SFT 指令微调
05_rl_grpo.py          GRPO 强化学习
06_chat.py             命令行对话
```

---

## ⚠️ 所有脚本的共同约定

### 1. 导入顺序：必须先 `import minilm`，再 `import torch`

```python
import sys
sys.path.insert(0, str(PROJECT_ROOT))
import minilm          # ← 必须在前！它设置 PYTORCH_CUDA_ALLOC_CONF
import torch           # ← 必须在后
```

`minilm/__init__.py` 会设置 `expandable_segments:True`，
这个环境变量只在 CUDA 分配器**首次初始化**时被读取。
顺序反了会退回默认分配器 → reserved 从 4.90GB 涨到 5.40GB → 触发显存超售。

### 2. 脚本可以从任意工作目录执行

每个脚本都会自己把项目根目录加进 `sys.path`。

---

## `00_setup.py` —— 环境自检 + 吞吐基准

**这是整个项目的决策门。**

```bash
python scripts/00_setup.py           # 完整自检 + 基准（约 5 分钟）
python scripts/00_setup.py --quick   # 只做环境和模型检查，跳过基准
```

做四件事：

1. **环境检查** —— Python 版本、11 个必需包是否安装
2. **GPU 检查** —— 显存、算力等级、mem_efficient 后端是否真的可用
3. **模型核对** —— 构建模型，**交叉校验解析式参数量和实测参数量**
   （开发时靠这个发现了"漏算输出层 RMSNorm"的 bug）
4. **吞吐基准** —— 用真实配置实测不同 micro_batch 的吞吐和显存，
   推算真实 ETA

**为什么必须先跑这个**：Windows 的显存超售是**静默的**（不报错，只是慢）。
实测超售时 305 tok/s，不超售 4618 tok/s —— 差 15 倍。
不先测就开训，可能跑 3 天才发现来不及。

输出示例：
```
  测试 micro_batch=2 ...    504 ms/step |  4061 tok/s | 峰值 3.41 GB
  测试 micro_batch=4 ...    887 ms/step |  4618 tok/s | 峰值 4.30 GB
  测试 micro_batch=8 ...   3478 ms/step |  2356 tok/s | 峰值 6.52 GB  <== 显存超售！

  最佳配置: micro_batch=4, 4618 tok/s
  → 预计预训练耗时: 120.3 小时 (5.0 天)
```

---

## `01_train_tokenizer.py` —— 训练分词器

```bash
python scripts/01_train_tokenizer.py
python scripts/01_train_tokenizer.py --vocab-size 32768 --n-docs 20000
python scripts/01_train_tokenizer.py --offline-samples data/raw/samples.txt
```

产出 `tokenizer/tokenizer.json`。

**为什么这一步必须在预训练之前，而且只能做一次**：
特殊 token（`<|im_start|>` 等）必须在预训练阶段就存在于词表中，
模型才能学到它们的语义。预训练完才加的话，它们的 embedding 是随机初始化的。

**词表一旦确定就不能改** —— 改了要重新预训练（5 天）。

脚本做的四件事：
1. 采样语料（本地文件或从魔塔流式采样）
2. 训练字节级 BPE
3. 配置后处理器（预训练编码时自动在末尾附加 `<|endoftext|>`）
4. **验证**：
   - 特殊 token 能编码成**单个** id（被拆成多字符的话对话格式会失效）
   - 无损往返（含 emoji、生僻字、代码）
   - 实测压缩率 → 反推需要多少原始语料

**实测压缩率：0.877 tokens/汉字，1.574 字符/token。**

---

## `02_prepare_data.py` —— 数据准备

```bash
python scripts/02_prepare_data.py --stage download   # 只下载
python scripts/02_prepare_data.py --stage tokenize   # 只清洗分词
python scripts/02_prepare_data.py --stage all        # 两步都做
python scripts/02_prepare_data.py --stage download --only wikipedia  # 只下某一源
```

**为什么分两个阶段**：下载是网络瓶颈（以小时计），分词是 CPU 瓶颈。
分开后各自可以独立重跑和排查。

### 阶段 A：下载

从魔塔并发下载到 `data/raw/<name>/`。

- **跳过已存在的文件**（断点续传），中断后重跑不会重复下载
- 按大小排序，**小文件先下**（能尽快拿到可用数据）
- 用 8 并发分块下载，聚合约 6 MB/s

### 阶段 B：清洗 + 分词 + 打包

产出 `data/pretrain/{train,val}.bin` + `meta.json`。

- 验证集从每个数据源的开头切一小段，保证分布和训练集一致
  （只用第一个源做验证的话，验证 loss 不能代表整体）
- 达到 token 配额就停止读取该源
- **检查词表是否超过 uint16 上限**（超了会静默截断 token id）

实测分词速度约 **51 万 tokens/s**。

---

## `03_pretrain.py` —— 预训练

```bash
python scripts/03_pretrain.py                    # 从头开始
python scripts/03_pretrain.py --resume           # 自动找最新 checkpoint 续训
python scripts/03_pretrain.py --resume-from checkpoints/pretrain/step_0005000.pt
python scripts/03_pretrain.py --max-steps 3000   # 覆盖总步数
python scripts/03_pretrain.py --micro-batch 2    # 覆盖 batch（OOM 时用）
```

### 断点续训是必需品

要连跑 5 天。桌面机断电、系统更新重启、误关窗口都是**大概率事件**。
没有续训，一次意外就要从头再来。

- 每 500 步存 checkpoint，滚动保留最近 3 个（每个约 1.3GB）
- `--resume` 自动找最新的一步接着跑
- **GradScaler 的状态也会恢复**（不恢复的话 loss scale 会从初始值重爬，
  续训后前几百步的梯度实际偏小，loss 会出现一个台阶）
- 数据采样器会推进到对应位置，保证数据顺序一致

### 监控指标

| 指标 | 异常信号 |
|---|---|
| `loss` | 不降 → 检查 lr / 数据 / 代码 |
| `grad_norm` | 突然暴涨 → 即将发散 |
| 吞吐 tok/s | 突然下降 → 显存碎片化或超售 |
| `mem` reserved | 超过物理空闲显存 → 超售 |

```bash
tensorboard --logdir checkpoints/pretrain/tb
```

### 内置保护

- 梯度裁剪（fp16 训练必须开，防梯度爆炸）
- **NaN/Inf 检测**：检测到非有限梯度就跳过该步，不让坏更新污染参数
- 显存超售警告

---

## `04_sft.py` —— SFT 指令微调

```bash
python scripts/04_sft.py
python scripts/04_sft.py --pretrained checkpoints/pretrain/final.pt
python scripts/04_sft.py --data data/raw/minimind/sft_t2t_mini.jsonl
python scripts/04_sft.py --max-samples 5000   # 调试用
```

**核心是 loss 掩码**：只在 assistant 回复上算 loss，prompt 部分标 `-100`。

不掩码的后果：模型会学着预测用户的提问，SFT 后倾向于复述用户的话。
**loss 曲线完全看不出来这个错误。**

脚本会自动检查"被监督 token 比例"：
- < 15% → 偏低，可能掩码有问题或回复太短
- \> 90% → 过高，可能掩码没生效

这两个警告能有效抓住 SFT 最常见的配置错误。

超参：lr `2e-5`（比预训练小一个量级，防止灾难性遗忘）、2 epochs。

---

## `05_rl_grpo.py` —— GRPO 强化学习

```bash
python scripts/05_rl_grpo.py
python scripts/05_rl_grpo.py --steps 200 --group-size 4
python scripts/05_rl_grpo.py --data data/raw/minimind/agent_rl_math.jsonl
```

### 为什么用 GRPO 而不是 PPO

PPO 需要和策略同尺寸的 **critic 网络**。6GB 显存装不下
policy + reference + critic 三个 0.3B 模型。

GRPO 用**同一道题一组采样的相对好坏**估计优势：
```
advantage_i = (reward_i - mean) / std
```
完全不需要 critic。来自 DeepSeek-R1，是小算力做 RL 的主流选择。

### 为什么用 LoRA

需要同时驻留 policy（训练中）和 reference（冻结）。全量微调放不下。
LoRA 把可训练参数降到 1% 量级。

**LoRA 是自己实现的**（`LoRALinear` 类），没依赖 `peft`。

### 奖励设计

| 部分 | 权重 | 说明 |
|---|---|---|
| 正确性 | 1.0 | 抽取答案与 gold 归一化后精确匹配 |
| 格式 | 0.1 | 有 `\boxed{}` 等规范标记 |

**格式权重必须远小于正确性**，否则模型会学会"输出格式漂亮的废话"骗分
（reward hacking）。

### 最重要的监控指标：有效题占比

GRPO 有个固有失效模式：**如果同一题的一组 rollout 全对或全错，
组内标准差为 0 → 优势全为 0 → 梯度为 0，这一步完全白跑。**

脚本统计"有效题占比"（组内既有对的也有错的）。
这个指标比 reward 本身更能反映 RL 是否在起作用。
低于 30% 会告警，说明题目难度和模型能力不匹配 —— 应该换数据，而不是硬跑。

### 收尾时会把 LoRA 合并回主权重

导出成可独立使用的模型，不需要额外的 LoRA 加载逻辑。

---

## `06_chat.py` —— 命令行对话

```bash
python scripts/06_chat.py                       # 交互模式
python scripts/06_chat.py --prompt "你好"        # 单次提问
python scripts/06_chat.py --model checkpoints/sft/final.pt
python scripts/06_chat.py --temperature 0.9 --repetition-penalty 1.1
```

自动按优先级找模型：`rl/final.pt` → `sft/final.pt` → `pretrain/final.pt`
→ 各自最新的 step checkpoint。

**必须使用和训练时完全一致的 chat template** ——
这也是为什么模板集中在 `minilm/chat_template.py`。

功能：
- 流式输出（逐 token 打印，不用等全部生成完）
- 多轮对话（保留最近 8 轮，防止超上下文）
- 命令：`/reset` 清空历史，`/exit` 退出
- 遇到 `<|im_end|>` 停止生成（这是模型学到的回合结束信号）

### 常见问题

**问：模型答非所问？**
先检查 chat template 是否一致 —— 训练和推理用了不同模板是最常见原因，
而且不会报错。

**问：模型一直复读？**
小模型的常见问题。调高 `--repetition-penalty`（如 1.1~1.2），
或降低 `--temperature`。
