"""
阶段 5：RL —— GRPO（Group Relative Policy Optimization）+ LoRA。

## 为什么用 GRPO 而不是 PPO

    PPO 需要一个和策略模型同等大小的**价值网络（critic）**来估计优势。
    在 6GB 显存上，policy + reference + critic 三个 0.3B 模型根本放不下。

    GRPO 的核心洞察：**用同一道题的一组采样结果的相对好坏来估计优势，
    从而完全不需要 critic。**

        advantage_i = (reward_i - mean(rewards)) / std(rewards)

    组内平均分当作基线。比平均好的回答被强化，比平均差的被抑制。
    这是 DeepSeek-R1 用的方法，现在是小算力做 RL 的主流选择。

## 为什么用 LoRA

    即使不要 critic，GRPO 也需要同时驻留：
        policy（要训练，有权重+梯度+优化器状态）
        reference（冻结，用于算 KL 惩罚）
    全量微调的话，显存需求约为 2 × (0.6 + 0.6 + 0.6) = 3.6GB，
    加上 rollout 时的 KV cache 和激活，6GB 卡会超售。

    LoRA 只训练低秩增量矩阵，可训练参数量降到 1% 量级，
    优化器状态几乎不占显存。reference 模型用 fp16 冻结常驻，约 0.6GB。

## 关于 reward 设计（这是 RL 成败的关键）

    **只有可程序化验证的奖励才靠得住。** 用另一个模型打分需要额外的模型和
    训练好的奖励模型，本项目没有条件。

    奖励分两部分，且**必须分离**：
        正确性奖励：答案和标准答案精确匹配 → 1.0，否则 0.0
        格式奖励  ：输出符合要求的格式 → 小权重加分
    关键约束：**格式奖励的权重必须远小于正确性奖励**。
    否则模型会学会"只输出格式正确的废话"来骗分。

## 关于难度匹配

    GRPO 有个致命的失效模式：如果同一道题的一组 rollout **全部答错**，
    那么组内 reward 全为 0，标准差为 0，优势全为 0 → **梯度为 0**，
    这一步完全白跑。全部答对同理（没有改进空间）。

    所以题目难度必须匹配模型当前能力。本脚本会统计"有效题目比例"
    （组内既有对的也有错的），这个指标比 reward 本身更能反映 RL 是否在起作用。

用法：
    python scripts/05_rl_grpo.py
    python scripts/05_rl_grpo.py --steps 200 --group-size 4
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import minilm  # noqa: E402,F401

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from minilm.chat_template import extract_response, render_conversation  # noqa: E402
from minilm.config import load_model_config  # noqa: E402
from minilm.model import build_model, force_mem_efficient_attention  # noqa: E402
from minilm.tokenizer import SpecialTokens, load_tokenizer  # noqa: E402
from minilm.utils import (  # noqa: E402
    check_vram_available,
    find_latest_checkpoint,
    format_gpu_memory,
    get_logger,
    human_int,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)


# ===========================================================================
# LoRA —— 自己实现，不依赖 peft
# ===========================================================================
class LoRALinear(nn.Module):
    """给已有的 Linear 层挂一个低秩增量分支。

    原理：
        原权重 W 冻结不动，额外学一个低秩分解 ΔW = B @ A，
        其中 A 的形状是 (r, in_features)，B 是 (out_features, r)，r << d。
        前向变成 Wx + (B@A)x * scaling。

    为什么省显存：
        r=8、d=1024 时，A 和 B 加起来只有 16K 参数，
        相比原来的 1M 参数降低了 60 倍。优化器状态也随之降到可忽略。

    为什么初始化成 A 用随机、B 用全零：
        这样训练开始时 ΔW = B@A = 0，模型行为和原模型完全一致。
        如果两个都随机初始化，一开始就会有一个大的扰动，
        可能直接把已经训好的模型打坏。
    """

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        # 冻结原权重
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        # scaling 用 alpha/r 是 LoRA 论文的做法：
        # 这样调整 r 的时候不必同步调 lr，alpha 保持不变即可
        self.scaling = alpha / r

        # ⚠️ 关键：必须把新参数建在**底座权重所在的设备**上。
        # apply_lora 是在模型已经 .to("cuda") 之后才调用的，
        # 而 nn.Parameter(torch.zeros(...)) 默认在 CPU 上创建，不会自动跟随。
        # 不指定 device 的话，前向时会报
        # "Expected all tensors to be on the same device, but found cuda:0 and cpu"。
        # （这和 RoPE 缓存必须注册成 buffer 是同一类坑：新建的张量不受 .to() 管辖）
        dev = base.weight.device
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=dev))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # A 用 Kaiming 初始化（保证前向方差稳定），B 保持全零
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B 已经是 zeros，不用再初始化

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.base(x)
        # 注意：即使 B 全零也要保留计算图，否则第一步梯度传不到 A
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return out + lora_out * self.scaling


def apply_lora(model, r: int = 8, alpha: float = 16.0, target_modules=("q_proj", "v_proj")) -> int:
    """给模型的指定 Linear 层挂上 LoRA，返回被替换的层数。

    默认只对 q_proj / v_proj 挂 LoRA，这是原论文的建议：
        注意力里 Q 和 V 的适配收益最高，对 MLP 挂 LoRA 收益小但显存开销大。
        小算力场景下少挂几层能显著省显存。

    ## ⚠️ 关键：必须先冻结整个底座模型（踩过的坑）

        只冻结被替换的那几个模块是**不够的** —— 模型其余部分
        （MLP、o_proj、k_proj、embedding）会保持 requires_grad=True，
        结果"可训练参数量"是 88% 而不是 LoRA 应有的 0.2%，
        等价于在做接近全参微调：
          - 优化器状态多占约 1.7GB（对 6GB 卡很危险）
          - 150 步 RL 会让模型显著偏离 SFT 权重 → 灾难性遗忘

        本项目的实测症状：日志打印
            "LoRA 已挂载: 48 个层, 可训练 279.66M / 317.4M (88.11%)"
        正确值应该是 **0.69M（0.22%）**。看到这个数字不对就该停下来查。

    做法：先 `for p in model.parameters(): p.requires_grad = False`
    把**整个模型**冻住，再挂 LoRA —— 新建的 lora_A/lora_B 默认
    requires_grad=True，于是它们自动成为唯一可训练的参数。
    """
    # ---- 第一步：冻结整个底座 ----
    n_frozen = 0
    for p in model.parameters():
        if p.requires_grad:
            p.requires_grad = False
            n_frozen += p.numel()

    # ---- 第二步：挂 LoRA（新参数默认可训练）----
    # 注意：先把要替换的模块收集起来再统一替换，
    # 不在遍历 named_modules() 的过程中修改模块树 —— 那样容易出意外。
    to_replace: list[tuple[str, nn.Linear]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(name.endswith(t) for t in target_modules):
            to_replace.append((name, module))

    for name, module in to_replace:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, LoRALinear(module, r=r, alpha=alpha))

    return len(to_replace)


def count_trainable(model) -> tuple[int, int]:
    """返回（可训练参数数, 总参数数）。"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# ===========================================================================
# 奖励函数
# ===========================================================================
# 从模型输出里抽取最终答案的正则。
#
# ## 为什么需要这么多模式（实测校准）
#
#   最初只用"取末尾数字"，结果在真实模型输出上大量误判。实测样例：
#       问 "5+5" → 模型输出 "5+5=10，直接输出答案为10"   → 末尾数字 10 ✅
#       问 "5+5" → 模型输出 "10 + 5 = 15，是正确的"      → 末尾数字 15 ❌
#       问 "5+5" → 模型输出 "5加上5都是10，所以应该是100" → 末尾数字 100 ❌
#
#   一个 0.3B 模型无法稳定输出规范格式，所以抽取必须**宽容**：
#   宁可多认几种写法，也不要因为格式没对上而把正确答案判成错误 ——
#   那会给 reward 注入噪声，直接毁掉 GRPO 的优势信号。
#
#   优先级：明确的答案标记 > 等号/等于后的数字 > 末尾数字
_RE_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_RE_ANSWER_CN = re.compile(r"答案[是为:：]\s*([^\n。，,]+)")
# "= 42" 或 "等于 42"：取**最后一个**匹配（模型的推理链里可能有中间结果）
_RE_EQ_NUM = re.compile(r"[=＝]\s*(-?\d+(?:\.\d+)?)")
_RE_EQUAL_CN_NUM = re.compile(r"等于\s*(-?\d+(?:\.\d+)?)")
_RE_LAST_NUM = re.compile(r"(-?\d+(?:\.\d+)?)")


def extract_answer(text: str) -> str | None:
    """从模型输出里抽取最终答案。

    按优先级依次尝试：
        1. \\boxed{...}          —— 数学推理的标准格式
        2. "答案是 X" / "答案：X"
        3. 最后一个 "= X" 或 "等于 X"  —— 覆盖模型自发的算式写法
        4. 文本里的最后一个数字      —— 兜底

    为什么要有多种模式：
        0.3B 的小模型没法稳定遵循某一种格式。放宽抽取规则能让更多回答
        被正确评判，否则会因为"格式没对上"把正确答案判成错误，
        导致 reward 信号充满噪声 —— 这在 GRPO 里是致命的。
    """
    if not text:
        return None

    m = _RE_BOXED.search(text)
    if m:
        return m.group(1).strip()

    m = _RE_ANSWER_CN.search(text)
    if m:
        return m.group(1).strip()

    # 只看答案部分（去掉思维链），避免把推理过程中的中间数字当答案
    _, answer = extract_response(text)
    tail = answer.strip() or text.strip()

    # 最后一个 "= X"
    eqs = _RE_EQ_NUM.findall(tail)
    if eqs:
        return eqs[-1]
    # 最后一个 "等于 X"
    cns = _RE_EQUAL_CN_NUM.findall(tail)
    if cns:
        return cns[-1]
    # 兜底：最后一个数字
    nums = _RE_LAST_NUM.findall(tail)
    if nums:
        return nums[-1]

    return None


def normalize_answer(ans: str) -> str:
    """归一化答案，用于比对。

    处理常见的等价写法差异：
        去掉空格、千分位逗号、末尾句号、"元/个/人"这类单位后缀
        1.0 → 1、0.50 → 0.5
    """
    if ans is None:
        return ""
    s = ans.strip()
    # 去掉常见的单位后缀
    s = re.sub(r"[元个只人条天次米千米公斤斤岁本页分秒%]$", "", s)
    s = s.replace(",", "").replace(" ", "").replace("$", "").rstrip("。.")
    # 数值归一化：去掉末尾多余的 0
    try:
        f = float(s)
        if f == int(f):
            return str(int(f))
        return f"{f:.6g}"
    except (ValueError, TypeError):
        return s


def compute_reward(
    response_text: str,
    golds: str | list[str],
    format_weight: float = 0.1,
    require_think: bool = False,
) -> tuple[float, dict]:
    """计算单条回答的奖励。

    返回 (总奖励, 明细字典)。

    ## 参数 golds 支持"多个可接受答案"

        数据里的 gt 是列表（如 ['13575','40878','4']），
        只要模型答出**其中任意一个**就算对。
        这是必要的：一道题可能有多种正确表示（整数/小数、带不带单位），
        只认一个会大量误判，让 reward 信号充满噪声。

    ## 奖励必须分离正确性和格式

        正确性：抽出的答案和任一 gold 归一化后相等 → 1.0
        格式  ：有 \\boxed{} 或规范的答案标记 → 一个小奖励

        为什么格式权重必须小（这里 0.1）：
            如果格式奖励够大，模型会发现"只要把格式写对就能拿分"，
            于是学会输出格式漂亮但内容胡说八道。这在 RL 里是经典的 reward hacking。
            0.1 的权重保证"答对"永远比"格式好"更值钱。
    """
    if isinstance(golds, str):
        golds = [golds]

    detail: dict = {}
    reward = 0.0

    pred = extract_answer(response_text)
    detail["extracted"] = pred

    # ---- 正确性：匹配任一可接受答案 ----
    ok = False
    if pred is not None:
        pred_norm = normalize_answer(pred)
        ok = any(pred_norm == normalize_answer(g) for g in golds)
    if ok:
        reward += 1.0
    detail["correct"] = ok

    # ---- 格式 ----
    fmt = 0.0
    if _RE_BOXED.search(response_text or ""):
        fmt += 1.0
    if require_think and "<think>" in (response_text or "") and "</think>" in (response_text or ""):
        fmt += 0.5
    detail["format"] = fmt
    reward += fmt * format_weight

    return reward, detail


# ===========================================================================
# 数据加载
# ===========================================================================
def load_rl_data(path: Path, tokenizer, max_prompt_len: int, logger) -> list[dict]:
    """加载 RL 训练数据，提取出 (提示词, 标准答案) 对。

    ## 两种数据布局（实测 minimind 的数据）

        agent_rl_math.jsonl : {"conversations": [...], "gt": "答案"}
                              → **有独立的 gt 字段，最理想**
        dpo.jsonl / 其它     : {"conversations": [...]}
                              → 只能从 assistant 的回复里反抽答案

    优先用 `gt` 字段。从回复里反抽答案是有风险的：
        - 抽取正则可能匹配错位置
        - assistant 的回复里可能包含多个数字，抓到错误的那个
        - 归一化规则不完美时会引入噪声
    独立的 gt 字段没有这些问题。

    ## 安全检查

        **prompt 里绝对不能包含 gold 答案** —— 否则模型直接抄，
        reward 恒为 1，RL 完全失效（而且 loss 曲线看起来"训练得很好"）。
        这里显式检查并丢弃这类样本。
    """
    samples = []
    n_skipped = 0
    n_from_gt = 0

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                n_skipped += 1
                continue

            convs = obj.get("conversations") or obj.get("messages") or []
            if not convs:
                n_skipped += 1
                continue

            # ---- 找 prompt（第一组 user 消息）----
            # 多轮数据里，用第一轮 user 提问作 prompt 最简单可靠。
            # 用更靠后的轮次会让 prompt 变长，在 6GB 显存下不划算。
            prompt_msgs = []
            for m in convs:
                if m.get("role") == "user":
                    prompt_msgs.append({"role": "user", "content": m.get("content") or ""})
                    break

            if not prompt_msgs:
                n_skipped += 1
                continue

            # ---- 取 gold 答案 ----
            # ⚠️ 关键：gt 字段是**列表**（含多个可接受答案），不是单个值。
            # 实测 agent_rl_math.jsonl：
            #     "算算7109*2920"  → gt = ['20758280']                 （单个答案）
            #     多问题目          → gt = ['2797','196','2929130']     （多个答案）
            #     "calculate 2**2" → gt = ['13575','40878','4']        （正确值是 4）
            # 如果按 str(gt) 处理会得到 "['20758280']" 这种字符串，
            # 和模型输出的 "20758280" **永远匹配不上** → reward 恒为 0
            # → GRPO 组内无差异 → 梯度为零 → RL 完全白跑，且看不出来。
            golds: list[str] = []
            for key in ("gt", "answer", "gold", "label", "target"):
                v = obj.get(key)
                if v is None:
                    continue
                if isinstance(v, (list, tuple)):
                    golds = [str(x).strip() for x in v if str(x).strip()]
                elif isinstance(v, str):
                    # 字符串可能是 JSON 列表的字面量，试着解析
                    s = v.strip()
                    if s.startswith("[") and s.endswith("]"):
                        try:
                            parsed = json.loads(s.replace("'", '"'))
                            golds = [str(x).strip() for x in parsed if str(x).strip()]
                        except json.JSONDecodeError:
                            golds = [s] if s else []
                    else:
                        golds = [s] if s else []
                else:
                    golds = [str(v).strip()] if str(v).strip() else []
                if golds:
                    n_from_gt += 1
                    break

            # 没有独立字段才从 assistant 回复里反抽
            if not golds:
                for m in convs:
                    if m.get("role") == "assistant":
                        content = m.get("content") or ""
                        got = extract_answer(content)
                        if got:
                            golds = [got]
                        break

            # 没有可验证的答案就跳过。
            # agent_rl.jsonl 里有大量 gt=[] 的样本（开放问答，无标准答案），
            # 拿去训练只会污染 reward 信号。
            if not golds:
                n_skipped += 1
                continue

            prompt_text = render_conversation(prompt_msgs, add_generation_prompt=True)
            ids = tokenizer.encode(prompt_text, add_special_tokens=False).ids

            if len(ids) > max_prompt_len:
                n_skipped += 1
                continue

            # 安全检查：任何一个 gold 出现在 prompt 里都算答案泄漏，必须丢弃
            if any(g in prompt_text for g in golds):
                n_skipped += 1
                continue

            samples.append({
                "prompt_ids": ids,
                "golds": golds,
                "prompt_text": prompt_text,
                "prompt_len": len(ids),
            })

    logger.info(f"  RL 数据: {len(samples):,} 条有效样本"
                f"（其中 {n_from_gt:,} 条用了独立的 gt 字段，跳过 {n_skipped:,}）")

    if samples:
        # 抽查前几条，确认 gold 抽取正确 —— 抽错了 RL 就完全学不到东西
        logger.info("  抽样检查（前 3 条）:")
        for s in samples[:3]:
            logger.info(f"    prompt 末尾: ...{s['prompt_text'][-45:]!r}")
            logger.info(f"    可接受答案 : {s['golds']}")

    return samples


# ===========================================================================
# GRPO 核心
# ===========================================================================
@torch.no_grad()
def rollout_group(
    model,
    tokenizer,
    prompt_ids: list[int],
    group_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_id: int,
) -> list[dict]:
    """对同一个 prompt 采样一组回答（GRPO 的基础操作）。

    返回每条回答的 input_ids、生成部分的 token 数、以及文本。

    为什么用较高的 temperature：
        GRPO 需要组内回答有**差异**才能算出非零优势。
        temperature 太低会导致一组回答几乎一样，reward 全同 → 优势全零 → 梯度为零。
        0.8~1.0 是比较合适的范围。
    """
    results = []
    for _ in range(group_size):
        ids = torch.tensor([prompt_ids], dtype=torch.long, device="cuda")
        generated = ids
        for _ in range(max_new_tokens):
            ctx = generated[:, -model.cfg.max_seq_len:]
            out = model(ctx, logits_last_only=True)
            logits = out["logits"][:, -1, :].float() / temperature

            # top-p 采样
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            probs = torch.softmax(sorted_logits, dim=-1)
            cum = probs.cumsum(dim=-1)
            remove = cum - probs > top_p
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = torch.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)

            if nxt.item() == eos_id:
                break
            generated = torch.cat([generated, nxt], dim=1)

        gen_ids = generated[0].tolist()
        text = tokenizer.decode(gen_ids[len(prompt_ids):], skip_special_tokens=False)
        results.append({
            "input_ids": gen_ids,
            "completion_len": len(gen_ids) - len(prompt_ids),
            "text": text,
        })
    return results


def compute_logprobs(model, input_ids: list[int], prompt_len: int) -> torch.Tensor:
    """算一条序列在 completion 部分的逐 token log 概率。

    传入的是**整个序列**（prompt + completion），返回的只保留 completion 部分。
    这是必要的 —— 因为注意力是因果的，completion 的每个 token 的概率
    都依赖于它前面的完整上下文（含 prompt）。
    """
    ids = torch.tensor([input_ids], dtype=torch.long, device="cuda")
    out = model(ids)
    logits = out["logits"][0]  # (seq, vocab)

    # 位置 i 的 logits 预测的是位置 i+1 的 token，
    # 所以要错开一位：用 logits[:-1] 对 input_ids[1:]
    logits = logits[:-1]
    targets = ids[0, 1:]

    logp = torch.log_softmax(logits.float(), dim=-1)
    token_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

    # 只保留 completion 部分。
    # 偏移关系：targets[k] 对应原序列位置 k+1，
    # 所以 completion 从原序列的 prompt_len 位置开始，对应 targets 的索引 prompt_len-1
    return token_logp[prompt_len - 1:]


def main() -> int:
    parser = argparse.ArgumentParser(description="阶段 5：GRPO 强化学习")
    parser.add_argument("--model-config", default=str(PROJECT_ROOT / "configs" / "model_0.3b.yaml"))
    parser.add_argument("--sft-model", default=None, help="SFT 后的模型，默认自动查找")
    parser.add_argument("--tokenizer", default=str(PROJECT_ROOT / "tokenizer" / "tokenizer.json"))
    parser.add_argument("--data", default=None, help="RL 数据 jsonl")
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "checkpoints" / "rl"))
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--group-size", type=int, default=4, help="每题采样几条回答")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--max-prompt-len", type=int, default=384)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--kl-coef", type=float, default=0.04, help="KL 惩罚系数")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--save-interval", type=int, default=50)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("rl", out_dir / "train.log")

    logger.info("=" * 68)
    logger.info("  Mini_LM 阶段 5：GRPO 强化学习")
    logger.info("=" * 68)

    set_seed(1337)
    force_mem_efficient_attention()
    logger.info("")
    logger.info(f"  {format_gpu_memory()}")
    check_vram_available(4.5, raise_on_fail=False)

    # ---- 分词器 ----
    tokenizer = load_tokenizer(args.tokenizer)
    eos_id = tokenizer.token_to_id(SpecialTokens.IM_END)

    mcfg = load_model_config(args.model_config)

    # ---- Policy 模型（要训练）----
    logger.info("")
    policy = build_model(mcfg, "cuda")

    # 找 SFT 权重
    if args.sft_model:
        sft_path = Path(args.sft_model)
    else:
        sft_path = PROJECT_ROOT / "checkpoints" / "sft" / "final.pt"
        if not sft_path.exists():
            sft_path = find_latest_checkpoint(PROJECT_ROOT / "checkpoints" / "sft")

    if sft_path is None or not sft_path.exists():
        logger.error("找不到 SFT 权重。请先完成阶段 4。")
        return 1

    logger.info(f"  从 SFT 权重初始化: {sft_path.name}")
    load_checkpoint(sft_path, policy)

    # ---- 挂 LoRA ----
    n_replaced = apply_lora(policy, r=args.lora_r, alpha=args.lora_r * 2)
    trainable, total = count_trainable(policy)
    logger.info(f"  LoRA 已挂载: {n_replaced} 个层, "
                f"可训练 {human_int(trainable)} / {total/1e6:.1f}M "
                f"({trainable/total*100:.2f}%)")

    if trainable == 0:
        logger.error("没有可训练参数，LoRA 挂载失败。")
        return 1

    # ---- Reference 模型（冻结，算 KL 用）----
    logger.info("")
    logger.info("  构建 reference 模型（冻结）...")
    ref = build_model(mcfg, "cuda")
    load_checkpoint(sft_path, ref)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad = False
    logger.info("  reference 已就绪")

    # ---- 优化器（只优化 LoRA 参数）----
    import bitsandbytes as bnb

    lora_params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = bnb.optim.AdamW8bit(lora_params, lr=args.lr, betas=(0.9, 0.95))

    # ---- 数据 ----
    logger.info("")
    if args.data:
        data_path = Path(args.data)
    else:
        candidates = sorted((PROJECT_ROOT / "data" / "raw").rglob("*rl*.jsonl"))
        candidates = [c for c in candidates if "math" in c.name] or candidates
        if not candidates:
            logger.error("找不到 RL 数据。请先下载，或用 --data 指定。")
            return 1
        data_path = candidates[0]

    samples = load_rl_data(data_path, tokenizer, args.max_prompt_len, logger)
    if args.max_samples:
        samples = samples[: args.max_samples]
    if not samples:
        logger.error("没有有效 RL 样本。")
        return 1

    logger.info("")
    logger.info(f"  GRPO 配置: group_size={args.group_size}, "
                f"max_new_tokens={args.max_new_tokens}, KL系数={args.kl_coef}")
    logger.info(f"  每题采样 {args.group_size} 条，理论上 {len(samples)} 题可跑 "
                f"{len(samples)} 步")

    # ---- 训练循环 ----
    logger.info("")
    logger.info("=" * 68)
    logger.info("  开始 GRPO 训练")
    logger.info("=" * 68)

    policy.train()
    t0 = time.time()
    stats_window: list[dict] = []

    for step in range(args.steps):
        sample = samples[step % len(samples)]

        # ---- 1. 采样一组回答 ----
        with torch.no_grad():
            group = rollout_group(
                policy, tokenizer, sample["prompt_ids"],
                group_size=args.group_size,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                eos_id=eos_id,
            )

        # ---- 2. 计算奖励 ----
        rewards, details = [], []
        for g in group:
            # golds 是列表（多个可接受答案），compute_reward 会匹配其中任意一个
            r, d = compute_reward(g["text"], sample["golds"])
            rewards.append(r)
            details.append(d)

        rewards_t = torch.tensor(rewards, dtype=torch.float32, device="cuda")

        # ---- 3. 组内归一化算优势（GRPO 的核心）----
        mean_r = rewards_t.mean()
        std_r = rewards_t.std(unbiased=False)

        # 关键失效检测：组内全同分（全对或全错）→ 标准差为 0 → 优势全零 → 无梯度。
        # 这不是 bug，是 GRPO 的固有特性，但必须显式统计出来，
        # 否则会看到"loss 一直是 0"却不知道为什么。
        degenerate = std_r < 1e-6
        if degenerate:
            advantages = torch.zeros_like(rewards_t)
        else:
            advantages = (rewards_t - mean_r) / (std_r + 1e-8)

        n_correct = sum(1 for d in details if d["correct"])
        stats_window.append({
            "reward": mean_r.item(),
            "std": std_r.item(),
            "n_correct": n_correct,
            "degenerate": degenerate,
        })

        # ---- 4. 只对非退化组做梯度更新 ----
        if degenerate:
            # 全对或全错，跳过更新（更新也没用，梯度是 0）
            if (step + 1) % 5 == 0:
                logger.info(
                    f"  step {step+1:>4}/{args.steps} | 组内无差异"
                    f"({'全对' if n_correct == args.group_size else '全错'})，跳过"
                )
            continue

        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        for g, adv in zip(group, advantages):
            if g["completion_len"] == 0:
                continue

            # ---- policy 的 log 概率 ----
            logp = compute_logprobs(policy, g["input_ids"], len(sample["prompt_ids"]))

            # ---- reference 的 log 概率（用于 KL 惩罚）----
            with torch.no_grad():
                ref_logp = compute_logprobs(ref, g["input_ids"], len(sample["prompt_ids"]))

            # ---- GRPO 损失 ----
            # 策略梯度项：-advantage * logp
            # 优势为正 → 降低 loss（提高该回答的概率）
            # 优势为负 → 提高 loss（降低该回答的概率）
            pg_loss = -(adv * logp).mean()

            # KL 惩罚项：防止策略偏离 reference 太远。
            # 不加这个约束，模型会为了骗 reward 而输出完全不自然的文本
            # （reward hacking），语言能力迅速退化。
            kl = (logp - ref_logp).mean()

            loss = pg_loss + args.kl_coef * kl
            loss.backward()
            total_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()

        # ---- 日志 ----
        if (step + 1) % 5 == 0 and stats_window:
            recent = stats_window[-20:]
            avg_r = sum(s["reward"] for s in recent) / len(recent)
            valid_rate = sum(1 for s in recent if not s["degenerate"]) / len(recent)
            avg_correct = sum(s["n_correct"] for s in recent) / (len(recent) * args.group_size)
            elapsed = time.time() - t0
            eta = (args.steps - step - 1) * elapsed / (step + 1) / 60

            logger.info(
                f"  step {step+1:>4}/{args.steps} | "
                f"reward {avg_r:.3f} | 正确率 {avg_correct*100:.0f}% | "
                f"有效题占比 {valid_rate*100:.0f}% | "
                f"loss {total_loss:.4f} | ETA {eta:.0f}分"
            )

            # 有效题占比是关键指标：太低说明题目难度和模型能力不匹配，
            # 需要换更简单/更难的题，而不是继续硬跑
            if valid_rate < 0.3 and step > 30:
                logger.warning(
                    f"    ⚠ 有效题占比只有 {valid_rate*100:.0f}%。"
                    f"GRPO 需要组内有对有错才有梯度。建议换难度匹配的数据。"
                )

        if (step + 1) % args.save_interval == 0:
            save_checkpoint(out_dir / f"step_{step+1:06d}.pt", policy, optimizer, step + 1)

    # ---- 收尾：把 LoRA 合并回主权重，导出可独立使用的模型 ----
    logger.info("")
    logger.info("  合并 LoRA 到主权重...")
    for name, module in list(policy.named_modules()):
        if isinstance(module, LoRALinear):
            with torch.no_grad():
                delta = (module.lora_B @ module.lora_A) * module.scaling
                module.base.weight.data += delta.to(module.base.weight.dtype)
            parent_name = name.rsplit(".", 1)[0]
            child_name = name.rsplit(".", 1)[1]
            parent = policy.get_submodule(parent_name) if parent_name else policy
            setattr(parent, child_name, module.base)

    save_checkpoint(out_dir / "final.pt", policy, None, args.steps)

    logger.info("")
    logger.info("=" * 68)
    logger.info("  GRPO 训练完成")
    logger.info("=" * 68)
    logger.info(f"  耗时: {(time.time()-t0)/60:.1f} 分钟")
    logger.info(f"  模型: {out_dir / 'final.pt'}")
    logger.info("")
    logger.info("  下一步：python scripts/06_chat.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
