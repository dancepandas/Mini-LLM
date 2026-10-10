"""
对话模板：把消息列表渲染成模型能理解的 token 序列。

## 为什么单独抽一个模块

    **训练和推理必须用完全相同的模板。** 这是 SFT 最容易犯、也最致命的错误：
      - 训练时用模板 A，推理时用模板 B
      - 模型学到的"什么格式该回答"的规律在推理时完全失效
      - 表现是：loss 正常下降，但推理时模型答非所问或复读

    这种错误不会报错，只会让效果莫名其妙地差。把模板集中到一个模块，
    训练和推理都从这里取，能从结构上杜绝不一致。

## 格式：ChatML

    采用 Qwen / OpenAI 的 ChatML 约定：

        <|im_start|>system
        你是一个有用的AI助手。<|im_end|>
        <|im_start|>user
        你好<|im_end|>
        <|im_start|>assistant
        你好！有什么可以帮你的吗？<|im_end|>

    注意 `<|im_start|>` 后面紧跟角色名，然后是换行，再是内容。
    这个换行的位置很关键 —— 少了它，角色名会和内容粘在一起。

## 关于思维链（reasoning）

    训练数据（如 Qwen3 蒸馏数据）里可能带 reasoning_content 字段。
    这里的处理方式是用 <think></think> 包起来放进 assistant 段。
    推理时可以按需解析出来展示给用户，或者直接保留在输出里。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .tokenizer import SpecialTokens


# ===========================================================================
# 消息结构与渲染
# ===========================================================================
@dataclass
class Message:
    """一条对话消息。

    role : "system" | "user" | "assistant"
    content : 正文
    reasoning : 思维链内容（可选）。只有 assistant 消息可能有。
    """

    role: str
    content: str
    reasoning: str | None = None

    def render(self) -> str:
        """渲染成 ChatML 文本片段（不含 <|im_start|> 前缀）。"""
        body = self.content
        # 把思维链包进 <think> 标签放在正文前面。
        # 用标签而不是直接拼接，是为了让模型学到"这段是内部推理"的边界，
        # 推理阶段可以选择不展示。
        if self.reasoning:
            body = f"<think>\n{self.reasoning.strip()}\n</think>\n\n{body}"
        return f"{SpecialTokens.IM_START}{self.role}\n{body}{SpecialTokens.IM_END}\n"


def render_conversation(
    messages: list[Message] | list[dict[str, Any]],
    add_generation_prompt: bool = False,
) -> str:
    """把消息列表渲染成完整的 ChatML 文本。

    参数
    ----
    messages : list
        消息列表。接受 Message 对象或形如 {"role":..., "content":...} 的字典。
    add_generation_prompt : bool
        是否在末尾追加 `<|im_start|>assistant\n` 以提示模型开始生成。
        **推理时必须为 True** —— 否则模型不知道轮到自己说话了。

    返回
    ----
    str
        渲染后的文本。
    """
    norm = [_normalize_message(m) for m in messages]
    text = "".join(m.render() for m in norm)

    if add_generation_prompt:
        # 注意结尾的换行：和 Message.render 里角色名后的换行保持一致，
        # 这样模型在训练时见到的"assistant\n内容"和推理时"assistant\n"的
        # 前缀形式完全一致
        text += f"{SpecialTokens.IM_START}{SpecialTokens.ROLE_ASSISTANT}\n"

    return text


def _normalize_message(m: Message | dict[str, Any]) -> Message:
    """把 dict 形式的消息统一成 Message 对象。"""
    if isinstance(m, Message):
        return m

    role = m.get("role", "user")
    content = m.get("content", "") or ""
    reasoning = m.get("reasoning_content") or m.get("reasoning")

    # 有些数据集把角色写成 "human"/"gpt"（ShareGPT 格式），转成标准角色
    role_map = {
        "human": "user",
        "gpt": "assistant",
        "bot": "assistant",
        "model": "assistant",
    }
    role = role_map.get(role, role)

    if role not in ("system", "user", "assistant"):
        role = "user"

    return Message(role=role, content=content, reasoning=reasoning)


# ===========================================================================
# 编码（带 loss mask）
# ===========================================================================
def encode_conversation(
    messages: list[Message] | list[dict[str, Any]],
    tokenizer,
    max_len: int | None = None,
    ignore_index: int = -100,
) -> dict[str, list[int]]:
    """把对话编码成 input_ids 和 labels，**并对 prompt 部分做 loss 掩码**。

    这是 SFT 最关键的一步。

    ## 为什么要掩码

        语言模型的标准训练目标是"预测下一个 token"。如果整个序列都算 loss，
        模型会同时学着去预测**用户的提问**。
        但我们只希望它学会"生成回答"，而不是"预测用户会问什么"。

        不掩码的后果：SFT 后模型倾向于复述用户的话，或者回答质量明显偏低。
        这个错误在 loss 曲线上完全看不出来（loss 一样会正常下降）。

    ## 实现方式

        遍历每条消息：
          - assistant 消息：这段 token 的 label 就是 token 本身（要学）
          - 其他消息（system/user）：这段的 token 全部标成 -100（不学）
          - **但 assistant 之前的那部分仍要作为输入喂进去**（模型需要看到问题）

        注意 `<|im_start|>assistant\n` 这个前缀本身也标成 -100：
        它是模板固定内容，不需要模型去"生成"它。

    返回
    ----
    dict，包含：
        "input_ids" : list[int]
        "labels"    : list[int]  （-100 表示该位置不参与 loss）
    """
    # 关键：用 tokenizer 实际编码这些片段，而不是硬编码它们的长度。
    # 因为 "<|im_start|>assistant\n" 可能被编码成多个 token，
    # 具体多少个取决于分词器的 merge 规则。硬编码必然出错。
    assistant_prefix = f"{SpecialTokens.IM_START}{SpecialTokens.ROLE_ASSISTANT}\n"
    prefix_ids = tokenizer.encode(assistant_prefix, add_special_tokens=False).ids

    input_ids: list[int] = []
    labels: list[int] = []

    norm = [_normalize_message(m) for m in messages]

    for msg in norm:
        if msg.role == "assistant":
            # 前缀部分：作为输入，但不计 loss
            head = f"{assistant_prefix}"
            head_ids = tokenizer.encode(head, add_special_tokens=False).ids
            input_ids.extend(head_ids)
            labels.extend([ignore_index] * len(head_ids))

            # 正文部分：作为输入，**计 loss**
            body = _render_assistant_body(msg)
            body_ids = tokenizer.encode(body, add_special_tokens=False).ids
            input_ids.extend(body_ids)
            labels.extend(body_ids)  # label = token 本身

            # 结束标记：也要计 loss！
            # 这一点容易被忽略 —— 模型必须学会"什么时候该停下"，
            # 而学习停止信号的方式就是预测 <|im_end|>。
            # 如果把 <|im_end|> 标成 -100，模型永远学不会停止，
            # 推理时会一直说下去直到撞上长度上限。
            end_ids = tokenizer.encode(SpecialTokens.IM_END + "\n", add_special_tokens=False).ids
            input_ids.extend(end_ids)
            labels.extend(end_ids)
        else:
            # system / user：整段作为输入，但不计 loss
            seg = msg.render()
            seg_ids = tokenizer.encode(seg, add_special_tokens=False).ids
            input_ids.extend(seg_ids)
            labels.extend([ignore_index] * len(seg_ids))

    # ---- 长度处理 ----
    if max_len is not None and len(input_ids) > max_len:
        # 从**左侧**截断：保留最后的对话内容。
        # 为什么不从右边截：对话的关键信息（最近的问答）在末尾，
        # 截掉末尾等于把要学的内容丢了。
        input_ids = input_ids[-max_len:]
        labels = labels[-max_len:]

    return {"input_ids": input_ids, "labels": labels}


def _render_assistant_body(msg: Message) -> str:
    """渲染 assistant 消息的正文部分（含思维链，不含前缀和结束标记）。"""
    body = msg.content
    if msg.reasoning:
        body = f"<think>\n{msg.reasoning.strip()}\n</think>\n\n{body}"
    return body


# ===========================================================================
# 输出解析
# ===========================================================================
def extract_response(text: str) -> tuple[str | None, str]:
    """从模型生成的原文本里解析出（思维链, 回答）。

    用于推理和 RL 阶段：需要把模型的输出切成思考和答案两部分，
    奖励函数通常只看答案，而格式奖励会检查思维链格式是否规范。

    返回
    ----
    (reasoning, answer)
        reasoning 为 None 表示输出里没有 <think> 标签。
    """
    if "<think>" in text and "</think>" in text:
        start = text.index("<think>") + len("<think>")
        end = text.index("</think>")
        reasoning = text[start:end].strip()
        answer = text[end + len("</think>"):].strip()
        return reasoning, answer
    return None, text.strip()


def is_turn_complete(generated_text: str) -> bool:
    """判断生成是否已经产生了一个完整的回合（出现 <|im_end|>）。"""
    return SpecialTokens.IM_END in generated_text
