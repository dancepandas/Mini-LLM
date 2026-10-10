"""
阶段 6：命令行对话。

## 这个脚本的用途

    这是验证"模型到底能不能对话"的最终手段。
    评测指标（loss、perplexity）只能说明训练过程正常，
    真正要判断效果还是得实际跟它聊几句。

## 关键的工程细节

    **必须使用和训练时完全一致的 chat template。**
    训练时用 <|im_start|>user\n...<|im_end|> 的格式，
    推理时也必须一字不差。任何差异（少个换行、角色名不同）
    都会让模型表现异常 —— 而且不会报错，只会答非所问。

    这就是为什么模板集中在 minilm/chat_template.py：
    训练和推理从同一个函数取，从结构上杜绝不一致。

用法：
    python scripts/06_chat.py                          # 交互式对话
    python scripts/06_chat.py --model checkpoints/sft/final.pt
    python scripts/06_chat.py --prompt "你好"           # 单次提问
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import minilm  # noqa: E402,F401

import torch  # noqa: E402

from minilm.chat_template import extract_response, render_conversation  # noqa: E402
from minilm.config import load_model_config  # noqa: E402
from minilm.model import build_model, force_mem_efficient_attention  # noqa: E402
from minilm.tokenizer import SpecialTokens, load_tokenizer  # noqa: E402
from minilm.utils import find_latest_checkpoint, load_checkpoint  # noqa: E402


def find_model(preferred: str | None) -> Path:
    """按优先级找一个可用的 checkpoint。

    优先级：命令行指定 > SFT final > SFT 最新 > 预训练 final > 预训练最新。
    先找 SFT 是因为 SFT 后的模型才是能对话的那个；
    只有预训练权重的话也能跑，但只会续写，不会对话。
    """
    if preferred:
        p = Path(preferred)
        if not p.exists():
            raise FileNotFoundError(f"指定的模型不存在: {p}")
        return p

    candidates = [
        PROJECT_ROOT / "checkpoints" / "sft" / "final.pt",
        PROJECT_ROOT / "checkpoints" / "rl" / "final.pt",
        PROJECT_ROOT / "checkpoints" / "pretrain" / "final.pt",
    ]
    for c in candidates:
        if c.exists():
            return c

    # 都没有就找最新的 step checkpoint
    for stage in ("rl", "sft", "pretrain"):
        latest = find_latest_checkpoint(PROJECT_ROOT / "checkpoints" / stage)
        if latest:
            return latest

    raise FileNotFoundError(
        "找不到任何模型权重。请先完成训练：\n"
        "  python scripts/03_pretrain.py\n"
        "  python scripts/04_sft.py"
    )


def build_prompt_ids(tokenizer, messages: list[dict]) -> torch.Tensor:
    """把消息列表渲染成 ChatML 并编码成 token id 张量。"""
    text = render_conversation(messages, add_generation_prompt=True)
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    return torch.tensor([ids], dtype=torch.long, device="cuda")


@torch.no_grad()
def generate_stream(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.05,
):
    """流式生成：逐 token 产出，边生成边打印。

    为什么要流式而不是一次性生成完再打印：
        小模型生成速度有限（0.3B 在 2060 上大约每秒 20~40 个 token），
        生成 300 个 token 要 10 秒以上。等全部生成完再显示，
        用户会觉得卡死了。流式输出体感好得多。

    停止条件：生成出 <|im_end|> 就停 —— 这是模型学会的"回合结束"信号。
    """
    eos_id = tokenizer.token_to_id(SpecialTokens.IM_END)
    im_end_str = SpecialTokens.IM_END

    generated = input_ids
    buffer_text = ""

    for _ in range(max_new_tokens):
        # 序列超长时从左侧截断（模型的最大上下文是 1024）
        ctx = generated[:, -model.cfg.max_seq_len :]

        out = model(ctx, logits_last_only=True)
        logits = out["logits"][:, -1, :].float()

        # 重复惩罚：小模型很容易陷入复读循环，这个参数实际很有用
        if repetition_penalty and repetition_penalty != 1.0:
            for tok in torch.unique(generated[0]):
                if logits[0, tok] > 0:
                    logits[0, tok] /= repetition_penalty
                else:
                    logits[0, tok] *= repetition_penalty

        if temperature <= 0:
            next_token = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k and top_k > 0:
                k = min(top_k, logits.size(-1))
                kth = torch.topk(logits, k, dim=-1).values[:, -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if top_p and 0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
                probs = torch.softmax(sorted_logits, dim=-1)
                cum = probs.cumsum(dim=-1)
                remove = cum - probs > top_p
                sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        tid = next_token.item()

        # 遇到 im_end 说明模型认为回合结束了
        if eos_id is not None and tid == eos_id:
            break

        generated = torch.cat([generated, next_token], dim=1)

        # 逐 token 解码会有"半个汉字"的问题（一个汉字可能占多个 token），
        # 所以维护一个缓冲区，只在解码出完整字符时才输出
        piece = tokenizer.decode([tid], skip_special_tokens=False)
        buffer_text += piece
        # 尝试解码缓冲区，如果末尾是不完整的多字节字符，decode 会给出替换字符
        if buffer_text and "�" not in buffer_text[-1:]:
            print(buffer_text, end="", flush=True)
            buffer_text = ""

    if buffer_text:
        print(buffer_text, end="", flush=True)

    return generated


def main() -> int:
    parser = argparse.ArgumentParser(description="阶段 6：命令行对话")
    parser.add_argument("--model", default=None, help="模型权重路径，默认自动查找")
    parser.add_argument("--model-config", default=str(PROJECT_ROOT / "configs" / "model_0.3b.yaml"))
    parser.add_argument("--tokenizer", default=str(PROJECT_ROOT / "tokenizer" / "tokenizer.json"))
    parser.add_argument("--prompt", default=None, help="单次提问（不给就进交互模式）")
    parser.add_argument("--system",
                        default="你是一个助手。直接回答用户的问题，不要输出思考过程，回答要简短。",
                        help="system 提示词。默认值经过实测调优，见下方说明")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--no-think", action="store_true",
                        help="不输出思维链（若模型会输出的话）")
    args = parser.parse_args()

    force_mem_efficient_attention()

    # ---- 加载 ----
    model_path = find_model(args.model)
    print(f"加载模型: {model_path}")
    print(f"加载分词器: {args.tokenizer}")

    tokenizer = load_tokenizer(args.tokenizer)
    mcfg = load_model_config(args.model_config)
    model = build_model(mcfg, "cuda")
    load_checkpoint(model_path, model)
    model.eval()

    print(f"模型已就绪（{model.num_parameters()/1e6:.0f}M 参数）")
    print("=" * 60)

    def chat_once(messages: list[dict]) -> str:
        """跑一轮生成，返回回答文本。"""
        input_ids = build_prompt_ids(tokenizer, messages)
        print("助手: ", end="", flush=True)
        out = generate_stream(
            model, tokenizer, input_ids,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
        )
        print()

        # 把生成的 token 解回文本，取出这一轮的回答
        new_ids = out[0, input_ids.shape[1]:].tolist()
        text = tokenizer.decode(new_ids, skip_special_tokens=False)
        reasoning, answer = extract_response(text)
        if reasoning and not args.no_think:
            print(f"\n[思维链] {reasoning[:300]}")
        return answer

    # ---- 单次提问模式 ----
    if args.prompt:
        msgs = [{"role": "system", "content": args.system},
                {"role": "user", "content": args.prompt}]
        chat_once(msgs)
        return 0

    # ---- 交互模式 ----
    print("输入问题开始对话。命令：")
    print("  /reset  清空对话历史")
    print("  /exit   退出")
    print("=" * 60)

    history: list[dict] = [{"role": "system", "content": args.system}]

    while True:
        try:
            user_input = input("\n你: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue
        if user_input == "/exit":
            print("再见！")
            break
        if user_input == "/reset":
            history = [{"role": "system", "content": args.system}]
            print("对话已清空。")
            continue

        history.append({"role": "user", "content": user_input})
        answer = chat_once(history)
        # 把回答加进历史，实现多轮对话
        history.append({"role": "assistant", "content": answer})

        # 控制历史长度：只保留最近若干轮，防止超出模型上下文
        if len(history) > 9:
            history = [history[0]] + history[-8:]

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
