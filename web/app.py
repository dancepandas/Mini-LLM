"""
网页版对话 Demo（Gradio）。

启动：
    python web/app.py
然后在浏览器打开终端里打印的地址（默认 http://127.0.0.1:7860）。

## 设计要点

1. **默认 system prompt 是经过实测调优的**
   见 `DEFAULT_SYSTEM_PROMPT` 的注释。用默认的那个，模型会直接回答问题；
   去掉的话它会陷入 `<think>` 式的自问自答。

2. **流式输出**
   0.3B 模型在 RTX 2060 上大约每秒 15~30 个 token，生成 200 个 token 要 10 秒以上。
   一次性返回会让用户以为卡死。这里逐 token 推送。

3. **模型只加载一次**
   放在模块级全局变量里，由 Gradio 的进程常驻。
   每次对话都重新加载 1.3GB 权重的话，每次要等十几秒。

4. **单例推理**
   GPU 显存有限（6GB），不支持并发请求。Gradio 默认按队列处理，符合需求。
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

# ---------------------------------------------------------------------------
# ⚠️ 必须在 import gradio 之前处理代理
# ---------------------------------------------------------------------------
# 本机开着系统代理（Clash 之类，端口 7897）。Gradio 启动时会对**自己的**
# localhost 地址发一个健康检查请求（/gradio_api/startup-events），
# 而这个请求会被系统代理截走 —— 代理不认识 127.0.0.1，返回 502，
# 于是启动直接失败，报错信息是误导性的
#     "Couldn't start the app because ... failed (code 502).
#      Check your network or proxy settings to ensure localhost is accessible."
#
# 解决：把 localhost 加进 NO_PROXY，并清掉 http_proxy/https_proxy。
# 这类"代理拦截本地回环"的坑在国内开发环境很常见。
for _k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
    os.environ.pop(_k, None)
os.environ["NO_PROXY"] = "localhost,127.0.0.1,::1,0.0.0.0"
os.environ["no_proxy"] = "localhost,127.0.0.1,::1,0.0.0.0"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import minilm  # noqa: E402,F401  (必须先于 torch：设置显存分配器)

import torch  # noqa: E402

from minilm.chat_template import render_conversation  # noqa: E402
from minilm.config import load_model_config  # noqa: E402
from minilm.model import build_model, force_mem_efficient_attention  # noqa: E402
from minilm.tokenizer import SpecialTokens, load_tokenizer  # noqa: E402
from minilm.utils import find_latest_checkpoint, load_checkpoint  # noqa: E402


# ===========================================================================
# 配置
# ===========================================================================
# ⚠️ 这个 system prompt 是实测调优出来的，不是随便写的。
#
#    SFT 数据（minimind 的 sft_t2t_mini）带思维链，模型因此学会了
#    "以 <think> 开头"，但**没学会闭合 </think> 并给出最终答案**。
#    不加 system prompt 时，约一半的事实类问题会变成自问自答：
#
#        问：中国首都是哪里？
#        答：<think> 好的，用户问中国首都是哪里。我需要先确定用户的需求是什么…
#            （一直说到撞上长度上限，给不出"北京"）
#
#    加上这句之后，同一批问题全部能直接答出。所以默认就用它。
DEFAULT_SYSTEM_PROMPT = "你是一个助手。直接回答用户的问题，不要输出思考过程，回答要简短。"

MODEL_CANDIDATES = [
    "checkpoints/rl/final.pt",       # 最终交付模型（GRPO 后）
    "checkpoints/sft/final.pt",      # SFT 模型
    "checkpoints/pretrain/final.pt", # 预训练底座（只会续写，不会对话）
]


# ===========================================================================
# 模型加载（全局单例）
# ===========================================================================
class ChatEngine:
    """封装模型加载与生成，供 Gradio 调用。"""

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.device = "cuda"
        self.model_path: str = ""
        # 生成是逐 token 串行的，用锁避免并发请求把显存打爆
        self._lock = threading.Lock()

    def load(self, tokenizer_path: str = "tokenizer/tokenizer.json") -> str:
        """加载模型和分词器，返回实际加载的模型路径。"""
        force_mem_efficient_attention()

        tok_path = PROJECT_ROOT / tokenizer_path
        if not tok_path.exists():
            raise FileNotFoundError(f"分词器不存在: {tok_path}")
        self.tokenizer = load_tokenizer(tok_path)

        # 按优先级找可用的模型
        chosen = None
        for rel in MODEL_CANDIDATES:
            p = PROJECT_ROOT / rel
            if p.exists():
                chosen = p
                break
        if chosen is None:
            fallback = find_latest_checkpoint(PROJECT_ROOT / "checkpoints" / "pretrain")
            if fallback is None:
                raise FileNotFoundError(
                    "找不到任何模型权重。请先运行训练：\n"
                    "  python scripts/03_pretrain.py\n"
                    "  python scripts/04_sft.py"
                )
            chosen = fallback

        cfg = load_model_config(str(PROJECT_ROOT / "configs" / "model_0.3b.yaml"))
        self.model = build_model(cfg, self.device)
        load_checkpoint(chosen, self.model)
        self.model.eval()
        self.model_path = str(chosen.relative_to(PROJECT_ROOT))
        return self.model_path

    @property
    def ready(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    @torch.no_grad()
    def stream_generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
    ):
        """流式生成，逐段 yield 新产生的文本。

        参数
        ----
        messages : list[dict]
            形如 [{"role": "system"/"user"/"assistant", "content": "..."}]

        产出
        ----
        str
            每次 yield 的是**新增的文本片段**（不是累积的全文），
            调用方自己拼接。
        """
        if not self.ready:
            yield "（模型未加载）"
            return

        tok = self.tokenizer
        eos_id = tok.token_to_id(SpecialTokens.IM_END)

        # 渲染成 ChatML 并编码
        text = render_conversation(messages, add_generation_prompt=True)
        ids = tok.encode(text, add_special_tokens=False).ids
        generated = torch.tensor([ids], dtype=torch.long, device=self.device)
        prompt_len = generated.shape[1]

        # 逐 token 查看是否触发了停止词，用一个小的滚动窗口
        produced_ids: list[int] = []

        with self._lock:  # 串行化，避免并发请求同时占用显存
            for _ in range(max_new_tokens):
                # 超出模型上下文时从左侧截断
                ctx = generated[:, -self.model.cfg.max_seq_len :]

                out = self.model(ctx, logits_last_only=True)
                logits = out["logits"][:, -1, :].float()

                # ---- 重复惩罚 ----
                # 小模型很容易陷入复读循环，这个参数实际很有用
                if repetition_penalty and repetition_penalty != 1.0:
                    for t in torch.unique(generated[0]):
                        if logits[0, t] > 0:
                            logits[0, t] /= repetition_penalty
                        else:
                            logits[0, t] *= repetition_penalty

                # ---- 采样 ----
                if temperature <= 0:
                    next_token = logits.argmax(dim=-1, keepdim=True)
                else:
                    logits = logits / max(temperature, 1e-5)

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

                # 遇到 <|im_end|> 说明模型认为这一轮说完了
                if eos_id is not None and tid == eos_id:
                    break

                generated = torch.cat([generated, next_token], dim=1)
                produced_ids.append(tid)

                # ---- 解码成文本 ----
                # 逐 token 解码会有"半个汉字"的问题（一个汉字可能跨多个 token），
                # 所以每步都对**完整序列**解码再取增量，而不是单独解码新 token。
                # 这样能正确处理多字节字符的边界。
                full = tok.decode(produced_ids, skip_special_tokens=False)
                # 把未完成的替换字符挡在最后，等下一个 token 补齐
                if full.endswith("�"):
                    continue
                yield full

        # 生成结束
        return


ENGINE = ChatEngine()


# ===========================================================================
# Gradio 界面
# ===========================================================================
def clean_for_display(text: str) -> str:
    """把模型输出处理成适合展示给用户的文本。

    ## 为什么要剥掉 <think> 块

        SFT 数据带思维链，模型因此学会了输出 `<|think|>` 式的推理块，
        但**经常不闭合**，或者推理很长把 token 预算吃光、最终答案挤不出来。

        直接展示给用户的后果：界面上是一大段自问自答，真正的答案
        （如果有）藏在最后甚至根本没有。

        处理规则：
          - 有完整的 <think>…</think>  → 只展示 </think> 之后的内容
          - 有 <think> 但没闭合        → 说明还在推理，显示"思考中"提示
          - 没有 <think>               → 原样展示

    ## 为什么在后端做而不是用 Gradio 的特性

        Gradio 6 有 `reasoning_tags` 参数可以折叠推理块，但依赖它意味着
        绑定特定版本的行为。在后端做完全可控，也不受 API 变动影响。
    """
    if not text:
        return ""

    has_open = "<think>" in text
    has_close = "</think>" in text

    if has_open and has_close:
        # 取最后一个 </think> 之后的内容 —— 模型有时会输出多段推理
        answer = text.rsplit("</think>", 1)[-1].strip()
        return answer if answer else "（模型只输出了思考过程，没有给出答案）"

    if has_open:
        # 推理还没结束。把已经想出来的部分以小字提示，让用户知道模型在干活
        partial = text.split("<think>", 1)[-1].strip()
        return f"⏳ *思考中…*（{len(partial)} 字）" if partial else "⏳ *思考中…*"

    return text.strip()


def build_ui():
    import gradio as gr

    def on_load():
        path = ENGINE.load()
        return f"✅ 模型已加载：`{path}`（{ENGINE.model.num_parameters()/1e6:.0f}M 参数）"

    def respond(message, history, system_prompt, temperature, top_p, top_k, rep_pen, max_tokens):
        """一次对话：把用户消息加进历史，然后流式生成回答。"""
        if not message or not message.strip():
            yield history, ""
            return

        if not ENGINE.ready:
            history = history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "（模型未加载，请稍候或检查终端）"},
            ]
            yield history, ""
            return

        # 组装发给模型的消息（system + 历史 + 当前问题）
        # 只保留最近若干轮，防止超出模型 1024 的上下文上限
        msgs = []
        if system_prompt and system_prompt.strip():
            msgs.append({"role": "system", "content": system_prompt.strip()})
        for h in history[-6:]:
            msgs.append({"role": h["role"], "content": h["content"]})
        msgs.append({"role": "user", "content": message})

        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": ""},
        ]
        yield history, ""

        for i, partial in enumerate(
            ENGINE.stream_generate(
                msgs,
                max_new_tokens=int(max_tokens),
                temperature=temperature,
                top_p=top_p,
                top_k=int(top_k),
                repetition_penalty=rep_pen,
            )
        ):
            # 剥掉 <think> 推理块，只把最终答案展示给用户
            history[-1]["content"] = clean_for_display(partial)
            # Gradio 需要每次 yield 一个新对象才会刷新界面
            yield [dict(m) for m in history], ""

        if not history[-1]["content"]:
            history[-1]["content"] = "（没有生成内容，可能是采样到了停止词）"
            yield history, ""

    def clear():
        return [], ""

    with gr.Blocks(title="Mini_LM 对话 Demo") as demo:
        gr.Markdown(
            "# Mini_LM 对话 Demo\n"
            "从零训练的 **0.3B 中文语言模型**（316.7M 参数，单张 RTX 2060 训练约 5 天）"
        )
        status = gr.Markdown("⏳ 正在加载模型…")

        chatbot = gr.Chatbot(
            label="对话",
            height=440,
            # 注意：Gradio 6 的 Chatbot **默认就是 messages 格式**
            # （list of {"role":..., "content":...}），不再有 `type` 参数。
            # 老代码里的 type="messages" 在 6.x 会直接报 TypeError。
        )

        with gr.Row():
            msg = gr.Textbox(
                label="输入",
                placeholder="输入问题后按回车…",
                scale=8,
                autofocus=True,
            )
            send = gr.Button("发送", variant="primary", scale=1)

        with gr.Accordion("生成参数", open=False):
            system_prompt = gr.Textbox(
                label="System Prompt",
                value=DEFAULT_SYSTEM_PROMPT,
                lines=2,
                info="⚠️ 不要清空 —— 实测清空后模型会陷入 <think> 式自问自答，"
                     "给出正确答案的概率大幅下降",
            )
            with gr.Row():
                temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.05, label="Temperature",
                                        info="越低越确定，越高越发散")
                top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
            with gr.Row():
                top_k = gr.Slider(0, 100, value=50, step=5, label="Top-k",
                                  info="0 表示不限制")
                rep_pen = gr.Slider(1.0, 1.5, value=1.05, step=0.01, label="重复惩罚",
                                    info="小模型容易复读，调高可缓解")
                # 默认给到 384：模型的 <think> 块可能吃掉 200 个 token，
                # 留太小的话答案还没输出就被截断了
                max_tokens = gr.Slider(32, 512, value=384, step=32, label="最多生成 token")

        with gr.Row():
            clear_btn = gr.Button("🗑 清空对话")

        # ---- 示例问题 ----
        # 用一排 Button 而不是 gr.Examples：
        #   gr.Examples 的参数在不同 Gradio 大版本间变过好几次
        #   （6.x 里它的签名被装饰器包住，无法用 inspect 确认），
        #   用 Button + 设置 textbox 值的写法完全版本无关，更稳。
        gr.Markdown("**示例问题**（点击填入输入框）")
        example_questions = [
            "你好，请介绍一下你自己",
            "中国首都是哪里？",
            "5+5等于几？",
            "你是谁训练的？",
            "用一句话解释什么是人工智能",
            "给我讲个笑话",
        ]
        with gr.Row():
            ex_buttons = [gr.Button(q, size="sm") for q in example_questions]

        # ---- 事件绑定 ----
        demo.load(on_load, outputs=status)

        # 示例按钮：点击后把问题填进输入框（不直接发送，方便用户先改）
        for btn, q in zip(ex_buttons, example_questions):
            btn.click(lambda q=q: q, outputs=msg)

        send.click(respond,
                   inputs=[msg, chatbot, system_prompt, temperature, top_p, top_k, rep_pen, max_tokens],
                   outputs=[chatbot, msg])
        msg.submit(respond,
                   inputs=[msg, chatbot, system_prompt, temperature, top_p, top_k, rep_pen, max_tokens],
                   outputs=[chatbot, msg])
        clear_btn.click(clear, outputs=[chatbot, msg])

    return demo


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Mini_LM 网页对话 Demo")
    parser.add_argument("--host", default="127.0.0.1", help="监听地址。用 0.0.0.0 可局域网访问")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="生成公网临时链接（需联网）")
    args = parser.parse_args()

    print("=" * 68)
    print("  Mini_LM 网页对话 Demo")
    print("=" * 68)

    demo = build_ui()

    print(f"\n  启动中… 请在浏览器打开： http://{args.host}:{args.port}\n")
    demo.queue(max_size=8).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
        inbrowser=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
