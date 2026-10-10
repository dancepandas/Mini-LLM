"""
MiniLM —— 从零训练的中文语言模型。

模块划分（详细说明见各自的 README）：

    config.py        配置系统：YAML ↔ dataclass，带启动期校验
    model.py         模型结构：RMSNorm / RoPE / GQA / SwiGLU / 梯度检查点
    tokenizer.py     字节级 BPE 分词器的训练与加载
    chat_template.py 对话模板（预训练/SFT/推理三处必须一致）
    generate.py      采样策略与生成辅助
    utils.py         日志、随机种子、显存监控、checkpoint 管理

    data/            数据管线：下载 → 清洗 → 分词 → memmap
    train/           三个训练阶段：pretrain / sft / rl_grpo
"""

import os as _os
import sys as _sys

__version__ = "0.1.0"


# ---------------------------------------------------------------------------
# 显存分配器配置（必须在 import torch 之前设置）
# ---------------------------------------------------------------------------
# `expandable_segments:True` 让 PyTorch 的显存分配器使用可扩展的内存段，
# 而不是为每种尺寸的张量维护独立的固定块。
#
# 本机实测（RTX 2060 6GB，micro_batch=4）：
#     默认分配器          : reserved 5.40 GB  → 超过 4.98 GB 空闲显存 → 轻度超售，4417 tok/s
#     expandable_segments : reserved 4.90 GB  → 不再超售              → 4618 tok/s
#
# 关掉超售后吞吐反而涨了 4.5%，说明超售的代价是实打实的。
#
# **这个设置必须在 import torch 之前生效**，因为 CUDA 分配器在第一次
# 分配显存时就会读取这个环境变量，之后再改设置就晚了。
# 所以它放在包的 __init__ 里，并且所有脚本都应该先 import minilm 再 import torch。
_os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ---------------------------------------------------------------------------
# 修复指向不存在文件的 SSL 证书环境变量
# ---------------------------------------------------------------------------
# conda 环境会设置 `SSL_CERT_FILE` 指向 `<env>/ssl/cacert.pem`。
# 本机这个文件**不存在**（环境被裁剪过或证书包没装上），后果是：
#
#     - `urllib` **不读**这个变量 → 走代理访问 HTTPS 正常
#     - `httpx` 在 trust_env=True（默认）时**会读**它，
#       直接抛 `FileNotFoundError: [Errno 2] No such file or directory`
#       → 所有基于 httpx 的库全部无法导入/使用
#
# 受影响的库包括 huggingface_hub、gradio、openai 等一大票。
# 症状极具误导性：同一个进程里 urllib 测网络是通的，
# 但 `import gradio` 直接崩，报错还指向证书而不是环境问题。
#
# 这个修复必须放在**包初始化**里，而不是某个子模块里 ——
# 因为任何模块都可能先触发 httpx 的导入。放在这里可以保证
# `import minilm` 之后所有 HTTP 库都能正常工作。
for _var in ("SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
    _val = _os.environ.get(_var)
    # 只在"设置了"且"指向的文件真的不存在"时才删，避免误伤有效配置
    if _val and not _os.path.exists(_val):
        del _os.environ[_var]


# ---------------------------------------------------------------------------
# Windows 控制台编码修正
# ---------------------------------------------------------------------------
# Windows 的默认控制台编码是 GBK（cp936）。它能编码汉字，但编不了 ✓ ⚠ → 这类符号，
# 一旦 print 到它们就会抛 UnicodeEncodeError 把程序打断。
#
# 与其在每个 print 里小心翼翼地避开这些字符，不如在包导入时把标准输出
# 统一改成 UTF-8 —— 这是 Windows 上做中文项目的通用做法，一次性解决所有输出问题。
#
# 用 try/except 包住是因为：
#   - 某些环境下 stdout 被重定向成不支持 reconfigure 的对象
#   - 这种做法失败也不该影响整个包被导入
if _sys.platform == "win32":
    for _stream in (_sys.stdout, _sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except (AttributeError, OSError):
            pass

