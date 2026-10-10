# `web/` —— 网页版对话 Demo

用 Gradio 做的聊天界面，可以在浏览器里直接体验模型效果。

## 启动

```bash
cd D:/chengs/9.project/Mini_LM
conda activate vmodel
python web/app.py
```

然后打开终端提示的地址：**http://127.0.0.1:7860**

常用参数：

```bash
python web/app.py --port 8080         # 换端口
python web/app.py --host 0.0.0.0      # 允许局域网内其他设备访问
python web/app.py --share             # 生成公网临时链接（需联网）
```

---

## 界面功能

| 区域 | 说明 |
|---|---|
| 对话区 | 多轮对话，流式输出（逐字显示） |
| 输入框 | 输入后按回车或点"发送" |
| 生成参数 | 可调 Temperature / Top-p / Top-k / 重复惩罚 / 最大长度 |
| System Prompt | **默认值经过实测调优，建议不要清空**（见下） |
| 示例问题 | 点击按钮填入输入框 |

---

## 两个关键的工程细节

### 1. System Prompt 不能清空

`DEFAULT_SYSTEM_PROMPT` 是实测调优出来的，不是随便写的：

```
你是一个助手。直接回答用户的问题，不要输出思考过程，回答要简短。
```

**为什么需要它**：SFT 数据（minimind 的 `sft_t2t_mini`）带思维链，
模型因此学会了"以 `<think>` 开头"，但**没学会闭合 `</think>` 并给出答案**。

不加时的实测表现（同一批问题）：

| 问题 | 加 system prompt | 不加 |
|---|---|---|
| 中国首都是哪里？ | 中国首都是北京。 | `<think>` 自问自答，给不出"北京" |
| 5+5等于几？ | 5 + 5 = 10 | 5+5=7…答案是7 |

加上之后，事实类问题的直接回答率从约 50% 提升到接近 100%。

### 2. 代理会拦截 localhost（本机特有）

本机开着系统代理（端口 7897）。Gradio 启动时会对自己的 localhost 地址
发一个健康检查请求，**这个请求会被系统代理截走**，代理不认识 `127.0.0.1`
就返回 502，启动直接失败，报错信息还很有误导性：

```
Couldn't start the app because 'http://127.0.0.1:7860/gradio_api/startup-events'
failed (code 502). Check your network or proxy settings to ensure localhost is accessible.
```

**代码里已经在 `import gradio` 之前处理好了**：
清掉 `http_proxy`/`https_proxy`，并把 localhost 加进 `NO_PROXY`。
如果你把这份代码挪到别的机器上遇到同样问题，就是这个原因。

---

## 输出后处理：剥掉 `<think>` 块

`clean_for_display()` 会把模型输出处理成适合展示的形式：

| 模型原始输出 | 展示给用户 |
|---|---|
| `<think>推理…</think>实际答案` | `实际答案` |
| `<think>推理中（还没结束）` | `⏳ 思考中…（N 字）` |
| `直接的回答` | `直接的回答` |

**为什么在后端做而不是用 Gradio 的 `reasoning_tags`**：
后者依赖特定版本的行为（Gradio 6 才有），在后端做完全可控、不受 API 变动影响。

---

## 已知限制

- **生成速度**：0.3B 模型在 RTX 2060 上约 15~30 token/s，
  一条 200 token 的回答需要 6~13 秒（界面是流式输出的，不会显得卡死）。

- **不支持并发**：6GB 显存只能装一份模型，代码里用锁串行化了请求。
  Gradio 的队列会排队处理，不会崩，但同时多人访问要等。

- **模型能力上限**：0.3B 参数量决定了它在复杂推理、多位数运算上很弱，
  偶尔会在简单算术上出错（如 `5+5=7`）。这是模型规模的客观限制，
  不是服务的问题。

- **示例问题按钮**：用的是普通 Button 而不是 `gr.Examples` ——
  后者的参数在 Gradio 大版本间变过好几次，用 Button 完全版本无关。
