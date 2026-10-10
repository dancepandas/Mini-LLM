"""
数据下载：通过代理从 HuggingFace 流式拉取中文语料。

## 为什么必须流式（streaming），不能先下载再处理

    本项目的候选数据集动辄几十上百 GB（Fineweb-Edu-Chinese V2.1 的 4_5 子集就有 74GB），
    但我们只需要其中的 2B tokens。全下载再抽样既浪费时间（网络 2.33MB/s）又浪费磁盘。
    流式读取可以边拉边处理，取够就停。

## 网络环境（本机实测）

    直连 huggingface.co **超时**，必须走本地代理 127.0.0.1:7897。
    走代理后 API 返回 200，实测下载速度 2.33 MB/s。
    2B tokens 对应约 5GB 原始文本，下载约 35 分钟，不是瓶颈。

## 各数据集的字段结构差异（实测确认，不是猜的）

    不同数据集的文本字段名完全不同，而且有些有坑：
        wikimedia/wikipedia      → 字段 "text"
        opencsg/Fineweb-...-V2.1 → 字段 "text"
        opencsg/chinese-cosmopedia → 字段 "text"
        liwu/MNBVC               → 字段 "text"（原始 gzip jsonl）
        wenge-research/yayi2_... → 字段 "content"   <-- 不叫 text！
        m-a-p/Matrix             → 字段 "text"

    所以抽取文本的逻辑必须按数据集配置，不能写死。
"""

from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

from tqdm import tqdm


# ===========================================================================
# 网络环境配置
# ===========================================================================
DEFAULT_PROXY = "http://127.0.0.1:7897"


@dataclass
class NetworkConfig:
    """HuggingFace 访问配置。

    本机直连 HF 不通，必须走代理。这个配置集中管理，避免在多个脚本里
    重复写代理地址（改端口时只改一处）。
    """

    proxy: str | None = DEFAULT_PROXY
    """本地代理地址。设为 None 表示直连（本机不可用，仅作兼容保留）。"""

    endpoint: str = "https://huggingface.co"
    """HF 端点。用官方站而非镜像 —— 实测走代理后直连可用（200, 1.6s），
    而 hf-mirror.com 在设了代理的情况下反而连不上。"""

    cache_dir: str = "data/hf_cache"
    """HF 下载缓存目录。流式读取时数据会缓存在这里。"""

    token: str | None = None
    """HF access token。当前选用的数据集都不需要授权，留空即可。"""

    max_retries: int = 5
    """单个分片的下载重试次数。实测代理链路偶发 SSL EOF，必须重试。"""

    def apply(self) -> None:
        """把配置写入环境变量。**必须在 import datasets 之前调用。**

        huggingface_hub 在导入时就会读取这些环境变量并据此构造 HTTP 客户端，
        之后再设置就无效了。这是个很容易踩的坑 ——
        表现为"明明设了代理还是连不上"。
        """
        self._fix_broken_ssl_cert_env()

        if self.proxy:
            os.environ["http_proxy"] = self.proxy
            os.environ["https_proxy"] = self.proxy
            os.environ["HTTP_PROXY"] = self.proxy
            os.environ["HTTPS_PROXY"] = self.proxy
        else:
            for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
                os.environ.pop(key, None)

        os.environ["HF_ENDPOINT"] = self.endpoint
        os.environ["HF_HOME"] = self.cache_dir
        # 用 hf-xet 的高速传输后端。新版 huggingface_hub 自带，
        # 大文件下载会明显快于单连接 HTTP 的 2.33 MB/s。
        os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
        if self.token:
            os.environ["HF_TOKEN"] = self.token

    @staticmethod
    def _fix_broken_ssl_cert_env() -> None:
        """清理指向不存在文件的 SSL 证书环境变量。

        ## 这个坑值得详细记下来，因为它极其隐蔽

        conda 环境会设置 `SSL_CERT_FILE` 指向 `<env>/ssl/cacert.pem`。
        如果该文件不存在（环境被裁剪过、证书包没装上、或从别处拷贝的环境），
        那么：

            - `urllib` **不读**这个变量 → 走代理能正常访问 HTTPS ✅
            - `httpx` 在 trust_env=True（默认）时**会读**它，
              并直接抛 `FileNotFoundError: [Errno 2] No such file or directory`
              → 所有 HTTPS 请求全部失败 ❌

        而 huggingface_hub 底层用的正是 httpx。

        现象因此非常反直觉：**同一个进程里，用 urllib 测网络是通的，
        但 datasets.load_dataset 就是连不上**，且报错信息是
        "Couldn't find ... on the Hugging Face Hub"，指向完全错误的方向
        （看起来像数据集不存在，实际是证书配置坏了）。

        本机实测就是这个情况：`SSL_CERT_FILE` 指向的
        `D:\\chengs\\7.miniconda\\envs\\vmodel/ssl/cacert.pem` 并不存在。

        处理方式：指向的文件不存在就删掉该变量，
        让 httpx 回落到 certifi 内置的证书包 —— 那才是正确的默认行为。
        """
        for var in ("SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
            val = os.environ.get(var)
            # 只在"设置了"且"指向的文件真的不存在"时才删，
            # 避免误伤用户有意配置的有效证书路径
            if val and not os.path.exists(val):
                del os.environ[var]

    def describe(self) -> str:
        return f"代理={self.proxy or '直连'}, 端点={self.endpoint}, 缓存={self.cache_dir}"


# ===========================================================================
# 数据集定义
# ===========================================================================
@dataclass
class DatasetSpec:
    """描述一个要下载的数据集及其抽取方式。

    把"怎么从一条记录里取出正文"做成可配置的，是因为各数据集的字段名不同，
    而且有些还有嵌套结构需要特殊处理。
    """

    repo_id: str
    """HuggingFace 数据集 id，如 "wikimedia/wikipedia" """

    text_field: str = "text"
    """正文所在的字段名。"""

    config_name: str | None = None
    """数据集的子配置名。wikimedia/wikipedia 需要指定 "20231101.zh"。"""

    split: str = "train"

    data_files: str | None = None
    """指定要读取的文件（支持 glob）。

    对 Fineweb V2.1 这类仓库很重要：它有五个不同质量的子目录
    （IndustryCorpus / Skypile / cci2 / map / tele），我们**只要 4_5 这个最高质量子集**。
    不指定的话会把整个 311GB 的仓库都纳入范围。
    """

    target_tokens: int = 0
    """这个数据集的目标 token 配额，用于控制混合比例。"""

    weight: float = 1.0
    """采样权重，用于在多个数据集间按比例混合。"""

    license_note: str = ""
    """许可证备注。有几个数据集的卡片标签和正文许可不一致，需要记录。"""

    extra_filter: Callable[[dict], bool] | None = None
    """自定义过滤函数，用于处理字段结构特殊的数据集。"""

    def describe(self) -> str:
        parts = [f"{self.repo_id}"]
        if self.config_name:
            parts.append(f"config={self.config_name}")
        if self.data_files:
            parts.append(f"files={self.data_files}")
        parts.append(f"字段={self.text_field}")
        return " | ".join(parts)


# 预训练语料的数据集清单。
# 配额的依据是 scripts/00_setup.py 之外的另一组调研结论（见各条注释）。
PRETRAIN_DATASETS: list[DatasetSpec] = [
    DatasetSpec(
        repo_id="wikimedia/wikipedia",
        config_name="20231101.zh",   # 注意：20241201.zh 这个配置**不存在**（实测 404）
        text_field="text",
        target_tokens=250_000_000,
        license_note="CC BY-SA 3.0 / GFDL，需署名且共享相同许可",
    ),
    DatasetSpec(
        repo_id="opencsg/Fineweb-Edu-Chinese-V2.1",
        # 只要 4_5 这个教育质量评分最高（4-5 分）的子集。
        # 整个仓库是 311GB，全下既没必要也没必要 —— 4_5 子集 9745 个 parquet 共 74GB。
        data_files="4_5/*.parquet",
        text_field="text",
        target_tokens=1_350_000_000,
        license_note="卡片标 Apache-2.0，但正文要求遵守 OpenCSG 社区许可，商用需另申请",
    ),
    DatasetSpec(
        repo_id="opencsg/chinese-cosmopedia",
        text_field="text",
        target_tokens=200_000_000,
        license_note="合成教材数据。占比控制在 10% 以内，避免合成数据主导语料分布",
    ),
    DatasetSpec(
        repo_id="liwu/MNBVC",
        # MNBVC 是超大型多来源语料，这里只取新闻子集来增加真实新闻文本，
        # 避免整个语料都是教育筛选网页 + 合成内容。
        data_files="news/*/*.jsonl.gz",
        text_field="text",
        target_tokens=200_000_000,
        license_note="仓库元数据标 MIT，但抓取原文的版权不能一概而论",
    ),
]


# ===========================================================================
# 流式读取
# ===========================================================================
def _import_datasets():
    """延迟导入 datasets。

    为什么不在文件顶部直接 import：
        `NetworkConfig.apply()` 必须在 datasets 导入**之前**调用，
        因为它要设置代理环境变量。放在函数里强制了这个顺序，
        避免以后有人在文件顶部 import 而破坏代理配置。
    """
    import datasets

    return datasets


def stream_dataset(
    spec: DatasetSpec,
    net: NetworkConfig,
    max_docs: int | None = None,
    shuffle_buffer: int = 10_000,
    seed: int = 1337,
    verbose: bool = True,
) -> Iterator[dict[str, Any]]:
    """流式读取一个数据集，逐条返回记录。

    参数
    ----
    max_docs : int, optional
        最多读多少条就停。None 表示读到没有为止。
        **流式模式下必须能中途退出** —— 否则会一直读到数据集末尾。
    shuffle_buffer : int
        乱序缓冲区大小。流式读取默认是按文件顺序的，直接取前 N 条会
        只拿到某个文件/某个来源的数据，产生严重的来源偏置。
        用缓冲区做局部打乱能显著缓解这个问题。
        设 0 表示不打乱。
    seed : int
        打乱用的随机种子。

    产出
    ----
    dict
        原始记录（未做字段抽取）。
    """
    datasets = _import_datasets()

    kwargs: dict[str, Any] = {
        "path": spec.repo_id,
        "split": spec.split,
        "streaming": True,  # 关键：流式，不落全量盘
    }
    if spec.config_name:
        kwargs["name"] = spec.config_name
    if spec.data_files:
        kwargs["data_files"] = spec.data_files

    if verbose:
        print(f"  [下载] {spec.describe()}")

    ds = _load_with_retry(datasets, kwargs, net, verbose=verbose)

    if shuffle_buffer and shuffle_buffer > 0:
        # shuffle 的 buffer_size 越大打乱越充分，但内存占用也越大。
        # 10000 条约占几十 MB，是个合理的折中。
        ds = ds.shuffle(buffer_size=shuffle_buffer, seed=seed)

    n = 0
    for record in ds:
        yield record
        n += 1
        if max_docs is not None and n >= max_docs:
            return


def _load_with_retry(datasets, kwargs: dict, net: NetworkConfig, verbose: bool = True):
    """带重试的 load_dataset。

    实测代理链路偶发 SSLEOFError（连接被对端中断），重试即可成功。
    没有重试的话，一个 5 天的下载任务会因为一次偶发网络抖动而整个失败。
    """
    last_err: Exception | None = None
    for attempt in range(1, net.max_retries + 1):
        try:
            return datasets.load_dataset(**kwargs)
        except Exception as e:  # noqa: BLE001 - 网络异常种类繁多，统一重试
            last_err = e
            if attempt < net.max_retries:
                # 指数退避：等待时间逐次翻倍，避免在对方限流时持续冲击
                wait = min(2**attempt, 30)
                if verbose:
                    print(f"        第 {attempt} 次失败 ({type(e).__name__})，{wait}s 后重试...")
                time.sleep(wait)

    raise RuntimeError(
        f"数据集加载失败，已重试 {net.max_retries} 次。\n"
        f"最后错误: {type(last_err).__name__}: {last_err}\n\n"
        f"排查建议：\n"
        f"  1. 确认代理在运行：curl -x {net.proxy} {net.endpoint}\n"
        f"  2. 确认数据集存在且无需授权（有些数据集是 gated，会返回 401）\n"
        f"  3. 检查 data_files 的路径 glob 是否匹配到文件"
    ) from last_err


def extract_text(record: dict[str, Any], spec: DatasetSpec) -> str | None:
    """从一条原始记录里抽取正文文本。

    处理各种字段结构的差异：
        - 正常情况：直接取 text_field 指定的字段
        - 嵌套结构：有些数据集把正文放在 list/dict 里
        - 字段缺失：返回 None，由调用方跳过
    """
    value = record.get(spec.text_field)

    # 字段不存在
    if value is None:
        return None

    # 字符串：最常见的情况
    if isinstance(value, str):
        return value if value.strip() else None

    # 列表：有些数据集把多段文本放在列表里，拼起来
    if isinstance(value, list):
        parts = [str(v) for v in value if isinstance(v, str) and v.strip()]
        return "\n".join(parts) if parts else None

    # 字典：尝试常见的正文字段名
    if isinstance(value, dict):
        for key in ("text", "content", "completion", "body"):
            v = value.get(key)
            if isinstance(v, str) and v.strip():
                return v
        return None

    return None


def sample_texts(
    specs: list[DatasetSpec],
    net: NetworkConfig,
    n_docs_per_spec: int,
    verbose: bool = True,
) -> list[str]:
    """从多个数据集各采样若干条文本，用于训练分词器。

    为什么不能拿整个语料训练分词器：
        分词器训练只需要统计字节对的共现频率，几百 MB 的文本就足够收敛了。
        用 2B tokens 的全量语料去训，耗时会长几十倍，但效果几乎没有区别。

    为什么要从**多个来源**采样：
        分词器的 merge 规则会偏向训练语料的内容分布。如果只用维基百科训练，
        分词器对新闻、口语化文本的压缩率就会很差。
        从最终语料的各个来源按比例采样，才能得到一个均衡的分词器。
    """
    all_texts: list[str] = []
    for spec in specs:
        if verbose:
            print(f"  采样 {spec.repo_id} ({n_docs_per_spec} 条)...")
        count = 0
        try:
            for record in stream_dataset(spec, net, max_docs=n_docs_per_spec * 2, verbose=False):
                text = extract_text(record, spec)
                if text and len(text) > 100:  # 跳过太短的碎片
                    all_texts.append(text)
                    count += 1
                    if count >= n_docs_per_spec:
                        break
        except Exception as e:  # noqa: BLE001
            print(f"    [警告] 采样失败，跳过该数据集: {type(e).__name__}: {str(e)[:100]}")

        if verbose:
            print(f"    实际采到 {count} 条")

    return all_texts


def check_connectivity(net: NetworkConfig) -> bool:
    """检查 HF 连通性，给出明确的诊断信息。

    单独做成函数是因为"连不上"是本项目最常见的阻塞点，
    而且原因有好几种（代理没开、代理端口错、数据集 gated），
    有个清晰的诊断能省很多排查时间。
    """
    import urllib.error
    import urllib.request

    url = f"{net.endpoint}/api/datasets?limit=1"
    proxy_handler = (
        urllib.request.ProxyHandler({"http": net.proxy, "https": net.proxy})
        if net.proxy
        else urllib.request.ProxyHandler({})
    )
    opener = urllib.request.build_opener(proxy_handler)

    try:
        t0 = time.time()
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with opener.open(req, timeout=15) as r:
            dt = time.time() - t0
            print(f"  [OK] HF 连通 ({net.endpoint}) —— HTTP {r.status}，耗时 {dt:.2f}s")
            return True
    except Exception as e:  # noqa: BLE001
        print(f"  [失败] 无法访问 {url}")
        print(f"         错误: {type(e).__name__}: {e}")
        print(f"\n  排查步骤：")
        print(f"    1. 确认代理软件正在运行，且端口是 {net.proxy}")
        print(f"    2. 手动测试: curl -x {net.proxy} {net.endpoint}")
        print(f"    3. 如果换了端口，修改 minilm/data/download.py 里的 DEFAULT_PROXY")
        return False
