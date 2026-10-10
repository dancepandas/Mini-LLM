"""
魔塔社区（ModelScope）数据源。

## 为什么要有这个后端

    实测 huggingface.co 直连超时，走代理虽然能通但很慢（2.33 MB/s），
    而且 datasets 的流式读取在列大目录时会卡住（MNBVC 有 2.4 万个文件）。
    魔塔社区在国内**直连可用且速度快**，是更实际的语料来源。

## 和 HF 后端的关键差异

    HF 的 `load_dataset(streaming=True)` 有原生的流式支持，
    可以对整个数据集做 shuffle、逐条产出。

    魔塔没有等价的接口，所以这里实现了**文件粒度的流式**：
        列出文件清单 → 逐文件下载到本地缓存 → 逐条解析 → 删掉已处理的文件
    这样内存占用是恒定的（一次只有一个文件在内存里），
    对 100GB 级别的语料也不会爆内存。

    parquet 文件内部还可以进一步分批读（`iter_batches`），
    连单个文件都不需要完整载入内存。

## 网络注意事项

    **魔塔不需要代理，必须直连。** 如果设置了 http_proxy 反而可能连不上，
    所以本模块在发起请求前会主动清掉代理环境变量。
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterator

from tqdm import tqdm


MODELSCOPE_HOST = "https://www.modelscope.cn"


def clear_proxy_env() -> None:
    """清除代理环境变量。

    魔塔是国内站点，直连即可，走代理反而会失败或变慢。
    每次发起请求前调用，防止上层代码设置的代理污染这里的请求。
    """
    for key in (
        "http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
        "all_proxy", "ALL_PROXY",
    ):
        os.environ.pop(key, None)


def _request_json(url: str, timeout: int = 30, retries: int = 4) -> Any:
    """发 GET 请求并解析 JSON，带重试。

    实测魔塔偶发 SSL UNEXPECTED_EOF，重试即可成功。
    """
    clear_proxy_env()
    last_err: Exception | None = None

    # ⚠️ 关键：不仅要清环境变量，还必须显式传一个空的 ProxyHandler。
    # 只清 os.environ 是不够的 —— urllib 在 Windows 上还会去读**系统级代理设置**
    # （注册表里的 WinINET 配置），如果系统代理指向一个处理不了该站点的代理，
    # 就会报 "SSL record layer failure" 这类底层错误。
    # 传 ProxyHandler({}) 能强制 urllib 完全绕过所有代理。
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with opener.open(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < retries:
                time.sleep(min(2**attempt, 15))

    raise RuntimeError(f"请求失败（重试 {retries} 次）: {url}\n最后错误: {last_err}") from last_err


# ===========================================================================
# 文件清单
# ===========================================================================
def list_files(
    repo_id: str,
    revision: str = "master",
    recursive: bool = True,
    verbose: bool = True,
    page_size: int = 100,
    max_pages: int = 500,
) -> list[dict[str, Any]]:
    """列出魔塔数据集的文件清单。

    参数
    ----
    repo_id : str
        形如 "AI-ModelScope/wikipedia-cn-20230720-filtered"
    recursive : bool
        是否递归列出子目录。大型语料通常按目录分片，需要 True。
    max_pages : int
        最多翻多少页，防止遇到超大仓库时无限翻页（每页 page_size 个文件）。

    返回
    ----
    list[dict]
        每项包含 "Path"（相对路径）和 "Size"（字节数，可能为 None）。

    ## 关于接口路径的坑（踩过）

        我一开始用的是 `/oss/tree/`，它**返回 HTTP 200 但 Data 是空数组**
        （`{"Code":200,"Data":[],"TotalCount":0}`）—— 不报错，就是没数据。
        这种"成功但为空"的响应极具误导性，会让人以为数据集是空的。

        正确端点是 **`/repo/tree`**，返回
        `{"Code":200,"Data":{"Files":[{"Path":..., "Size":...}]}}`。

        优先用 SDK 的 `get_dataset_files()`，它内部处理了分页；
        只有 SDK 不可用时才回退到手写 HTTP + 翻页。
    """
    # ---- 主路径：手写 HTTP + 翻页 ----
    # 为什么不优先用 SDK 的 get_dataset_files：
    #   实测它在 page_size=1000 时**只返回第一页就停了**（拿到 999 个文件），
    #   而 Fineweb-Edu-Chinese-V2.1 实际有 29286 个文件。
    #   这种"成功但结果不全"的行为很危险 —— 不报错，
    #   但会导致我们只看到字母序靠前的 2_3/ 分片，看不到需要的 4_5/ 分片。
    #   自己翻页虽然朴素，但行为完全可控。
    collected: list[dict] = []
    seen_paths: set[str] = set()
    total_count: int | None = None

    for page in range(1, max_pages + 1):
        url = (
            f"{MODELSCOPE_HOST}/api/v1/datasets/{repo_id}/repo/tree"
            f"?Revision={revision}&Recursive={str(recursive).lower()}"
            f"&PageNumber={page}&PageSize={page_size}"
        )
        data = _request_json(url)
        payload = data.get("Data") or {}

        # Data 可能直接是 list，也可能是 {"Files": [...]}
        if isinstance(payload, list):
            batch = payload
        elif isinstance(payload, dict):
            batch = payload.get("Files", []) or []
        else:
            batch = []

        if total_count is None:
            total_count = data.get("TotalCount")

        if not batch:
            break

        for e in batch:
            p = e.get("Path") or e.get("Name")
            # 用路径去重，防止分页接口在边界上重复返回同一条
            if p and p not in seen_paths:
                seen_paths.add(p)
                collected.append(e)

        # 终止判断：收齐了，或这一页不满（说明是最后一页）
        if total_count is not None and len(seen_paths) >= total_count:
            break
        if len(batch) < page_size:
            break

    files = _normalize_file_entries(collected)

    if verbose:
        total_bytes = sum(f.get("Size", 0) or 0 for f in files)
        print(f"  [魔塔] {repo_id}: {len(files)} 个文件, 共 {total_bytes/1e9:.2f} GB"
              + (f" (仓库报告 {total_count} 项)" if total_count else ""))

    # 兜底：HTTP 路径也拿不到时，才退回 SDK
    if not files:
        try:
            from modelscope.hub.api import HubApi

            clear_proxy_env()
            api = HubApi()
            raw = api.get_dataset_files(
                repo_id, revision=revision, root_path="/",
                recursive=recursive, page_number=1, page_size=page_size,
            )
            files = _normalize_file_entries(raw)
            if verbose and files:
                print(f"  [魔塔] (SDK 兜底) {len(files)} 个文件")
        except Exception as e:  # noqa: BLE001
            if verbose:
                print(f"  [魔塔] SDK 兜底也失败: {type(e).__name__}: {str(e)[:80]}")

    return files


def _normalize_file_entries(entries: list) -> list[dict[str, Any]]:
    """把不同来源的文件条目统一成 {"Path": str, "Size": int}。

    魔塔的接口在不同端点/版本下字段名不一致（Name / Path / Size / FileSize），
    统一规范化后，上层代码就不用关心这些差异了。

    只保留 blob（文件），过滤掉 tree（目录）—— 目录条目没有 Size，
    混进列表里会让大小统计和下载逻辑出错。
    """
    out: list[dict[str, Any]] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        # 过滤目录
        etype = e.get("Type")
        if etype is not None and etype not in ("blob", "file", "BLOB"):
            continue

        path = e.get("Path") or e.get("Name")
        if not path:
            continue
        # 有些版本把路径放在 Key 里
        if not e.get("Path") and e.get("Key"):
            path = e["Key"]

        size = e.get("Size") or e.get("FileSize") or 0
        try:
            size = int(size)
        except (TypeError, ValueError):
            size = 0

        out.append({"Path": path, "Size": size})
    return out


def match_files(files: list[dict], pattern: str) -> list[dict]:
    """按 glob 模式筛选文件。

    用 pathlib 的 match 语义，支持 "4_5/*.parquet" 这类模式。
    通过只下载需要的子集来控制语料来源和体积 ——
    比如 Fineweb 系列有多个质量档位，我们只要最高质量的那一档。
    """
    from fnmatch import fnmatch

    matched = [f for f in files if fnmatch(f.get("Path", ""), pattern)]
    return matched


# ===========================================================================
# 下载
# ===========================================================================
def download_file(
    repo_id: str,
    file_path: str,
    dest_dir: str | Path,
    revision: str = "master",
    retries: int = 4,
    verbose: bool = False,
) -> Path:
    """下载单个文件到本地。

    ## 为什么用直连 HTTP 而不是官方 SDK 的 snapshot_download

        实测 SDK 有几个问题，在批量下载场景下很麻烦：
          1. 它把文件塞进 `<cache_dir>/datasets/<repo>/snapshots/master/` 这样的深层
             嵌套目录，返回的路径和传入的 dest_dir 不一致，调用方要额外处理
          2. 它内部走 requests，而 requests 在 Windows 上会读**系统代理设置**，
             实测会导致下载变慢甚至卡死
          3. 它没有暴露并发控制

        直连 HTTP + `ProxyHandler({})` 完全绕开这些问题，行为可控。
        实测速度 1.2~1.6 MB/s，和 SDK 相当或更快。

    ## 支持断点续传

        如果目标文件已存在且大小 > 0，直接复用，不重复下载。
        对动辄几百 MB 的分片来说，断点续传能省下大量时间。
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    # 用 __ 替换路径分隔符，避免不同子目录下的同名文件互相覆盖
    dest = dest_dir / file_path.replace("/", "__")

    if dest.exists() and dest.stat().st_size > 0:
        if verbose:
            print(f"    复用: {dest.name}")
        return dest

    url = f"{MODELSCOPE_HOST}/api/v1/datasets/{repo_id}/repo?Revision={revision}&FilePath={file_path}"
    clear_proxy_env()
    # 必须显式禁用代理：清环境变量不够，urllib 还会读 Windows 系统代理
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    tmp = dest.with_suffix(dest.suffix + ".part")

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with opener.open(req, timeout=180) as r, tmp.open("wb") as f:
                shutil.copyfileobj(r, f, length=1 << 20)
            os.replace(tmp, dest)
            if verbose:
                print(f"    下载完成: {file_path} ({dest.stat().st_size/1e6:.1f} MB)")
            return dest
        except Exception as e:  # noqa: BLE001
            last_err = e
            # 清掉半成品，避免下次误判为"已下载"
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            if attempt < retries:
                time.sleep(min(2**attempt, 20))

    raise RuntimeError(
        f"文件下载失败: {repo_id}/{file_path}\n"
        f"最后错误: {type(last_err).__name__}: {last_err}"
    ) from last_err


# ===========================================================================
# 解析
# ===========================================================================
def iter_parquet(path: Path, batch_size: int = 1000) -> Iterator[dict[str, Any]]:
    """分批迭代 parquet 文件，逐行产出 dict。

    用 ParquetFile.iter_batches 而不是 pq.read_table：
        read_table 会把整个文件读进内存。大文件（几百 MB）会占用大量内存，
        而 iter_batches 只保持一个 batch（1000 行）在内存里，占用恒定。
    """
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size):
        # to_pylist 把 Arrow 结构转成 Python dict 列表
        yield from batch.to_pylist()


def iter_jsonl(path: Path, encoding: str = "utf-8") -> Iterator[dict[str, Any]]:
    """逐行迭代 JSONL 文件（自动识别 .gz 压缩）。

    逐行读而不是 json.load 整个文件：
        语料文件动辄几百 MB 到几 GB，整体载入会爆内存（本机 32GB）。
        逐行流式读取内存占用是常数。
    """
    name = path.name.lower()
    if name.endswith(".gz"):
        f = gzip.open(path, "rt", encoding=encoding, errors="replace")
    else:
        f = path.open("r", encoding=encoding, errors="replace")

    with f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # 语料里难免有损坏的行，跳过而不是中断整个流程
                continue
            if isinstance(obj, dict):
                yield obj
            # 顶层是数组/标量的行直接跳过（格式不符）


def iter_file_records(path: Path) -> Iterator[dict[str, Any]]:
    """按扩展名自动选择解析方式。"""
    name = path.name.lower()
    if name.endswith(".parquet"):
        yield from iter_parquet(path)
    elif name.endswith((".jsonl", ".jsonl.gz", ".json", ".json.gz")):
        yield from iter_jsonl(path)
    else:
        raise ValueError(f"不支持的文件格式: {path.name}（目前支持 parquet / jsonl）")


# ===========================================================================
# 高层接口
# ===========================================================================
def stream_records(
    repo_id: str,
    file_pattern: str = "**/*.parquet",
    cache_dir: str | Path = "data/ms_cache",
    max_docs: int | None = None,
    tmp_dir: str | Path | None = None,
    keep_files: bool = False,
    verbose: bool = True,
) -> Iterator[dict[str, Any]]:
    """从魔塔数据集流式读取记录。

    工作方式（文件粒度流式）：
        for 每个匹配的文件:
            下载 → 逐条解析产出 → （默认）删除该文件

    这样内存占用恒定，磁盘占用也只和单个文件相当，
    可以处理远超本地磁盘容量的数据集。

    参数
    ----
    max_docs : int, optional
        最多产出多少条记录就停。None 表示读完全部（慎用，可能是几亿条）。
    keep_files : bool
        是否保留下载的文件。默认 False —— 读完就删，避免磁盘被撑爆。
        调试时设为 True 便于反复查看。
    """
    files = list_files(repo_id, verbose=verbose)
    matched = match_files(files, file_pattern)

    if not matched:
        raise ValueError(
            f"没有文件匹配模式 {file_pattern!r}（共 {len(files)} 个文件）。\n"
            f"可用路径样例: {[f.get('Path') for f in files[:8]]}"
        )

    total_bytes = sum(f.get("Size", 0) or 0 for f in matched)
    if verbose:
        print(f"  [魔塔] 匹配到 {len(matched)} 个文件, {total_bytes/1e9:.2f} GB")

    tmp_dir = Path(tmp_dir or (Path(cache_dir) / "_tmp"))
    tmp_dir.mkdir(parents=True, exist_ok=True)

    n = 0
    pbar = tqdm(matched, desc=f"  读取 {repo_id}", unit="file", disable=not verbose) if verbose else matched

    for fmeta in pbar:
        fpath = fmeta.get("Path")
        if not fpath:
            continue

        try:
            local = download_file(repo_id, fpath, tmp_dir, verbose=False)
        except Exception as e:  # noqa: BLE001
            # 单个文件失败不该中断整个流程 —— 语料有几千个文件，
            # 因为一个文件损坏就全盘放弃是不可接受的
            print(f"    [跳过] {fpath}: {type(e).__name__}: {str(e)[:100]}")
            continue

        try:
            for record in iter_file_records(local):
                yield record
                n += 1
                if max_docs is not None and n >= max_docs:
                    return
        except Exception as e:  # noqa: BLE001
            print(f"    [解析失败] {fpath}: {type(e).__name__}: {str(e)[:100]}")
        finally:
            # 读完即删，控制磁盘占用。
            # 这是文件粒度流式的关键一步 —— 不删的话，处理 100GB 语料就会需要 100GB 磁盘。
            if not keep_files:
                try:
                    local.unlink()
                except OSError:
                    pass


def download_file_chunked(
    repo_id: str,
    file_path: str,
    dest: str | Path,
    workers: int = 8,
    chunk_mb: int = 64,
    revision: str = "master",
    verbose: bool = True,
) -> Path:
    """用 HTTP Range 把**单个大文件**分块并行下载。

    ## 为什么需要这个

        魔塔单连接只有约 1.5 MB/s。minimind 的 `pretrain_t2t.jsonl` 有 8.3 GB，
        单连接下载要 92 分钟。

        但实测该端点**支持 HTTP Range 请求**（`Content-Range` 头正常返回），
        所以可以把文件切成多块并发下载再拼起来。
        8 个并发能把 92 分钟压到 10 分钟出头。

        这和 `download_files_parallel` 的区别：
          - 前者解决"很多小文件"（如 Fineweb 的 9745 个分片）
          - 这个解决"一个巨大文件"（如 8.3GB 的单个 jsonl）
        两者互补，本项目两种场景都会遇到。

    ## 实现要点

        1. 先发一个 Range 请求探测文件总大小
        2. 按 chunk_mb 切块，用线程池并发下载
        3. 每块写到独立的 .part 文件，避免并发写同一文件句柄（Windows 上会出错）
        4. 全部完成后按顺序拼接成最终文件

        "先写独立分块再拼接"比"多线程 seek 写同一文件"更稳：
        后者需要文件锁，且任何一块失败都会留下损坏的文件。
    """
    import math

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size > 0:
        if verbose:
            print(f"    复用已存在: {dest.name}")
        return dest

    url = f"{MODELSCOPE_HOST}/api/v1/datasets/{repo_id}/repo?Revision={revision}&FilePath={file_path}"
    clear_proxy_env()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    # ---- 第 1 步：探测总大小 ----
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0", "Range": "bytes=0-0"}
    )
    with opener.open(req, timeout=60) as r:
        content_range = r.headers.get("Content-Range")  # 形如 "bytes 0-0/8275123456"
        if not content_range or "/" not in content_range:
            raise RuntimeError(f"该文件不支持 Range 请求，无法分块下载: {file_path}")
        total_size = int(content_range.split("/")[-1])

    chunk = chunk_mb * 1024 * 1024
    n_chunks = math.ceil(total_size / chunk)

    if verbose:
        print(f"    分块下载 {file_path}: {total_size/1e9:.2f} GB, "
              f"{n_chunks} 块 × {chunk_mb} MB, {workers} 并发")

    parts_dir = dest.parent / f".{dest.name}.parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    done_bytes = [0] * n_chunks
    lock = threading.Lock()
    pbar = tqdm(total=total_size, desc=f"      {dest.name}", unit="B",
                unit_scale=True, disable=not verbose)

    def fetch_chunk(i: int) -> None:
        start = i * chunk
        end = min(start + chunk - 1, total_size - 1)
        part = parts_dir / f"{i:05d}.part"

        # 断点续传：该块已完成就直接跳过
        expected = end - start + 1
        if part.exists() and part.stat().st_size == expected:
            with lock:
                done_bytes[i] = expected
                pbar.update(expected)
            return

        last_err = None
        for attempt in range(1, 5):
            try:
                # 关键：每次请求都必须重新禁用代理，
                # 因为线程池里的线程可能在不同时机执行
                op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                req = urllib.request.Request(
                    url, headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes={start}-{end}"}
                )
                with op.open(req, timeout=180) as r, part.open("wb") as f:
                    shutil.copyfileobj(r, f, length=1 << 20)
                with lock:
                    done_bytes[i] = expected
                    pbar.update(expected)
                return
            except Exception as e:  # noqa: BLE001
                last_err = e
                if part.exists():
                    try:
                        part.unlink()
                    except OSError:
                        pass
                time.sleep(min(2**attempt, 15))

        with lock:
            pbar.set_postfix({"失败块": i})
        raise RuntimeError(f"分块 {i} 下载失败: {last_err}")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(fetch_chunk, i) for i in range(n_chunks)]
        for f in as_completed(futures):
            f.result()  # 任何一块失败就让整体失败，避免拼出损坏的文件

    pbar.close()

    # ---- 第 3 步：按顺序拼接 ----
    tmp = dest.with_suffix(dest.suffix + ".assembling")
    with tmp.open("wb") as out:
        for i in range(n_chunks):
            part = parts_dir / f"{i:05d}.part"
            with part.open("rb") as pf:
                shutil.copyfileobj(pf, out, length=1 << 22)
    os.replace(tmp, dest)

    # 清理分块
    try:
        shutil.rmtree(parts_dir)
    except OSError:
        pass

    actual = dest.stat().st_size
    if actual != total_size:
        raise RuntimeError(
            f"拼接后大小不符：期望 {total_size:,} 字节，实际 {actual:,} 字节。"
            f"文件可能已损坏，请删除后重下。"
        )

    if verbose:
        print(f"    完成: {dest.name} ({actual/1e9:.2f} GB)")

    return dest


# ===========================================================================
# 并发下载 —— 实测单连接只有 ~1 MB/s，必须靠并发才能拿到可用带宽
# ===========================================================================
def check_speed(repo_id: str, file_path: str, mb: int = 4, timeout: int = 60) -> float:
    """实测单个文件的下载速度（MB/s），不落盘。

    用途：
      1. 在开始大规模下载前判断当前网络是否正常
      2. 决定需要开多少并发（并发数 ≈ 目标带宽 / 单连接带宽）
    """
    url = f"{MODELSCOPE_HOST}/api/v1/datasets/{repo_id}/repo?Revision=master&FilePath={file_path}"
    clear_proxy_env()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    limit = mb * 1024 * 1024
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0", "Range": f"bytes=0-{limit-1}"}
    )
    t0 = time.time()
    n = 0
    with opener.open(req, timeout=timeout) as r:
        while n < limit:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            n += len(chunk)
    dt = time.time() - t0
    return (n / 1e6 / dt) if dt > 0 else 0.0


def dataset_info(repo_id: str) -> dict[str, Any]:
    """获取数据集基本信息（用于确认存在性和许可证）。

    返回的 Data 里有 License、Downloads、Likes、ReadmeContent 等字段。
    在决定用一个数据集之前应该先看 License —— 有些数据集标着宽松许可，
    但正文里另有社区许可或商用限制。
    """
    url = f"{MODELSCOPE_HOST}/api/v1/datasets/{repo_id}"
    try:
        return _request_json(url)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def download_files_chunked_parallel(
    repo_id: str,
    file_specs: list[tuple[str, int]],
    dest_dir: str | Path,
    workers: int = 8,
    chunk_mb: int = 64,
    revision: str = "master",
    verbose: bool = True,
) -> list[Path]:
    """顺序下载多个大文件，每个文件内部用分块并行。

    ## 为什么不把所有文件的所有块放进一个线程池

        那会让并发数失控（10 个文件 × 8 块 = 80 个并发连接），
        大概率被服务端限流，而且本地文件句柄也会耗尽。

        逐个文件处理，每个文件内部开 `workers` 个并发，
        并发度始终受控，行为可预测。

    参数
    ----
    file_specs : list[tuple[str, int]]
        [(文件路径, 字节大小), ...]。大小用于进度显示和完成度估算。
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    total_bytes = sum(sz for _, sz in file_specs)
    done_bytes = 0
    t0 = time.time()
    results: list[Path] = []

    for i, (fpath, size) in enumerate(file_specs, 1):
        print(f"\n  [{i}/{len(file_specs)}] {fpath}  ({size/1e9:.2f} GB)")
        dest = dest_dir / fpath.replace("/", "__")

        # 已完成的大文件直接跳过（断点续传）
        if dest.exists() and dest.stat().st_size == size:
            print(f"    已存在，跳过")
            results.append(dest)
            done_bytes += size
            continue

        try:
            p = download_file_chunked(
                repo_id, fpath, dest,
                workers=workers, chunk_mb=chunk_mb,
                revision=revision, verbose=verbose,
            )
            results.append(p)
        except Exception as e:  # noqa: BLE001
            print(f"    [失败] {type(e).__name__}: {str(e)[:150]}")
            continue

        done_bytes += size
        elapsed = time.time() - t0
        rate = done_bytes / 1e6 / elapsed if elapsed > 0 else 0
        remain = (total_bytes - done_bytes) / (rate * 1e6) if rate > 0 else 0
        print(f"    累计进度 {done_bytes/1e9:.2f}/{total_bytes/1e9:.2f} GB, "
              f"{rate:.2f} MB/s, 剩余约 {remain/60:.1f} 分钟")

    return results


def download_files_parallel(
    repo_id: str,
    file_paths: list[str],
    dest_dir: str | Path,
    workers: int = 8,
    revision: str = "master",
    verbose: bool = True,
    stop_after_bytes: int | None = None,
) -> list[Path]:
    """并发下载多个文件。

    ## 为什么要并发（实测依据）

        实测魔塔**单连接下载只有约 1 MB/s**，而且不稳定
        （有时会卡到 0）。照这个速度下 5GB 语料要一个多小时。

        但限制在单连接上，不在总带宽上 —— 开 8 个并发后总吞吐能到 6~8 MB/s，
        下载时间缩短到十分之一。对于由大量小分片组成的数据集
        （如 Fineweb 的 9745 个 parquet），并发是唯一实际可行的方案。

    ## 关于线程数

        设 8 是保守的折中：够用、不会因为并发过高被限流或耗尽文件句柄。
        设太大反而会因为服务端限流而变慢。

    ## 关于 stop_after_bytes

        用于"下够就停"的场景：我们只需要 2B tokens，
        不需要把 74GB 的 Fineweb 全下下来。达到字节目标后，
        已在途的任务会完成，但不再提交新任务。
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    results: list[Path] = []
    downloaded_bytes = 0
    lock = threading.Lock()
    stop_flag = threading.Event()
    failed: list[str] = []

    pbar = tqdm(total=len(file_paths), desc=f"  下载 {repo_id}", unit="file", disable=not verbose)

    def worker(fpath: str) -> None:
        nonlocal downloaded_bytes
        if stop_flag.is_set():
            pbar.update(1)
            return
        try:
            p = download_file(repo_id, fpath, dest_dir, revision=revision, verbose=False)
            with lock:
                results.append(p)
                downloaded_bytes += p.stat().st_size
                pbar.set_postfix({"GB": f"{downloaded_bytes/1e9:.2f}"})
                # 达到字节目标就通知其他 worker 停止提交新任务
                if stop_after_bytes and downloaded_bytes >= stop_after_bytes:
                    stop_flag.set()
        except Exception as e:  # noqa: BLE001
            with lock:
                failed.append(f"{fpath}: {type(e).__name__}")
        finally:
            pbar.update(1)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker, fp) for fp in file_paths]
        for f in as_completed(futures):
            # 逐个等待完成，避免异常被吞掉
            try:
                f.result()
            except Exception:  # noqa: BLE001
                pass

    pbar.close()

    if failed and verbose:
        print(f"  [警告] {len(failed)} 个文件下载失败（已跳过），例如: {failed[:3]}")

    return results
