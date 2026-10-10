"""
数据管线：从 HuggingFace 下载中文语料到训练用的二进制文件。

处理流程：

    download.py    流式下载 + 字段抽取（带代理、重试）
        ↓
    clean.py       清洗：长度过滤、去重、乱码过滤
        ↓
    build_bin.py   分词 → 拼接 → 写入 uint16 memmap（.bin 文件）
        ↓
    dataset.py     从 memmap 随机裁剪出训练批次

**为什么要把数据预处理成二进制 memmap，而不是训练时在线分词：**

    在线分词意味着每个 step 都要跑一遍 BPE 分词。BPE 是 CPU 密集的 Python 逻辑，
    在 4618 tok/s 的训练吞吐下会成为严重瓶颈（GPU 会一直等 CPU）。
    预先分好词、存成扁平数组后，训练时只需要做一次内存切片，开销可以忽略。
    这也是 nanoGPT 等项目的标准做法。

    代价是磁盘：2B tokens × 2 字节（uint16）= 约 4GB，完全可以接受。
"""
