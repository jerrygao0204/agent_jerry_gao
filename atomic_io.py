# memory_growth/atomic_io.py
"""
atomic_io.py - 跨进程安全的原子 JSON 写入工具

给 extractor.py / layer_mapper.py 共用，解决两个问题：
1. 并发写同一文件互相覆盖 —— 用 filelock 按"目标文件路径"加锁，
   锁文件路径 = 目标文件路径 + .lock，天然跨模块/跨进程共享同一把锁。
2. 写入中途崩溃损坏原文件 —— 先写临时文件，再用 os.replace() 原子替换。
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Union

from filelock import FileLock

LOCK_SUFFIX = ".lock"
LOCK_TIMEOUT = 30  # 秒，避免异常情况下无限等待死锁


def file_lock_for(path: Union[str, Path], timeout: int = LOCK_TIMEOUT) -> FileLock:
    """返回锁住 path 对应文件的 FileLock。
    调用方应该把"读取旧数据 -> 合并 -> 写入"整个流程都包在这个锁的
    with 块里，而不是只锁写入这一步——否则两个进程可能各自基于
    同一份旧数据算出不同的合并结果，后写的会覆盖掉先写的。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return FileLock(str(target) + LOCK_SUFFIX, timeout=timeout)


def atomic_dump_json(path: Union[str, Path], data: Any, indent: int = 2) -> None:
    """把 data 原子写入 path。
    注意：此函数本身不加锁，需要在调用方持有的 file_lock_for(path)
    临界区内调用，负责的是"写入方式"而不是"互斥"。
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(target.parent), prefix=target.name + ".", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=indent)
        os.replace(tmp_path, target)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise