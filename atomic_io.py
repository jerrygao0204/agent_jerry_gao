# # atomic_io.py
# """
# atomic_io.py - 跨进程安全的原子 JSON 写入工具

# 给 extractor.py / layer_mapper.py 共用，解决两个问题：
# 1. 并发写同一文件互相覆盖 —— 用 filelock 按"目标文件路径"加锁，
#    锁文件路径 = 目标文件路径 + .lock，天然跨模块/跨进程共享同一把锁。
# 2. 写入中途崩溃损坏原文件 —— 先写临时文件，再用 os.replace() 原子替换。
# """

# import json
# import os
# import tempfile
# from pathlib import Path
# from typing import Any, Union

# from filelock import FileLock

# LOCK_SUFFIX = ".lock"
# LOCK_TIMEOUT = 30  # 秒，避免异常情况下无限等待死锁


# def file_lock_for(path: Union[str, Path], timeout: int = LOCK_TIMEOUT) -> FileLock:
#     """返回锁住 path 对应文件的 FileLock。
#     调用方应该把"读取旧数据 -> 合并 -> 写入"整个流程都包在这个锁的
#     with 块里，而不是只锁写入这一步——否则两个进程可能各自基于
#     同一份旧数据算出不同的合并结果，后写的会覆盖掉先写的。
#     """
#     target = Path(path)
#     target.parent.mkdir(parents=True, exist_ok=True)
#     return FileLock(str(target) + LOCK_SUFFIX, timeout=timeout)


# def atomic_dump_json(path: Union[str, Path], data: Any, indent: int = 2) -> None:
#     """把 data 原子写入 path。
#     注意：此函数本身不加锁，需要在调用方持有的 file_lock_for(path)
#     临界区内调用，负责的是"写入方式"而不是"互斥"。
#     """
#     target = Path(path)
#     target.parent.mkdir(parents=True, exist_ok=True)
#     fd, tmp_path = tempfile.mkstemp(
#         dir=str(target.parent), prefix=target.name + ".", suffix=".tmp"
#     )
#     try:
#         with os.fdopen(fd, "w", encoding="utf-8") as f:
#             json.dump(data, f, ensure_ascii=False, indent=indent)
#         os.replace(tmp_path, target)
#     except Exception:
#         if os.path.exists(tmp_path):
#             os.remove(tmp_path)
#         raise

# atomic_io.py
"""
atomic_io.py - 跨进程/跨线程安全的原子 JSON 写入与可重入锁工具

解决问题：
1. 原生 FileLock 不可重入导致的同线程嵌套调用死锁。
2. 跨进程写同一文件互相覆盖。
3. 写入中途崩溃损坏原文件（临时文件 + os.replace 原子替换）。
"""

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Union
from filelock import FileLock

LOCK_SUFFIX = ".lock"
LOCK_TIMEOUT = 30  # 秒

# 线程局部变量，维护当前线程持有的锁状态
_thread_local = threading.local()


class ReentrantFileLock:
    """
    支持同线程可重入的跨进程 FileLock 包装器
    """
    def __init__(self, lock_file_path: str, timeout: int = LOCK_TIMEOUT):
        self.lock_file_path = os.path.abspath(lock_file_path)
        self.timeout = timeout
        self._raw_lock = FileLock(self.lock_file_path, timeout=self.timeout)

    def __enter__(self):
        if not hasattr(_thread_local, "locks"):
            _thread_local.locks = {}

        # 若当前线程已持有该锁，自增计数器并直接返回（实现重入）
        if self.lock_file_path in _thread_local.locks:
            _thread_local.locks[self.lock_file_path]["depth"] += 1
            return self

        # 首次获取物理锁
        self._raw_lock.acquire()
        _thread_local.locks[self.lock_file_path] = {
            "depth": 1,
            "raw_lock": self._raw_lock
        }
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(_thread_local, "locks") and self.lock_file_path in _thread_local.locks:
            _thread_local.locks[self.lock_file_path]["depth"] -= 1
            # 当重入层数降为 0 时，释放物理锁
            if _thread_local.locks[self.lock_file_path]["depth"] == 0:
                del _thread_local.locks[self.lock_file_path]
                self._raw_lock.release()


def file_lock_for(path: Union[str, Path], timeout: int = LOCK_TIMEOUT) -> ReentrantFileLock:
    """返回支持可重入的文件锁实例"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lock_path = str(target) + LOCK_SUFFIX
    return ReentrantFileLock(lock_path, timeout=timeout)


def atomic_dump_json(path: Union[str, Path], data: Any, indent: int = 2) -> None:
    """把 data 原子写入 path（必须在 file_lock_for 临界区内调用）"""
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