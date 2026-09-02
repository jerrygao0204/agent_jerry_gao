# tests/test_atomic_io.py
import json
import os
import threading
import time

import pytest

from atomic_io import atomic_dump_json, file_lock_for


def test_atomic_dump_json_writes_correct_content(tmp_path):
    """原子写入后，文件内容应完整且正确"""
    target = tmp_path / "sample.json"
    payload = {"foo": "bar", "list": [1, 2, 3]}

    atomic_dump_json(target, payload)

    assert target.exists()
    with open(target, "r", encoding="utf-8") as f:
        assert json.load(f) == payload


def test_atomic_dump_json_no_leftover_tmp_file(tmp_path):
    """写入成功后，不应该在目录里留下任何 .tmp 临时文件"""
    target = tmp_path / "sample.json"
    atomic_dump_json(target, {"a": 1})

    leftover = [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert leftover == [], f"写入成功后仍残留临时文件: {leftover}"


def test_atomic_dump_json_overwrite_replaces_old_content(tmp_path):
    """多次写入应该是整份替换，而不是残留旧内容"""
    target = tmp_path / "sample.json"
    atomic_dump_json(target, {"version": 1, "extra_field": "old"})
    atomic_dump_json(target, {"version": 2})

    with open(target, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert data == {"version": 2}
    assert "extra_field" not in data


def test_file_lock_blocks_concurrent_access(tmp_path):
    """同一路径的锁应该互斥：第二个线程必须等第一个释放锁才能进入"""
    target = tmp_path / "locked.json"
    events = []

    def worker(name, hold_seconds):
        with file_lock_for(target, timeout=5):
            events.append(f"{name}_enter")
            time.sleep(hold_seconds)
            events.append(f"{name}_exit")

    t1 = threading.Thread(target=worker, args=("A", 0.3))
    t2 = threading.Thread(target=worker, args=("B", 0.0))
    t1.start()
    time.sleep(0.05)  # 确保 A 先拿到锁
    t2.start()
    t1.join()
    t2.join()

    # 关键断言：A 必须完整地 enter->exit 之后，B 才能 enter
    # 也就是不应该出现 A_enter, B_enter, A_exit 这种交叉的顺序
    assert events.index("A_exit") < events.index("B_enter"), f"锁未生效，执行顺序: {events}"


def test_atomic_dump_json_concurrent_writes_no_corruption(tmp_path):
    """并发写同一文件 50 次，最终文件必须是合法 JSON 且是某一次完整的写入结果"""
    target = tmp_path / "concurrent.json"

    def worker(i):
        with file_lock_for(target):
            atomic_dump_json(target, {"writer": i})

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 文件必须存在、必须是合法 JSON（不能是损坏的半份文件）
    with open(target, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "writer" in data
    assert 0 <= data["writer"] < 50