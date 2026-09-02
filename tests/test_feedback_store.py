# tests/test_feedback_store.py
"""
memory/feedback_store.py 的测试

覆盖：
1. 单次写入内容是否完整、字段默认值是否正确
2. 多次写入是否按 JSONL 顺序追加，互不覆盖
3. 并发调用 record_feedback 是否不丢失、不损坏任何一行
4. 写入失败时是否优雅返回 False，而不是让异常往外抛
"""
import json
import threading

from memory.feedback_store import FeedbackStore


def _read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_record_feedback_writes_correct_content(tmp_path):
    store = FeedbackStore(log_dir=str(tmp_path))

    ok = store.record_feedback(
        user_id="gaozheng",
        query="怎么创建预警？",
        response="进入管理系统...",
        rating="like",
        feedback_text="回答很清楚",
        citations=["数据预警.pdf_h_6"],
    )

    assert ok is True
    records = _read_jsonl(store.log_file)
    assert len(records) == 1
    record = records[0]
    assert record["user_id"] == "gaozheng"
    assert record["query"] == "怎么创建预警？"
    assert record["rating"] == "like"
    assert record["feedback_text"] == "回答很清楚"
    assert record["citations"] == ["数据预警.pdf_h_6"]
    assert "timestamp" in record


def test_record_feedback_default_fields(tmp_path):
    """user_id 为空时应回退为 default_user；citations 为 None 时应回退为空列表"""
    store = FeedbackStore(log_dir=str(tmp_path))

    store.record_feedback(user_id="", query="q", response="r", rating="dislike")

    records = _read_jsonl(store.log_file)
    assert records[0]["user_id"] == "default_user"
    assert records[0]["citations"] == []
    assert records[0]["feedback_text"] == ""


def test_record_feedback_multiple_appends_preserve_order(tmp_path):
    store = FeedbackStore(log_dir=str(tmp_path))

    for i in range(5):
        store.record_feedback(user_id="u", query=f"q{i}", response="r", rating="like")

    records = _read_jsonl(store.log_file)
    assert [r["query"] for r in records] == [f"q{i}" for i in range(5)]


def test_concurrent_record_feedback_no_lost_writes(tmp_path):
    """50 个线程并发写入，最终应该一条不丢、每一行都是合法 JSON"""
    store = FeedbackStore(log_dir=str(tmp_path))

    def worker(i):
        store.record_feedback(user_id="u", query=f"query_{i}", response="r", rating="like")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    records = _read_jsonl(store.log_file)
    assert len(records) == 50, f"预期 50 条记录，实际 {len(records)} 条——存在丢失写入"

    queries = {r["query"] for r in records}
    expected = {f"query_{i}" for i in range(50)}
    assert queries == expected, "记录内容不完整或有重复"


def test_record_feedback_returns_false_on_write_failure(tmp_path, monkeypatch):
    """写入过程中如果抛异常，应该被捕获并返回 False，而不是让异常往外抛"""
    store = FeedbackStore(log_dir=str(tmp_path))

    def broken_open(*args, **kwargs):
        raise IOError("模拟磁盘写入失败")

    monkeypatch.setattr("builtins.open", broken_open)

    ok = store.record_feedback(user_id="u", query="q", response="r", rating="like")
    assert ok is False