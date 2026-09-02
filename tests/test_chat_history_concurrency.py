# tests/test_chat_history_concurrency.py
import threading

from memory.chat_history_file import ChatHistoryFileStorage


def test_concurrent_add_message_no_lost_writes(tmp_path):
    """同一 session 并发写 50 条消息，最终应该一条不丢、顺序不乱"""
    storage = ChatHistoryFileStorage(base_dir=str(tmp_path))
    storage.create_session("test_user", "sess_1")

    def worker(i):
        storage.add_message("test_user", "sess_1", "user", f"message_{i}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    messages = storage.get_session_messages("test_user", "sess_1")
    assert len(messages) == 50, f"预期 50 条消息，实际 {len(messages)} 条——存在丢失写入"

    contents = {m["content"] for m in messages}
    expected = {f"message_{i}" for i in range(50)}
    assert contents == expected, "消息内容不完整或有重复"