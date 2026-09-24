# tests/test_session_registry.py
"""
SessionMemoryRegistry 并发安全测试（不依赖 torch / gradio，可在任何环境跑）。

覆盖：
  1. 同一个 (用户, 会话) 并发首次访问只创建一个实例（旧实现是"先检查后赋值"，会创建多个）
  2. 同一用户的不同会话拿到不同实例；一个会话的切换不会改动另一个会话实例的 session_id
  3. 不同用户之间不会拿到对方的实例
  4. 无 session_id 的旧式调用返回稳定实例，不会反复创建
  5. LRU 淘汰、evict
  6. 真实 MemoryManager：两个"标签页"交错写入 / 事务回滚，不会写错会话、不会误删对方消息
  7. reload_session 会从磁盘原地刷新短期记忆（switch_session(同id) 不会）
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from memory.session_registry import SessionMemoryRegistry


class _FakeMgr:
    created = 0
    _lock = threading.Lock()

    def __init__(self, user_id, session_id):
        time.sleep(0.02)  # 放大"先检查后创建"的竞态窗口
        with _FakeMgr._lock:
            _FakeMgr.created += 1
        self.user_id = user_id
        self.session_id = session_id


@pytest.fixture(autouse=True)
def _reset_counter():
    _FakeMgr.created = 0
    yield


def _registry(**kw):
    return SessionMemoryRegistry(factory=lambda user_id, session_id: _FakeMgr(user_id, session_id), **kw)


def test_concurrent_first_access_creates_exactly_one_instance():
    reg = _registry()
    with ThreadPoolExecutor(max_workers=32) as pool:
        mgrs = list(pool.map(lambda _: reg.get("alice", "s1"), range(32)))
    assert _FakeMgr.created == 1
    assert all(m is mgrs[0] for m in mgrs)


def test_sessions_of_same_user_are_isolated_instances():
    reg = _registry()
    a1 = reg.get("alice", "s1")
    a2 = reg.get("alice", "s2")
    assert a1 is not a2
    assert (a1.session_id, a2.session_id) == ("s1", "s2")
    # 再次取 s1，不会因为"另一个标签页用了 s2"而改动 s1 实例
    assert reg.get("alice", "s1") is a1 and a1.session_id == "s1"


def test_users_never_share_instances_under_concurrency():
    reg = _registry()
    users = [f"user-{i}" for i in range(20)]

    def one(u):
        m = reg.get(u, "same-session-id")
        return u, m

    with ThreadPoolExecutor(max_workers=20) as pool:
        for _ in range(5):
            for u, m in pool.map(one, users):
                assert m.user_id == u, f"{u} 拿到了 {m.user_id} 的实例"
    assert _FakeMgr.created == 20


def test_sessionless_call_is_stable_and_follows_last_explicit_session():
    reg = _registry()
    a = reg.get("bob")
    assert reg.get("bob") is a and _FakeMgr.created == 1  # 稳定占位实例，不反复创建
    s = reg.get("bob", "s9")
    assert reg.get("bob") is s  # 之后无 session_id 的调用跟随最近使用的会话
    assert reg.current_session("bob") == "s9"


def test_lru_eviction_and_explicit_evict():
    reg = _registry(max_cached=3)
    for i in range(5):
        reg.get("u", f"s{i}")
    assert len(reg) == 3
    reg.evict("u", "s4")
    assert len(reg) == 2
    assert reg.current_session("u") is None
    # 被淘汰后再取会重建，而不是报错
    assert reg.get("u", "s0").session_id == "s0"


# --------------------------------------------------------------------------- #
# 真实 MemoryManager（存储重定向到临时目录，不碰项目 data/）
# --------------------------------------------------------------------------- #
@pytest.fixture
def real_registry(tmp_path, monkeypatch):
    from memory.memory_manager import MemoryManager
    # MemoryManager 内部用 `from chat_history_file import ...`（sys.path 方式导入），
    # 与 `memory.chat_history_file` 是两个不同的模块对象，必须 patch 它实际使用的那个。
    import chat_history_file as chf

    class _Cfg:
        @staticmethod
        def get_config_dict():
            return {"data_root": str(tmp_path)}

    monkeypatch.setattr(chf, "config_loader", _Cfg)

    # 保险：确认存储确实落在临时目录，绝不写进项目真实的 data/
    probe = MemoryManager(user_id="__probe__", session_id="__probe__", max_messages=1)
    assert probe.history_storage.base_dir == str(tmp_path), "测试存储未重定向到临时目录，拒绝继续"

    return SessionMemoryRegistry(
        factory=lambda user_id, session_id: MemoryManager(user_id=user_id, session_id=session_id, max_messages=20)
    )


def test_two_tabs_same_user_do_not_cross_write_or_rollback_each_other(real_registry):
    reg = real_registry
    tab_a = reg.get("alice", "sess-A")   # 标签 A：agent 正在推理
    tab_b = reg.get("alice", "sess-B")   # 标签 B：用户切到/新建了另一个会话

    tab_a.begin_transaction()
    tab_a.process_user_input("A 的问题")

    # A 推理期间，B 也在同一用户下发消息并提交
    tab_b.begin_transaction()
    tab_b.process_user_input("B 的问题")
    tab_b.process_assistant_output("B 的回答")
    tab_b.commit()

    # A 推理失败回滚：只应回滚 A 自己刚写入的那条
    tab_a.rollback()

    a_msgs = [m["content"] for m in tab_a.history_storage.get_session_messages("alice", "sess-A")]
    b_msgs = [m["content"] for m in tab_b.history_storage.get_session_messages("alice", "sess-B")]
    assert a_msgs == [], f"A 回滚后不应残留: {a_msgs}"
    assert b_msgs == ["B 的问题", "B 的回答"], f"B 的消息被 A 的回滚/写入影响: {b_msgs}"
    assert [m["content"] for m in tab_b.short_term.get_messages()] == ["B 的问题", "B 的回答"]


def test_concurrent_writes_from_many_tabs_land_in_their_own_session(real_registry):
    reg = real_registry
    n = 8

    def tab(i):
        m = reg.get("alice", f"s{i}")
        for k in range(5):
            m.process_user_input(f"s{i}-q{k}")
            m.process_assistant_output(f"s{i}-a{k}")
        return i

    with ThreadPoolExecutor(max_workers=n) as pool:
        list(pool.map(tab, range(n)))

    storage = reg.get("alice", "s0").history_storage
    for i in range(n):
        msgs = [m["content"] for m in storage.get_session_messages("alice", f"s{i}")]
        assert len(msgs) == 10 and all(c.startswith(f"s{i}-") for c in msgs), f"会话 s{i} 混入了别的会话消息: {msgs}"


def test_reload_session_refreshes_from_disk_but_switch_same_id_does_not(real_registry):
    reg = real_registry
    m = reg.get("alice", "s1")
    m.process_user_input("q1")
    m.process_assistant_output("a1")
    m.process_user_input("q2")

    # 模拟"编辑重跑"：磁盘上把 q2 软删除
    m.history_storage.soft_delete_messages_after(user_id="alice", session_id="s1", keep_active_count=2)

    m.switch_session("s1")  # 相同 id：防重校验直接返回，内存仍是旧的
    assert [x["content"] for x in m.short_term.get_messages()] == ["q1", "a1", "q2"]

    m.reload_session()
    assert [x["content"] for x in m.short_term.get_messages()] == ["q1", "a1"]
