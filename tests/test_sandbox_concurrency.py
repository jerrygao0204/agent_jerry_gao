# tests/test_sandbox_concurrency.py
"""
沙箱多用户并发安全测试。

覆盖的风险点：
  1. 不同用户同时执行代码时，result / stdout 不能串线（各拿各的）
  2. 工具调用桥：不同用户的 ToolDispatcher(user_role) 不能被别人的调用"配错"
  3. 一个用户的死循环 / 卡死工具 / 疯狂调用工具，不能拖累或误杀其他用户
  4. 并发上限生效，超限排队超时返回 busy，而不是无限 fork
  5. 异常路径（结果不可序列化、stdout 爆量）不会让主进程卡住或撑爆
  6. 跑完后没有遗留的僵尸/孤儿子进程
  7. ASTCodeChecker 被多线程共用时违规记录不串线
"""
import multiprocessing
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import agent.sandbox as sandbox_mod
from agent.sandbox import SandboxExecutor
from agent.security import ASTCodeChecker
from agent.tool_transport import ToolDispatcher
import tests.test_sandbox_concurrency as self_mod


# --------------------------------------------------------------------------- #
# 测试替身：与真实 HierarchicalToolFactory 一样，工具是全局共享的单例，
# 用户身份只通过 ToolDispatcher(user_role=...) 传入。
# --------------------------------------------------------------------------- #
class _EchoTool:
    def run(self, query: str = "", **kwargs):
        time.sleep(random.uniform(0.0, 0.05))  # 制造交错
        return {"query": query}


class _EchoFactory:
    def __init__(self):
        self.tool = _EchoTool()  # 共享单例

    def get_tool(self, name, user_role=None):
        # 把调用时的 user_role 一并回显，用于检验"没有配错人"
        role = user_role

        class _Bound:
            def run(_self, **kw):
                out = self.tool.run(**kw)
                out["role"] = role
                return out

        return _Bound()


def _settle_children(max_wait: float = 3.0):
    """等待子进程被回收（active_children 会顺带 join 已结束的进程）"""
    deadline = time.time() + max_wait
    while time.time() < deadline:
        if not multiprocessing.active_children():
            return []
        time.sleep(0.05)
    return multiprocessing.active_children()


# --------------------------------------------------------------------------- #
# Top-level helper definition for spawn multiprocessing (pickle safety)
# --------------------------------------------------------------------------- #
_FLAKY_MANAGER_FLAG = None
_REAL_TARGET = sandbox_mod._isolated_execution_target


class _NoopConn:
    def close(self):
        pass


def _top_level_flaky_target(*args, **kwargs):
    """顶层 _flaky 目标，支持跨进程 Value 共享状态"""
    global _FLAKY_MANAGER_FLAG, _REAL_TARGET
    if _FLAKY_MANAGER_FLAG is not None:
        with _FLAKY_MANAGER_FLAG.get_lock():
            if not _FLAKY_MANAGER_FLAG.value:
                _FLAKY_MANAGER_FLAG.value = True  # 第一次尝试：标记已执行并卡死
                time.sleep(30)
                return
    return _REAL_TARGET(*args, **kwargs)


def _top_level_slow_start_target(code_str, global_vars, conn, parent_conn, *rest):
    """顶层 _slow_start 目标，模拟慢速启动"""
    global _REAL_TARGET
    try:
        parent_conn.close()
    except Exception:
        pass
    time.sleep(1.2)
    return _REAL_TARGET(code_str, global_vars, conn, _NoopConn(), *rest)


def test_concurrent_pure_code_results_do_not_cross():
    sb = SandboxExecutor(timeout=5)
    n = 24

    def one(i):
        code = (
            f"marker = 'user-{i}'\n"
            f"print(marker)\n"
            f"FINAL_RESULT = (marker, {i} * {i})\n"
        )
        return i, sb.run(code)

    with ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(one, range(n)))

    for i, res in results:
        assert res["status"] == "success", res
        assert res["result"] == (f"user-{i}", i * i), f"用户 {i} 拿到了别人的结果: {res['result']}"
        assert res["stdout"].strip() == f"user-{i}", f"用户 {i} 的 stdout 串线: {res['stdout']!r}"
    assert _settle_children() == []


def test_concurrent_tool_calls_route_to_the_right_user():
    factory = _EchoFactory()
    n = 16

    def one(i):
        role = f"role-{i}"
        dispatcher = ToolDispatcher(tool_factory=factory, user_role=role)
        sb = SandboxExecutor(timeout=5, tool_timeout=5)
        code = (
            "out = []\n"
            "for k in range(3):\n"
            f"    out.append(echo(query='q-{i}-' + str(k)))\n"
            "print(out)\n"
            "FINAL_RESULT = out\n"
        )
        return i, sb.run(code, tool_names=["echo"], tool_dispatcher=dispatcher)

    with ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(one, range(n)))

    for i, res in results:
        assert res["status"] == "success", res
        assert len(res["result"]) == 3
        for k, item in enumerate(res["result"]):
            assert item["role"] == f"role-{i}", f"用户 {i} 的工具调用被配给了 {item['role']}"
            assert item["query"] == f"q-{i}-{k}", f"用户 {i} 收到了别人的响应: {item}"
    assert _settle_children() == []


def test_infinite_loop_only_kills_its_own_sandbox():
    sb = SandboxExecutor(timeout=2)

    def bad():
        return sb.run("while True:\n    pass")

    def good(i):
        return sb.run(f"FINAL_RESULT = {i} + 1")

    with ThreadPoolExecutor(max_workers=8) as pool:
        bad_f = pool.submit(bad)
        good_fs = [pool.submit(good, i) for i in range(6)]
        bad_res = bad_f.result()
        good_res = [f.result() for f in good_fs]

    assert bad_res["status"] == "timeout"
    for i, r in enumerate(good_res):
        assert r["status"] == "success" and r["result"] == i + 1, r
    assert _settle_children() == []


def test_stuck_tool_does_not_block_or_delay_kill_and_others_unaffected():
    class _Blocked:
        def run(self, **kw):
            time.sleep(6)
            return "不应返回"

    class _BlockedFactory:
        def get_tool(self, name, user_role=None):
            return _Blocked()

    stuck_sb = SandboxExecutor(timeout=2, tool_timeout=1)
    stuck_disp = ToolDispatcher(tool_factory=_BlockedFactory())
    fast_sb = SandboxExecutor(timeout=5)

    def stuck():
        t0 = time.time()
        res = stuck_sb.run("hang()", tool_names=["hang"], tool_dispatcher=stuck_disp)
        return res, time.time() - t0

    def fast(i):
        t0 = time.time()
        res = fast_sb.run(f"FINAL_RESULT = {i}")
        return res, time.time() - t0

    with ThreadPoolExecutor(max_workers=6) as pool:
        sf = pool.submit(stuck)
        ff = [pool.submit(fast, i) for i in range(4)]
        (stuck_res, stuck_elapsed) = sf.result()
        fast_out = [f.result() for f in ff]

    assert stuck_res["status"] == "runtime_error" and "执行超时" in stuck_res["error"], stuck_res
    assert stuck_elapsed < 4.0, f"卡死的工具拖住了 run() {stuck_elapsed:.1f}s"
    for i, (r, _) in enumerate(fast_out):
        assert r["status"] == "success" and r["result"] == i, r


def test_tool_call_loop_is_bounded():
    factory = _EchoFactory()
    disp = ToolDispatcher(tool_factory=factory, user_role="r")
    sb = SandboxExecutor(timeout=2, tool_timeout=2, max_tool_calls=5)
    t0 = time.time()
    res = sb.run("while True:\n    echo(query='x')", tool_names=["echo"], tool_dispatcher=disp)
    assert res["status"] == "runtime_error" and "上限" in res["error"], res
    assert time.time() - t0 < 10


def test_tool_not_in_whitelist_is_rejected():
    factory = _EchoFactory()
    disp = ToolDispatcher(tool_factory=factory, user_role="r")
    sb = SandboxExecutor(timeout=3, tool_timeout=2)
    res = sb.run("other(query='x')", tool_names=["echo"], tool_dispatcher=disp)
    assert res["status"] == "runtime_error"


def test_concurrency_cap_returns_busy_instead_of_forking_unbounded(monkeypatch):
    monkeypatch.setattr(sandbox_mod, "_SLOTS", threading.BoundedSemaphore(1))
    sb_long = SandboxExecutor(timeout=5)
    sb_wait = SandboxExecutor(timeout=5, queue_timeout=0.2)

    started = threading.Event()

    def long_run():
        started.set()
        return sb_long.run("import time\ntime.sleep(1.2)\nFINAL_RESULT = 'done'")

    with ThreadPoolExecutor(max_workers=2) as pool:
        f = pool.submit(long_run)
        started.wait(2)
        time.sleep(0.3)
        busy = sb_wait.run("FINAL_RESULT = 1")
        assert f.result()["result"] == "done"

    assert busy["status"] == "busy", busy
    assert SandboxExecutor(timeout=5).run("FINAL_RESULT = 2")["result"] == 2


def test_unserializable_result_does_not_hang():
    sb = SandboxExecutor(timeout=5)
    res = sb.run("FINAL_RESULT = (lambda: 1)")
    assert res["status"] == "success"
    assert isinstance(res["result"], str)


def test_huge_stdout_is_capped():
    sb = SandboxExecutor(timeout=10, max_output_chars=1000)
    res = sb.run("for i in range(200000):\n    print('x' * 50)\nFINAL_RESULT = 1")
    assert res["status"] == "success"
    assert len(res["stdout"]) < 1200 and "已截断" in res["stdout"]


def test_ast_checker_is_thread_safe_when_shared():
    checker = ASTCodeChecker()
    bad = "import os\nos.system('x')"
    good = "import math\nprint(math.sqrt(4))"
    errors = []

    def worker(i):
        for _ in range(200):
            if i % 2 == 0:
                ok, violations, _ = checker.check_code(bad)
                if ok or not violations:
                    errors.append(("bad 被放行", i))
            else:
                ok, violations, _ = checker.check_code(good)
                if not ok or violations:
                    errors.append(("good 被误拦", i, violations))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors, errors[:3]


# --------------------------------------------------------------------------- #
# 启动握手：多线程进程里 fork 可能让子进程卡在启动阶段
# --------------------------------------------------------------------------- #
def test_child_hung_at_startup_is_reported_as_startup_failure_not_code_timeout(monkeypatch):
    def _hang_before_handshake(*args, **kwargs):
        time.sleep(30)

    monkeypatch.setattr(sandbox_mod, "_isolated_execution_target", _hang_before_handshake)
    sb = SandboxExecutor(timeout=5, startup_timeout=0.4)
    t0 = time.time()
    res = sb.run("FINAL_RESULT = 1")
    assert res["status"] == "runtime_error" and "启动失败" in res["error"], res
    assert time.time() - t0 < 4, "启动失败应在 2*startup_timeout 左右返回，而不是等满代码超时"
    assert _settle_children() == []


def test_one_off_startup_hang_is_retried_and_succeeds(monkeypatch):
    # 使用轻量级 multiprocessing.Value ('b' 即 bool 类型)，不依赖额外的 SyncManager 服务进程
    shared_flag = multiprocessing.Value('b', False)

    self_mod._FLAKY_MANAGER_FLAG = shared_flag
    monkeypatch.setattr(sandbox_mod, "_isolated_execution_target", _top_level_flaky_target)

    try:
        sb = SandboxExecutor(timeout=5, startup_timeout=0.4)
        res = sb.run("FINAL_RESULT = 41 + 1")

        assert res["status"] == "success" and res["result"] == 42, res
    finally:
        self_mod._FLAKY_MANAGER_FLAG = None

    assert _settle_children() == []


def test_slow_startup_does_not_consume_the_code_time_budget(monkeypatch):
    monkeypatch.setattr(sandbox_mod, "_isolated_execution_target", _top_level_slow_start_target)

    sb = SandboxExecutor(timeout=1, startup_timeout=5)  # 代码时限 1s < 启动耗时 1.2s
    res = sb.run("FINAL_RESULT = 'ok'")

    assert res["status"] == "success" and res["result"] == "ok", res
    assert _settle_children() == []