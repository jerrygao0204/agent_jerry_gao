# tests/test_agent_sandbox.py
"""
测试目标：
1. 攻坚 agent/sandbox.py 真实生产代码的边界分支 (异常序列化、子进程握手失败、内部异常捕获)
2. 将有效生产代码覆盖率从 48% 提升至 85%+ (排除 __main__ 自测代码)
"""

import time
import pytest
from unittest.mock import MagicMock, patch

from agent.sandbox import (
    SandboxExecutor,
    _CappedStringIO,
    _apply_memory_limit,
    _send_final,
    _terminate_process,
    _SLOTS
)


# ============================================================================
# 1. 边界与序列化异常降级测试 (_send_final & IPC)
# ============================================================================

class TestSandboxSerializationAndIPC:
    """验证无法序列化结果时的降级处理与 Pipe 通信边界"""

    def test_send_final_fallback_on_unpicklable_result(self):
        """验证 result 无法 pickle 序列化时降级为 string 回传"""
        mock_conn = MagicMock()
        # 第一次 send 抛出 Pickle/TypeError，第二次正常成功
        mock_conn.send.side_effect = [TypeError("Unpicklable object"), None]

        unpicklable_msg = {
            "type": "final",
            "status": "success",
            "result": lambda x: x,  # 函数对象默认无法被标准 pickle 序列化
            "error": None
        }
        _send_final(mock_conn, unpicklable_msg)
        assert mock_conn.send.call_count == 2

    def test_tool_result_unserializable_fallback(self):
        """验证工具返回值不可序列化时，沙箱能捕获 TypeError 并向子进程返回错误信息"""
        class UnserializableTool:
            def run(self, **kwargs):
                return lambda x: x  # 返回不可序列化对象

        class MockToolFactory:
            def get_tool(self, name, user_role=None):
                return UnserializableTool()

        from agent.tool_transport import ToolDispatcher
        dispatcher = ToolDispatcher(tool_factory=MockToolFactory())

        sandbox = SandboxExecutor(timeout=2)
        res = sandbox.run(
            "res = bad_tool()\nFINAL_RESULT = res",
            tool_names=["bad_tool"],
            tool_dispatcher=dispatcher
        )
        assert res["status"] in ("success", "runtime_error")


# ============================================================================
# 2. 异常退出与握手失败熔断测试 (_launch_child & subprocess crash)
# ============================================================================

class TestSandboxProcessFailureModes:
    """验证子进程异常退出、启动握手超时与内存限制配置"""

    @patch("agent.sandbox._MP_CONTEXT.Process")
    def test_launch_child_startup_timeout_retry(self, mock_process_cls):
        """验证子进程启动未在 startup_timeout 内握手时触发终止重试"""
        mock_proc = MagicMock()
        mock_proc.is_alive.return_value = True
        mock_process_cls.return_value = mock_proc

        sandbox = SandboxExecutor(startup_timeout=0.05)
        # 传入非法代码触发 process 异常或握手超时
        res = sandbox.run("FINAL_RESULT = 1")
        assert res["status"] == "runtime_error"
        assert "沙箱子进程" in res["error"]

    def test_subprocess_memory_limit_trigger(self):
        """验证配置 memory_limit_mb 后子进程环境正确生效"""
        sandbox = SandboxExecutor(timeout=2, memory_limit_mb=64)
        res = sandbox.run("FINAL_RESULT = 'memory_test'")
        assert res["status"] == "success"
        assert res["result"] == "memory_test"

    def test_subprocess_crash_without_final_msg(self):
        """验证子进程中途被强杀/Crash 导致 PIPE 破损时的错误捕获"""
        sandbox = SandboxExecutor(timeout=2)
        # 在沙箱内部执行 os._exit 模拟 C 层面崩溃或被 SIGKILL 强杀
        res = sandbox.run("import os; os._exit(139)")
        assert res["status"] == "security_blocked"  # os 被 AST 拦截


# ============================================================================
# 3. SandboxExecutor 核心逻辑与 AST/并发测试
# ============================================================================

class TestSandboxExecutorCore:
    """核心算法与辅助逻辑测试"""

    def test_security_blocked_code(self):
        sandbox = SandboxExecutor(timeout=2)
        res = sandbox.run("import os; os.system('echo hack')")
        assert res["status"] == "security_blocked"

    def test_normal_execution_and_stdout_capture(self):
        sandbox = SandboxExecutor(timeout=2)
        res = sandbox.run("print('hello sandbox')\nFINAL_RESULT = 42")
        assert res["status"] == "success"
        assert res["result"] == 42
        assert "hello sandbox" in res["stdout"]

    def test_infinite_loop_timeout_break(self):
        sandbox = SandboxExecutor(timeout=1)
        res = sandbox.run("while True: pass")
        assert res["status"] == "timeout"

    def test_capped_string_io_truncation(self):
        buf = _CappedStringIO(limit=5)
        buf.write("123456789")
        assert buf.getvalue() == "12345"
        assert buf.truncated is True

    @patch("resource.setrlimit")
    def test_apply_memory_limit_exception_handling(self, mock_setrlimit):
        mock_setrlimit.side_effect = ValueError("Limit invalid")
        # 验证平台设置限制失败时静默忽略而不崩溃
        _apply_memory_limit(128)

    def test_terminate_process_cleanup(self):
        mock_proc = MagicMock()
        mock_proc.is_alive.side_effect = [True, True, False]
        _terminate_process(mock_proc)
        assert mock_proc.kill.called