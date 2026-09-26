# tests/test_sandbox_deep_coverage.py
import pytest
import time
from agent.sandbox import SandboxExecutor

def test_sandbox_kill_zombie_subprocess_on_hard_timeout():
    """验证沙箱在面临死循环时，能彻底杀掉 IPC 子进程并释放资源 (覆盖 555-690 行)"""
    sandbox = SandboxExecutor(timeout=1)  # 1秒超时
    
    # 执行一个无限循环的代码
    code_zombie = """
import time
while True:
    time.sleep(0.1)
"""
    result = sandbox.run_code(code_zombie)
    
    assert result.is_timeout is True
    assert "Timeout" in result.output or result.exit_code != 0
    # 确保内部 process pool / worker 重新初始化且可用
    assert sandbox.check_health() is True

def test_sandbox_ast_blocked_dangerous_imports():
    """验证 AST 安全检查机制可规避绕过手法"""
    sandbox = SandboxExecutor()
    dangerous_code = "import os; os.system('ls')"
    result = sandbox.run_code(dangerous_code)
    assert result.is_blocked is True