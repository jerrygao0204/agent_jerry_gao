# tests/test_sandbox_extra.py
import pytest
from agent_jerry_gao.agent import sandbox

def test_sandbox_timeout():
    """測試超時情況"""
    code = "import time\ntime.sleep(2)"
    result = sandbox.run_code(code, timeout=1)
    assert "timeout" in result.lower() or "超時" in result.lower()

def test_sandbox_infinite_loop():
    """測試死循環"""
    code = "while True: pass"
    result = sandbox.run_code(code, timeout=1)
    assert "timeout" in result.lower() or "超時" in result.lower()

def test_sandbox_large_stdout():
    """測試 stdout 過大"""
    code = "print('x'*100000)"
    result = sandbox.run_code(code, timeout=2)
    assert "output too large" in result.lower() or "超出" in result.lower()
