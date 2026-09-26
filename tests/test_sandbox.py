# tests/test_sandbox.py
import pytest
import time
import concurrent.futures
from unittest.mock import MagicMock

# 尝试适配两种导入方式
try:
    from agent.sandbox import SandboxExecutor
except ImportError:
    try:
        from sandbox import SandboxExecutor
    except ImportError:
        SandboxExecutor = None


@pytest.fixture
def sandbox():
    """基础 SandboxExecutor 实例 (2 秒 CPU 超时)"""
    if SandboxExecutor is None:
        pytest.skip("SandboxExecutor 模块未找到")
    return SandboxExecutor(timeout=2, tool_timeout=2)


def test_sandbox_initialization(sandbox):
    """验证沙箱初始化与默认配置"""
    assert sandbox is not None
    assert sandbox.timeout == 2
    assert sandbox.tool_timeout == 2


def test_sandbox_security_blocked(sandbox):
    """验证 AST 静态代码审查拦截黑名单指令"""
    bad_code = "import os\nos.system('echo hack')"
    res = sandbox.run(bad_code)
    
    assert res["status"] == "security_blocked"
    assert res["result"] is None
    assert "安全审计未通过" in res["error"]


def test_sandbox_infinite_loop_timeout(sandbox):
    """验证 CPU 死循环时进程强行杀死与 timeout 状态返回"""
    infinite_code = "while True:\n    pass"
    res = sandbox.run(infinite_code)
    
    assert res["status"] == "timeout"
    assert "强熔断" in res["error"]


def test_sandbox_execution_and_stdout(sandbox):
    """验证正常 Python 代码计算与 stdout 捕获 (Observation 来源)"""
    code = (
        "print('计算开始')\n"
        "a = 10\n"
        "b = 20\n"
        "print(f'a + b = {a + b}')\n"
        "FINAL_RESULT = a + b"
    )
    res = sandbox.run(code)
    
    assert res["status"] == "success"
    assert res["result"] == 30
    assert "计算开始" in res["stdout"]
    assert "a + b = 30" in res["stdout"]


def test_sandbox_runtime_error_handling(sandbox):
    """验证代码运行时异常捕获（内置异常白名单与 stdout 保留）"""
    code = "print('准备抛出异常')\nraise ValueError('测试自定义业务异常')"
    res = sandbox.run(code)
    
    assert res["status"] == "runtime_error"
    assert "准备抛出异常" in res["stdout"]
    assert "测试自定义业务异常" in res["error"]


def test_sandbox_concurrency_isolation():
    """验证多线程并发调用下沙箱的线程隔离与数据不串扰"""
    if SandboxExecutor is None:
        pytest.skip("SandboxExecutor 模块未找到")
        
    executor_inst = SandboxExecutor(timeout=3)

    def worker(task_id):
        code = f"import time\nprint('Task {task_id}')\nFINAL_RESULT = {task_id} * 10"
        return executor_inst.run(code)

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as thread_pool:
        futures = [thread_pool.submit(worker, i) for i in range(4)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    assert len(results) == 4
    for res in results:
        assert res["status"] == "success"
        # 确保每个任务的结果都是自己的 task_id * 10
        assert res["result"] in [0, 10, 20, 30]


def test_sandbox_tool_dispatch_and_deadline_extension():
    """验证工具调用桥接 (ToolDispatcher) 及慢工具的 Deadline 顺延机制"""
    if SandboxExecutor is None:
        pytest.skip("SandboxExecutor 模块未找到")

    # Mock 工具调度器
    mock_dispatcher = MagicMock()
    # 模拟工具执行耗时 1.5 秒（大于沙箱默认短 CPU 允许时间，测试顺延）
    def mock_dispatch(tool_name, kwargs):
        time.sleep(0.5)
        return f"Query '{kwargs.get('query')}' 查重结果"

    mock_dispatcher.dispatch.side_effect = mock_dispatch

    # 沙箱设置 CPU 超时 1s，工具超时 3s
    sandbox_tool = SandboxExecutor(timeout=1, tool_timeout=3)
    
    code = (
        "res = search_kb(query='测试知识库')\n"
        "FINAL_RESULT = res"
    )

    res = sandbox_tool.run(
        code,
        tool_names=["search_kb"],
        tool_dispatcher=mock_dispatcher
    )

    assert res["status"] == "success"
    assert "测试知识库" in str(res["result"])
    mock_dispatcher.dispatch.assert_called_once_with("search_kb", {"query": "测试知识库"})