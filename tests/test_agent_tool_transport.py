"""
测试目标：
针对 agent/tool_transport.py 中的真实类进行 100% 覆盖率测试：
1. ToolDispatcher (正常调度、权限拦截、run/execute 方法适配、前置/后置钩子处理)
2. ToolTransportClient (抽象类实例化与默认 NotImplementedError 覆盖)
3. build_tool_proxies & _ToolProxy (工具代理生成、Pickle 兼容性、__call__ 转发)
"""

import pickle
import pytest
from unittest.mock import MagicMock

from agent.tool_transport import (
    ToolDispatcher,
    ToolTransportClient,
    build_tool_proxies,
    _ToolProxy,
)


# ============================================================================
# 1. ToolDispatcher 服务端调度器单元测试
# ============================================================================

class TestToolDispatcher:
    """测试 ToolDispatcher 的动态路由、钩子预处理与异常分类"""

    def test_dispatch_run_method_success(self):
        """验证带有 run() 方法的工具调用成功"""
        mock_tool = MagicMock()
        mock_tool.run.return_value = "run_result"

        mock_factory = MagicMock()
        mock_factory.get_tool.return_value = mock_tool

        dispatcher = ToolDispatcher(tool_factory=mock_factory, user_role="admin")
        res = dispatcher.dispatch("search_tool", {"query": "python"})

        assert res == "run_result"
        mock_factory.get_tool.assert_called_once_with("search_tool", user_role="admin")
        mock_tool.run.assert_called_once_with(query="python")

    def test_dispatch_execute_method_success(self):
        """验证无 run() 但带有 execute() 方法的工具调用成功"""
        class ExecuteOnlyTool:
            def execute(self, val: int):
                return val * 2

        mock_factory = MagicMock()
        mock_factory.get_tool.return_value = ExecuteOnlyTool()

        dispatcher = ToolDispatcher(tool_factory=mock_factory)
        res = dispatcher.dispatch("math_tool", {"val": 21})
        assert res == 42

    def test_dispatch_tool_not_found_raises_value_error(self):
        """验证工具不存在或无权限时抛出 ValueError"""
        mock_factory = MagicMock()
        mock_factory.get_tool.return_value = None

        dispatcher = ToolDispatcher(tool_factory=mock_factory, user_role="guest")
        with pytest.raises(ValueError, match="未找到工具 .* 无权限访问"):
            dispatcher.dispatch("secret_tool", {})

    def test_dispatch_invalid_tool_interface_raises_type_error(self):
        """验证工具既没有 run() 也没有 execute() 时抛出 TypeError (使用 raw string 修复警告)"""
        class InvalidTool:
            pass

        mock_factory = MagicMock()
        mock_factory.get_tool.return_value = InvalidTool()

        dispatcher = ToolDispatcher(tool_factory=mock_factory)
        with pytest.raises(TypeError, match=r"既没有 run\(\) 也没有 execute\(\)"):
            dispatcher.dispatch("bad_tool", {})

    def test_dispatch_preprocessor_and_postprocessor_hooks(self):
        """验证 kwargs_preprocessor 与 result_postprocessor 钩子按顺序生效"""
        def preprocessor(tool_name: str, kwargs: dict) -> dict:
            kwargs["top_k"] = 10
            return kwargs

        def postprocessor(tool_name: str, result: any) -> any:
            return f"Processed: {result}"

        mock_tool = MagicMock()
        mock_tool.run.return_value = "raw_data"

        mock_factory = MagicMock()
        mock_factory.get_tool.return_value = mock_tool

        dispatcher = ToolDispatcher(
            tool_factory=mock_factory,
            kwargs_preprocessor=preprocessor,
            result_postprocessor=postprocessor,
        )

        res = dispatcher.dispatch("query_tool", {"input": "test"})
        assert res == "Processed: raw_data"
        mock_tool.run.assert_called_once_with(input="test", top_k=10)


# ============================================================================
# 2. ToolTransportClient 与 _ToolProxy 客户端代理测试
# ============================================================================

class TestToolProxyAndClient:
    """测试客户端工具代理的包装、调用转发与 Pickle 序列化"""

    class DummyTransport(ToolTransportClient):
        def __init__(self):
            self.called_with = None

        def call(self, tool_name: str, kwargs: dict):
            self.called_with = (tool_name, kwargs)
            return f"response_from_{tool_name}"

    def test_transport_client_abstract_class_raises_on_instantiation(self):
        """验证未实现抽象方法的子类实例化抛出 TypeError"""
        class UnimplementedTransport(ToolTransportClient):
            pass

        with pytest.raises(TypeError):
            UnimplementedTransport()

    def test_abstract_transport_client_call_raises_not_implemented(self):
        """直接触发 ToolTransportClient 基类 call() 默认实现，覆盖第 87 行"""
        class ConcreteTransport(ToolTransportClient):
            def call(self, tool_name: str, kwargs: dict):
                return super().call(tool_name, kwargs)

        client = ConcreteTransport()
        with pytest.raises(NotImplementedError):
            client.call("test_tool", {})

    def test_build_tool_proxies_and_execution(self):
        """验证 build_tool_proxies 批量构建代理函数并正确转发调用"""
        transport = self.DummyTransport()
        proxies = build_tool_proxies(["search", "calculator"], transport)

        assert "search" in proxies
        assert "calculator" in proxies
        assert isinstance(proxies["search"], _ToolProxy)

        res = proxies["search"](query="pytest", limit=5)
        assert res == "response_from_search"
        assert transport.called_with == ("search", {"query": "pytest", "limit": 5})

    def test_tool_proxy_pickle_compatibility(self):
        """验证 _ToolProxy 实例能够正常被 Pickle 序列化/反序列化 (兼容 spawn/fork)"""
        transport = self.DummyTransport()
        proxy = _ToolProxy("search_tool", transport)

        # 序列化再反序列化
        serialized = pickle.dumps(proxy)
        deserialized: _ToolProxy = pickle.loads(serialized)

        assert deserialized._tool_name == "search_tool"
        res = deserialized(key="val")
        assert res == "response_from_search_tool"