# tests/test_tool_factory.py
import pytest
from agent_jerry_gao.factory import tool_factory

def test_tool_factory_invalid_tool():
    """測試未知工具類型"""
    with pytest.raises(Exception):
        tool_factory.create_tool("nonexistent_tool")

def test_tool_factory_register_and_get():
    """測試工具註冊與獲取"""
    def dummy_tool():
        return "ok"
    tool_factory.register_tool("dummy", dummy_tool)
    tool = tool_factory.get_tool("dummy")
    assert tool() == "ok"
