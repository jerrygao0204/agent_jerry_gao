# tests/test_tool_registry.py
import pytest
from agent_jerry_gao.factory import tool_registry

def test_tool_registry_lookup_invalid():
    """測試查找不存在的工具"""
    with pytest.raises(KeyError):
        tool_registry.get_tool("invalid_tool")

def test_tool_registry_register_and_lookup():
    """測試註冊後查找"""
    def dummy_tool():
        return "ok"
    tool_registry.register_tool("dummy", dummy_tool)
    tool = tool_registry.get_tool("dummy")
    assert tool() == "ok"
