# tests/test_rag_tool.py
import pytest
from agent_jerry_gao.factory.tools import rag_tool

def test_rag_tool_empty_query():
    """測試空查詢返回空結果"""
    result = rag_tool.search("")
    assert result == []

def test_rag_tool_invalid_query():
    """測試無效查詢"""
    with pytest.raises(Exception):
        rag_tool.search(None)
