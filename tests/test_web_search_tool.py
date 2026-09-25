# tests/test_web_search_tool.py
import pytest
from agent_jerry_gao.factory.tools import web_search_tool

def test_web_search_tool_empty_query():
    """測試空查詢"""
    result = web_search_tool.search("")
    assert result == []

def test_web_search_tool_invalid_query():
    """測試無效查詢"""
    with pytest.raises(Exception):
        web_search_tool.search(None)
