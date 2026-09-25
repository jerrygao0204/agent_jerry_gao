# tests/test_react_agent_extra.py
import pytest
from agent_jerry_gao.agent import react_agent

def test_react_agent_invalid_tool():
    """測試選擇不存在的工具"""
    agent = react_agent.ReactAgent()
    with pytest.raises(Exception):
        agent.use_tool("nonexistent_tool", "query")

def test_react_agent_decision_loop_empty():
    """測試 decision loop 空輸入"""
    agent = react_agent.ReactAgent()
    result = agent.run("")
    assert isinstance(result, str)
