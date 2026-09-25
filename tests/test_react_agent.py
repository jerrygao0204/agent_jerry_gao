# tests/test_react_agent.py
import pytest
from agent.react_agent import ReActAgent

class DummyLLMClient:
    """簡化版 LLMClient，用於測試覆蓋率"""
    def __init__(self, mode="DIRECT_CHAT"):
        self.mode = mode

    def stream_generate(self, query=None, context=None, messages=None):
        # 模擬不同模式的返回
        if query and "意圖分類" in query:
            if self.mode == "DIRECT_CHAT":
                yield "DIRECT_CHAT"
            else:
                yield "NEED_TOOLS"
        elif messages:
            if self.mode == "DIRECT_CHAT":
                yield "你好，我是測試助手"
            else:
                yield "這是一個工具路由測試"
        else:
            yield "NEED_TOOLS"

@pytest.fixture
def agent_direct():
    return ReActAgent(llm_client=DummyLLMClient(mode="DIRECT_CHAT"))

@pytest.fixture
def agent_tools():
    return ReActAgent(llm_client=DummyLLMClient(mode="NEED_TOOLS"))

def test_short_query_interception(agent_direct):
    query = "你好"
    steps = list(agent_direct.run_stream(query))
    step_types = [s["type"] for s in steps]
    assert "thought" in step_types
    assert step_types[-1] == "final_answer"
    assert "你好" in steps[-1]["content"]

def test_need_tools_path(agent_tools):
    query = "請幫我查詢數據"
    steps = list(agent_tools.run_stream(query))
    step_types = [s["type"] for s in steps]
    assert step_types.count("thought") >= 1
    assert step_types[-1] in ["thought", "final_answer"]

def test_is_short_query_true(agent_direct):
    assert agent_direct._is_short_query("你好") is True

def test_is_short_query_false(agent_tools):
    assert agent_tools._is_short_query("請幫我查詢數據") is False

def test_prepare_tool_kwargs(agent_tools):
    kwargs = agent_tools._prepare_tool_kwargs("search_knowledge_base", {"query": "test"})
    assert "top_k" in kwargs
    assert "top_k_rerank" in kwargs
    assert "filter" in kwargs

def test_postprocess_tool_result_low_score(agent_tools):
    result = [{"rerank_score": 0.2}]
    processed = agent_tools._postprocess_tool_result("search_knowledge_base", result)
    assert "无匹配结果" in processed or "無匹配結果" in processed

def test_postprocess_tool_result_high_score(agent_tools):
    result = [{"rerank_score": 0.9}]
    processed = agent_tools._postprocess_tool_result("search_knowledge_base", result)
    assert processed == result

def test_extract_reflection_and_code(agent_tools):
    """覆蓋 _extract_reflection_and_code"""
    response_text = "<reflection>檢查代碼錯誤</reflection>\n```python\nprint('hello')\n```"
    reflection, code = agent_tools._extract_reflection_and_code(response_text)
    assert reflection == "檢查代碼錯誤"
    assert "print('hello')" in code

def test_build_observation_success(agent_tools):
    """覆蓋 _build_observation 成功情況"""
    sandbox_res = {"status": "success", "stdout": "ok", "result": "42"}
    obs = agent_tools._build_observation(sandbox_res)
    assert "[stdout]" in obs
    assert "FINAL_RESULT" in obs

def test_build_observation_failure(agent_tools):
    """覆蓋 _build_observation 失敗情況"""
    sandbox_res = {"status": "error", "error": "boom"}
    obs = agent_tools._build_observation(sandbox_res)
    assert "[执行异常]" in obs
