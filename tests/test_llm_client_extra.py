# tests/test_llm_client_extra.py
import pytest
from agent_jerry_gao.generator import llm_client

def test_llm_client_invalid_model():
    """測試無效模型名稱"""
    client = llm_client.LLMClient()
    with pytest.raises(Exception):
        client.generate("hello", model_name="nonexistent-model")

def test_llm_client_stream_generate_error():
    """測試 stream_generate 異常"""
    client = llm_client.LLMClient()
    client._call_api = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("API error"))
    with pytest.raises(RuntimeError):
        list(client.stream_generate([{"role": "user", "content": "hi"}], model_name="qwen3-32b"))
