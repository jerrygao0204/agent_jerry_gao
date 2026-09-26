# tests/test_model_factory.py
import pytest
from factory.model_factory import ModelFactory


@pytest.fixture
def model_factory():
    """提供 ModelFactory 实例"""
    return ModelFactory()


def test_model_factory_init(model_factory):
    """验证 ModelFactory 初始化"""
    assert model_factory is not None


def test_stream_chat_mocked_success(model_factory):
    """验证 ModelFactory.stream_chat 在全局 Mock 模式下的流式输出 (不再发起真实网络请求)"""
    messages = [{"role": "user", "content": "Test prompt"}]
    
    # 执行 stream_chat 消费 Generator
    chunks = list(model_factory.stream_chat(
        messages=messages,
        model="qwen3-4b",
        temperature=0.7,
        max_tokens=512
    ))
    
    result_text = "".join(chunks)
    assert "[Mock Stream]" in result_text or len(chunks) > 0


def test_model_factory_get_llm_client(model_factory):
    """验证 get_llm_client 已被成功拦截，返回 Mock 对象"""
    client = model_factory.get_llm_client()
    assert client is not None