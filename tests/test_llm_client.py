# tests/test_llm_client.py
import pytest
from generator.llm_client import LLMClient


def test_llm_client_init():
    """验证 LLMClient 默认模型初始化"""
    client = LLMClient()
    assert client.default_model_name == "qwen3-4b"

    custom_client = LLMClient(default_model_name="qwen3-32b")
    assert custom_client.default_model_name == "qwen3-32b"


def test_stream_generate_with_query_only(mocker):
    """验证仅传入 query 时的流式生成逻辑"""
    client = LLMClient()
    
    # Mock 工厂层的 stream_chat Generator
    mock_stream_chat = mocker.patch.object(
        client.factory,
        "stream_chat",
        return_value=iter(["RAG", " 架构", " 核心"])
    )

    chunks = list(client.stream_generate(query="简述 RAG", model_name="qwen3-4b"))
    assert "".join(chunks) == "RAG 架构 核心"

    # 验证传入 ModelFactory 的消息结构
    mock_stream_chat.assert_called_once_with(
        messages=[{"role": "user", "content": "简述 RAG"}],
        model="qwen3-4b",
        temperature=0.7,
        max_tokens=1024
    )


def test_stream_generate_with_context(mocker):
    """验证包含 RAG 背景知识 (context) 时的 Prompt 拼接逻辑"""
    client = LLMClient()
    mock_stream_chat = mocker.patch.object(
        client.factory,
        "stream_chat",
        return_value=iter(["回答"])
    )

    list(client.stream_generate(
        query="如何优化检索？",
        context="上下文内容123",
        model_name="qwen3-vl-4b"
    ))

    # 获取底层调用的 messages 参数
    called_messages = mock_stream_chat.call_args.kwargs["messages"]
    user_content = called_messages[0]["content"]

    assert "【背景知识】\n上下文内容123" in user_content
    assert "【用户问题】\n如何优化检索？" in user_content


def test_stream_generate_with_messages_override(mocker):
    """验证显式传入 messages 时直接透传，覆盖 query/context"""
    client = LLMClient()
    mock_stream_chat = mocker.patch.object(
        client.factory,
        "stream_chat",
        return_value=iter(["OK"])
    )

    custom_messages = [{"role": "system", "content": "You are a helpful assistant."}]
    list(client.stream_generate(messages=custom_messages))

    mock_stream_chat.assert_called_once_with(
        messages=custom_messages,
        model="qwen3-4b",
        temperature=0.7,
        max_tokens=1024
    )


def test_stream_generate_missing_query_and_messages_raises_error():
    """验证 query 与 messages 均未传入时触发 ValueError 异常"""
    client = LLMClient()
    with pytest.raises(ValueError) as exc_info:
        list(client.stream_generate())
    
    assert "stream_generate 需要传入 query 或 messages 其中之一" in str(exc_info.value)