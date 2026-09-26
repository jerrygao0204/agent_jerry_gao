# tests/conftest.py
import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock

RAG_ROOT_DIR = str(Path(__file__).resolve().parent.parent.parent)
if RAG_ROOT_DIR not in sys.path:
    sys.path.insert(0, RAG_ROOT_DIR)

@pytest.fixture(autouse=True)
def mock_env_vars(monkeypatch):
    """自动设置测试用环境变量，隔离真实配置"""
    monkeypatch.setenv("OPENAI_API_KEY", "mock-test-key-123456")
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost:8000/v1")
    monkeypatch.setenv("EMBEDDING_MODEL_NAME", "text-embedding-v3")
    monkeypatch.setenv("LLM_MODEL_NAME", "qwen3-32b")

@pytest.fixture(autouse=True)
def mock_embedding_client(monkeypatch):
    """全局 Mock Embedding 接口，拦截真实网络调用，防止 400 Bad Request"""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=[0.1] * 1024)]
    mock_client.embeddings.create.return_value = mock_response

    try:
        monkeypatch.setattr("agent_jerry_gao.rag.get_embedding_client", lambda: mock_client, raising=False)
        monkeypatch.setattr("agent_jerry_gao.utils.embedding.get_client", lambda: mock_client, raising=False)
    except Exception:
        pass
    
    return mock_client

@pytest.fixture(autouse=True)
def mock_llm_client(monkeypatch):
    """全局 Mock LLM Chat Completion 接口"""
    mock_client = MagicMock()
    
    # 构造标准 Response 模拟
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="[Mock Response] Agent Jerry Gao Test Suite"))
    ]
    
    # 构造流式 Chunk 模拟 (供 stream=True 时迭代)
    chunk1 = MagicMock()
    chunk1.choices = [MagicMock(delta=MagicMock(content="[Mock "))]
    chunk2 = MagicMock()
    chunk2.choices = [MagicMock(delta=MagicMock(content="Stream]"))]
    
    # 支持非流式与流式调用
    mock_client.chat.completions.create.side_effect = lambda *args, **kwargs: (
        iter([chunk1, chunk2]) if kwargs.get("stream") else mock_response
    )

    # 【关键修覆】注入到 factory.model_factory 以及全局引用路径
    try:
        monkeypatch.setattr("factory.model_factory.ModelFactory.get_llm_client", lambda self: mock_client, raising=False)
        monkeypatch.setattr("agent_jerry_gao.llm.get_llm_client", lambda: mock_client, raising=False)
        monkeypatch.setattr("agent_jerry_gao.utils.llm.get_client", lambda: mock_client, raising=False)
    except Exception:
        pass

    return mock_client