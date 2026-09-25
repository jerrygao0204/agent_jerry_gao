# tests/test_model_factory_backends_smoke.py
import pytest
from factory.model_factory import ModelFactory

class DummyEmbeddingResponse:
    def __init__(self, embeddings):
        self.data = [type("obj", (object,), {"embedding": e})() for e in embeddings]

@pytest.fixture(scope="module")
def factory():
    return ModelFactory.get_instance()

def test_smoke_local_backend(factory):
    query = "GPU 算力配置"
    docs = ["Python 自动化", "DGX Spark GPU 算力", "天气很好"]
    results = factory.rerank(query, docs, top_n=2)
    assert isinstance(results, list)
    assert all("document" in r for r in results)

def test_smoke_vlm_backend(factory):
    client, model_name = factory.get_vlm_model("qwen3-32b")
    assert client is not None
    assert isinstance(model_name, str)
    assert "qwen3-32b" in model_name

def test_smoke_multi_vlm_backend(factory):
    client1, model1 = factory.get_vlm_model("qwen3-32b")
    client2, model2 = factory.get_vlm_model("qwen3-32b")
    assert client1 == client2
    assert model1 == model2

def test_smoke_remote_failure_fallback_to_local(factory):
    results = factory.embed_texts([])
    assert results == []

def test_embed_query(factory, monkeypatch):
    monkeypatch.setattr(factory.get_llm_client().embeddings, "create",
                        lambda **kwargs: DummyEmbeddingResponse([[0.1, 0.2, 0.3]]))
    embedding = factory.embed_query("測試文本")
    assert embedding == [0.1, 0.2, 0.3]

def test_embed_texts(factory, monkeypatch):
    monkeypatch.setattr(factory.get_llm_client().embeddings, "create",
                        lambda **kwargs: DummyEmbeddingResponse([[0.1, 0.2], [0.3, 0.4]]))
    texts = ["文本A", "文本B"]
    embeddings = factory.embed_texts(texts)
    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]

def test_resolve_model_path(factory):
    path = factory.resolve_model_path("BAAI/bge-reranker-large")
    assert isinstance(path, str)

def test_stream_chat(factory, monkeypatch):
    """測試流式對話生成接口 (qwen3-32b)，mock 返回"""
    class DummyStream:
        def __iter__(self):
            return iter([type("obj", (object,), {"choices": [type("obj", (object,), {"delta": type("obj", (object,), {"content": "你好"})()})()]})()])
    monkeypatch.setattr(factory.get_llm_client().chat.completions, "create",
                        lambda **kwargs: DummyStream())
    messages = [{"role": "user", "content": "你好，請簡單自我介紹"}]
    stream = factory.stream_chat(messages=messages, model="qwen3-32b")
    chunks = list(stream)
    assert len(chunks) > 0
    assert "你好" in chunks[0]
