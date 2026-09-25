# tests/test_retriever_extra.py
import pytest
from agent_jerry_gao.search import retriever

def test_retriever_hybrid_search_empty():
    """測試 hybrid_search 無結果"""
    r = retriever.Retriever()
    result = r.hybrid_search("nonexistent query")
    assert result == [] or result is None

def test_retriever_embedding_failure(monkeypatch):
    """測試 embedding 失敗"""
    r = retriever.Retriever()

    # 模擬 embedding 函數拋出異常
    def fake_embed(*args, **kwargs):
        raise RuntimeError("embedding error")

    monkeypatch.setattr(r, "embed_query", fake_embed)

    with pytest.raises(RuntimeError):
        r.hybrid_search("query")
