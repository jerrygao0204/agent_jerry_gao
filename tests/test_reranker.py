# tests/test_reranker.py
from agent_jerry_gao.search import reranker

def test_reranker_empty_documents():
    """測試空列表"""
    result = reranker.rerank("query", [])
    assert result == []

def test_reranker_sorting():
    """測試分數排序"""
    docs = [{"content": "a", "score": 0.1}, {"content": "b", "score": 0.9}]
    result = reranker.rerank("query", docs)
    assert result[0]["score"] >= result[1]["score"]
