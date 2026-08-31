# tests/test_retriever.py
#
# 根据仓库真实结构调整：
# - 仓库里是 search/retriever.py 的 FineBIRetriever，而不是原稿假设的
#   rag.retriever.Retriever；返回的也不是带 .page_content 属性的
#   LangChain Document 对象，而是普通 dict（含 chunk_id / content /
#   base_content / up_content / down_content / hierarchy 等字段）。
# - FineBIRetriever.__init__ 会直接连接 Milvus，且 embedding 依赖真实的
#   Qwen3-Embedding 模型 + GPU（transformers/torch）。这些在普通 CI/本地
#   环境里既连不上也跑不动，所以这里用 unittest.mock 把 ModelFactory 和
#   MilvusClient 都替换掉，只对"纯逻辑"部分做单元测试：
#     1) generate_sparse_vector 的稀疏向量生成是否稳定、可复现
#     2) hybrid_search 对 Milvus 返回结果的解析、层级拼接、前后置
#        Chunk 上下文拓展逻辑是否正确
#   真正端到端连 Milvus + 真实模型的检索效果验证，建议放在单独标记的
#   集成测试（如 @pytest.mark.integration）里，按需在有 GPU/Milvus 的
#   环境手动跑，而不是放进默认的单元测试套件。

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def retriever_inst():
    """构造一个不需要真实 Milvus 连接、不加载真实 Embedding 模型的 FineBIRetriever"""
    with patch("search.retriever.ModelFactory") as MockModelFactory, \
         patch("search.retriever.MilvusClient") as MockMilvusClient:

        mock_client = MagicMock()
        MockMilvusClient.return_value = mock_client
        MockModelFactory.return_value = MagicMock()

        from search.retriever import FineBIRetriever

        r = FineBIRetriever(
            milvus_host="mock-host",
            milvus_port="19530",
            collection_name="finebi_knowledge_chunks_test",
        )
        yield r


def test_generate_sparse_vector_is_deterministic_and_nonempty():
    """稀疏向量生成对同一文本应结果一致，且不应为空"""
    from search.retriever import FineBIRetriever

    vec1 = FineBIRetriever.generate_sparse_vector("怎么创建预警用户")
    vec2 = FineBIRetriever.generate_sparse_vector("怎么创建预警用户")

    assert vec1 == vec2
    assert len(vec1) > 0
    assert all(isinstance(v, float) for v in vec1.values())


def test_hybrid_search_parses_hit_and_expands_context(retriever_inst, monkeypatch):
    """验证命中结果的层级拼接、以及前置/后置 Chunk 上下文拓展是否正确"""
    # 绕开真实的稠密向量模型推理
    monkeypatch.setattr(retriever_inst, "get_dense_embedding", lambda text: [0.1, 0.2, 0.3])

    mock_hit = {
        "entity": {
            "chunk_id": "c2",
            "content": "重置密码：进入个人中心 -> 忘记密码 -> 按提示重置密码。",
            "section_id": "sec_01",
            "file_name": "finebi_faq.md",
            "full_hierarchy_array": {"data": ["FAQ", "账号相关", "重置密码"]},
            "biz_summary": "密码重置指引",
            "next_chunk_id": "c3",
            "prev_chunk_id": "c1",
        },
        "distance": 0.87,
    }
    retriever_inst.client.hybrid_search.return_value = [[mock_hit]]
    retriever_inst.client.query.return_value = [
        {"chunk_id": "c1", "content": "个人中心入口说明"},
        {"chunk_id": "c3", "content": "重置成功后的提示文案"},
    ]

    results = retriever_inst.hybrid_search(query="如何重置密码", top_k=3, expand_context=True)

    assert len(results) == 1
    top = results[0]
    assert top["chunk_id"] == "c2"
    assert "重置密码" in top["content"]
    assert top["hierarchy"] == "FAQ > 账号相关 > 重置密码"
    assert top["up_content"] == "个人中心入口说明"
    assert top["down_content"] == "重置成功后的提示文案"

    retriever_inst.client.load_collection.assert_called_once_with(
        collection_name="finebi_knowledge_chunks_test"
    )


def test_hybrid_search_returns_empty_list_when_no_hits(retriever_inst, monkeypatch):
    """Milvus 无召回结果时应返回空列表，而不是抛异常"""
    monkeypatch.setattr(retriever_inst, "get_dense_embedding", lambda text: [0.1])

    retriever_inst.client.hybrid_search.return_value = []

    results = retriever_inst.hybrid_search(query="这是一个不存在的问题")

    assert results == []
