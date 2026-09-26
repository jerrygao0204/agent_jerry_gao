# tests/test_rag_tool.py
import pytest
from unittest.mock import MagicMock
from factory.tools.rag_tool import RAGKnowledgeSearchTool, RAGSearchInput


def test_rag_tool_metadata():
    """验证 RAGKnowledgeSearchTool 的元数据配置与 Pydantic Schema"""
    tool = RAGKnowledgeSearchTool()
    assert tool.name == "search_knowledge_base"
    assert tool.domain == "rag_knowledge"
    assert tool.is_read_only is True
    assert tool.args_schema == RAGSearchInput

    # 验证输入 Schema 验证
    input_data = RAGSearchInput(query="报错排查", top_k=5)
    assert input_data.query == "报错排查"
    assert input_data.top_k == 5


def test_rag_tool_run_without_retriever():
    """验证未注入 retriever 时的模拟兜底逻辑"""
    tool = RAGKnowledgeSearchTool(retriever=None)
    result = tool.run(query="数据库连接失败")
    assert "[模拟 RAG 结果]" in result
    assert "数据库连接失败" in result


def test_rag_tool_full_pipeline_return_raw():
    """验证完整链路：hybrid_search -> rerank -> 格式化结构化列表 (return_raw=True)"""
    # 1. Mock Retriever
    mock_retriever = MagicMock()
    mock_raw_chunks = [
        {"chunk_id": "c1", "content": "MySQL 连接驱动设置", "score": 0.8},
        {"chunk_id": "c2", "content": "Oracle 数据源配置", "score": 0.6}
    ]
    mock_retriever.hybrid_search.return_value = mock_raw_chunks

    # 2. Mock Reranker
    mock_reranker = MagicMock()
    mock_reranker.rerank.return_value = [
        {"chunk_id": "c1", "content": "MySQL 连接驱动设置", "rerank_score": 0.95, "source_file": "db_guide.pdf"}
    ]

    tool = RAGKnowledgeSearchTool(retriever=mock_retriever, reranker=mock_reranker, top_k_retrieval=10)
    
    results = tool.run(query="MySQL 怎么配置？", top_k=1, return_raw=True)

    # 断言调用参数与结果
    mock_retriever.hybrid_search.assert_called_once_with(query="MySQL 怎么配置？", top_k=10)
    mock_reranker.rerank.assert_called_once_with(query="MySQL 怎么配置？", documents=mock_raw_chunks, top_n=1)
    
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]["rerank_score"] == 0.95
    assert results[0]["chunk_id"] == "c1"


def test_rag_tool_full_pipeline_return_formatted_text():
    """验证 Prompt 文本格式化输出 (return_raw=False)"""
    # 【修正点】使用 spec 限制 Mock 属性，或直接显式移除 hybrid_search 属性
    mock_retriever = MagicMock(spec=["retrieve"])
    mock_retriever.retrieve.return_value = [
        {"chunk_id": "c1", "content": "FineBI 安装手册", "score": 0.88, "doc_name": "install.md"}
    ]

    # 未传入 reranker，验证警告降级直接截取逻辑
    tool = RAGKnowledgeSearchTool(retriever=mock_retriever, reranker=None)
    
    formatted_text = tool.run(query="FineBI 安装", top_k=3, return_raw=False)

    assert isinstance(formatted_text, str)
    assert "[1] 文档: install.md (Score: 0.8800)" in formatted_text
    assert "内容: FineBI 安装手册" in formatted_text


def test_rag_tool_empty_retrieval():
    """验证初召回为空时的响应边界"""
    mock_retriever = MagicMock()
    mock_retriever.hybrid_search.return_value = []

    tool = RAGKnowledgeSearchTool(retriever=mock_retriever)

    # Raw 模式返回空列表
    raw_res = tool.run(query="不存在的内容", return_raw=True)
    assert raw_res == []

    # Text 模式返回未检索到提示
    text_res = tool.run(query="不存在的内容", return_raw=False)
    assert text_res == "未检索到相关文档。"