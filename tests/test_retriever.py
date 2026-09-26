# tests/test_retriever.py
import pytest
import torch
from unittest.mock import MagicMock
from search.reranker import Reranker


@pytest.fixture
def mock_reranker(mocker):
    """构建注入 Mock Tokenizer 与 Mock Model 的 Reranker 实例"""
    # 1. 模拟 Tokenizer 输出
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value.to.return_value = {}

    # 2. 模拟 Model 的 Logits 输出
    mock_model = MagicMock()
    # 预设 Logits 输出 模拟得分 (例如 [3.0, 0.5, -2.0])
    mock_logits = torch.tensor([3.0, 0.5, -2.0])
    mock_outputs = MagicMock()
    mock_outputs.logits.view.return_value.float.return_value = mock_logits
    mock_model.return_value = mock_outputs

    # 3. 拦截 ModelFactory 的静态属性注入
    mocker.patch("factory.model_factory.ModelFactory._rerank_tokenizer", mock_tokenizer)
    mocker.patch("factory.model_factory.ModelFactory._rerank_model", mock_model)
    mocker.patch("factory.model_factory.ModelFactory._device", "cpu")

    reranker = Reranker(strict_mode=True)
    return reranker


def test_build_context_aware_text(mock_reranker):
    """验证上下文感知的文本拼接逻辑"""
    chunk_data = {
        "chunk_id": "c2",
        "hierarchy": "文档1 > 章节2",
        "biz_summary": "测试摘要",
        "content": "核心内容C2",
        "up_content": "c1",
        "down_content": "c3"
    }
    chunk_map = {
        "c1": "上文内容C1",
        "c2": "核心内容C2",
        "c3": "下文内容C3"
    }

    text = mock_reranker._build_context_aware_text(chunk_data, chunk_map, enable_surrounding_context=True)

    assert "[文档层级 (Hierarchy)]: 文档1 > 章节2" in text
    assert "[业务摘要 (Summary)]: 测试摘要" in text
    assert "[上文补充 (Up Content - c1)]:\n上文内容C1" in text
    assert "[核心内容 (Core Content - c2)]:\n核心内容C2" in text
    assert "[下文补充 (Down Content - c3)]:\n下文内容C3" in text


def test_rerank_filtering_and_sorting(mock_reranker):
    """验证 Reranker 的重排序、Sigmoid 转换与得分过滤逻辑"""
    documents = [
        {"chunk_id": "c1", "content": "FineBI 支持多种数据源"},
        {"chunk_id": "c2", "content": "帆软报表安装教程"},
        {"chunk_id": "c3", "content": "无关文档内容"}
    ]
    query = "FineBI 连接"

    # 执行 rerank
    results = mock_reranker.rerank(query=query, documents=documents, top_n=2)

    # Logits [3.0, 0.5, -2.0] -> Sigmoid 后 top1 明显最高
    assert len(results) <= 2
    assert results[0]["chunk_id"] == "c1"
    assert "rerank_score" in results[0]
    assert "rerank_prob" in results[0]


def test_rerank_empty_documents(mock_reranker):
    """验证空文档列表的边界处理"""
    results = mock_reranker.rerank(query="test", documents=[], top_n=5)
    assert results == []


def test_rerank_low_probability_blocking(mock_reranker, mocker):
    """验证当 Top-1 概率未达到 min_prob 门槛时的阻断熔断"""
    # 模拟非常低的分数 Logits [-5.0, -6.0]，Sigmoid 后概率接近 0
    mock_logits = torch.tensor([-5.0, -6.0])
    mock_outputs = MagicMock()
    mock_outputs.logits.view.return_value.float.return_value = mock_logits
    mocker.patch.object(mock_reranker, "model", return_value=mock_outputs)

    # 在实例属性上设置门槛值
    mock_reranker.min_prob = 0.25

    documents = [
        {"chunk_id": "c1", "content": "内容1"},
        {"chunk_id": "c2", "content": "内容2"}
    ]

    # 【修正点】移除 rerank 调用的非法关键字参数 min_prob
    results = mock_reranker.rerank(query="不相关问题", documents=documents)
    # 概率低未过门槛，触发阻断并返回 []
    assert results == []


def test_rerank_disable_filtering(mock_reranker):
    """验证 disable_filtering=True 时无视门槛强制截取 Top-N"""
    documents = [
        {"chunk_id": "c1", "content": "内容1"},
        {"chunk_id": "c2", "content": "内容2"},
        {"chunk_id": "c3", "content": "内容3"}
    ]

    results = mock_reranker.rerank(
        query="任意查询", 
        documents=documents, 
        top_n=2, 
        disable_filtering=True
    )

    assert len(results) == 2