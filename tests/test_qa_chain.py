# tests/test_qa_chain.py
import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch

# 📌 1. 将项目根目录动态添加到 Python 模块搜索路径中
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# 📌 2. 正式导入待测试的真实业务类
from generator.qa_chain import QAChain


# 📌 3. 测试 QAChain 初始化与超时逻辑
def test_qa_chain_initialization():
    """验证 QAChain 对象的实例创建与属性设置"""
    # 模拟底层的 retriever 与 reranker，避免测试时真的加载大模型与向量库
    with patch("qa_chain.get_retriever") as mock_retriever, \
         patch("qa_chain.get_reranker") as mock_reranker:
        
        chain = QAChain(llm_choice="Qwen3-VL", top_k_ret=5, top_k_rerank=3)
        
        assert chain is not None
        assert chain.top_k_ret == 5
        assert chain.top_k_rerank == 3

def test_qa_chain_stream_answer_timeout():
    """测试 QAChain 流式回答在超时情况下的拦截行为"""
    with patch("qa_chain.get_retriever"), patch("qa_chain.get_reranker"):
        chain = QAChain(llm_choice="Qwen3-VL", top_k_ret=5, top_k_rerank=3)
        
        # Mock 底层 LLM 生成器，模拟卡死/延迟场景
        mock_llm = MagicMock()
        chain.llm = mock_llm
        
        def mock_blocked_stream(*args, **kwargs):
            yield "Token 1"
            raise TimeoutError("LLM 推理超时！")

        mock_llm.stream = mock_blocked_stream

        # 执行测试并捕获超时异常
        with pytest.raises(TimeoutError) as exc_info:
            gen = chain.stream_answer(query="测试问题", history=[], filter_expr=None)
            list(gen) # 消费生成器触发异常
            
        assert "LLM 推理超时" in str(exc_info.value)