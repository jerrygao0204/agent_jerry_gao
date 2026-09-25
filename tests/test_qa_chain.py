# tests/test_qa_chain.py
import pytest
from agent_jerry_gao.generator.qa_chain import QAChain
from utils.timeout_ctx import TimeoutException
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2].parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# -------------------------------
# Dummy 模擬類
# -------------------------------
class DummyRetriever:
    """模擬檢索器"""
    def retrieve(self, query):
        return [{"text": "mocked result", "score": 0.9}]

    def hybrid_search(self, query, top_k=10, filter_expr=None):
        return [{"content": "mocked content", "score": 0.9}]

class DummyReranker:
    """模擬重排序器"""
    def rerank(self, query, documents, top_n=3):
        return documents[:top_n]

# -------------------------------
# Fixture
# -------------------------------
@pytest.fixture
def qa_chain():
    """返回一個帶有假 retriever/reranker 的 QAChain"""
    chain = QAChain(llm_model_name="qwen3-32b")
    if hasattr(chain, "retriever"):
        chain.retriever = DummyRetriever()
    if hasattr(chain, "reranker"):
        chain.reranker = DummyReranker()
    return chain

# -------------------------------
# 基本測試
# -------------------------------
def test_qa_chain_initialization(qa_chain):
    """測試 QAChain 初始化"""
    assert qa_chain is not None

def test_qa_chain_run_basic(qa_chain):
    """測試 QAChain 基本運行"""
    query = "測試問題"
    result = qa_chain.run(query)
    assert isinstance(result, str)
    assert "測試" in result or "mocked" in result

def test_qa_chain_empty_query(qa_chain):
    """測試空輸入情況"""
    result = qa_chain.run("")
    assert isinstance(result, str)

def test_qa_chain_timeout_handling(qa_chain):
    """測試超時處理（模擬 retriever 無響應）"""
    def fake_hybrid_search(*args, **kwargs):
        raise TimeoutException("模擬超時")
    qa_chain.retriever.hybrid_search = fake_hybrid_search
    qa_chain.reranker.rerank = lambda query, documents, top_n=3: []
    responses = list(qa_chain.stream_answer("超時測試"))
    assert any(r["type"] == "sources" and r["data"] == [] for r in responses)
    assert any("超时" in r["data"] for r in responses)

# -------------------------------
# 擴展測試
# -------------------------------
def test_format_prompt_user_assistant_dict(qa_chain):
    contexts = [{"file_url": "doc.pdf", "section_id": "sec1", "content": "內容"}]
    history = [{"user": "【用户问题】: 舊問題", "assistant": "舊回答"}]
    messages = qa_chain.format_prompt("新問題", contexts, history)
    assert any(m["role"] == "user" and "舊問題" in m["content"] for m in messages)
    assert any(m["role"] == "assistant" and "舊回答" in m["content"] for m in messages)

def test_stream_answer_with_pre_retrieved_chunks(qa_chain):
    pre_chunks = [{"file_url": "doc.pdf", "section_id": "sec1", "content": "內容"}]
    qa_chain.llm_client.stream_generate = lambda messages, model_name: ["回答"]
    responses = list(qa_chain.stream_answer("測試", pre_retrieved_chunks=pre_chunks))
    assert any(r["type"] == "sources" for r in responses)
    assert any(r["type"] == "text" for r in responses)

def test_stream_answer_disable_compliance(qa_chain):
    qa_chain.llm_client.stream_generate = lambda messages, model_name: ["token1", "token2"]
    responses = list(qa_chain.stream_answer("測試", enable_compliance_check=False))
    assert any(r["data"] == "token1" for r in responses)

def test_stream_answer_sanitize_text(qa_chain):
    qa_chain.compliance_checker.sanitize_text = lambda t: "已脫敏"
    qa_chain.llm_client.stream_generate = lambda messages, model_name: ["敏感"]
    responses = list(qa_chain.stream_answer("測試"))
    assert any(r["data"] == "已脫敏" for r in responses)

def test_stream_answer_audit_passed(qa_chain):
    qa_chain.compliance_checker.audit_and_sanitize = lambda text: {"passed": True}
    qa_chain.llm_client.stream_generate = lambda messages, model_name: ["正常回答"]
    responses = list(qa_chain.stream_answer("測試"))
    assert all(r["type"] != "security_block" for r in responses)

def test_run_with_security_block(qa_chain):
    qa_chain.compliance_checker.audit_and_sanitize = lambda text: {
        "passed": False,
        "blocked_by": "敏感詞",
        "sanitized_text": "已過濾"
    }
    qa_chain.llm_client.stream_generate = lambda messages, model_name: ["敏感信息"]
    result = qa_chain.run("測試")
    assert "已過濾" in result
