# tests/test_qa_chain_extra.py
import pytest
from generator.qa_chain import QAChain
from utils.timeout_ctx import TimeoutException

class DummyRetriever:
    def hybrid_search(self, query, top_k=10, filter_expr=None):
        return [{"content": "mocked content", "score": 0.9}]

class DummyReranker:
    def rerank(self, query, documents, top_n=3):
        return documents[:top_n]

@pytest.fixture
def qa_chain():
    chain = QAChain(llm_model_name="qwen3-32b")
    chain.retriever = DummyRetriever()
    chain.reranker = DummyReranker()
    return chain

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
    # 不應該返回 security_block
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
