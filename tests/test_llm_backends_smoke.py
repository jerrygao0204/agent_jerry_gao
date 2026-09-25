# tests/test_llm_backends_smoke.py
import pytest
from agent_jerry_gao.factory import llm_backends

def test_llm_backends_init_smoke():
    """Smoke test: 確保 llm_backends 模塊可以導入"""
    assert hasattr(llm_backends, "__doc__")

def test_llm_backends_invalid_model():
    """測試無效模型名稱"""
    with pytest.raises(Exception):
        llm_backends.load_model("nonexistent-model")
