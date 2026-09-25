# tests/test_qa_admin_smoke.py
import importlib

def test_qa_admin_importable():
    """Smoke test: 確保 qa_admin 模塊可以導入"""
    module = importlib.import_module("agent_jerry_gao.qa_admin")
    assert module is not None
