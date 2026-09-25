# tests/test_app_admin.py
import importlib

def test_app_admin_importable():
    """Smoke test: 確保 app_admin 模塊可以導入"""
    module = importlib.import_module("agent_jerry_gao.app_admin")
    assert module is not None
