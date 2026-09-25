# tests/test_mcp_server.py
import importlib

def test_mcp_server_importable():
    """Smoke test: 確保 mcp_server 模塊可以導入"""
    module = importlib.import_module("agent_jerry_gao.mcp_server")
    assert module is not None
