# tests/test_memory_manager_extra.py
import pytest
from agent_jerry_gao.memory import memory_manager

def test_memory_manager_delete_nonexistent_session():
    """測試刪除不存在的 session"""
    mgr = memory_manager.MemoryManager()
    with pytest.raises(Exception):
        mgr.delete("nonexistent_session")

def test_memory_manager_get_nonexistent_session():
    """測試獲取不存在的 session"""
    mgr = memory_manager.MemoryManager()
    with pytest.raises(Exception):
        mgr.get("nonexistent_session")

def test_memory_manager_update_nonexistent_session():
    """測試更新不存在的 session"""
    mgr = memory_manager.MemoryManager()
    with pytest.raises(Exception):
        mgr.update("nonexistent_session", {"key": "value"})
