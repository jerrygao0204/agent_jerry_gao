import pytest
from agent_jerry_gao.factory import model_factory

def test_model_factory_invalid_model_path():
    """測試無效模型路徑"""
    with pytest.raises(Exception):
        model_factory.load_model("nonexistent_path")

def test_model_factory_invalid_model_type():
    """測試無效模型類型"""
    with pytest.raises(Exception):
        model_factory.create_model("invalid_type")

def test_model_factory_embed_query_invalid():
    """測試 embed_query 異常"""
    with pytest.raises(Exception):
        model_factory.embed_query("hello", model_name="nonexistent-model")

def test_model_factory_resolve_model_path_invalid():
    """測試 resolve_model_path 異常"""
    with pytest.raises(Exception):
        model_factory.resolve_model_path("invalid_model")
