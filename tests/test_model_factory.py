import pytest
import torch
from factory.model_factory import ModelFactory

def test_system_gc_execution():
    """测试底层垃圾回收与 CUDA 清空方法"""
    try:
        ModelFactory._trigger_system_gc()
        assert True
    except Exception as e:
        pytest.fail(f"_trigger_system_gc 执行抛出异常: {e}")

def test_destroy_all_models_cls():
    """测试全量物理销毁逻辑，确保不会引发未捕获异常"""
    try:
        ModelFactory.destroy_all_models_cls()
        # 验证单例句柄已被清空
        if hasattr(ModelFactory, "_instance"):
            assert ModelFactory._instance is None
    except Exception as e:
        pytest.fail(f"destroy_all_models_cls 执行失败: {e}")