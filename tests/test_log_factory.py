# tests/test_log_factory.py
from agent_jerry_gao.factory import log_factory

def test_log_factory_get_logger():
    logger = log_factory.get_logger("test")
    assert logger is not None
    logger.info("這是一條測試日誌")
