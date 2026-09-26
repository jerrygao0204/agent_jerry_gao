# tests/test_log_factory.py
import logging
import os
import pytest
from pathlib import Path
from factory.log_factory import setup_logger, RAGContextLoggerAdapter


@pytest.fixture(autouse=True)
def clean_logger_handlers():
    """测试前/后清理 Logger Handlers，保证测试隔离"""
    logger = logging.getLogger("RAG_System_Test")
    logger.handlers.clear()
    yield
    logger.handlers.clear()


def test_setup_logger_handlers_and_singleton(tmp_path):
    """验证 setup_logger 防止重复添加 Handler 以及文件创建"""
    log_file = tmp_path / "logs" / "rag_system.log"
    logger = setup_logger(name="RAG_System_Test", log_file=str(log_file))

    # 验证 Handler 数量 (1个 FileHandler + 1个 StreamHandler)
    assert len(logger.handlers) == 2

    # 重载同名 Logger，验证防重复添加机制
    logger_duplicate = setup_logger(name="RAG_System_Test", log_file=str(log_file))
    assert len(logger_duplicate.handlers) == 2
    assert logger is logger_duplicate


def test_setup_logger_file_write(tmp_path):
    """验证按天轮转日志文件写操作与内容落盘"""
    log_file = tmp_path / "logs" / "test_write.log"
    logger = setup_logger(name="RAG_System_Test", log_file=str(log_file))

    test_msg = "Jerry Gao Agent Log System Test"
    logger.info(test_msg)

    # 刷新 handler 缓冲区
    for handler in logger.handlers:
        handler.flush()

    assert os.path.exists(log_file)
    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()
        assert test_msg in content
        assert "[INFO]" in content


def test_rag_context_logger_adapter():
    """验证 RAGContextLoggerAdapter 的 User与 Query 上下文格式化功能"""
    base_logger = logging.getLogger("RAG_System_Test")
    adapter = RAGContextLoggerAdapter(
        base_logger, extra={"user_id": "Jerry", "question": "What is Python?"}
    )

    # 1. 使用默认 Adapter extra
    formatted_msg, kwargs = adapter.process("Initial prompt processing", {})
    assert formatted_msg == "[User:Jerry] [Query:What is Python?] Initial prompt processing"

    # 2. 运行时覆盖 extra
    override_kwargs = {
        "extra": {"user_id": "Gao", "question": "Explain RAG Pipeline"}
    }
    formatted_msg_override, _ = adapter.process("Overridden prompt", override_kwargs)
    assert formatted_msg_override == "[User:Gao] [Query:Explain RAG Pipeline] Overridden prompt"