# factory/log_factory.py
import os
import logging
from logging.handlers import RotatingFileHandler

def setup_logger(
    name: str = "RAG_System",
    log_file: str = "logs/rag_system.log",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB 自動輪轉
    backup_count: int = 5,              # 保留 5 個歷史日誌檔
    level=logging.INFO
) -> logging.Logger:
    """
    創建/獲取統一輪轉日誌記錄器
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重複添加 Handler
    if not logger.handlers:
        # 1. 輪轉檔案 Handler
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=max_bytes, 
            backupCount=backup_count, 
            encoding="utf-8"
        )
        formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - [%(filename)s:%(lineno)d] - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 2. 控制台 Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger