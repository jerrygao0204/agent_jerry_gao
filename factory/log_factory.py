# # factory/log_factory.py
# import os
# import logging
# from logging.handlers import RotatingFileHandler

# def setup_logger(
#     name: str = "RAG_System",
#     log_file: str = "logs/rag_system.log",
#     max_bytes: int = 10 * 1024 * 1024,  # 10 MB 自動輪轉
#     backup_count: int = 5,              # 保留 5 個歷史日誌檔
#     level=logging.INFO
# ) -> logging.Logger:
#     """
#     創建/獲取統一輪轉日誌記錄器
#     """
#     os.makedirs(os.path.dirname(log_file), exist_ok=True)
#     logger = logging.getLogger(name)
#     logger.setLevel(level)

#     # 避免重複添加 Handler
#     if not logger.handlers:
#         # 1. 輪轉檔案 Handler
#         file_handler = RotatingFileHandler(
#             log_file, 
#             maxBytes=max_bytes, 
#             backupCount=backup_count, 
#             encoding="utf-8"
#         )
#         formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - [%(filename)s:%(lineno)d] - %(message)s")
#         file_handler.setFormatter(formatter)
#         logger.addHandler(file_handler)

#         # 2. 控制台 Console Handler
#         console_handler = logging.StreamHandler()
#         console_handler.setFormatter(formatter)
#         logger.addHandler(console_handler)

#     return logger


# factory/log_factory.py
import logging
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler

class RAGContextLoggerAdapter(logging.LoggerAdapter):
    """
    動態注入 RAG 系統上下文（user_id 與 question）的 Logger Adapter
    """
    def process(self, msg, kwargs):
        extra = kwargs.get("extra", {})
        # 提取或給予預設值
        user_id = extra.get("user_id", self.extra.get("user_id", "Anonymous"))
        question = extra.get("question", self.extra.get("question", "N/A"))
        
        # 統一格式化進訊息中，或透過 Formatter 渲染
        kwargs["extra"] = extra
        
        # 我們也可以直接改寫 msg，讓 Formatter 方便印出
        formatted_msg = f"[User:{user_id}] [Query:{question}] {msg}"
        return formatted_msg, kwargs


def setup_logger(
    name: str = "RAG_System",
    log_file: str = None,
    backup_count: int = 30,             # 改為保留 30 天的歷史紀錄
    level=logging.INFO
) -> logging.Logger:
    """
    創建/獲取支援按天輪轉並包含上下文的日誌記錄器
    """
    if log_file is None:
        project_root = Path(__file__).resolve().parent.parent
        log_file = project_root / "logs" / "rag_system.log"
    else:
        log_file = Path(log_file)

    # 確保日誌資料夾存在
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 避免重複添加 Handler
    if not logger.handlers:
        # 1. 按天輪轉 Handler (每天午夜切換一次，保留 backup_count 天)
        file_handler = TimedRotatingFileHandler(
            filename=str(log_file),
            when="midnight",
            interval=1,
            backupCount=backup_count,
            encoding="utf-8"
        )
        # 設定檔名字尾格式（例如：rag_system.log.2026-09-18）
        file_handler.suffix = "%Y-%m-%d"
        
        formatter = logging.Formatter(
            "%(asctime)s - [%(levelname)s] - [%(filename)s:%(lineno)d] - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # 2. 控制台 Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger