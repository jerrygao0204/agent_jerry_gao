# import logging
# import signal
# from contextlib import contextmanager

# logger = logging.getLogger(__name__)

# class TimeoutException(Exception):
#     """自定义超时异常 (Custom Timeout Exception)"""
#     pass

# @contextmanager
# def timeout(seconds: int = 30, label: str = "Operation"):
#     """
#     通用超时保护上下文管理器 (仅适用于 Unix/Linux 主线程)
    
#     :param seconds: 超时时间（秒）
#     :param label: 操作标识名称，用于日志输出
#     """
#     def _handle_timeout(signum, frame):
#         raise TimeoutException(f"【超时告警】{label} 执行超过 {seconds} 秒")

#     # 1. 注册 SIGALRM 信号处理器
#     signal.signal(signal.SIGALRM, _handle_timeout)
#     # 2. 启动定时器
#     signal.alarm(seconds)
    
#     try:
#         yield
#     except TimeoutException as e:
#         logger.error(f"[Timeout] {e}")
#         raise e
#     finally:
#         # 3. 无论正常结束还是抛出异常，均重置定时器，取消 SIGALRM
#         signal.alarm(0)

# utils/timeout_ctx.py
import signal
import threading
import logging
from contextlib import contextmanager

class TimeoutException(Exception):
    pass

def _handle_timeout(signum, frame):
    raise TimeoutException("⏰ 操作執行超時！")

@contextmanager
def timeout(seconds: int, label: str = "Operation"):
    # ⚠️ 檢查是否為主執行緒，非主執行緒跳過 SIGALRM 註冊，避免 ValueError
    is_main_thread = threading.current_thread() is threading.main_thread()
    
    if is_main_thread and seconds > 0:
        old_handler = signal.signal(signal.SIGALRM, _handle_timeout)
        signal.alarm(seconds)
    else:
        if not is_main_thread:
            logging.debug(f"ℹ️ [{label}] 處於子執行緒，自動跳過 SIGALRM 超時監控。")

    try:
        yield
    finally:
        if is_main_thread and seconds > 0:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)