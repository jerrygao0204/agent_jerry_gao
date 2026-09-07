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