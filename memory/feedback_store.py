# memory/feedback_store.py

import json
import time
from pathlib import Path
from threading import Lock

class FeedbackStore:
    def __init__(self, log_dir: str = "data/feedback"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / "user_feedback.jsonl"
        self._lock = Lock()

    def record_feedback(
        self, 
        user_id: str,           # 👈 确认此处已添加 user_id 参数
        query: str, 
        response: str, 
        rating: str, 
        feedback_text: str = "", 
        citations: list = None
    ) -> bool:
        """
        记录用户反馈 (rating: 'like' | 'dislike')，包含 user_id 追溯
        """
        record = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "user_id": user_id or "default_user",
            "query": query,
            "response": response,
            "rating": rating,
            "feedback_text": feedback_text,
            "citations": citations or []
        }
        
        with self._lock:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                return True
            except Exception as e:
                print(f"❌ 写入反馈日志失败: {e}")
                return False

# 全局单例
feedback_store = FeedbackStore()