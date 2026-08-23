# memory/short_term_memory.py
import logging
from typing import List, Dict, Any

logger = logging.getLogger("ShortTermMemory")

class ShortTermMemory:
    """短期对话上下文 Memory"""

    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self.messages: List[Dict[str, str]] = []

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})
        self._trim()

    def add_assistant_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})
        self._trim()

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})
        self._trim()

    def get_messages(self) -> List[Dict[str, str]]:
        return [msg.copy() for msg in self.messages]

    def set_messages(self, messages: List[Dict[str, str]]):
        self.messages = [msg.copy() for msg in messages]

    def _trim(self):
        """窗口截断逻辑"""
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]

    def clear(self):
        self.messages.clear()