# memory/memory_manager.py
import os
import sys
import copy
import uuid
import logging
from typing import Dict, Any, List, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from short_term_memory import ShortTermMemory
from entity_memory import EntityMemory
# 导入测试期文件存储引擎
from chat_history_file import ChatHistoryFileStorage

logger = logging.getLogger("MemoryManager")

class MemoryManager:
    """多用户 & 多会话 Memory 管理器（测试阶段：JSON 文件存储）"""

    def __init__(self, user_id: str = "default", session_id: Optional[str] = None, max_messages: int = 20, config_path: Optional[str] = None):
        self.user_id = user_id
        self.session_id = session_id or str(uuid.uuid4())
        
        self.short_term = ShortTermMemory(max_messages=max_messages)
        self.entity = EntityMemory(user_id=self.user_id, config_path=config_path)
        
        # 🧪 测试期间使用基于 JSON 的文件存储引擎（后期只需切换此处的 Storage 类即可）
        self.history_storage = ChatHistoryFileStorage()

        self._snapshot_messages: Optional[List[Dict[str, str]]] = None
        self._snapshot_entities: Optional[Dict[str, Any]] = None

        # 加载当前 Session 历史记录
        self._load_session_history()

        logger.info(f"🧠 MemoryManager 已初始化 | 用户: [{self.user_id}] | 会话: [{self.session_id}] (模式: File JSON)")

    def _load_session_history(self):
        """加载当前 user_id 及 session_id 的历史记录"""
        messages = self.history_storage.get_session_messages(self.user_id, self.session_id)
        for msg in messages:
            if msg.get("role") == "user":
                self.short_term.add_user_message(msg["content"])
            elif msg.get("role") == "assistant":
                self.short_term.add_assistant_message(msg["content"])

    def switch_session(self, new_session_id: str):
        """切换活跃会话"""
        self.session_id = new_session_id
        self.short_term.clear()
        self._load_session_history()
        logger.info(f"🔄 用户 [{self.user_id}] 已切换至会话: [{self.session_id}]")

    def process_user_input(self, user_text: str):
        """处理用户输入并保存至 JSON 文件"""
        self.short_term.add_user_message(user_text)
        self.entity.extract_and_update(user_text)
        self.history_storage.add_message(self.user_id, self.session_id, "user", user_text)

    def process_assistant_output(self, assistant_text: str):
        """处理模型回复并保存至 JSON 文件"""
        self.short_term.add_assistant_message(assistant_text)
        self.history_storage.add_message(self.user_id, self.session_id, "assistant", assistant_text)

    def get_recent_sessions_list(self) -> List[Dict[str, Any]]:
        """获取当前用户的最近对话清单"""
        return self.history_storage.get_user_sessions(self.user_id)

    def get_context_for_llm(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "messages": self.short_term.get_messages(),
            "entities": self.entity.get_entities()
        }

    def begin_transaction(self):
        self._snapshot_messages = copy.deepcopy(self.short_term.get_messages())
        self._snapshot_entities = copy.deepcopy(self.entity.get_entities())

    def commit(self):
        self._snapshot_messages = None
        self._snapshot_entities = None

    def rollback(self):
        if self._snapshot_messages is not None and self._snapshot_entities is not None:
            self.short_term.set_messages(self._snapshot_messages)
            self.entity.set_entities(self._snapshot_entities)
            self._snapshot_messages = None
            self._snapshot_entities = None
            logger.warning(f"🚨 [{self.user_id}] 触发 Rollback！已恢复内存快照")

    def clear_all(self):
        self.short_term.clear()
        self.entity.clear()
        # 传入 user_id 进行删除
        self.history_storage.delete_session(self.user_id, self.session_id)
        self._snapshot_messages = None
        self._snapshot_entities = None