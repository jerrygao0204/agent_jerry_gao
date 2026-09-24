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
# 导入文件存储引擎
from chat_history_file import ChatHistoryFileStorage

logger = logging.getLogger("MemoryManager")

class MemoryManager:
    """多用户 & 多会话 Memory 管理器（支持 Soft Delete Schema）"""

    def __init__(self, user_id: str = "default", session_id: Optional[str] = None, max_messages: int = 20, config_path: Optional[str] = None):
        self.user_id = user_id
        self.session_id = session_id or str(uuid.uuid4())
        
        self.short_term = ShortTermMemory(max_messages=max_messages)
        self.entity = EntityMemory(user_id=self.user_id, config_path=config_path)
        
        # 基于 JSON 的文件存储引擎
        self.history_storage = ChatHistoryFileStorage()

        self._snapshot_messages: Optional[List[Dict[str, str]]] = None
        self._snapshot_entities: Optional[Dict[str, Any]] = None

        # 加载当前 Session 历史记录（底层 get_session_messages 已支持自动过滤软删消息）
        self._load_session_history()

        logger.info(f"🧠 MemoryManager 已初始化 | 用户: [{self.user_id}] | 会话: [{self.session_id}] (模式: File JSON)")

    def _load_session_history(self):
        """加载当前 user_id 及 session_id 的历史记录（自动只载入未软删的消息）"""
        messages = self.history_storage.get_session_messages(self.user_id, self.session_id)
        for msg in messages:
            if msg.get("role") == "user":
                self.short_term.add_user_message(msg["content"])
            elif msg.get("role") == "assistant":
                self.short_term.add_assistant_message(msg["content"])

    def reload_session(self):
        """强制从持久化存储重新载入【当前】会话的 active 消息，丢弃内存中的短期记忆。

        switch_session(同一个 id) 会因为"防重校验"直接返回而不重新载入，
        需要"原地刷新"时（切回已缓存的会话、软删除/编辑重建之后）应调用本方法。
        """
        self.short_term.clear()
        self._load_session_history()

    def switch_session(self, new_session_id: str):
        """切换活跃会话（加防重校验与实体状态刷新）"""
        if not new_session_id or new_session_id == self.session_id:
            return

        self.session_id = new_session_id
        self.short_term.clear()
        
        # 若 Entity Memory 为 Session 级别，切换时需要重置/重新加载
        if hasattr(self.entity, "reload_for_session"):
            self.entity.reload_for_session(new_session_id)
            
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
        """获取当前用户的最近对话清单（自动过滤软删会话）"""
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
        """回滚内存与存储状态至事务开始前的快照"""
        if self._snapshot_messages is not None and self._snapshot_entities is not None:
            # 1. 恢复内存数据状态
            self.short_term.set_messages(self._snapshot_messages)
            self.entity.set_entities(self._snapshot_entities)
            
            # 2. 如果之前在 process_user_input 时已同步写入磁盘，需要从磁盘中移除未提交的末尾消息
            if hasattr(self.history_storage, "rollback_session_messages"):
                self.history_storage.rollback_session_messages(
                    self.user_id, self.session_id, expected_count=len(self._snapshot_messages)
                )
                
            self._snapshot_messages = None
            self._snapshot_entities = None
            logger.warning(f"🚨 用户 [{self.user_id}] 会话 [{self.session_id}] 触发 Rollback！已恢复内存快照")

    # ==========================================
    # 🗑️ 软清空与物理清空操作
    # ==========================================
    def soft_clear_all(self):
        """软删除当前 Session（仅标记删除索引，保留全局用户实体）"""
        # 1. 清空当前内存上下文
        self.short_term.clear()
        
        # 2. 若实体具备 Session 隔离能力，执行 Session 级别的清理
        if hasattr(self.entity, "clear_session_entities"):
            self.entity.clear_session_entities(self.session_id)
            
        # 3. 触发存储层的软删除（仅标记 is_deleted=True / status='deleted'）
        self.history_storage.soft_delete_session(self.user_id, self.session_id)
        
        # 4. 清空事务快照
        self._snapshot_messages = None
        self._snapshot_entities = None
        logger.info(f"🗑️ [软删除成功] 用户 [{self.user_id}] 会话 [{self.session_id}] 已标记删除")

    def clear_all(self):
        """物理删除当前 Session（清理内存与磁盘文件，安全保留用户全局实体）"""
        self.short_term.clear()
        if hasattr(self.entity, "clear_session_entities"):
            self.entity.clear_session_entities(self.session_id)

        self.history_storage.delete_session(self.user_id, self.session_id)
        self._snapshot_messages = None
        self._snapshot_entities = None
        logger.info(f"💥 [物理删除成功] 用户 [{self.user_id}] 会话 [{self.session_id}] 磁盘文件已移除")