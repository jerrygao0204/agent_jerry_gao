import os
import json
import datetime
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config_loader import config_loader    
from atomic_io import file_lock_for, atomic_dump_json

logger = logging.getLogger("ChatHistoryFile")


class ChatHistoryFileStorage:
    """基于 JSON 文件存储历史对话与会话清单（按 user_id 划分独立存储路径，支持 Soft Delete Schema）"""

    def __init__(self, base_dir: Optional[str] = None):
        if base_dir is None:
            self.base_dir = config_loader.get_config_dict()["data_root"]
        else:
            self.base_dir = base_dir

    def _get_user_dir(self, user_id: str) -> str:
        user_dir = os.path.join(self.base_dir, user_id)
        os.makedirs(user_dir, exist_ok=True)
        return user_dir

    def _get_sessions_file(self, user_id: str) -> str:
        return os.path.join(self._get_user_dir(user_id), "sessions_index.json")

    def _get_session_file_path(self, user_id: str, session_id: str) -> str:
        return os.path.join(self._get_user_dir(user_id), f"session_{session_id}.json")

    # ==========================================
    # 🔄 向下兼容归一化函数 (Backward Compatibility)
    # ==========================================
    def _normalize_session_record(self, session: Dict[str, Any]) -> Dict[str, Any]:
        """为旧版 sessions_index.json 节点补齐软删除默认字段"""
        if isinstance(session, dict):
            session.setdefault("is_deleted", False)
            session.setdefault("status", "active")
        return session

    def _normalize_message_record(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """为旧版 session_xxxx.json Message 节点补齐状态默认字段"""
        if isinstance(message, dict):
            message.setdefault("is_active", True)
            message.setdefault("status", "valid")
        return message

    def _load_json(self, path: str) -> Any:
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ 读取文件失败 ({path}): {e}")
            return {}

    def _load_sessions(self, user_id: str) -> Dict[str, Any]:
        """安全读取 sessions_index.json 并执行向下兼容归一化"""
        sessions_file = self._get_sessions_file(user_id)
        data = self._load_json(sessions_file)
        if not isinstance(data, dict):
            return {}
        
        # 对每一个 session 节点执行归一化
        for s_id, s_data in data.items():
            data[s_id] = self._normalize_session_record(s_data)
        return data

    def _load_messages(self, user_id: str, session_id: str) -> List[Dict[str, Any]]:
        """安全读取 session_xxxx.json 并执行 Message 节点归一化（带读锁保护）"""
        session_file = self._get_session_file_path(user_id, session_id)
        if not os.path.exists(session_file):
            return []
            
        with file_lock_for(session_file):
            data = self._load_json(session_file)
            
        if not isinstance(data, list):
            return []
            
        return [self._normalize_message_record(msg) for msg in data]
    
    # ==========================================
    # 🛠️ 会话与消息读写操作
    # ==========================================
    def create_session(self, user_id: str, session_id: str, title: str = "新对话"):
        """创建新会话索引（自动包含 is_deleted 与 status 字段）"""
        sessions_file = self._get_sessions_file(user_id)
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with file_lock_for(sessions_file):
            sessions = self._load_sessions(user_id)
            sessions[session_id] = {
                "session_id": session_id,
                "user_id": user_id,
                "title": title,
                "created_at": now,
                "updated_at": now,
                "is_deleted": False,      # P0 Soft Delete Schema
                "status": "active"        # P0 Soft Delete Schema
            }
            atomic_dump_json(sessions_file, sessions)

        session_file = self._get_session_file_path(user_id, session_id)
        with file_lock_for(session_file):
            if not os.path.exists(session_file):
                atomic_dump_json(session_file, [])

    def add_message(self, user_id: str, session_id: str, role: str, content: str):
        """追加问答消息到指定用户的 session JSON 文件中（自动包含 is_active 与 status 字段）"""
        sessions_file = self._get_sessions_file(user_id)
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 1. 维护会话标题与更新时间
        with file_lock_for(sessions_file):
            sessions = self._load_sessions(user_id)
            if session_id not in sessions:
                title = content[:15] + "..." if len(content) > 15 else content
                sessions[session_id] = {
                    "session_id": session_id,
                    "user_id": user_id,
                    "title": title,
                    "created_at": now,
                    "updated_at": now,
                    "is_deleted": False,
                    "status": "active"
                }
            else:
                if sessions[session_id].get("title") == "新对话" and role == "user":
                    sessions[session_id]["title"] = content[:15] + "..." if len(content) > 15 else content
                sessions[session_id]["updated_at"] = now
                
            atomic_dump_json(sessions_file, sessions)

        # 2. 追加消息内容
        session_file = self._get_session_file_path(user_id, session_id)
        with file_lock_for(session_file):
            messages = self._load_messages(user_id, session_id)
            messages.append({
                "user_id": user_id,
                "role": role,
                "content": content,
                "timestamp": now,
                "is_active": True,        # P0 Soft Delete Schema
                "status": "valid"         # P0 Soft Delete Schema
            })
            atomic_dump_json(session_file, messages)

    # ==========================================
    # 🗑️ 软删除操作 (Soft Delete Methods)
    # ==========================================
    def soft_delete_session(self, user_id: str, session_id: str):
        """软删除会话：仅标记 is_deleted/status，不物理删除文件，保留用于审计"""
        sessions_file = self._get_sessions_file(user_id)
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with file_lock_for(sessions_file):
            sessions = self._load_sessions(user_id)
            if session_id in sessions:
                sessions[session_id]["is_deleted"] = True
                sessions[session_id]["status"] = "deleted"
                sessions[session_id]["updated_at"] = now
                atomic_dump_json(sessions_file, sessions)

    def soft_delete_messages_after(self, user_id: str, session_id: str, keep_active_count: int):
        """软删除指定编辑点及之后的消息，解耦文件锁避免死锁"""
        session_file = self._get_session_file_path(user_id, session_id)
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        modified_count = 0

        # 1. 操作 session 消息文件锁
        with file_lock_for(session_file):
            raw_messages = self._load_json(session_file)
            if isinstance(raw_messages, list):
                active_count = 0
                for msg in raw_messages:
                    self._normalize_message_record(msg)
                    if not msg.get("is_active", True) or msg.get("status") == "deleted":
                        continue

                    active_count += 1
                    if active_count > keep_active_count:
                        msg["is_active"] = False
                        msg["status"] = "deleted"
                        msg["deleted_at"] = now
                        modified_count += 1

                if modified_count > 0:
                    atomic_dump_json(session_file, raw_messages)

        # 2. 独立操作 sessions 索引文件锁（避免与 session_file 形成交叉嵌套锁）
        if modified_count > 0:
            sessions_file = self._get_sessions_file(user_id)
            with file_lock_for(sessions_file):
                sessions = self._load_sessions(user_id)
                if session_id in sessions:
                    sessions[session_id]["updated_at"] = now
                    atomic_dump_json(sessions_file, sessions)
                    
            logger.info(f"🗑️ [消息软截断] 用户 [{user_id}] 会话 [{session_id[:8]}] 保留前 {keep_active_count} 条 active 消息，已软删 {modified_count} 条消息")

    def rollback_session_messages(self, user_id: str, session_id: str, expected_count: int):
        """事务回滚专用：确保磁盘消息数量不超过 expected_count"""
        messages = self.get_session_messages(user_id, session_id)
        if len(messages) > expected_count:
            self.soft_delete_messages_after(user_id, session_id, keep_active_count=expected_count)
            logger.warning(f"🚨 [存储回滚] 用户 [{user_id}] 会话 [{session_id[:8]}] 消息数已回滚至 {expected_count} 条")

    # ==========================================
    # 🔄 内部私有加载函数 (内部不加锁，避免重入死锁)
    # ==========================================
    def _load_sessions_unlocked(self, user_id: str) -> Dict[str, Any]:
        """无锁安全读取 sessions_index.json"""
        sessions_file = self._get_sessions_file(user_id)
        data = self._load_json(sessions_file)
        if not isinstance(data, dict):
            return {}
        for s_id, s_data in data.items():
            data[s_id] = self._normalize_session_record(s_data)
        return data

    def _load_messages_unlocked(self, user_id: str, session_id: str) -> List[Dict[str, Any]]:
        """无锁安全读取 session_xxxx.json"""
        session_file = self._get_session_file_path(user_id, session_id)
        data = self._load_json(session_file)
        if not isinstance(data, list):
            return []
        return [self._normalize_message_record(msg) for msg in data]
    
    # ==========================================
    # 🔍 读取与查询操作 (更新为带过滤功能)
    # ==========================================
    def get_user_sessions(self, user_id: str, limit: int = 30, include_deleted: bool = False) -> List[Dict[str, Any]]:
        """获取指定用户的最近对话清单（带读锁保护）"""
        sessions_file = self._get_sessions_file(user_id)
        with file_lock_for(sessions_file):
            sessions = self._load_sessions_unlocked(user_id)
            
        user_sessions = list(sessions.values())
        if not include_deleted:
            user_sessions = [
                s for s in user_sessions
                if not s.get("is_deleted", False) and s.get("status", "active") != "deleted"
            ]

        user_sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return user_sessions[:limit]

    def get_session_messages(self, user_id: str, session_id: str, include_inactive: bool = False) -> List[Dict[str, Any]]:
        """获取指定用户的某个会话聊天记录（带读锁保护）"""
        session_file = self._get_session_file_path(user_id, session_id)
        with file_lock_for(session_file):
            messages = self._load_messages_unlocked(user_id, session_id)
            
        if include_inactive:
            return messages
        return [m for m in messages if m.get("is_active", True) and m.get("status", "valid") == "valid"]

    def delete_session(self, user_id: str, session_id: str):
        """删除某个会话索引及对应的 JSON 文件"""
        sessions_file = self._get_sessions_file(user_id)
        with file_lock_for(sessions_file):
            sessions = self._load_sessions(user_id)
            if session_id in sessions:
                del sessions[session_id]
                atomic_dump_json(sessions_file, sessions)

        session_file = self._get_session_file_path(user_id, session_id)
        if os.path.exists(session_file):
            try:
                os.remove(session_file)
            except Exception as e:
                logger.error(f"⚠️ 删除会话文件失败 ({session_file}): {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    storage = ChatHistoryFileStorage()
    print(f"✅ 当前自适应解析出的 Data 根目录: {storage.base_dir}")