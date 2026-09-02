# # memory/chat_history_file.py
# import os
# import json
# import datetime
# import logging
# import sys
# from typing import List, Dict, Any, Optional
# from pathlib import Path
# from filelock import FileLock

# logger = logging.getLogger("ChatHistoryFile")

# # 1. 动态获取项目根目录并注入 sys.path (确保跨环境/跨目录执行时导入不崩)
# PROJECT_ROOT = Path(__file__).resolve().parent.parent  # 指向 '问答机器人' 项目根目录
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

# logger = logging.getLogger("ChatHistoryFile")

# class ChatHistoryFileStorage:
#     """基于 JSON 文件存储历史对话与会话清单（按 user_id 划分独立存储路径）"""

#     def __init__(self, base_dir: Optional[str] = None):
#         # 默认基础路径设置为用户指定的目录
#         if base_dir is None:
#             self.base_dir = str(PROJECT_ROOT / "data")
#         else:
#             self.base_dir = base_dir

#     def _get_user_dir(self, user_id: str) -> str:
#         """获取并自动创建指定 user_id 的存储目录"""
#         user_dir = os.path.join(self.base_dir, user_id)
#         os.makedirs(user_dir, exist_ok=True)
#         return user_dir

#     def _get_user_lock(self, user_id: str) -> FileLock:
#         """获取指定用户的并发文件锁（锁文件存放在用户目录下）"""
#         user_dir = self._get_user_dir(user_id)
#         lock_path = os.path.join(user_dir, ".user_storage.lock")
#         return FileLock(lock_path, timeout=10)
    
#     def _get_sessions_file(self, user_id: str) -> str:
#         """获取指定用户的会话索引文件路径"""
#         return os.path.join(self._get_user_dir(user_id), "sessions_index.json")

#     def _get_session_file_path(self, user_id: str, session_id: str) -> str:
#         """获取指定用户某个会话的具体消息文件路径"""
#         return os.path.join(self._get_user_dir(user_id), f"session_{session_id}.json")

#     def _load_json(self, path: str) -> Dict[str, Any]:
#         if not os.path.exists(path):
#             return {}
#         try:
#             with open(path, "r", encoding="utf-8") as f:
#                 return json.load(f)
#         except Exception as e:
#             logger.error(f"❌ 读取文件失败 ({path}): {e}")
#             return {}

#     def _save_json(self, path: str, data: Any):
#         try:
#             with open(path, "w", encoding="utf-8") as f:
#                 json.dump(data, f, ensure_ascii=False, indent=2)
#         except Exception as e:
#             logger.error(f"❌ 写入文件失败 ({path}): {e}")

#     def create_session(self, user_id: str, session_id: str, title: str = "新对话"):
#         """创建新会话索引"""
#         lock = self._get_user_lock(user_id)
#         with lock:
#             sessions_file = self._get_sessions_file(user_id)
#             sessions = self._load_json(sessions_file)
#             now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#             sessions[session_id] = {
#                 "session_id": session_id,
#                 "user_id": user_id,
#                 "title": title,
#                 "created_at": now,
#                 "updated_at": now
#             }
#             self._save_json(sessions_file, sessions)

#             # 初始化会话消息文件
#             session_file = self._get_session_file_path(user_id, session_id)
#             if not os.path.exists(session_file):
#                 self._save_json(session_file, [])

#     def add_message(self, user_id: str, session_id: str, role: str, content: str):
#         """追加问答消息到指定用户的 session JSON 文件中"""
#         lock = self._get_user_lock(user_id)
#         with lock:
#             sessions_file = self._get_sessions_file(user_id)
#             sessions = self._load_json(sessions_file)
#             now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

#             # 1. 自动维护会话标题与更新时间
#             if session_id not in sessions:
#                 title = content[:15] + "..." if len(content) > 15 else content
#                 sessions[session_id] = {
#                     "session_id": session_id,
#                     "user_id": user_id,
#                     "title": title,
#                     "created_at": now,
#                     "updated_at": now
#                 }
#             else:
#                 if sessions[session_id].get("title") == "新对话" and role == "user":
#                     sessions[session_id]["title"] = content[:15] + "..." if len(content) > 15 else content
#                 sessions[session_id]["updated_at"] = now

#             self._save_json(sessions_file, sessions)

#             # 2. 追加消息内容
#             session_file = self._get_session_file_path(user_id, session_id)
#             messages = self._load_json(session_file)
#             if not isinstance(messages, list):
#                 messages = []

#             messages.append({
#                 "user_id": user_id,
#                 "role": role,
#                 "content": content,
#                 "timestamp": now
#             })
#             self._save_json(session_file, messages)

#     def get_user_sessions(self, user_id: str, limit: int = 30) -> List[Dict[str, Any]]:
#         """获取指定用户的最近对话清单"""
#         lock = self._get_user_lock(user_id)
#         with lock:
#             sessions_file = self._get_sessions_file(user_id)
#             sessions = self._load_json(sessions_file)
            
#             user_sessions = list(sessions.values())
#             user_sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
#             return user_sessions[:limit]

#     def get_session_messages(self, user_id: str, session_id: str) -> List[Dict[str, str]]:
#         """获取指定用户的某个会话全量聊天记录"""
#         lock = self._get_user_lock(user_id)
#         with lock:
#             session_file = self._get_session_file_path(user_id, session_id)
#             messages = self._load_json(session_file)
#             return messages if isinstance(messages, list) else []

#     def delete_session(self, user_id: str, session_id: str):
#         """删除某个会话索引及对应的 JSON 文件"""
#         lock = self._get_user_lock(user_id)
#         with lock:
#             sessions_file = self._get_sessions_file(user_id)
#             sessions = self._load_json(sessions_file)
            
#             if session_id in sessions:
#                 del sessions[session_id]
#                 self._save_json(sessions_file, sessions)

#             session_file = self._get_session_file_path(user_id, session_id)
#             if os.path.exists(session_file):
#                 try:
#                     os.remove(session_file)
#                 except Exception as e:
#                     logger.error(f"⚠️ 删除会话文件失败 ({session_file}): {e}")

# if __name__ == "__main__":
#     logging.basicConfig(level=logging.INFO)
#     storage = ChatHistoryFileStorage()
#     print(f"✅ 当前自适应解析出的 Data 根目录: {storage.base_dir}")

# memory/chat_history_file.py

import os
import json
import datetime
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

from config.config_loader import config_loader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from atomic_io import file_lock_for, atomic_dump_json

logger = logging.getLogger("ChatHistoryFile")


class ChatHistoryFileStorage:
    """基于 JSON 文件存储历史对话与会话清单（按 user_id 划分独立存储路径）"""

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

    def _load_json(self, path: str) -> Any:
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ 读取文件失败 ({path}): {e}")
            return {}

    def create_session(self, user_id: str, session_id: str, title: str = "新对话"):
        """创建新会话索引"""
        sessions_file = self._get_sessions_file(user_id)
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with file_lock_for(sessions_file):
            sessions = self._load_json(sessions_file)
            sessions[session_id] = {
                "session_id": session_id,
                "user_id": user_id,
                "title": title,
                "created_at": now,
                "updated_at": now,
            }
            atomic_dump_json(sessions_file, sessions)

        session_file = self._get_session_file_path(user_id, session_id)
        with file_lock_for(session_file):
            if not os.path.exists(session_file):
                atomic_dump_json(session_file, [])

    def add_message(self, user_id: str, session_id: str, role: str, content: str):
        """追加问答消息到指定用户的 session JSON 文件中"""
        sessions_file = self._get_sessions_file(user_id)
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # 1. 维护会话标题与更新时间：读-改-写整段包在锁里
        with file_lock_for(sessions_file):
            sessions = self._load_json(sessions_file)
            if session_id not in sessions:
                title = content[:15] + "..." if len(content) > 15 else content
                sessions[session_id] = {
                    "session_id": session_id,
                    "user_id": user_id,
                    "title": title,
                    "created_at": now,
                    "updated_at": now,
                }
            else:
                if sessions[session_id].get("title") == "新对话" and role == "user":
                    sessions[session_id]["title"] = content[:15] + "..." if len(content) > 15 else content
                sessions[session_id]["updated_at"] = now
            atomic_dump_json(sessions_file, sessions)

        # 2. 追加消息内容：读-改-写整段包在锁里
        session_file = self._get_session_file_path(user_id, session_id)
        with file_lock_for(session_file):
            messages = self._load_json(session_file)
            if not isinstance(messages, list):
                messages = []
            messages.append({
                "user_id": user_id,
                "role": role,
                "content": content,
                "timestamp": now,
            })
            atomic_dump_json(session_file, messages)

    def get_user_sessions(self, user_id: str, limit: int = 30) -> List[Dict[str, Any]]:
        """获取指定用户的最近对话清单"""
        sessions_file = self._get_sessions_file(user_id)
        sessions = self._load_json(sessions_file)
        user_sessions = list(sessions.values())
        user_sessions.sort(key=lambda x: x.get("updated_at", ""), reverse=True)
        return user_sessions[:limit]

    def get_session_messages(self, user_id: str, session_id: str) -> List[Dict[str, str]]:
        """获取指定用户的某个会话全量聊天记录"""
        session_file = self._get_session_file_path(user_id, session_id)
        messages = self._load_json(session_file)
        return messages if isinstance(messages, list) else []

    def delete_session(self, user_id: str, session_id: str):
        """删除某个会话索引及对应的 JSON 文件"""
        sessions_file = self._get_sessions_file(user_id)
        with file_lock_for(sessions_file):
            sessions = self._load_json(sessions_file)
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