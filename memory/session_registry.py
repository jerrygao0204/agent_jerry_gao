# memory/session_registry.py
"""
按 (user_id, session_id) 维度缓存 MemoryManager 的并发安全注册表。

为什么不能"每个用户一个 MemoryManager"：
    MemoryManager 里有【可变的会话状态】——session_id、short_term、以及
    begin_transaction/commit/rollback 用的快照。同一个用户只要同时有两个浏览器标签页
    （或者 agent 正在流式回答时点了"切换/新建会话"），共用一个实例就会：
      * A 标签的后续写入落进 B 标签刚切过去的会话（消息写错会话）
      * 两个请求的事务快照互相覆盖，rollback() 把对方的消息"回滚"掉
    所以这里让每个 (user_id, session_id) 拥有独立实例：请求拿到实例后，
    它的 session_id 永远不会被别的请求改掉。

并发安全：
    * 注册表内部用锁保护；同一个 key 并发首次访问只会创建一个实例（不再"先检查后赋值"）
    * 创建实例（读磁盘）在全局锁外进行，不会因为某个用户的慢 IO 卡住其他用户
    * LRU 淘汰：缓存有上限。被淘汰的实例如果仍被某个进行中的请求持有，请求可以继续用完；
      消息在 process_* 时已写入磁盘，下次访问会从磁盘重新载入，不会丢数据
"""
import logging
import threading
import uuid
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional, Tuple

logger = logging.getLogger("SessionMemoryRegistry")


class SessionMemoryRegistry:
    def __init__(self, factory: Callable[..., Any], max_cached: int = 256):
        """
        :param factory: factory(user_id=..., session_id=...) -> MemoryManager
        :param max_cached: 最多缓存多少个 (用户, 会话) 实例
        """
        self._factory = factory
        self._max_cached = max_cached
        self._lock = threading.RLock()
        self._cache: "OrderedDict[Tuple[str, str], Any]" = OrderedDict()
        self._key_locks: Dict[Tuple[str, str], threading.Lock] = {}
        # 每个用户"最近一次显式使用的会话"，仅供【没带 session_id 的旧式调用】（列会话清单、登录）兜底
        self._current: Dict[str, str] = {}
        # 用户从未显式使用过任何会话时的稳定占位 id（保证无 session_id 的调用不会反复创建新实例）
        self._anchor: Dict[str, str] = {}

    # ------------------------------------------------------------------
    def get(self, user_id: str, session_id: Optional[str] = None):
        with self._lock:
            if session_id:
                self._current[user_id] = session_id
            else:
                session_id = self._current.get(user_id) or self._anchor.setdefault(user_id, str(uuid.uuid4()))

            key = (user_id, session_id)
            mgr = self._cache.get(key)
            if mgr is not None:
                self._cache.move_to_end(key)
                return mgr
            key_lock = self._key_locks.setdefault(key, threading.Lock())

        # 同一个 key 的并发创建串行化；不同 key 之间互不阻塞
        with key_lock:
            with self._lock:
                mgr = self._cache.get(key)
                if mgr is not None:
                    self._cache.move_to_end(key)
                    return mgr

            logger.info(f"🛠️ 为账号 [{user_id}] 会话 [{session_id[:8]}] 初始化 MemoryManager...")
            mgr = self._factory(user_id=user_id, session_id=session_id)

            with self._lock:
                self._cache[key] = mgr
                self._key_locks.pop(key, None)
                while len(self._cache) > self._max_cached:
                    self._cache.popitem(last=False)
            return mgr

    def evict(self, user_id: str, session_id: str) -> None:
        """会话被删除/软删除后，移除对应缓存。"""
        if not session_id:
            return
        with self._lock:
            self._cache.pop((user_id, session_id), None)
            if self._current.get(user_id) == session_id:
                self._current.pop(user_id, None)

    def current_session(self, user_id: str) -> Optional[str]:
        with self._lock:
            return self._current.get(user_id)

    def __len__(self) -> int:
        with self._lock:
            return len(self._cache)
