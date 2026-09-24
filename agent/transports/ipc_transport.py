# agent/transports/ipc_transport.py
"""
本机 IPC 传输实现（当前唯一真正可用的 ToolTransportClient 实现）。

原理：
    沙箱代码运行在独立子进程里；主进程持有 tool_factory / 数据库连接 / LLM client 等有状态资源。
    每一次 SandboxExecutor.run() 都会创建一条【该次执行专属】的 multiprocessing.Pipe(duplex=True)：

        子进程 proxy.call(tool_name, kwargs)
            -> conn.send({"type": "tool_call", "call_id", "tool_name", "kwargs"})
            -> conn.recv() 阻塞等待 call_id 匹配的 {"type": "tool_result", ...}

        主进程（SandboxExecutor.run 的轮询循环）
            -> 收到 tool_call，交给 ToolDispatcher.dispatch(...) 真正执行
            -> conn.send({"type": "tool_result", "call_id", "result", "error"})

多用户/并发安全性：
    * 通道是"每次 run 一条"的私有 Pipe，不同用户、不同请求之间物理上没有共享的队列，
      所以响应不可能被别的请求"抢走"或"配错人"；call_id 只是在同一条通道内的二次校验。
    * 子进程里不再启动任何后台线程。之前的实现在主进程里创建 transport 并启动分发线程，
      但线程不会跟随 fork 进入子进程，导致子进程永远收不到响应（工具调用必定超时）。
    * 本类不持有 Lock / Thread，因此可被 pickle，spawn / forkserver 也能用。
"""
import uuid
from typing import Any, Dict, Optional

try:  # 作为包导入（qa_admin / react_agent）
    from agent.tool_transport import ToolTransportClient
except ImportError:  # 直接运行 agent/sandbox.py 自测时
    from tool_transport import ToolTransportClient


class IPCPipeToolTransport(ToolTransportClient):
    def __init__(self, conn, response_timeout: Optional[float] = None):
        """
        :param conn: multiprocessing.Connection（子进程侧的那一端）
        :param response_timeout: 等待主进程响应的上限(秒)。None 表示一直等
            （主进程会负责 kill 超时的子进程；设置上限只是为了主进程异常时子进程能自行退出）。
        """
        self._conn = conn
        self._response_timeout = response_timeout

    def call(self, tool_name: str, kwargs: Dict[str, Any]) -> Any:
        call_id = uuid.uuid4().hex
        try:
            self._conn.send(
                {"type": "tool_call", "call_id": call_id, "tool_name": tool_name, "kwargs": kwargs}
            )
        except (TypeError, AttributeError, ValueError) as e:
            # 参数里带了不可序列化的对象（如函数、文件句柄）
            raise RuntimeError(f"工具 [{tool_name}] 的参数无法序列化传给主进程: {e}") from e

        while True:
            if self._response_timeout is not None and not self._conn.poll(self._response_timeout):
                raise TimeoutError(f"等待工具 [{tool_name}] 响应超时 ({self._response_timeout}s)")
            resp = self._conn.recv()
            if resp.get("type") != "tool_result" or resp.get("call_id") != call_id:
                # 同一条私有通道上同一时刻只有一个在途请求，正常情况下不会走到这里；
                # 出现陌生响应就丢弃，继续等属于自己的那一条。
                continue
            if resp.get("error"):
                # 主进程回传了 error（工具超时 / 业务报错 / 权限不足），抛给沙箱代码
                raise RuntimeError(resp["error"])
            return resp.get("result")
