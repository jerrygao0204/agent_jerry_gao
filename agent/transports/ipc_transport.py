# agent/transports/ipc_transport.py
"""
本机 IPC 队列传输实现（当前唯一真正可用的 ToolTransportClient 实现）。

原理：
    沙箱代码运行在 fork 出来的子进程里；主进程持有 tool_factory / 数据库连接 /
    LLM client 等有状态资源。两者之间通过一对 multiprocessing.Manager().Queue()
    做同步请求/响应桥：

        子进程 proxy.call(tool_name, kwargs)
            -> request_queue.put({call_id, tool_name, kwargs})
            -> 阻塞等待 response_queue 里 call_id 匹配的响应

        主进程（SandboxExecutor.run 的轮询循环里）
            -> 从 request_queue 取出请求
            -> 调用 ToolDispatcher.dispatch(tool_name, kwargs)  【真正执行】
            -> 把结果 put 回 response_queue

    这样真正的工具执行（向量库连接等）永远不会跨越 fork 边界，避免了直接把工具对象
    塞进子进程导致的连接卡死/状态损坏风险。
"""
import uuid
from typing import Any, Dict

from tool_transport import ToolTransportClient


class IPCQueueToolTransport(ToolTransportClient):
    def __init__(self, request_queue, response_queue):
        """
        :param request_queue: multiprocessing.Manager().Queue()，子进程 -> 主进程 的请求通道
        :param response_queue: multiprocessing.Manager().Queue()，主进程 -> 子进程 的响应通道
        """
        self.request_queue = request_queue
        self.response_queue = response_queue

    def call(self, tool_name: str, kwargs: Dict[str, Any]) -> Any:
        call_id = str(uuid.uuid4())
        self.request_queue.put({"call_id": call_id, "tool_name": tool_name, "kwargs": kwargs})

        # 理论上沙箱代码单线程同步执行，同一时刻只会有一个请求在途，
        # 这里的 call_id 匹配只是防御性写法（万一未来支持并发工具调用）。
        while True:
            resp = self.response_queue.get()  # 阻塞直到主进程处理完
            if resp.get("call_id") != call_id:
                self.response_queue.put(resp)  # 不是自己的响应，放回去给别人
                continue
            if resp.get("error"):
                raise RuntimeError(f"工具调用失败 [{tool_name}]: {resp['error']}")
            return resp.get("result")