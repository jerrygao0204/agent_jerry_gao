# agent/transports/http_transport.py
"""
[骨架/未实现] HTTP 工具调用传输 —— 预留给未来分布式部署使用。

使用场景：
    当沙箱不再是本机 fork 出来的子进程，而是跑在独立的远程 worker（比如另一台机器
    或独立容器）上时，IPCQueueToolTransport 依赖的 multiprocessing.Manager().Queue()
    就不可用了（Queue 代理只在同一批由该 Manager 派生的进程间有效）。
    这种情况下，注入到远程 worker 里的代理函数需要通过 HTTP 请求把
    "工具名 + 参数" 发回主进程（或一个专门的 Tool Server），由主进程执行真正的
    ToolDispatcher.dispatch() 后把结果序列化返回。

    这层 HTTP 调用发生在【可信的 harness 代码】里（即本文件），而不是 LLM 生成的
    代码里——LLM 生成的代码依然只能调用被注入的 Python 函数（如
    search_knowledge_base(query=...)），本身不允许 import requests/socket，
    AST 安全审查规则不需要因为这个改动。

落地时还需要补的东西（当前均未实现）：
    1. 主进程侧要起一个实际的 HTTP 服务（如 FastAPI），暴露一个
       POST /tool_call 端点，内部调用 ToolDispatcher.dispatch()。
    2. 鉴权：远程 worker 和主进程之间需要一个共享密钥/短期 token，防止被冒用调用工具。
    3. 超时与重试策略需要和 SandboxExecutor 的整体 timeout 预算对齐。
    4. 序列化边界：kwargs / result 必须是 JSON 可序列化的，复杂对象需要额外约定协议。
"""
from typing import Any, Dict, Optional

from tool_transport import ToolTransportClient


class HTTPToolTransport(ToolTransportClient):
    """
    占位实现。构造函数先把将来会用到的配置项定义出来，方便按需接入；
    call() 抛 NotImplementedError，等真正需要拆分布式部署时再实现。
    """

    def __init__(self, endpoint_url: str, auth_token: Optional[str] = None, timeout: float = 10.0):
        self.endpoint_url = endpoint_url
        self.auth_token = auth_token
        self.timeout = timeout

    def call(self, tool_name: str, kwargs: Dict[str, Any]) -> Any:
        raise NotImplementedError(
            "HTTPToolTransport 尚未实现：需要先在主进程侧起一个 /tool_call HTTP 服务 "
            "（包装 agent.tool_transport.ToolDispatcher），并补充鉴权与超时策略后再启用。"
        )