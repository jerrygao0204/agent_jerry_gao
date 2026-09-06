# agent/transports/grpc_transport.py
"""
[骨架/未实现] gRPC 工具调用传输 —— 预留给未来分布式部署使用（HTTP 方案的高性能替代）。

和 http_transport.py 解决的是同一个问题（远程沙箱 worker 如何把工具调用请求
传回主进程执行），区别只是传输协议换成 gRPC，适合调用频次高、对延迟更敏感的场景。

落地时还需要补的东西（当前均未实现）：
    1. 定义 .proto 文件（ToolCallRequest{tool_name, kwargs_json} / ToolCallResponse{result_json, error}），
       用 grpc_tools.protoc 生成 *_pb2.py / *_pb2_grpc.py 桩代码。
    2. 主进程侧要起一个 grpc.server()，把 agent.tool_transport.ToolDispatcher 包装成
       对应的 Servicer 实现。
    3. kwargs / result 目前设计里是任意 Python 对象，gRPC 场景下需要收敛到
       JSON 字符串或 protobuf Struct 这类可序列化格式。
    4. Channel 复用、超时、TLS/鉴权策略需要和 SandboxExecutor 的 timeout 预算对齐。
"""
from typing import Any, Dict, Optional

from tool_transport import ToolTransportClient


class GRPCToolTransport(ToolTransportClient):
    """占位实现。call() 抛 NotImplementedError，等 .proto 和生成的桩代码就绪后再补齐。"""

    def __init__(self, channel_target: str, timeout: float = 10.0, credentials: Optional[Any] = None):
        self.channel_target = channel_target
        self.timeout = timeout
        self.credentials = credentials

    def call(self, tool_name: str, kwargs: Dict[str, Any]) -> Any:
        raise NotImplementedError(
            "GRPCToolTransport 尚未实现：需要先定义 .proto 并生成桩代码，"
            "在主进程侧起 grpc.server() 包装 ToolDispatcher 后再启用。"
        )