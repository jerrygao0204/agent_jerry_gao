# agent/tool_transport.py
"""
CodeAct 工具调用传输层抽象。

设计目的：
    CodeAct 模式下，LLM 生成的 Python 代码需要像调用普通函数一样调用业务工具
    （如 search_knowledge_base(query="...")）。这些工具的真正执行逻辑（向量库检索、
    网络搜索等）必须留在"可信的主进程"里跑——因为它们持有数据库连接、LLM client
    等有状态资源，直接塞进 fork 出来的沙箱子进程会有连接卡死/状态损坏的风险。

    所以拆成两端：
        - ToolDispatcher（服务端）：永远在主进程里跑，负责"拿到 tool_name + kwargs
          之后，真正调用 tool_factory 执行"。不关心请求是怎么传过来的。
        - ToolTransportClient（客户端接口）：注入到沙箱子进程里的代理函数只认这个
          接口的 call()，不关心背后是进程内 IPC 队列，还是未来的 HTTP/gRPC 请求。

    今天只有 IPCQueueToolTransport 是真正实现（见 agent/transports/ipc_transport.py），
    HTTPToolTransport / GRPCToolTransport 是为以后拆分布式部署预留的骨架
    （见 agent/transports/http_transport.py、grpc_transport.py）。切换时只需要在
    组装 SandboxExecutor 的地方换一个 Transport 实现，ToolDispatcher 和业务代码不用动。
"""
import logging
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("ToolTransport")


class ToolDispatcher:
    """
    服务端调度器：真正执行工具调用的地方，永远跑在主进程（拥有 tool_factory / 数据库连接 / LLM client 的那个进程）。
    不关心请求是通过 IPC 队列、HTTP 还是 gRPC 传过来的。
    """

    def __init__(
        self,
        tool_factory: Any,
        user_role: Optional[str] = None,
        kwargs_preprocessor: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None,
        result_postprocessor: Optional[Callable[[str, Any], Any]] = None,
    ):
        """
        :param tool_factory: HierarchicalToolFactory 实例
        :param user_role: 当前用户角色，用于 RBAC 可见性校验
        :param kwargs_preprocessor: 可选的参数预处理钩子 (tool_name, raw_kwargs) -> final_kwargs
            用于保留旧版 _prepare_tool_kwargs 里"自动补全 top_k/top_k_rerank/filter"这类逻辑，
            避免在这里重复业务规则。
        """
        self.tool_factory = tool_factory
        self.user_role = user_role
        self.kwargs_preprocessor = kwargs_preprocessor
        self.result_postprocessor = result_postprocessor

    def dispatch(self, tool_name: str, kwargs: Dict[str, Any]) -> Any:
        """
        执行一次工具调用。抛出的异常会被 Transport 捕获并序列化为 error 字段传回沙箱侧，
        因此这里可以直接 raise，不需要自己 try/except 包一层。
        """
        final_kwargs = kwargs
        if self.kwargs_preprocessor:
            final_kwargs = self.kwargs_preprocessor(tool_name, kwargs)

        tool_obj = self.tool_factory.get_tool(tool_name, user_role=self.user_role)
        if tool_obj is None:
            raise ValueError(f"未找到工具 [{tool_name}]，或当前角色 [{self.user_role}] 无权限访问")

        if hasattr(tool_obj, "run"):
            result = tool_obj.run(**final_kwargs)
        elif hasattr(tool_obj, "execute"):
            result = tool_obj.execute(**final_kwargs)
        else:
            raise TypeError(f"工具 [{tool_name}] 既没有 run() 也没有 execute() 方法")

        if self.result_postprocessor:
            result = self.result_postprocessor(tool_name, result)
        return result


class ToolTransportClient(ABC):
    """
    客户端传输接口：注入到沙箱（或未来的远程 worker）里的代理函数只依赖这一个方法。
    """

    @abstractmethod
    def call(self, tool_name: str, kwargs: Dict[str, Any]) -> Any:
        """发起一次工具调用并阻塞等待结果，失败时应 raise。"""
        raise NotImplementedError


def build_tool_proxies(tool_names: List[str], transport: ToolTransportClient) -> Dict[str, Callable[..., Any]]:
    """
    把工具名列表转换成一组可调用的 Python 函数，供注入沙箱 global_vars。
    LLM 生成的代码里可以直接写 `search_knowledge_base(query="xxx")`，
    内部实际转发给 transport.call(...)，对上层完全透明。

    注意：这里生成的闭包函数在 fork 场景下可以正常工作（fork 直接复制内存，
    不需要 pickle）；如果未来改用 'spawn' 启动方式，闭包无法被 pickle，
    需要改成模块级可序列化的类实例（如 functools.partial 或自定义 __call__ 类）。
    """
    proxies: Dict[str, Callable[..., Any]] = {}
    for name in tool_names:
        proxies[name] = _make_tool_proxy(name, transport)
    return proxies


def _make_tool_proxy(tool_name: str, transport: ToolTransportClient) -> Callable[..., Any]:
    def _proxy(**kwargs) -> Any:
        return transport.call(tool_name, kwargs)

    _proxy.__name__ = tool_name
    _proxy.__doc__ = f"CodeAct 工具代理: 转发调用到主进程执行的 [{tool_name}]"
    return _proxy