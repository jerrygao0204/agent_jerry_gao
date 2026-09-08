# agent/sandbox.py
import os
import sys
import io
import time
import queue
import logging
import contextlib
import multiprocessing
from typing import Dict, Any, List, Tuple, Optional

# 导入静态审查器
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from security import ASTCodeChecker
from tool_transport import ToolDispatcher, build_tool_proxies
from transports.ipc_transport import IPCQueueToolTransport

logger = logging.getLogger("SandboxExecutor")

def _isolated_execution_target(code_str: str, global_vars: Dict[str, Any], return_dict: Dict[str, Any]):
    """子进程独立执行目标函数"""
    # 💡 [stdout 捕获]: 用 StringIO 接管子进程内的 print() 输出，作为 CodeAct 的 Observation 来源
    stdout_buffer = io.StringIO()
    try:
        # 构建安全受限的环境作用域
        safe_globals = {
            "__builtins__": {
                "__import__": __import__,  # 允许 Python 底层执行 import 语句（已被 AST 审查白名单保护）
                "print": print,
                "range": range,
                "len": len,
                "int": int,
                "float": float,
                "str": str,
                "list": list,
                "dict": dict,
                "set": set,
                "tuple": tuple,
                "bool": bool,
                "sum": sum,
                "max": max,
                "min": min,
                "abs": abs,
                "round": round,
                "enumerate": enumerate,
                "zip": zip,
                "isinstance": isinstance,
                # 💡 [异常类白名单]: 允许沙箱代码正常 raise/except 常见内置异常，
                # 否则代码里一 raise 就先炸在 "异常类未定义" 上，掩盖了真实的业务异常信息。
                "Exception": Exception,
                "ValueError": ValueError,
                "TypeError": TypeError,
                "KeyError": KeyError,
                "IndexError": IndexError,
                "AttributeError": AttributeError,
                "ZeroDivisionError": ZeroDivisionError,
                "RuntimeError": RuntimeError,
                "StopIteration": StopIteration,
                "NameError": NameError,
                "ArithmeticError": ArithmeticError,
                "LookupError": LookupError,
            }
        }
        if global_vars:
            safe_globals.update(global_vars)

        local_vars = {}
        with contextlib.redirect_stdout(stdout_buffer):
            exec(code_str, safe_globals, local_vars)

        final_res = local_vars.get("FINAL_RESULT", local_vars)
        return_dict["status"] = "success"
        return_dict["result"] = final_res
        return_dict["local_vars"] = {k: str(v) for k, v in local_vars.items()}
        return_dict["error"] = None
        return_dict["stdout"] = stdout_buffer.getvalue()
    except Exception as e:
        # 即便运行时报错，也要把报错前已经打印出来的内容一并带回，方便 LLM 定位问题
        return_dict["status"] = "runtime_error"
        return_dict["result"] = None
        return_dict["error"] = f"运行时错误: {str(e)}"
        return_dict["stdout"] = stdout_buffer.getvalue()

class SandboxExecutor:
    """受限安全沙箱隔离执行器 (子进程强熔断版)"""

    def __init__(self, config_path: str = None, timeout: int = 10):
        self.checker = ASTCodeChecker(config_path=config_path)
        self.timeout = timeout

    def run(
        self,
        code_str: str,
        global_vars: Optional[Dict[str, Any]] = None,
        tool_names: Optional[List[str]] = None,
        tool_dispatcher: Optional[ToolDispatcher] = None,
    ) -> Dict[str, Any]:
        """
        沙箱安全执行入口
        :param code_str: 待执行 Python 代码
        :param global_vars: 注入的变量上下文
        :param tool_names: 本轮允许调用的工具名列表（如 ["search_knowledge_base"]）。
            传入后会自动生成对应的可调用代理函数注入沙箱，LLM 代码里可以直接
            `search_knowledge_base(query="...")` 调用。
        :param tool_dispatcher: 负责真正执行工具的调度器（跑在主进程），
            与 tool_names 配套使用；不传则不启用工具调用能力。
        :return: 包含 status, result, execution_time, error 等信息的字典
        """
        start_time = time.time()

        # 1. AST 静态安全审查
        is_safe, violations, warnings = self.checker.check_code(code_str)
        if not is_safe:
            logger.warning(f"🛡️ 代码被安全沙箱拦截: {violations}")
            return {
                "status": "security_blocked",
                "result": None,
                "error": f"安全审计未通过: {'; '.join(violations)}",
                "stdout": "",
                "warnings": [],
                "execution_time": round(time.time() - start_time, 4)
            }

        # 2. 独立子进程执行与超时强行 kill
        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        # 💡 [工具调用桥]: 有工具需要暴露时，建一对请求/响应队列，
        # 把工具代理函数注入沙箱 global_vars；真正的工具执行留在本方法所在的主进程完成。
        merged_globals = dict(global_vars or {})
        request_queue = None
        response_queue = None
        if tool_names and tool_dispatcher is not None:
            request_queue = manager.Queue()
            response_queue = manager.Queue()
            transport = IPCQueueToolTransport(request_queue, response_queue)
            merged_globals.update(build_tool_proxies(tool_names, transport))

        process = multiprocessing.Process(
            target=_isolated_execution_target,
            args=(code_str, merged_globals, return_dict)
        )
        process.start()

        # 3. 轮询：在超时预算内，一边把子进程发来的工具调用请求转发给 ToolDispatcher 执行，
        #    一边检查子进程是否已经跑完。
        deadline = start_time + self.timeout
        poll_interval = 0.05
        while True:
            remaining = deadline - time.time()
            if remaining <= 0:
                break

            if request_queue is not None:
                try:
                    req = request_queue.get(timeout=min(poll_interval, max(remaining, 0.001)))
                except queue.Empty:
                    req = None
                if req is not None:
                    try:
                        result = tool_dispatcher.dispatch(req["tool_name"], req["kwargs"])
                        response_queue.put({"call_id": req["call_id"], "result": result, "error": None})
                    except Exception as tool_err:
                        logger.warning(f"⚙️ 工具调用 [{req.get('tool_name')}] 执行失败: {tool_err}")
                        response_queue.put({"call_id": req["call_id"], "result": None, "error": str(tool_err)})
            else:
                process.join(timeout=min(poll_interval, max(remaining, 0.001)))

            if not process.is_alive():
                process.join()
                break

        # 判定是否超时
        if process.is_alive():
            process.terminate()  # 强行终止卡死的子进程
            process.join()
            elapsed = round(time.time() - start_time, 4)
            logger.error(f"⏱️ 沙箱代码执行超时，已强行杀死子进程 (耗时 {elapsed}s)")
            return {
                "status": "timeout",
                "result": None,
                "error": f"代码执行超时 ({self.timeout}秒强熔断)",
                # 💡 子进程被强行 kill，缓冲区内容无法跨进程边界同步，只能返回空
                "stdout": "",
                "warnings": warnings,
                "execution_time": elapsed
            }

        elapsed = round(time.time() - start_time, 4)
        status = return_dict.get("status", "runtime_error")
        result = return_dict.get("result", None)
        error = return_dict.get("error", None)
        stdout = return_dict.get("stdout", "")

        return {
            "status": status,
            "result": result,
            "error": error,
            "stdout": stdout,
            "warnings": warnings,
            "execution_time": elapsed
        }

if __name__ == "__main__":
    sandbox = SandboxExecutor(timeout=2)

    print("\n--- 1. 拦截测试 ---")
    res1 = sandbox.run("import os; os.system('echo hack')")
    print(res1)

    print("\n--- 2. 死循环超时熔断测试 ---")
    res2 = sandbox.run("while True: pass")
    print(res2)

    print("\n--- 3. 正常计算测试 ---")
    res3 = sandbox.run("import math\na = 10\nb = 20\nFINAL_RESULT = math.sqrt(a + b)")
    print(res3)

    print("\n--- 4. stdout 捕获测试 (CodeAct Observation 来源) ---")
    res4 = sandbox.run("print('正在计算...')\nx = 3 * 7\nprint(f'x = {x}')\nFINAL_RESULT = x")
    print(res4)
    assert "正在计算" in res4["stdout"] and "x = 21" in res4["stdout"], "stdout 捕获失败"
    print("✅ stdout 捕获断言通过")

    print("\n--- 5. 运行时报错也应带回已打印的 stdout ---")
    res5 = sandbox.run("print('开始执行')\nraise ValueError('模拟异常')")
    print(res5)
    assert "开始执行" in res5["stdout"] and res5["status"] == "runtime_error", "报错分支 stdout 捕获失败"
    assert "模拟异常" in res5["error"], "raise 的内置异常类应能被正常识别，而不是报 NameError"
    print("✅ 报错分支 stdout 捕获断言通过，且异常类白名单生效")

    print("\n--- 6. 无 print() 观测点应触发非阻断 warning ---")
    res6 = sandbox.run("FINAL_RESULT = 1 + 1")
    print(res6)
    assert res6["status"] == "success" and len(res6["warnings"]) == 1, "无 print 场景应正常执行并带一条 warning"
    print("✅ 观测点 warning 断言通过")

    print("\n--- 7. CodeAct 工具调用桥测试 (IPC Queue Transport) ---")

    class _MockKnowledgeTool:
        def run(self, query: str, top_k: int = 5, **kwargs):
            return f"【模拟检索结果】关于 '{query}' 的 Top-{top_k} 条 FineBI 文档"

    class _MockToolFactory:
        def get_tool(self, name, user_role=None):
            if name == "search_knowledge_base":
                return _MockKnowledgeTool()
            return None

    dispatcher = ToolDispatcher(tool_factory=_MockToolFactory(), user_role="analyst")
    res7 = sandbox.run(
        "res = search_knowledge_base(query='怎么创建预警用户', top_k=3)\n"
        "print(res)\n"
        "FINAL_RESULT = res",
        tool_names=["search_knowledge_base"],
        tool_dispatcher=dispatcher,
    )
    print(res7)
    assert res7["status"] == "success", "工具调用桥应能正常执行"
    assert "怎么创建预警用户" in res7["result"] and "Top-3" in res7["result"], "工具调用结果应正确透传回沙箱"
    print("✅ CodeAct 工具调用桥测试通过：主进程真正执行了工具，沙箱侧像调用普通函数一样拿到结果")

    print("\n--- 8. 工具调用桥：工具执行报错应正确透传 ---")

    class _FailingTool:
        def run(self, **kwargs):
            raise ValueError("向量库连接超时")

    class _FailingToolFactory:
        def get_tool(self, name, user_role=None):
            return _FailingTool()

    dispatcher2 = ToolDispatcher(tool_factory=_FailingToolFactory())
    res8 = sandbox.run(
        "search_knowledge_base(query='任意问题')",
        tool_names=["search_knowledge_base"],
        tool_dispatcher=dispatcher2,
    )
    print(res8)
    assert res8["status"] == "runtime_error" and "向量库连接超时" in res8["error"], "工具执行异常应正确透传回沙箱侧"
    print("✅ 工具调用报错透传测试通过")
