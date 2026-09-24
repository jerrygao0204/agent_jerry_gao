# agent/sandbox.py
import os
import sys
import io
import time
import uuid
import logging
import threading
import contextlib
import multiprocessing
from typing import Dict, Any, List, Optional

# 兼容两种导入方式：作为包 (agent.sandbox) 与直接运行 agent/sandbox.py 自测
try:
    from agent.security import ASTCodeChecker
    from agent.tool_transport import ToolDispatcher, build_tool_proxies
    from agent.transports.ipc_transport import IPCPipeToolTransport
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    from security import ASTCodeChecker
    from tool_transport import ToolDispatcher, build_tool_proxies
    from transports.ipc_transport import IPCPipeToolTransport

logger = logging.getLogger("SandboxExecutor")

# =============================================================================
# 进程级并发控制
# =============================================================================
# 💡 [子进程启动方式]: 显式使用 fork，避免 Python 3.14 起 Linux 默认改为 forkserver 后，
# 因为要重新 import 入口脚本(qa_admin.py 等)而带来的副作用。可用环境变量覆盖。
_start_method = os.environ.get("SANDBOX_START_METHOD", "spawn")
if _start_method not in multiprocessing.get_all_start_methods():
    _start_method = "spawn"
_MP_CONTEXT = multiprocessing.get_context(_start_method)

# 💡 [并发上限]: 同时存活的沙箱子进程数。不设上限时，N 个用户同时执行代码会同时 fork N 个进程，
# CPU 被摊薄后，每个"2 秒超时"的沙箱都可能被饿死而误判超时（好代码被判超时，且谁碰上谁倒霉）。
# 超出上限的请求在这里排队，排队时间不计入该次执行的超时。
_MAX_CONCURRENCY = int(os.environ.get("SANDBOX_MAX_CONCURRENCY", "0")) or min(16, max(4, (os.cpu_count() or 2) * 2))
_SLOTS = threading.BoundedSemaphore(_MAX_CONCURRENCY)

_SAFE_BUILTINS: Dict[str, Any] = {
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


class _CappedStringIO(io.StringIO):
    """带上限的 stdout 缓冲：防止一段 print 死循环把子进程/主进程内存撑爆。"""

    def __init__(self, limit: int):
        super().__init__()
        self._limit = limit
        self._written = 0
        self.truncated = False

    def write(self, s: str) -> int:
        n = len(s)
        room = self._limit - self._written
        if room <= 0:
            self.truncated = True
            return n
        if n > room:
            s = s[:room]
            self.truncated = True
        self._written += len(s)
        super().write(s)
        return n


def _apply_memory_limit(memory_limit_mb: Optional[int]) -> None:
    """限制子进程可用地址空间（仅 Linux/macOS）。防止某个用户的代码吃光整机内存拖垮所有人。"""
    if not memory_limit_mb:
        return
    try:
        import resource
        limit = int(memory_limit_mb) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    except Exception as e:  # 平台不支持就忽略，不影响执行
        logger.warning(f"设置子进程内存上限失败，已忽略: {e}")


def _send_final(conn, msg: Dict[str, Any]) -> None:
    """把最终结果送回主进程；result 无法序列化时降级为 str，保证主进程一定能收到一条结果。"""
    try:
        conn.send(msg)  # send() 先 pickle 再写管道，序列化失败不会写出半截数据
        return
    except Exception as e:
        msg = dict(msg)
        msg["result"] = str(msg.get("result"))
        msg["status"] = msg.get("status") or "success"
        logger.debug(f"结果无法序列化，已降级为字符串: {e}")
    try:
        conn.send(msg)
    except Exception:
        pass


def _isolated_execution_target(
    code_str: str,
    global_vars: Dict[str, Any],
    conn,
    parent_conn,
    max_output_chars: int,
    memory_limit_mb: Optional[int],
):
    """子进程独立执行目标函数。结果通过该次执行专属的 Pipe 回传，不使用任何跨请求共享的容器。"""
    # 子进程只需要自己这一端；把继承来的主进程端关掉，主进程异常退出时子进程能立刻感知 EOF
    try:
        parent_conn.close()
    except Exception:
        pass

    # 💡 [启动握手]: 告诉主进程"子进程已经跑起来了"。多线程进程里 fork 可能让子进程继承到别的线程
    # 持有的锁而卡死在启动阶段；主进程收不到握手就能判定"启动失败"并重试，而不是把它误报成"代码超时"。
    try:
        conn.send({"type": "started"})
    except Exception:
        return

    # 💡 [stdout 捕获]: 用 StringIO 接管子进程内的 print() 输出，作为 CodeAct 的 Observation 来源
    stdout_buffer = _CappedStringIO(max_output_chars)
    msg: Dict[str, Any]
    try:
        _apply_memory_limit(memory_limit_mb)

        # 构建安全受限的环境作用域（每次执行独享一份 builtins 字典）
        safe_globals = {"__builtins__": dict(_SAFE_BUILTINS)}
        if global_vars:
            safe_globals.update(global_vars)

        local_vars: Dict[str, Any] = {}
        with contextlib.redirect_stdout(stdout_buffer):
            exec(code_str, safe_globals, local_vars)

        msg = {
            "type": "final",
            "status": "success",
            "result": local_vars.get("FINAL_RESULT", local_vars),
            "error": None,
        }
    except BaseException as e:
        # 即便运行时报错，也要把报错前已经打印出来的内容一并带回，方便 LLM 定位问题
        msg = {
            "type": "final",
            "status": "runtime_error",
            "result": None,
            "error": f"运行时错误: {str(e)}",
        }

    out = stdout_buffer.getvalue()
    if stdout_buffer.truncated:
        out += f"\n...[输出超过 {max_output_chars} 字符，已截断]"
    msg["stdout"] = out
    _send_final(conn, msg)
    try:
        conn.close()
    except Exception:
        pass


def _call_in_daemon_thread(fn, timeout: float):
    """
    在独立的守护线程里执行 fn，最多等 timeout 秒。
    返回 (finished, result, exception)。

    为什么不用线程池：线程池的 worker 卡死后，后续任务会排在它后面一起卡住，
    且 `with ThreadPoolExecutor` 退出时会等它结束——这会把"超时"变成"无限等待"，
    并且延后对沙箱子进程的 kill。每次调用独占一个守护线程则互不影响，
    卡死的线程也不会阻止进程退出。
    """
    box: Dict[str, Any] = {}

    def _target():
        try:
            box["result"] = fn()
        except BaseException as e:  # noqa: BLE001 - 需要把任何异常带回
            box["error"] = e

    t = threading.Thread(target=_target, daemon=True, name="sandbox-tool-call")
    t.start()
    t.join(timeout)
    if t.is_alive():
        return False, None, None
    return True, box.get("result"), box.get("error")


def _terminate_process(process) -> None:
    """先 terminate，仍未退出再 kill，最后 join 回收，避免僵尸进程。"""
    try:
        if process.is_alive():
            process.terminate()
            process.join(1.0)
            if process.is_alive():
                process.kill()
        process.join(2.0)
    except Exception as e:
        logger.warning(f"回收沙箱子进程时出错: {e}")


class SandboxExecutor:
    """受限安全沙箱隔离执行器 (子进程强熔断版，多用户并发安全)"""

    def __init__(
        self,
        config_path: str = None,
        timeout: int = 10,
        tool_timeout: int = 15,
        max_tool_calls: int = 20,
        max_output_chars: int = 20000,
        memory_limit_mb: Optional[int] = None,
        queue_timeout: float = 30.0,
        startup_timeout: float = 5.0,
    ):
        """
        :param timeout: 代码自身(CPU)执行时限，工具调用耗时不计入
        :param tool_timeout: 单次工具调用的最大等待超时
        :param max_tool_calls: 单次 run 内允许的工具调用次数上限。
            工具耗时会顺延 deadline，如果不限次数，`while True: tool()` 之类的代码永远不会超时。
        :param max_output_chars: stdout 回传上限
        :param memory_limit_mb: 子进程内存上限(MB)，None 表示不限制
        :param queue_timeout: 并发已满时最多排队等待多久(秒)，超过返回 status="busy"
        :param startup_timeout: 等待子进程启动握手的上限(秒)。超时会杀掉并自动重试一次；
            代码执行的 timeout 从握手之后才开始计时，启动耗时不算在内
        """
        self.checker = ASTCodeChecker(config_path=config_path)
        self.timeout = timeout
        self.tool_timeout = tool_timeout  # 单次工具调用的最大等待超时
        self.max_tool_calls = max_tool_calls
        self.max_output_chars = max_output_chars
        self.memory_limit_mb = memory_limit_mb
        self.queue_timeout = queue_timeout
        self.startup_timeout = startup_timeout

    # ------------------------------------------------------------------
    def run(
        self,
        code_str: str,
        global_vars: Optional[Dict[str, Any]] = None,
        tool_names: Optional[List[str]] = None,
        tool_dispatcher: Optional[ToolDispatcher] = None,
    ) -> Dict[str, Any]:
        """
        沙箱安全执行入口（线程安全：可被多个线程/多个用户同时调用）
        :param code_str: 待执行 Python 代码
        :param global_vars: 注入的变量上下文
        :param tool_names: 本轮允许调用的工具名列表（如 ["search_knowledge_base"]）。
            传入后会自动生成对应的可调用代理函数注入沙箱，LLM 代码里可以直接
            `search_knowledge_base(query="...")` 调用。
        :param tool_dispatcher: 负责真正执行工具的调度器（跑在主进程），
            与 tool_names 配套使用；不传则不启用工具调用能力。
        :return: 包含 status, result, execution_time, error 等信息的字典
        """
        t_enter = time.monotonic()

        # 1. AST 静态安全审查（checker 内部无共享可变状态）
        is_safe, violations, warnings = self.checker.check_code(code_str)
        if not is_safe:
            logger.warning(f"🛡️ 代码被安全沙箱拦截: {violations}")
            return {
                "status": "security_blocked",
                "result": None,
                "error": f"安全审计未通过: {'; '.join(violations)}",
                "stdout": "",
                "warnings": [],
                "execution_time": round(time.monotonic() - t_enter, 4),
            }

        # 2. 申请并发名额（排队时间不计入代码执行时限）
        if not _SLOTS.acquire(timeout=self.queue_timeout):
            logger.error(f"沙箱并发已满 (上限 {_MAX_CONCURRENCY})，排队超过 {self.queue_timeout}s")
            return {
                "status": "busy",
                "result": None,
                "error": f"沙箱当前并发已满(上限 {_MAX_CONCURRENCY})，请稍后重试",
                "stdout": "",
                "warnings": warnings,
                "execution_time": round(time.monotonic() - t_enter, 4),
            }
        try:
            return self._run_in_subprocess(code_str, global_vars, tool_names, tool_dispatcher, warnings)
        finally:
            _SLOTS.release()

    # ------------------------------------------------------------------
    def _run_in_subprocess(
        self,
        code_str: str,
        global_vars: Optional[Dict[str, Any]],
        tool_names: Optional[List[str]],
        tool_dispatcher: Optional[ToolDispatcher],
        warnings: List[str],
    ) -> Dict[str, Any]:
        run_id = uuid.uuid4().hex[:8]
        start_time = time.monotonic()

        allowed_tools = set(tool_names or [])
        tools_enabled = bool(allowed_tools) and tool_dispatcher is not None

        process, parent_conn = self._launch_child(run_id, code_str, global_vars, allowed_tools, tools_enabled)
        if process is None:
            return {
                "status": "runtime_error",
                "result": None,
                "error": "沙箱子进程启动失败（已重试），请稍后重试",
                "stdout": "",
                "warnings": warnings,
                "execution_time": round(time.monotonic() - start_time, 4),
            }

        final: Optional[Dict[str, Any]] = None
        timed_out = False
        tool_calls = 0

        try:
            # 3. 动态 Deadline 轮询（带工具执行时间补偿与卡死隔离）；子进程已握手，从现在开始计时
            deadline = time.monotonic() + self.timeout
            poll_interval = 0.05

            while final is None:
                now = time.monotonic()
                expired = now >= deadline

                # 💡 用 poll(timeout) 阻塞等待数据，而不是空转；到期后仍做一次 poll(0)，
                # 避免结果恰好在 deadline 边缘到达却被误判为超时。
                try:
                    ready = parent_conn.poll(0 if expired else min(poll_interval, deadline - now))
                except (OSError, EOFError):
                    break

                if ready:
                    try:
                        msg = parent_conn.recv()
                    except (EOFError, OSError):
                        break  # 子进程已关闭通道且没有留下结果
                    except Exception as e:
                        logger.error(f"[{run_id}] 无法解析子进程消息: {e}")
                        break

                    mtype = msg.get("type")
                    if mtype == "final":
                        final = msg
                    elif mtype == "tool_call":
                        tool_calls += 1
                        deadline = self._serve_tool_call(
                            run_id, parent_conn, msg, tool_dispatcher, allowed_tools,
                            tool_calls, deadline,
                        )
                    continue

                if expired:
                    timed_out = True
                    break

                if not process.is_alive():
                    # 子进程已退出：把管道里可能残留的最终结果读干净后再判定
                    while final is None and parent_conn.poll(0):
                        try:
                            m = parent_conn.recv()
                        except Exception:
                            break
                        if m.get("type") == "final":
                            final = m
                    break
        finally:
            # 无论正常/超时/主进程异常，都保证子进程被回收、管道被关闭
            _terminate_process(process)
            try:
                parent_conn.close()
            except Exception:
                pass

        elapsed = round(time.monotonic() - start_time, 4)

        if final is None and timed_out:
            logger.error(f"⏱️ [{run_id}] 沙箱代码执行超时，已强行杀死子进程 (耗时 {elapsed}s)")
            return {
                "status": "timeout",
                "result": None,
                "error": f"代码执行超时 ({self.timeout}秒强熔断)",
                # 💡 子进程被强行 kill，缓冲区内容无法跨进程边界同步，只能返回空
                "stdout": "",
                "warnings": warnings,
                "execution_time": elapsed,
            }

        if final is None:
            logger.error(f"[{run_id}] 沙箱子进程异常退出且无结果 (exitcode={process.exitcode})")
            return {
                "status": "runtime_error",
                "result": None,
                "error": f"沙箱子进程异常退出 (exitcode={process.exitcode})，可能触发了内存上限或被系统终止",
                "stdout": "",
                "warnings": warnings,
                "execution_time": elapsed,
            }

        return {
            "status": final.get("status", "runtime_error"),
            "result": final.get("result"),
            "error": final.get("error"),
            "stdout": final.get("stdout", ""),
            "warnings": warnings,
            "execution_time": elapsed,
        }

    # ------------------------------------------------------------------
    def _launch_child(self, run_id, code_str, global_vars, allowed_tools, tools_enabled):
        """
        启动沙箱子进程并等待启动握手；握手超时则杀掉重试一次。
        返回 (process, parent_conn)，两次都失败返回 (None, None)。
        """
        for attempt in (1, 2):
            # 💡 [专属通道]: 每次执行一条私有 Pipe。不同用户/请求之间不共享任何队列或字典，
            # 结果与工具响应不可能串到别人的执行里。
            parent_conn, child_conn = _MP_CONTEXT.Pipe(duplex=True)

            merged_globals = dict(global_vars or {})
            if tools_enabled:
                transport = IPCPipeToolTransport(child_conn, response_timeout=self.tool_timeout + 5)
                merged_globals.update(build_tool_proxies(list(allowed_tools), transport))

            process = _MP_CONTEXT.Process(
                target=_isolated_execution_target,
                args=(code_str, merged_globals, child_conn, parent_conn, self.max_output_chars, self.memory_limit_mb),
                daemon=True,  # 主进程退出时不会遗留孤儿沙箱
            )
            started = False
            try:
                process.start()
                child_conn.close()  # 主进程只保留自己这一端
                deadline = time.monotonic() + self.startup_timeout
                while not started:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not parent_conn.poll(remaining):
                        break
                    try:
                        msg = parent_conn.recv()
                    except (EOFError, OSError):
                        break
                    if msg.get("type") == "started":
                        started = True
            except Exception as e:
                logger.error(f"[{run_id}] 启动沙箱子进程异常 (第 {attempt} 次): {e}")

            if started:
                return process, parent_conn

            logger.error(f"[{run_id}] 沙箱子进程未在 {self.startup_timeout}s 内完成启动握手 (第 {attempt}/2 次)，已终止")
            _terminate_process(process)
            for c in (parent_conn, child_conn):
                try:
                    c.close()
                except Exception:
                    pass
        return None, None

    # ------------------------------------------------------------------
    def _serve_tool_call(
        self,
        run_id: str,
        conn,
        req: Dict[str, Any],
        tool_dispatcher: Optional[ToolDispatcher],
        allowed_tools: set,
        tool_calls: int,
        deadline: float,
    ) -> float:
        """在主进程执行一次工具调用并把结果写回该次执行的专属通道。返回顺延后的 deadline。"""
        call_id = req.get("call_id")
        tool_name = req.get("tool_name")
        result: Any = None
        error: Optional[str] = None
        is_tool_timeout = False
        tool_start = time.monotonic()

        if tool_dispatcher is None or tool_name not in allowed_tools:
            # 纵深防御：即使子进程伪造了请求，也只能调用本轮白名单里的工具
            error = f"工具 [{tool_name}] 不在本轮允许的工具列表中"
        elif tool_calls > self.max_tool_calls:
            error = f"工具调用次数超过上限 ({self.max_tool_calls})"
        else:
            kwargs = req.get("kwargs", {}) or {}
            finished, result, exc = _call_in_daemon_thread(
                lambda: tool_dispatcher.dispatch(tool_name, kwargs), self.tool_timeout
            )
            if not finished:
                is_tool_timeout = True
                error = f"工具 [{tool_name}] 执行超时 ({self.tool_timeout}s)"
                logger.warning(f"⚙️ [{run_id}] {error}")
            elif exc is not None:
                error = str(exc)
                logger.warning(f"⚙️ [{run_id}] 工具调用 [{tool_name}] 执行失败: {exc}")

        payload = {"type": "tool_result", "call_id": call_id, "result": result if error is None else None, "error": error}
        try:
            conn.send(payload)
        except (TypeError, AttributeError, ValueError) as e:
            # 工具返回值无法序列化：如实告诉沙箱代码，而不是让它一直等
            try:
                conn.send({"type": "tool_result", "call_id": call_id, "result": None,
                           "error": f"工具 [{tool_name}] 的返回值无法序列化: {e}"})
            except Exception:
                pass
        except (OSError, EOFError):
            pass  # 子进程已经没了

        now = time.monotonic()
        duration = now - tool_start
        if is_tool_timeout:
            # 工具卡死不顺延 CPU deadline；只给子进程一小段时间处理这个错误并收尾
            deadline = max(deadline, now + 1.0)
        else:
            # 💡 [关键修复]: 仅在工具正常响应（未触发 tool_timeout 卡死）时顺延 CPU deadline
            deadline += duration
            logger.info(f"⏱️ [{run_id}] 工具 [{tool_name}] 耗时 {round(duration, 2)}s，沙箱 Deadline 顺延。")
        return deadline


if __name__ == "__main__":
    import time

    # 初始化基础沙箱：CPU 代码逻辑超时 2s，单次工具执行超时 2s
    sandbox = SandboxExecutor(timeout=2, tool_timeout=2)

    print("\n--- 1. 拦截测试 ---")
    res1 = sandbox.run("import os; os.system('echo hack')")
    print(res1)
    assert res1["status"] == "security_blocked", "拦截测试应触发 security_blocked"

    print("\n--- 2. 死循环超时熔断测试 ---")
    res2 = sandbox.run("while True: pass")
    print(res2)
    assert res2["status"] == "timeout", "死循环应触发 timeout 熔断"

    print("\n--- 3. 正常计算测试 ---")
    res3 = sandbox.run("import math\na = 10\nb = 20\nFINAL_RESULT = math.sqrt(a + b)")
    print(res3)
    assert res3["status"] == "success", "正常计算测试应成功"

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
            return f"【模拟检索结果】关于 '{query}' 的 Top-{top_k} 条文档"

    class _MockToolFactory:
        def get_tool(self, name, user_role=None):
            if name == "search_knowledge_base":
                return _MockKnowledgeTool()
            return None

    dispatcher = ToolDispatcher(tool_factory=_MockToolFactory(), user_role="analyst")
    # 显式配置 tool_timeout=5，确保 IPC 响应窗口充裕，防止卡死
    sandbox_bridge = SandboxExecutor(timeout=2, tool_timeout=5)
    res7 = sandbox_bridge.run(
        "res = search_knowledge_base(query='怎么创建预警用户', top_k=3)\n"
        "print(res)\n"
        "FINAL_RESULT = res",
        tool_names=["search_knowledge_base"],
        tool_dispatcher=dispatcher,
    )
    print(res7)
    assert res7["status"] == "success", "工具调用桥应能正常执行"
    assert "怎么创建预警用户" in res7["result"] and "Top-3" in res7["result"], "工具调用结果应正确透传回沙箱"
    print("✅ CodeAct 工具调用桥测试通过")

    print("\n--- 8. 工具调用桥：工具执行报错应正确透传 ---")

    class _FailingTool:
        def run(self, **kwargs):
            raise ValueError("向量库连接超时")

    class _FailingToolFactory:
        def get_tool(self, name, user_role=None):
            return _FailingTool()

    dispatcher2 = ToolDispatcher(tool_factory=_FailingToolFactory())
    sandbox_failing = SandboxExecutor(timeout=2, tool_timeout=5)
    res8 = sandbox_failing.run(
        "search_knowledge_base(query='任意问题')",
        tool_names=["search_knowledge_base"],
        tool_dispatcher=dispatcher2,
    )
    print(res8)
    assert res8["status"] == "runtime_error" and "向量库连接超时" in res8["error"], "工具执行异常应正确透传回沙箱侧"
    print("✅ 工具调用报错透传测试通过")

    print("\n--- 9. 慢工具耗时补偿测试 (工具耗时 3s > 沙箱 timeout 2s) ---")

    class _SlowTool:
        def run(self, **kwargs):
            time.sleep(3)  # 工具耗时 3s
            return "慢工具执行完毕"

    class _SlowToolFactory:
        def get_tool(self, name, user_role=None):
            return _SlowTool()

    dispatcher3 = ToolDispatcher(tool_factory=_SlowToolFactory())
    
    # 💡 重新实例化沙箱：CPU 代码超时 2s，但允许单次工具执行最多 5s
    sandbox_slow = SandboxExecutor(timeout=2, tool_timeout=5)
    res9 = sandbox_slow.run(
        "res = search_knowledge_base(query='慢查询')\n"
        "print(res)\n"
        "FINAL_RESULT = res",
        tool_names=["search_knowledge_base"],
        tool_dispatcher=dispatcher3,
    )
    print(res9)
    assert res9["status"] == "success" and "慢工具执行完毕" in res9["result"], "工具时间补偿失败，沙箱误杀慢工具！"
    print("✅ 慢工具时间补偿测试通过：工具耗时 3s 未触发沙箱误杀，Deadline 顺延生效")

    print("\n--- 10. 工具卡死隔离测试 (工具耗时 5s > tool_timeout 2s) ---")

    class _BlockedTool:
        def run(self, **kwargs):
            time.sleep(5)  # 模拟网络卡死 5s
            return "不应该返回"

    class _BlockedToolFactory:
        def get_tool(self, name, user_role=None):
            return _BlockedTool()

    dispatcher4 = ToolDispatcher(tool_factory=_BlockedToolFactory())
    
    # 💡 重新实例化沙箱：单次工具执行限制 2s
    sandbox_blocked = SandboxExecutor(timeout=2, tool_timeout=2)
    res10 = sandbox_blocked.run(
        "search_knowledge_base(query='卡死查询')",
        tool_names=["search_knowledge_base"],
        tool_dispatcher=dispatcher4,
    )
    print(res10)
    assert res10["status"] == "runtime_error" and "执行超时" in res10["error"], "工具卡死隔离失败！"
    print("✅ 工具卡死隔离测试通过：工具单次超时触发，并把错误透传给沙箱")