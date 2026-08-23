# agent/sandbox.py
import os
import sys
import time
import logging
import multiprocessing
from typing import Dict, Any, Tuple, Optional

# 导入静态审查器
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from security import ASTCodeChecker

logger = logging.getLogger("SandboxExecutor")

def _isolated_execution_target(code_str: str, global_vars: Dict[str, Any], return_dict: Dict[str, Any]):
    """子进程独立执行目标函数"""
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
            }
        }
        if global_vars:
            safe_globals.update(global_vars)

        local_vars = {}
        exec(code_str, safe_globals, local_vars)

        final_res = local_vars.get("FINAL_RESULT", local_vars)
        return_dict["status"] = "success"
        return_dict["result"] = final_res
        return_dict["local_vars"] = {k: str(v) for k, v in local_vars.items()}
        return_dict["error"] = None
    except Exception as e:
        return_dict["status"] = "runtime_error"
        return_dict["result"] = None
        return_dict["error"] = f"运行时错误: {str(e)}"

class SandboxExecutor:
    """受限安全沙箱隔离执行器 (子进程强熔断版)"""

    def __init__(self, config_path: str = None, timeout: int = 10):
        self.checker = ASTCodeChecker(config_path=config_path)
        self.timeout = timeout

    def run(self, code_str: str, global_vars: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        沙箱安全执行入口
        :param code_str: 待执行 Python 代码
        :param global_vars: 注入的变量上下文
        :return: 包含 status, result, execution_time, error 等信息的字典
        """
        start_time = time.time()

        # 1. AST 静态安全审查
        is_safe, violations = self.checker.check_code(code_str)
        if not is_safe:
            logger.warning(f"🛡️ 代码被安全沙箱拦截: {violations}")
            return {
                "status": "security_blocked",
                "result": None,
                "error": f"安全审计未通过: {'; '.join(violations)}",
                "execution_time": round(time.time() - start_time, 4)
            }

        # 2. 独立子进程执行与超时强行 kill
        manager = multiprocessing.Manager()
        return_dict = manager.dict()

        process = multiprocessing.Process(
            target=_isolated_execution_target,
            args=(code_str, global_vars or {}, return_dict)
        )
        process.start()
        process.join(timeout=self.timeout)

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
                "execution_time": elapsed
            }

        elapsed = round(time.time() - start_time, 4)
        status = return_dict.get("status", "runtime_error")
        result = return_dict.get("result", None)
        error = return_dict.get("error", None)

        return {
            "status": status,
            "result": result,
            "error": error,
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