# agent/security.py
import ast
import os
import sys
import yaml
import logging
from typing import List, Dict, Any, Tuple, Set

logger = logging.getLogger("ASTCodeChecker")

class ASTCodeChecker(ast.NodeVisitor):
    """基于 AST 静态语法树的代码安全审查器"""

    def __init__(self, config_path: str = None):
        if config_path is None:
            # 默认指向 config/patterns.yaml
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            config_path = os.path.join(project_root, "config", "patterns.yaml")

        self.allowed_imports: Set[str] = set()
        self.forbidden_calls: Set[str] = set()
        self.forbidden_attrs: Set[str] = set()
        
        self._load_config(config_path)
        self.violations: List[str] = []

    def _load_config(self, config_path: str):
        """动态读取 patterns.yaml 中的安全配置规则"""
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                sec_cfg = cfg.get("security_sandbox", {})
                self.allowed_imports = set(sec_cfg.get("allowed_ast_imports", []))
                self.forbidden_calls = set(sec_cfg.get("forbidden_ast_calls", []))
                self.forbidden_attrs = set(sec_cfg.get("forbidden_ast_attributes", []))
            except Exception as e:
                logger.error(f"⚠️ 读取安全配置文件失败 ({config_path}): {e}")
        else:
            # 兜底默认规则
            self.allowed_imports = {"math", "datetime", "time", "json", "re"}
            self.forbidden_calls = {"eval", "exec", "open", "__import__"}
            self.forbidden_attrs = {"os", "sys", "subprocess", "__builtins__"}

    def visit_Import(self, node: ast.Import):
        """审查 import 语句"""
        for alias in node.names:
            mod_root = alias.name.split('.')[0]
            if mod_root not in self.allowed_imports:
                self.violations.append(f"禁止导入未授权模块: '{alias.name}' (仅允许: {list(self.allowed_imports)})")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """审查 from ... import ... 语句"""
        if node.module:
            mod_root = node.module.split('.')[0]
            if mod_root not in self.allowed_imports:
                self.violations.append(f"禁止从未授权模块导入: '{node.module}'")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """审查函数调用"""
        # 匹配直接调用，如 eval(...) / open(...)
        if isinstance(node.func, ast.Name):
            if node.func.id in self.forbidden_calls:
                self.violations.append(f"禁止调用高危函数: '{node.func.id}()'")
            elif node.func.id in self.forbidden_attrs:
                self.violations.append(f"禁止调用高危模块/对象: '{node.func.id}'")
        # 匹配属性调用，如 os.system(...)
        elif isinstance(node.func, ast.Attribute):
            if node.func.attr in self.forbidden_calls:
                self.violations.append(f"禁止调用高危方法: '.{node.func.attr}()'")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        """审查属性访问，如 os.path"""
        if isinstance(node.value, ast.Name):
            if node.value.id in self.forbidden_attrs:
                self.violations.append(f"禁止访问危险对象/模块属性: '{node.value.id}.{node.attr}'")
        if node.attr in self.forbidden_attrs:
            self.violations.append(f"禁止访问敏感属性: '.{node.attr}'")
        self.generic_visit(node)

    def check_code(self, code_str: str) -> Tuple[bool, List[str]]:
        """
        审查入口：解析并检查代码字符串
        :return: (is_safe, violations_list)
        """
        self.violations = []
        try:
            tree = ast.parse(code_str)
            self.visit(tree)
            is_safe = len(self.violations) == 0
            return is_safe, self.violations
        except SyntaxError as e:
            return False, [f"SyntaxError 语法错误，拒绝执行: {e}"]
        except Exception as e:
            return False, [f"AST 解析异常: {str(e)}"]

# 单元测试桩
if __name__ == "__main__":
    checker = ASTCodeChecker()
    
    # 恶性代码测试
    bad_code = "import os; os.system('rm -rf /')"
    safe, msgs = checker.check_code(bad_code)
    print(f"恶意代码拦截测试 -> Safe: {safe}, Msgs: {msgs}")
    
    # 良性代码测试
    good_code = "import math\nx = math.sqrt(16)\nFINAL_RESULT = x"
    safe, msgs = checker.check_code(good_code)
    print(f"合法代码通过测试 -> Safe: {safe}, Msgs: {msgs}")