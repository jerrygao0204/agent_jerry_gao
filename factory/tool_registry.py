# factory/tool_registry.py
import os
import sys

# ==========================================
# 0. 动态修复 Python 模块搜索路径 (sys.path)
# ==========================================
# 获取当前文件所在目录的上一级目录（即项目根目录）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import yaml
from factory.tool_factory import tool_factory
import importlib
from typing import Any, Dict, Optional
import inspect

logger = logging.getLogger("ToolsModule")

def init_tools(retriever: Optional[Any] = None, reranker: Optional[Any] = None) -> None:
    """初始化工具工厂：加载配置并动态注册工具"""
    yaml_path = os.path.join(project_root, "config", "tools.yaml")

    if not os.path.exists(yaml_path):
        logger.error(f"❌ 找不到配置文件: {yaml_path}")
        return

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. 先载入 YAML 定义的 Domain 与 Package 元数据
    for domain, dmeta in config.get("domains", {}).items():
        tool_factory.register_domain_meta(domain, dmeta.get("description", ""))
        for pkg, desc in dmeta.get("packages", {}).items():
            tool_factory.register_package_meta(domain, pkg, desc)

    # 2. 构造依赖注入上下文
    injection_context: Dict[str, Any] = {
        "retriever": retriever,
        "reranker": reranker
    }

    # 3. 动态实例化并注册工具
    for entry in config.get("tools", []):
        if not entry.get("enabled", True):
            continue
            
        module = importlib.import_module(entry["module"])
        cls = getattr(module, entry["class"])

        # 动态检测构造函数参数，仅注入需要的依赖（消除硬编码判断）
        sig = inspect.signature(cls.__init__)
        tool_kwargs = {
            k: v for k, v in injection_context.items() 
            if k in sig.parameters and v is not None
        }

        tool_instance = cls(**tool_kwargs)
        tool_factory.register_tool(tool_instance)
