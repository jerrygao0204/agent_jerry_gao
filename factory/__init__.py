# factory/__init__.py
import logging

# 1. 重命名导入各自的工厂类，彻底消除类名冲突
from factory.model_factory import ModelFactory

# 2. 导入工具工厂与初始化函数
from factory.tool_factory import tool_factory
from factory.tool_registry import init_tools

# 3. 从 tools/ 子包导出原子工具类
from factory.tools.rag_tool import RAGKnowledgeSearchTool

logger = logging.getLogger("FactoryPackage")

__all__ = [
    "ModelFactory",
    "tool_factory",
    "init_tools",
    "RAGKnowledgeSearchTool"
]