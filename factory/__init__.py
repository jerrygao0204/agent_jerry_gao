# factory/__init__.py
import logging
from factory.model_factory import ModelFactory
from factory.tool_factory import tool_factory

# 🌟 关键点：明确从 tools.py 单独文件导入 init_tools，避免与 tools/ 文件夹混淆
from factory.tool_registry import init_tools

# 从 tools/ 包导出原子工具类
from factory.tools.rag_tool import RAGKnowledgeSearchTool

logger = logging.getLogger("FactoryPackage")

__all__ = [
    "ModelFactory",
    "tool_factory",
    "init_tools",
    "RAGKnowledgeSearchTool"
]