# factory/tool_registry.py
import logging
from typing import Any, Optional
from factory.tool_factory import tool_factory

# 🌟 从具体实现模块导入定义，避免重复定义
from factory.tools.rag_tool import RAGKnowledgeSearchTool
from factory.tools.api_tool import FineBIDashboardTool
from factory.tools.web_search_tool import WebSearchTool
from factory.tools.dataset_summary import DatasetSummaryTool


logger = logging.getLogger("ToolsModule")


def init_tools(retriever: Optional[Any] = None, reranker: Optional[Any] = None) -> None:
    """集中注册所有的原子工具到全局单例 tool_factory"""
    try:
        # 1. 注册带 Reranker 能力的 RAG 工具
        rag_tool = RAGKnowledgeSearchTool(retriever=retriever, reranker=reranker)
        tool_factory.register_tool(rag_tool)

        # 2. 注册 API 工具
        bi_tool = FineBIDashboardTool()
        tool_factory.register_tool(bi_tool)

        # 3. 注册 web 工具
        web_tool = WebSearchTool()
        tool_factory.register_tool(web_tool)

        # 💡 4. 注册 数据集摘要工具 (Dataset Summary Tool)
        tool_factory.register_tool(DatasetSummaryTool())

        logger.info("✅ [init_tools] 所有工具类注册完毕。")
    except Exception as e:
        logger.error(f"❌ 工具初始化失败: {e}")
        raise e