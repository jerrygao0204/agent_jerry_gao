# # factory/tool_registry.py
# import logging
# from typing import Any, Optional
# from pydantic import BaseModel, Field
# from factory.tool_factory import BaseTool, tool_factory
# from factory.tools.rag_tool import RAGKnowledgeSearchTool
# from factory.tools.api_tool import FineBIDashboardTool

# logger = logging.getLogger("ToolsModule")


    
# # ==========================================
# # 1. 定义具体工具的输入 Schema 与工具类
# # ==========================================
# class RAGSearchInput(BaseModel):
#     query: str = Field(description="知识库检索关键词或用户提出的问题")
#     top_k: int = Field(default=3, description="返回的相关文档数量")

# class RAGKnowledgeSearchTool(BaseTool):
#     """FineBI 知识库检索工具"""
#     name: str = "search_knowledge_base"
#     description: str = "检索 FineBI 用户手册、FAQ 及故障排查文档"
#     domain: str = "rag_knowledge"
#     package: str = "knowledge_search_pkg"
#     args_schema = RAGSearchInput
#     is_read_only: bool = True

#     def __init__(self, retriever: Optional[Any] = None):
#         super().__init__()
#         self.retriever = retriever
        
        
#     def run(self, query: str, top_k: int = 3, **kwargs) -> Any:
#         logger.info(f"🔍 [RAGKnowledgeSearchTool] 执行检索: query='{query}', top_k={top_k}")
#         if self.retriever:
#             # 优先调用混合检索 hybrid_search
#             if hasattr(self.retriever, "hybrid_search"):
#                 return self.retriever.hybrid_search(query=query, top_k=top_k)
#             elif hasattr(self.retriever, "retrieve"):
#                 return self.retriever.retrieve(query, top_k=top_k)
#         return f"[模拟检索结果] 关于 '{query}' 的 FineBI 相关文档内容"
    

# # ==========================================
# # 2. 全局工具链初始化函数 (init_tools)
# # ==========================================
# def init_tools(retriever: Optional[Any] = None) -> None:
#     """集中注册所有的原子工具到全局单例 tool_factory"""
#     try:
#         # 1. 注册 RAG 工具
#         rag_tool = RAGKnowledgeSearchTool(retriever=retriever)
#         tool_factory.register_tool(rag_tool)

#         # 2. 注册 API 工具
#         bi_tool = FineBIDashboardTool()
#         tool_factory.register_tool(bi_tool)

#         logger.info("✅ [init_tools] 所有工具类注册完毕。")
#     except Exception as e:
#         logger.error(f"❌ 工具初始化失败: {e}")
#         raise e

# factory/tool_registry.py
import logging
from typing import Any, Optional
from factory.tool_factory import tool_factory

# 🌟 从具体实现模块导入定义，避免重复定义
from factory.tools.rag_tool import RAGKnowledgeSearchTool
from factory.tools.api_tool import FineBIDashboardTool

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

        logger.info("✅ [init_tools] 所有工具类注册完毕。")
    except Exception as e:
        logger.error(f"❌ 工具初始化失败: {e}")
        raise e