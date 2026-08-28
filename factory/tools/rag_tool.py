# factory/tools/rag_tool.py
import logging
from typing import Any, Optional, List, Dict
from pydantic import BaseModel, Field
from factory.tool_factory import BaseTool

logger = logging.getLogger("RAGTool")

class RAGSearchInput(BaseModel):
    query: str = Field(description="用户针对 FineBI 用户手册、FAQ 或故障排查指南提出的问题")
    top_k: int = Field(default=3, description="期望返回的相关文档 Chunk 数量")

class RAGKnowledgeSearchTool(BaseTool):
    name: str = "search_knowledge_base"  # 🌟 统一名称为 search_knowledge_base
    description: str = "检索 FineBI 系统官方文档、报错排查指南、FAQ 及最佳实践"
    domain: str = "rag_knowledge"
    package: str = "knowledge_search_pkg"
    args_schema = RAGSearchInput
    is_read_only: bool = True

    # 🌟 注入 reranker 与初检索候选数 top_k_retrieval
    def __init__(
        self, 
        retriever: Optional[Any] = None, 
        reranker: Optional[Any] = None, 
        top_k_retrieval: int = 10
    ):
        super().__init__()
        self.retriever = retriever
        self.reranker = reranker
        self.top_k_retrieval = top_k_retrieval

    def run(self, query: str, top_k: int = 3, return_raw: bool = True, **kwargs) -> Any:
        """
        :param query: 检索关键词
        :param top_k: 重排后最终返回的文档数
        :param return_raw: 为 True 时返回 Dict 列表（包含 rerank_score，方便 Agent 做阈值判定）；为 False 时返回拼接字符串
        """
        logger.info(f"🔍 [RAGTool] 触发知识库检索: query='{query}', top_k={top_k}")
        
        if not self.retriever:
            return f"[模拟 RAG 结果] 关于 '{query}' 的 FineBI 配置说明"

        # 1. 混合检索初召回
        raw_chunks = []
        if hasattr(self.retriever, "hybrid_search"):
            raw_chunks = self.retriever.hybrid_search(query=query, top_k=self.top_k_retrieval)
        elif hasattr(self.retriever, "retrieve"):
            raw_chunks = self.retriever.retrieve(query, top_k=self.top_k_retrieval)

        if not raw_chunks:
            return [] if return_raw else "未检索到相关文档。"

        # 2. 交叉重排计算交叉语义得分 (FineBIReranker)
        if self.reranker and hasattr(self.reranker, "rerank"):
            logger.info("⚡ [RAGTool] 执行 FineBIReranker 交叉重排...")
            reranked_chunks = self.reranker.rerank(
                query=query, 
                documents=raw_chunks, 
                top_n=top_k
            )
        else:
            logger.warning("⚠️ [RAGTool] 未接入 Reranker，直接截取混合检索结果！")
            reranked_chunks = raw_chunks[:top_k]

        # 3. 规范化字段处理，确保带有 rerank_score (0~1)
        formatted_results = []
        for item in reranked_chunks:
            chunk_dict = item if isinstance(item, dict) else item.__dict__
            score_val = chunk_dict.get("rerank_score", chunk_dict.get("score", 0.0))
            chunk_dict["rerank_score"] = float(score_val)
            formatted_results.append(chunk_dict)

        # 若上层（如 Agent 阀门校验）需要原始 List 数据结构
        if return_raw:
            return formatted_results

        # 若需要转换为 Prompt 文本格式
        formatted_chunks = []
        for idx, item in enumerate(formatted_results, 1):
            source = item.get("source_file") or item.get("doc_name") or "未知文档"
            formatted_chunks.append(
                f"[{idx}] 文档: {source} (Score: {item['rerank_score']:.4f})\n内容: {item.get('content', '')}"
            )
        return "\n\n".join(formatted_chunks)