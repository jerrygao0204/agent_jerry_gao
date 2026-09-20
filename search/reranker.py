# reranker.py 重排序
import os
import sys
import torch
import logging
from typing import List, Dict, Any, Union, Optional

# 📂 动态计算项目根目录，将其注入系统路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 统一从中央工厂引入枢纽
from factory.model_factory import ModelFactory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

class Reranker:
    def __init__(
        self,
        cache_dir: str = "/workspace/hf-conda/hf_cache/hub",
        max_length: int = 512,
        batch_size: int = 32,
        min_prob: float = 0.25,        # 概率过滤硬门槛
        max_score_gap: float = 3.5,    # 与 Top-1 的最大 Logits 允许分差
        strict_mode: bool = True,
        factory: ModelFactory = None
    ):
        """
        初始化 Reranker，直接复用 ModelFactory 内存中常驻的 Rerank 模型，避免重复加载
        """
        self.max_length = max_length
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.min_prob = min_prob
        self.max_score_gap = max_score_gap
        self.strict_mode = strict_mode

        # 1. 统一获取或初始化工厂单例（此时工厂已在内部常驻了 Rerank 模型）
        self.factory = factory if factory is not None else ModelFactory(cache_dir=cache_dir)
        self.device = ModelFactory._device

        # 2. 直接复用工厂内部常驻的 Tokenizer 与 Model，实现零重复加载
        self.tokenizer = ModelFactory._rerank_tokenizer
        self.model = ModelFactory._rerank_model

        if self.model is None or self.tokenizer is None:
            logging.warning("⚠️ 警告：ModelFactory 中的 Rerank 模型未就绪，尝试触发初始化...")
            self.factory._init_rerank_model()
            self.tokenizer = ModelFactory._rerank_tokenizer
            self.model = ModelFactory._rerank_model

        logging.info(f"✅ [Zero-Copy Load] Reranker 成功挂载复用 ModelFactory 常驻模型 (Device: {self.device})")

    def _build_context_aware_text(
        self, 
        chunk_data: Dict[str, Any], 
        chunk_map: Dict[str, str],
        enable_surrounding_context: bool = True
    ) -> str:
        """针对纯文本 KV 型 chunk_map 进行高效上下文反查与组装"""
        hierarchy: str = chunk_data.get("hierarchy", "").strip() or "全局文档"
        biz_summary: str = chunk_data.get("biz_summary", "").strip()
        core_content: str = (chunk_data.get("content") or chunk_data.get("base_content") or "").strip()

        header_lines = [f"[文档层级 (Hierarchy)]: {hierarchy}"]
        if biz_summary:
            header_lines.append(f"[业务摘要 (Summary)]: {biz_summary}")
        header_str = "\n".join(header_lines)

        if not enable_surrounding_context or not chunk_map:
            return f"{header_str}\n\n[核心内容 (Core Content)]:\n{core_content}"

        up_id = chunk_data.get("up_content")
        down_id = chunk_data.get("down_content")

        up_text = chunk_map.get(up_id, "").strip() if up_id else ""
        down_text = chunk_map.get(down_id, "").strip() if down_id else ""

        body_parts = []
        if up_text:
            body_parts.append(f"[上文补充 (Up Content - {up_id})]:\n{up_text}")
            
        body_parts.append(f"[核心内容 (Core Content - {chunk_data.get('chunk_id', 'Current')})]:\n{core_content}")
        
        if down_text:
            body_parts.append(f"[下文补充 (Down Content - {down_id})]:\n{down_text}")

        return f"{header_str}\n\n" + "\n\n".join(body_parts)
        
    def rerank(
        self, 
        query: str, 
        documents: List[Dict[str, Any]], 
        top_n: int = 5, 
        disable_filtering: bool = False
    ) -> List[Dict[str, Any]]:
        """
        對初篩撈出來的文檔進行精排打分：計算概率 -> Sigmoid 歸一化 -> 閾值過濾 -> 排序截取 Top-N
        """
        if not documents:
            logging.warning("传入的重排文档列表为空。")
            return []
        
        chunk_map = {doc["chunk_id"]: doc.get("content", "") for doc in documents if "chunk_id" in doc}
        doc_texts = [self._build_context_aware_text(doc, chunk_map) for doc in documents]
        pairs = [[query, doc_text] for doc_text in doc_texts]

        scores = []
        probs = []
        
        with torch.no_grad():
            for i in range(0, len(pairs), self.batch_size):
                batch_pairs = pairs[i : i + self.batch_size]
                inputs = self.tokenizer(
                    batch_pairs,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)
                
                batch_logits = self.model(**inputs).logits.view(-1).float()
                batch_probs = torch.sigmoid(batch_logits).cpu().tolist()
                batch_scores = batch_logits.cpu().tolist()
                
                scores.extend(batch_scores)
                probs.extend(batch_probs)

        for i in range(len(documents)):
            documents[i]["rerank_score"] = scores[i]
            documents[i]["rerank_prob"] = probs[i]

        sorted_docs = sorted(documents, key=lambda x: x["rerank_prob"], reverse=True)

        if disable_filtering or not self.strict_mode:
            final_results = sorted_docs[:top_n]
            return final_results

        top_1_prob = sorted_docs[0]["rerank_prob"]
        top_1_score = sorted_docs[0]["rerank_score"]
        
        if top_1_prob < self.min_prob:
            logging.warning(
                f"❌ [Reranker 阻断] 最高相关度概率为 {top_1_prob:.2%} (Logit: {top_1_score:.4f})，"
                f"未达到门槛 {self.min_prob:.2%}，判定为无相关匹配文本。"
            )
            return []

        valid_docs = [
            doc for doc in sorted_docs 
            if doc["rerank_prob"] >= self.min_prob and (top_1_score - doc["rerank_score"]) <= self.max_score_gap
        ]

        final_results = valid_docs[:top_n]
        logging.info(
            f"重排完成：初筛 {len(documents)} 个 Chunk -> 过滤保留 {len(valid_docs)} 个高置信度 Chunk (Top-1 概率: {top_1_prob:.2%}) -> 截取 Top-{len(final_results)}。"
        )
        
        return final_results

if __name__ == "__main__":
    mock_retrieved_docs = [
        {"chunk_id": "c1", "content": "FineBI 支持多种数据源连接，包括 MySQL, Oracle 以及各种大数据库平台。"},
        {"chunk_id": "c2", "content": "帆软报表软件的安装教程请参考官方支持文档，并确保本地 JDK 环境配置正确。"},
        {"chunk_id": "c3", "content": "在 FineBI 管理系统中，用户可以通过新建数据集并选择对应的数据库驱动来完成数据库的连接配置。"}
    ]
    
    test_query = "FineBI 怎么连接数据库？"
    
    # 实例化 Reranker（会自动共享 ModelFactory 中常驻的模型，不会重复加载）
    reranker = Reranker()
    
    ranked_results = reranker.rerank(query=test_query, documents=mock_retrieved_docs, top_n=2)
    
    print("\n" + "="*20 + " 重排序精排结果 " + "="*20)
    for idx, doc in enumerate(ranked_results):
        print(f"Top {idx+1} [精排得分: {doc['rerank_score']:.4f}] -> {doc['content']}")