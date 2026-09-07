# reranker.py 重排序
import os
import sys
import torch
import logging
from typing import List, Dict, Any, Union, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# 📂 动态计算项目根目录，将其注入系统路径中，确保全局工厂能被顺利导入
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 统一从中央工厂引入枢纽
from factory.model_factory import ModelFactory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

class FineBIReranker:
    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-reranker-large",
        cache_dir: str = "/workspace/hf-conda/hf_cache/hub",
        cuda_device: str = "0",
        max_length: int = 512,
        batch_size: int = 32,
        min_prob: float = 0.25,        # 新增：概率过滤硬门槛 (60%)
        max_score_gap: float = 3.5,    # 新增：与 Top-1 的最大 Logits 允许分差
        strict_mode: float = True,
        factory: ModelFactory = None
    ):
        """
        初始化 Reranker 模型，整合 ModelFactory 寻址与算力管理
        :param model_name_or_path: 模型名称或本地物理路径
        :param cache_dir: HuggingFace 缓存根目录
        :param cuda_device: 指定 GPU 卡号 (如 "0", "1")
        :param max_length: 最大文本截断长度
        :param batch_size: 批处理大小，防止爆显存
        :param min_prob: 相关度概率过滤门槛 (0.0~1.0)
        :param max_score_gap: 允许与最高得分 (Top-1) 的最大得分差距
        :param factory: 可选传入已初始化的 ModelFactory 实例
        """
        self.max_length = max_length
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.min_prob = min_prob
        self.max_score_gap = max_score_gap
        self.strict_mode = strict_mode

        # 1. 初始化或获取工厂单例
        self.factory = factory if factory is not None else ModelFactory(cache_dir=cache_dir)
        
        # 2. 统一设置 CUDA 设备
        self.device = ModelFactory.setup_cuda_device(cuda_device)
        logging.info(f"Reranker 正在加载模型至设备: {self.device}")

        # 3. 通过工厂统一路径寻址算法，定位本地 Snapshot 物理路径
        real_model_path = self.factory.resolve_model_path(model_name_or_path)
        logging.info(f"🚀 [Offline Load] Reranker 锁定物理路径: {real_model_path}")

        # 严格使用离线模式加载物理文件
        self.tokenizer = AutoTokenizer.from_pretrained(
            real_model_path, 
            local_files_only=True,
            trust_remote_code=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            real_model_path, 
            local_files_only=True,
            trust_remote_code=True
        )
        
        
        self.model.to(self.device)
        self.model.eval()
        logging.info("Reranker 模型加载成功。")

    def _build_context_aware_text(
        self, 
        chunk_data: Dict[str, Any], 
        chunk_map: Dict[str, str],
        enable_surrounding_context: bool = True
    ) -> str:
        """
        针对纯文本 KV 型 chunk_map 进行高效上下文反查与组装
        
        :param chunk_data: 当前要处理的 Chunk 节点数据
        :param chunk_map: 全局 chunk_id 到 content 的映射字典
        :param enable_surrounding_context: 是否拼接上下文
        :return: 格式化后的上下文增强文本
        """
        # 1. 基础字段提取
        hierarchy: str = chunk_data.get("hierarchy", "").strip() or "全局文档"
        biz_summary: str = chunk_data.get("biz_summary", "").strip()
        core_content: str = (chunk_data.get("content") or chunk_data.get("base_content") or "").strip()

        # 2. 构建结构化 Header
        header_lines = [f"[文档层级 (Hierarchy)]: {hierarchy}"]
        if biz_summary:
            header_lines.append(f"[业务摘要 (Summary)]: {biz_summary}")
        header_str = "\n".join(header_lines)

        # 3. 关闭增强或无 chunk_map 时降级返回
        if not enable_surrounding_context or not chunk_map:
            return f"{header_str}\n\n[核心内容 (Core Content)]:\n{core_content}"

        # 4. 执行上下文 ID 反查
        up_id = chunk_data.get("up_content")
        down_id = chunk_data.get("down_content")

        # 直接通过 Dict 获取纯文本，找不到则返回空字符串
        up_text = chunk_map.get(up_id, "").strip() if up_id else ""
        down_text = chunk_map.get(down_id, "").strip() if down_id else ""

        # 5. 组装完整 Body
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
        :param disable_filtering: 評測模式下可設為 True，停用硬性阻斷，僅做純 Top-N 排序
        """
        if not documents:
            logging.warning("传入的重排文档列表为空。")
            return []  # 保持返回 List
        
        chunk_map = {doc["chunk_id"]: doc.get("content", "") for doc in documents if "chunk_id" in doc}
        # doc_texts = [doc.get("content", "") for doc in documents]
        doc_texts = [self._build_context_aware_text(doc, chunk_map) for doc in documents]
        pairs = [[query, doc_text] for doc_text in doc_texts]

        scores = []
        probs = []
        # 分批次推断防止 GPU 显存溢出 (CUDA OOM)
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
                
                # 计算 Raw Logits
                batch_logits = self.model(**inputs).logits.view(-1).float()
                # 通过 Sigmoid 映射为 0~1 的置信度概率
                batch_probs = torch.sigmoid(batch_logits).cpu().tolist()
                batch_scores = batch_logits.cpu().tolist()
                
                scores.extend(batch_scores)
                probs.extend(batch_probs)

        # 1. 挂载原始得分与归一化概率
        for i in range(len(documents)):
            documents[i]["rerank_score"] = scores[i]
            documents[i]["rerank_prob"] = probs[i]

        # 2. 步骤一：按得分/概率降序排列，以便先确定 Top-1 得分
        sorted_docs = sorted(documents, key=lambda x: x["rerank_prob"], reverse=True)

        # 🛠️ 若顯示停用過濾（如離線評測時），直接截取 Top-N 返回
        if disable_filtering or not self.strict_mode:
            final_results = sorted_docs[:top_n]
            logging.debug(f"純排序模式（無阻斷）：截取 Top-{len(final_results)}")
            return final_results

        # 3. 步骤二：硬性绝对阈值校验 (Check Top-1 Probability)
        top_1_prob = sorted_docs[0]["rerank_prob"]
        top_1_score = sorted_docs[0]["rerank_score"]
        
        if top_1_prob < self.min_prob:
            logging.warning(
                f"❌ [Reranker 阻断] 最高相关度概率为 {top_1_prob:.2%} (Logit: {top_1_score:.4f})，"
                f"未达到门槛 {self.min_prob:.2%}，判定为无相关匹配文本。"
            )
            return []  # 保持返回空列表，避免下游抛出 AttributeError

        # 4. 步骤三：结合绝对门槛与相对分差 (Gap) 过滤合格 Chunk
        valid_docs = [
            doc for doc in sorted_docs 
            if doc["rerank_prob"] >= self.min_prob and (top_1_score - doc["rerank_score"]) <= self.max_score_gap
        ]

        # 5. 步骤四：截取前 Top-N
        final_results = valid_docs[:top_n]
        logging.info(
            f"重排完成：初筛 {len(documents)} 个 Chunk -> 过滤保留 {len(valid_docs)} 个高置信度 Chunk (Top-1 概率: {top_1_prob:.2%}) -> 截取 Top-{len(final_results)}。"
        )
        
        return final_results

if __name__ == "__main__":
    # 模拟从 `retriever.py` 混合检索出来的初筛数据
    mock_retrieved_docs = [
        {"chunk_id": "c1", "content": "FineBI 支持多种数据源连接，包括 MySQL, Oracle 以及各种大数据库平台。"},
        {"chunk_id": "c2", "content": "帆软报表软件的安装教程请参考官方支持文档，并确保本地 JDK 环境配置正确。"},
        {"chunk_id": "c3", "content": "在 FineBI 管理系统中，用户可以通过新建数据集并选择对应的数据库驱动来完成数据库的连接配置。"}
    ]
    
    test_query = "FineBI 怎么连接数据库？"
    
    # 初始化 Reranker 实例
    reranker = FineBIReranker(
        model_name_or_path="BAAI/bge-reranker-large",
        cache_dir="/workspace/hf-conda/hf_cache/hub",
        cuda_device="0"
    )
    
    # 执行精排
    ranked_results = reranker.rerank(query=test_query, documents=mock_retrieved_docs, top_n=2)
    
    print("\n" + "="*20 + " 重排序精排结果 " + "="*20)
    for idx, doc in enumerate(ranked_results):
        print(f"Top {idx+1} [精排得分: {doc['rerank_score']:.4f}] -> {doc['content']}")