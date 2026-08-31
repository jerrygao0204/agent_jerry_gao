# search/retriever.py 向量/混合检索
import os
import sys
import re
import logging
from collections import Counter
from typing import Dict, Any, List, Optional

# 📂 动态计算项目根目录并强行注入系统路径，确保全局工程内 factory 模块可见
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 统一从中央工厂引入枢纽
from factory.model_factory import ModelFactory

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pymilvus", "transformers", "torch"])
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")


class FineBIRetriever:
    def __init__(
        self,
        milvus_host: str = "172.17.0.1",
        milvus_port: str = "19530",
        collection_name: str = "finebi_knowledge_chunks",
        model_short_name: str = "Qwen/Qwen3-Embedding-8B",
        prompt_hub_path: str = "prompt_hub.yaml",
        cuda_device: str = "0"
    ):
        """
        统一采用底座工厂管理的高性能混合检索召回器
        """
        self.collection_name = collection_name
        self.model_short_name = model_short_name
        
        # 1. 初始化中央工厂（自动构建环境软链接）
        self.factory = ModelFactory(prompt_hub_path=prompt_hub_path)
        
        # 2. 绑定硬件卡号
        self.device = self.factory.setup_cuda_device(cuda_device)
        
        self.model = None
        self.tokenizer = None
        
        # 3. 建立物理向量数据库长连接
        self.client = MilvusClient(uri=f"http://{milvus_host}:{milvus_port}")
        logging.info(f"⚡ 标准规范检索器成功绑定 Milvus 数据库连接集群。")

    def _init_embedding_engine(self):
        """单例加载与预热稠密向量化模型"""
        if self.model is None or self.tokenizer is None:
            # 委派工厂进行统一的离线物理哈希寻址
            model_path = self.factory.resolve_model_path(self.model_short_name)
            logging.info(f"🚀 [Offline Load] 正在冷启动加载稠密向量化模型: {model_path}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path, 
                local_files_only=True, 
                trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, 
                torch_dtype=torch.float16, 
                device_map="auto", 
                local_files_only=True, 
                trust_remote_code=True
            )
            
            # 执行静态预热，保证显存池空间稳定
            warmup_prompt = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant"
            inputs = self.tokenizer(warmup_prompt, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                _ = self.model.generate(**inputs, max_new_tokens=5)
            logging.info("✅ Embedding 物理计算引擎预热完成。")

    def get_dense_embedding(self, text: str) -> List[float]:
        """计算高维稠密语义向量"""
        self._init_embedding_engine()
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # 提取最顶层 Hidden States 的均值池化作为文本表征
            embeddings = outputs.hidden_states[-1].mean(dim=1)
        return embeddings[0].cpu().numpy().tolist()

    @staticmethod
    def generate_sparse_vector(text: str) -> Dict[int, float]:
        """动态生成轻量级稀疏词频特征向量 (哈希映射防越界)"""
        tokens = re.findall(r'\w+', text.lower())
        counts = Counter(tokens)
        return {abs(hash(token)) % 2**31: float(count) for token, count in counts.items()}

    def _fetch_chunks_by_ids(self, chunk_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """批量从向量数据库追溯前置与后置块节点"""
        if not chunk_ids:
            return {}
        valid_ids = [cid for cid in chunk_ids if cid and cid != "SNULL"]
        if not valid_ids:
            return {}
            
        records = self.client.query(
            collection_name=self.collection_name,
            filter=f"chunk_id in {valid_ids}",
            output_fields=["chunk_id", "content", "full_hierarchy_array", "next_chunk_id", "prev_chunk_id"]
        )
        return {r["chunk_id"]: r for r in records}

    def hybrid_search(self, query: str, top_k: int = 3, expand_context: bool = True, filter_expr: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        稠密+稀疏多路混合双轨召回，使用 Milvus 底层物理 RFRanker 实现融合
        """
        self.client.load_collection(collection_name=self.collection_name)
        
        # 1. 并发计算双轨特征
        dense_vec = self.get_dense_embedding(query)
        sparse_vec = self.generate_sparse_vector(query)

        # 2. 组装多路检索请求
        dense_req = AnnSearchRequest(
            data=[dense_vec], anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"ef": 64}}, limit=top_k * 2
        )
        sparse_req = AnnSearchRequest(
            data=[sparse_vec], anns_field="sparse_vector",
            param={"metric_type": "IP", "params": {}}, limit=top_k * 2
        )

        # 3. 依赖底层内置 RFRanker 机制执行交叉评分级联融合
        results = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=[dense_req, sparse_req],
            ranker=RRFRanker(k=60),
            limit=top_k,
            output_fields=[
                "chunk_id", "content", "section_id", "file_url", 
                "full_hierarchy_array", "biz_summary", "next_chunk_id", "prev_chunk_id"
            ]
        )

        if not results or len(results) == 0:
            return []

        hits = results[0]
        
        # 4. 拓扑延伸：收集前置与后置块 ID，拉取周边上下文
        ids_to_fetch = set()
        for hit in hits:
            entity = hit.get("entity", {})
            for key in ["next_chunk_id", "prev_chunk_id"]:
                cid = entity.get(key)
                if cid and cid != "SNULL":
                    ids_to_fetch.add(cid)

        fetched_blocks = self._fetch_chunks_by_ids(list(ids_to_fetch)) if expand_context else {}

        # 5. 结构化解析回填
        parsed_chunks = []
        for hit in hits:
            entity = hit.get("entity", {})
            h_dict = entity.get("full_hierarchy_array", {})
            h_array = h_dict.get('data', []) if isinstance(h_dict, dict) else h_dict
            hierarchy_str = " > ".join(h_array) if isinstance(h_array, list) else ""
            
            chunk_id = entity.get("chunk_id")
            base_content = entity.get("content", "")
            prev_id = entity.get("prev_chunk_id")
            next_id = entity.get("next_chunk_id")
            
            up_content = ""
            down_content = ""
            
            if expand_context:
                if prev_id in fetched_blocks:
                    p_content = fetched_blocks[prev_id].get("content", "").strip()
                    if p_content:
                        up_content = p_content
                if next_id in fetched_blocks:
                    n_content = fetched_blocks[next_id].get("content", "").strip()
                    if n_content:
                        down_content = n_content

            parsed_chunks.append({
                "chunk_id": chunk_id,
                "score": hit.get("distance"),  # RFR 物理物理得分
                "base_content": base_content,
                "up_content": up_content,
                "down_content": down_content,
                "section_id": entity.get("section_id"),
                "file_url": entity.get("file_url"),
                "hierarchy": hierarchy_str,
                "biz_summary": entity.get("biz_summary"),
                "content": base_content  # 映射通用接口
            })

        return parsed_chunks


# =====================================================================
# 🧪 本地链路检索实战验证
# =====================================================================
if __name__ == "__main__":
    retriever = FineBIRetriever(
        milvus_host="172.17.0.1",
        collection_name="finebi_knowledge_chunks"
    )
    
    def print_helper(hits: list):
        for idx, item in enumerate(hits):
            print(f"\n[排名 Top-{idx+1}] 底层融合分: {item['score']:.4f} | 🧭 结构树: {item['hierarchy']}")
            
            up = item["up_content"]
            base = item["base_content"]
            down = item["down_content"]
            
            if up.strip():
                display_up = up[:300] + "...\n(已截断过长文本)" if len(up) > 300 else up
                print(f" ├ ── ⬆️ 前置延伸:\n{display_up.strip()}")
                
            if base.strip():
                display_base = base[:500] + "...\n(已截断过长文本)" if len(base) > 500 else base
                print(f" 📝 核心正文(匹配到的块):\n{display_base.strip()}")
                
            if down.strip():
                display_down = down[:300] + "...\n(已截断过长文本)" if len(down) > 300 else down
                print(f" └ ── ⬇️ 后置延伸:\n{display_down.strip()}")
                
            print("-" * 50)

    print("\n" + "="*20 + " 🔍 测试问法：怎么创建预警用户？ 🔍 " + "="*20)
    hits_1 = retriever.hybrid_search(query="怎么创建预警用户?", top_k=2, expand_context=True)
    print_helper(hits_1)