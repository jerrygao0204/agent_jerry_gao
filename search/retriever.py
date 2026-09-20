# search/retriever.py 向量/混合检索 (完全对接 ModelFactory 统一网关)
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

# 统一从中央工厂引入 ModelFactory（唯一 LLM/Embedding 网关入口，与 llm_client.py 保持一致）
from factory.model_factory import ModelFactory

try:
    from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pymilvus"])
    from pymilvus import MilvusClient, AnnSearchRequest, RRFRanker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")


class Retriever:
    def __init__(
        self,
        milvus_host: str = "172.17.0.1",
        milvus_port: str = "19530",
        collection_name: str = "finebi_knowledge_chunks",
        default_model_name: str = "qwen3-embedding-8b",  # 对应你在网关配置的向量模型代号
        factory: Optional[ModelFactory] = None
    ):
        """
        轻量级混合检索召回器：通过 ModelFactory 统一网关获取稠密向量，零本地大模型依赖
        """
        self.collection_name = collection_name
        self.default_model_name = default_model_name
        
        # 1. 统一采用 ModelFactory 管理底层路由与网关
        self.factory = factory if factory is not None else ModelFactory()
        
        # 2. 建立物理向量数据库长连接
        self.client = MilvusClient(uri=f"http://{milvus_host}:{milvus_port}")
        logging.info(f"⚡ 标准规范检索器成功绑定 Milvus 数据库连接集群。")

    def get_dense_embedding(self, text: str, model_name: Optional[str] = None) -> List[float]:
        """
        透过 ModelFactory 计算高维稠密语义向量
        """
        target_model = model_name or self.default_model_name
        
        try:
            return self.factory.embed_query(text, model=target_model)
        except Exception as e:
            logging.error(f"❌ 通过 ModelFactory 获取 Embedding 失败 (模型: {target_model}): {e}")
            raise e

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
                "score": hit.get("distance"),
                "base_content": base_content,
                "up_content": up_content,
                "down_content": down_content,
                "section_id": entity.get("section_id"),
                "file_url": entity.get("file_url"),
                "hierarchy": hierarchy_str,
                "biz_summary": entity.get("biz_summary"),
                "content": base_content
            })

        return parsed_chunks


# =====================================================================
# 🧪 本地链路检索实战验证
# =====================================================================
if __name__ == "__main__":
    # 🌍 环境自动适配：若在 Docker 容器内部运行，检查并修正网关地址（ModelFactory 读取 OPENAI_BASE_URL）
    if os.path.exists("/workspace") and not os.environ.get("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = "http://172.17.0.1:4000/v1"
        print("🔧 [自动适配] 检测到处于容器内部，已将 LiteLLM 网关自动重定向至宿主机: http://172.17.0.1:4000/v1")

    retriever = Retriever(
        milvus_host="172.17.0.1",
        collection_name="finebi_knowledge_chunks",
        default_model_name="qwen3-embedding-4b"  # 对应你 factory 中定义的默认 embedding 模型名称
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
    try:
        hits_1 = retriever.hybrid_search(query="怎么创建预警用户?", top_k=2, expand_context=True)
        print_helper(hits_1)
    except Exception as e:
        print(f"❌ 检索测试失败: {e}")