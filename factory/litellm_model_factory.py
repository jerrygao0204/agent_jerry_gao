# factory/litellm_model_factory.py
import os
import yaml
import logging
import threading
import requests
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse
from dotenv import load_dotenv
from openai import OpenAI
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")


class ModelFactory:
    """
    模型与环境配置中心工厂 (ModelFactory - LiteLLM Mode)
    全局接管：Prompt Hub 资产管理、LiteLLM API 句柄分发、Rerank 与 Embedding 统一调用
    """
    _instance: Optional["ModelFactory"] = None
    _lock: threading.Lock = threading.Lock()
    _openai_client: Optional[OpenAI] = None
    
    # 🌟 類別級別全域快取：確保模型與分詞器在整個生命週期中只被載入一次
    _rerank_tokenizer = None
    _rerank_model = None
    _device = "cuda" if torch.cuda.is_available() else "cpu"

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelFactory, cls).__new__(cls)
        return cls._instance

    def __init__(self, prompt_hub_path: str = "prompt_hub.yaml"):
        if getattr(self, "_initialized", False):
            return

        with self._lock:
            if getattr(self, "_initialized", False):
                return

            load_dotenv()

            # 1. 配置 Endpoint & Key
            self.base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:4000/v1").rstrip("/")
            self.api_key = os.getenv("OPENAI_API_KEY", "sk-1234")

            # 2. 解析 Prompt Hub 路径
            self.prompt_hub_path = self._resolve_path(prompt_hub_path)
            self.prompts = self._load_prompts()

            # 3. 初始化全局 OpenAI Client
            ModelFactory._openai_client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key
            )

            # 4. 初始化 requests.Session 连接池
            self.session = requests.Session()
            retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
            self.session.mount("http://", HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=retries))
            self.session.mount("https://", HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=retries))

            # 🌟 5. 確保在啟動初始化時「僅載入一次」Rerank 模型到 GPU
            self._init_rerank_model()

            self._initialized = True
            logging.info(f"🚀 ModelFactory (LiteLLM Mode) 初始化完成 | Endpoint: {self.base_url}")

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        factory_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(factory_dir)
        return os.path.join(project_root, "config", os.path.basename(path))

    def _load_prompts(self) -> Dict[str, str]:
        if not os.path.exists(self.prompt_hub_path):
            logging.warning(f"⚠️ Prompt Hub 文件不存在: {self.prompt_hub_path}")
            return {}

        try:
            with open(self.prompt_hub_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if not data or "prompts" not in data:
                logging.warning(f"⚠️ Prompt Hub 格式无效: {self.prompt_hub_path}")
                return {}

            prompts_dict = {p["name"]: p["content"] for p in data.get("prompts", []) if "name" in p and "content" in p}
            logging.info(f"📂 Prompt Hub 资产加载成功: [{self.prompt_hub_path}] | Keys: {list(prompts_dict.keys())}")
            return prompts_dict
        except Exception as e:
            logging.error(f"❌ 加载 Prompt Hub 失败 ({self.prompt_hub_path}): {e}")
            return {}

    @classmethod
    def get_instance(cls, *args, **kwargs) -> "ModelFactory":
        return cls(*args, **kwargs)

    def get_llm_client(self) -> OpenAI:
        if not ModelFactory._openai_client:
            raise RuntimeError("ModelFactory 未正确初始化 OpenAI Client")
        return ModelFactory._openai_client

    def _init_rerank_model(self):
        """內部方法：在啟動時預先載入 Rerank 模型，避免每次查詢重複載入"""
        if ModelFactory._rerank_model is not None:
            return

        model_path = Path("/workspace/hf-conda/hf_cache/hub/models--BAAI--bge-reranker-large/snapshots/55611d7bca2a7133960a6d3b71e083071bbfc312")
        if not model_path.exists():
            host_fallback = Path("/home/gaozheng/venv/hf-conda/hf_cache/hub/models--BAAI--bge-reranker-large/snapshots/55611d7bca2a7133960a6d3b71e083071bbfc312")
            if host_fallback.exists():
                model_path = host_fallback

        try:
            logging.info(f"⏳ 正在初始化本地 Rerank 模型至設備: {ModelFactory._device}...")
            ModelFactory._rerank_tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            ModelFactory._rerank_model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
            ModelFactory._rerank_model.eval()
            
            ModelFactory._rerank_model.to(ModelFactory._device)
            logging.info(f"✅ Rerank 模型已成功常駐於 {ModelFactory._device}")
        except Exception as e:
            logging.error(f"❌ 初始化 Rerank 模型失敗: {e}")

    def rerank(self, query: str, documents: List[str], top_n: int = 3) -> List[Dict[str, Any]]:
        """
        使用已常駐記憶體的 Rerank 模型進行高速重排序 (純 Forward 推理)
        """
        if not documents:
            return []

        if ModelFactory._rerank_model is None or ModelFactory._rerank_tokenizer is None:
            logging.warning("⚠️ Rerank 模型未就緒，啟動降級策略。")
            return [{"index": idx, "document": doc, "relevance_score": 0.0} for idx, doc in enumerate(documents[:top_n])]

        try:
            pairs = [[query, doc] for doc in documents]
            
            with torch.no_grad():
                inputs = ModelFactory._rerank_tokenizer(
                    pairs, 
                    padding=True, 
                    truncation=True, 
                    return_tensors="pt", 
                    max_length=512
                )
                inputs = {k: v.to(ModelFactory._device) for k, v in inputs.items()}
                
                # 直接調用常駐模型進行推理，極速返回
                scores = ModelFactory._rerank_model(**inputs).logits.squeeze(-1).float().cpu().tolist()
                
                if isinstance(scores, float):
                    scores = [scores]

            results = []
            for idx, (doc, score) in enumerate(zip(documents, scores)):
                results.append({
                    "index": idx,
                    "document": doc,
                    "relevance_score": float(score)
                })

            results = sorted(results, key=lambda x: x["relevance_score"], reverse=True)[:top_n]
            return results

        except Exception as e:
            logging.error(f"❌ 本地 Rerank 推理失敗: {e}")
            return [
                {"index": idx, "document": doc, "relevance_score": 0.0}
                for idx, doc in enumerate(documents[:top_n])
            ]

    def get_embeddings(self, input_texts: List[str], model: str = "Qwen/Qwen3-Embedding-8B", batch_size: int = 64) -> List[List[float]]:
        if not input_texts:
            return []

        all_embeddings = []
        try:
            for i in range(0, len(input_texts), batch_size):
                batch_texts = input_texts[i:i + batch_size]
                response = ModelFactory._openai_client.embeddings.create(
                    model=model,
                    input=batch_texts
                )
                all_embeddings.extend([data.embedding for data in response.data])
            return all_embeddings
        except Exception as e:
            logging.error(f"❌ Embedding 请求失败 (模型: {model}): {e}")
            raise e


# =====================================================================
# 🧪 快速验证测试
# =====================================================================
if __name__ == "__main__":
    factory = ModelFactory.get_instance()
    
    # 1. 测试 Prompt Hub
    print("📂 Prompts 資源:", list(factory.prompts.keys()))

    # 2. 测试 LLM 文本生成
    client = factory.get_llm_client()
    res = client.chat.completions.create(
        model="qwen3-4b",
        messages=[{"role": "user", "content": "你好，请自我介绍"}]
    )
    print("🤖 LLM 输出:", res.choices[0].message.content)
    print("== LLM 测试完成 ==", "=" * 30)

    # 3. 测试 Qwen/Qwen3-Embedding-8B 向量化 (使用一致的 client 風格)
    sample_texts = [
        "Python 自动化架构设计与模块化脚本",
        "DGX Spark GPU 算力配置与显存管理",
        "大模型 RAG 检索增强生成与本地 Rerank 优化"
    ]
    print(f"🔍 正在對 {len(sample_texts)} 筆文本調用 Qwen/Qwen3-Embedding-8B 進行向量化...")
    
    embedding_res = client.embeddings.create(
        model="Qwen/Qwen3-Embedding-8B",
        input=sample_texts
    )
    embeddings = [data.embedding for data in embedding_res.data]
    print(f"🎯 Embedding 輸出成功！向量維度: {len(embeddings[0])} | 總筆數: {len(embeddings)}")
    print("== Embedding 测试完成 ==", "=" * 30)

    # 4. 測試本地直連 Reranker 
    docs = ["Python 自动化", "DGX Spark GPU 算力", "天气很好"]
    query = "GPU 算力配置"
    
    print(f"🔍 正在對 Query: '{query}' 進行本地 Rerank 重排序...")
    ranked_docs = factory.rerank(query, docs, top_n=3)
    
    print("🎯 Rerank 本地推理結果:")
    for item in ranked_docs:
        print(f"  - [Score: {item['relevance_score']:.4f}] 索引: {item['index']} | 文檔: {item['document']}")
    print("== Rerank 测试完成 ==", "=" * 30)