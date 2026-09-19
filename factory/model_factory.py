# factory/model_factory.py
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
    模型与环境配置中心工厂 (ModelFactory - LiteLLM Unified Mode)
    全局接管：Prompt Hub 资产管理、LiteLLM API 句柄分发、Rerank 与 Embedding 统一调用
    """
    _instance: Optional["ModelFactory"] = None
    _lock: threading.Lock = threading.Lock()
    _openai_client: Optional[OpenAI] = None

    # 🌟 类级别全局缓存：确保 Rerank 模型与分词器在整个生命周期中只被加载一次
    _rerank_tokenizer = None
    _rerank_model = None
    _device = "cuda" if torch.cuda.is_available() else "cpu"

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelFactory, cls).__new__(cls)
        return cls._instance

    def __init__(self, prompt_hub_path: str = "prompt_hub.yaml", cache_dir: str = "/workspace/hf-conda/hf_cache/hub"):
        if getattr(self, "_initialized", False):
            return
        with self._lock:
            if getattr(self, "_initialized", False):
                return
            load_dotenv()

            # 1. 配置 Endpoint & Key (默认指向 LiteLLM 统一代理)
            self.base_url = os.getenv("OPENAI_BASE_URL", "http://172.17.0.1:4000/v1").rstrip("/")
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

            # 🌟 5. 确保在启动初始化时「仅加载一次」Rerank 模型到 GPU
            self._init_rerank_model()

            self._initialized = True
            logging.info(f"🚀 ModelFactory (LiteLLM Mode) 初始化完成 | Endpoint: {self.base_url}")

    def _resolve_path(self, path: str) -> str:
        if os.path.isabs(path):
            return path
        factory_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(factory_dir)
        filename = os.path.basename(path)
        return os.path.join(project_root, "config", filename)

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

    def setup_cuda_device(self, device_str: str = "0") -> str:
        """旧接口兼容方法"""
        return ModelFactory._device

    def _init_rerank_model(self):
        """内部方法：在启动时预先加载 Rerank 模型，避免每次查询重复加载"""
        if ModelFactory._rerank_model is not None:
            return
        model_path = Path("/workspace/hf-conda/hf_cache/hub/models--BAAI--bge-reranker-large/snapshots/55611d7bca2a7133960a6d3b71e083071bbfc312")
        if not model_path.exists():
            host_fallback = Path("/home/gaozheng/venv/hf-conda/hf_cache/hub/models--BAAI--bge-reranker-large/snapshots/55611d7bca2a7133960a6d3b71e083071bbfc312")
            if host_fallback.exists():
                model_path = host_fallback

        try:
            logging.info(f"⏳ 正在初始化本地 Rerank 模型至设备: {ModelFactory._device}...")
            ModelFactory._rerank_tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            ModelFactory._rerank_model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
            ModelFactory._rerank_model.eval()
            ModelFactory._rerank_model.to(ModelFactory._device)
            logging.info(f"✅ Rerank 模型已成功常驻于 {ModelFactory._device}")
        except Exception as e:
            logging.error(f"❌ 初始化 Rerank 模型失败: {e}")

    def rerank(self, query: str, documents: List[str], top_n: int = 3) -> List[Dict[str, Any]]:
        """使用已常驻内存的 Rerank 模型进行高速重排序 (纯 Forward 推理)"""
        if not documents:
            return []
        if ModelFactory._rerank_model is None or ModelFactory._rerank_tokenizer is None:
            logging.warning("⚠️ Rerank 模型未就绪，启动降级策略。")
            return [{"index": idx, "document": doc, "relevance_score": 0.0} for idx, doc in enumerate(documents[:top_n])]

        try:
            pairs = [[query, doc] for doc in documents]
            with torch.no_grad():
                inputs = ModelFactory._rerank_tokenizer(
                    pairs, padding=True, truncation=True, return_tensors="pt", max_length=512
                )
                inputs = {k: v.to(ModelFactory._device) for k, v in inputs.items()}
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
            logging.error(f"❌ Rerank 执行失败: {e}")
            return [{"index": idx, "document": doc, "relevance_score": 0.0} for idx, doc in enumerate(documents[:top_n])]

    def embed_query(self, text: str, model: str = "qwen3-embedding-4b") -> List[float]:
        """單條文本向量化（原 VLLMModelFactory.get_embedding_client().embed_query 合並於此）"""
        response = self.get_llm_client().embeddings.create(model=model, input=text)
        return response.data[0].embedding

    def embed_texts(self, input_texts: List[str], model: str = "qwen3-embedding-4b", batch_size: int = 64) -> List[List[float]]:
        """批量文本向量化（原 litellm_model_factory.get_embeddings 合並於此）"""
        if not input_texts:
            return []
        all_embeddings: List[List[float]] = []
        client = self.get_llm_client()
        for i in range(0, len(input_texts), batch_size):
            batch = input_texts[i:i + batch_size]
            response = client.embeddings.create(model=model, input=batch)
            all_embeddings.extend([data.embedding for data in response.data])
        return all_embeddings

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        """流式對話生成（原 VLLMTextClient.stream_invoke 合並於此，走 OpenAI SDK 而非手拼 urllib）"""
        stream = self.get_llm_client().chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta and delta.content:
                yield delta.content

    def get_vlm_model(self, vlm_short_name: str = "Qwen/Qwen3-VL-32B-Instruct"):
        """
        [LiteLLM 模式適配] 
        原本返回本地 VLM 模型與 Processor，現統一返回 LiteLLM OpenAI Client 
        以及對應的模型名稱，供上層流水線透過標準 Chat Completions (Vision) 進行多模態推理。
        """
        logging.info(f"ℹ️ 委託 ModelFactory: VLM 模型 [{vlm_short_name}] 已轉由 LiteLLM 統一代理調度。")
        return self.get_llm_client(), vlm_short_name

    # 兼容性空实现或显存监控方法，防范旧逻辑调用报错
    @classmethod
    def destroy_all_models_cls(cls):
        logging.info("ℹ️ LiteLLM 模式下无本地 LLM/VLM 模型显存需要物理销毁。")

    def resolve_model_path(self, model_name_or_path: str) -> str:
        """
        兼容 Reranker / Embedding 等本地物理快照路径解析
        """
        # 1. 如果传入的本身就是绝对路径或相对路径且存在，直接返回
        if os.path.exists(model_name_or_path):
            return model_name_or_path
            
        # 2. 如果请求的是 bge-reranker-large，直接返回工厂内部已经验证过的物理路径
        if "bge-reranker-large" in model_name_or_path:
            # 复用工厂内部已有的路径检查逻辑
            primary_path = Path("/workspace/hf-conda/hf_cache/hub/models--BAAI--bge-reranker-large/snapshots/55611d7bca2a7133960a6d3b71e083071bbfc312")
            if primary_path.exists():
                return str(primary_path)
            host_fallback = Path("/home/gaozheng/venv/hf-conda/hf_cache/hub/models--BAAI--bge-reranker-large/snapshots/55611d7bca2a7133960a6d3b71e083071bbfc312")
            if host_fallback.exists():
                return str(host_fallback)
                
        # 3. 兜底：直接返回原字符串
        return model_name_or_path

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
    print(f"🔍 正在對 {len(sample_texts)} 筆文本調用 Qwen3-Embedding-4B 進行向量化...")
    
    embedding_res = client.embeddings.create(
        model="qwen3-embedding-4b",
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

    # 5. 測試 resolve_model_path 物理路徑解析
    print(f"🔍 正在測試 ModelFactory.resolve_model_path 路徑解析...")
    
    test_cases = [
        "BAAI/bge-reranker-large",                          # 預期命中本地快照路徑
        "/workspace/hf-conda/hf_cache/hub",                 # 預期返回原路徑（存在）
        "unknown/model-name-xyz"                            # 預期走兜底邏輯返回原字符串
    ]
    
    for case in test_cases:
        resolved_path = factory.resolve_model_path(case)
        print(f"  - 原始輸入: {case} ---> 解析結果: {resolved_path}")
        
    print("== resolve_model_path 测试完成 ==", "=" * 30)