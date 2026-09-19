# factory/vllm_model_factory.py
import os
import json
import logging
import base64
import urllib.request
import time
from io import BytesIO
from typing import List, Dict, Any, Optional, Generator
from PIL import Image
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

# =====================================================================
# 1. Base / Text / Vision Clients (保持原有 HTTP 與 SSE 實現)
# =====================================================================
class BaseVLLMClient:
    def __init__(self, endpoint_url: str, model_name: str, default_temperature: float = 0.2):
        clean_base = endpoint_url.rstrip("/")
        if clean_base.endswith("/v1/chat/completions"):
            self.endpoint_url = clean_base
        elif clean_base.endswith("/v1"):
            self.endpoint_url = f"{clean_base}/chat/completions"
        else:
            self.endpoint_url = f"{clean_base}/v1/chat/completions"
            
        self.model_name = model_name
        self.default_temperature = default_temperature

    def _post(self, payload: Dict[str, Any], timeout: int = 120) -> str:
        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(
            self.endpoint_url, 
            data=json.dumps(payload).encode('utf-8'), 
            headers=headers
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                return result['choices'][0]['message']['content']
        except Exception as e:
            logging.error(f"❌ LiteLLM 閘道器請求失敗 [{self.endpoint_url}]: {e}")
            raise RuntimeError(f"LiteLLM Gateway Client Error: {e}")

    def _post_stream(self, payload: Dict[str, Any], timeout: int = 120) -> Generator[str, None, None]:
        headers = {"Content-Type": "application/json"}
        payload["stream"] = True
        req = urllib.request.Request(
            self.endpoint_url, 
            data=json.dumps(payload).encode('utf-8'), 
            headers=headers
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                for line in resp:
                    line_str = line.decode('utf-8').strip()
                    if not line_str or line_str.startswith(":"):
                        continue
                    if line_str.startswith("data: "):
                        data_content = line_str[6:]
                        if data_content == "[DONE]":
                            break
                        try:
                            chunk_json = json.loads(data_content)
                            if chunk_json['choices'] and 'delta' in chunk_json['choices'][0]:
                                delta = chunk_json['choices'][0]['delta']
                                if 'content' in delta and delta['content']:
                                    yield delta['content']
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logging.error(f"❌ LiteLLM SSE 流式請求失敗 [{self.endpoint_url}]: {e}")
            raise RuntimeError(f"LiteLLM Gateway Stream Error: {e}")


class VLLMTextClient(BaseVLLMClient):
    def invoke(self, prompt: str, temperature: Optional[float] = None, max_tokens: int = 2048) -> str:
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature if temperature is not None else self.default_temperature,
            "max_tokens": max_tokens
        }
        return self._post(payload)

    def stream_invoke(
        self, 
        messages: List[Dict[str, str]], 
        temperature: Optional[float] = None, 
        max_tokens: int = 2048
    ) -> Generator[str, None, None]:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature if temperature is not None else self.default_temperature,
            "max_tokens": max_tokens
        }
        yield from self._post_stream(payload)


class VLLMVisionClient(BaseVLLMClient):
    @staticmethod
    def _image_to_base64(image: Image.Image) -> str:
        buf = BytesIO()
        image.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def invoke(
        self, 
        prompt: str, 
        images: Optional[List[Image.Image]] = None, 
        temperature: Optional[float] = None, 
        max_tokens: int = 2048
    ) -> str:
        content_payload: List[Dict[str, Any]] = []
        if images:
            for img in images:
                b64_str = self._image_to_base64(img)
                content_payload.append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/png;base64,{b64_str}"}
                })
        content_payload.append({"type": "text", "text": prompt})

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": content_payload}],
            "temperature": temperature if temperature is not None else self.default_temperature,
            "max_tokens": max_tokens
        }
        return self._post(payload)

class VLLMEmbeddingClient(BaseVLLMClient):
    def __init__(self, endpoint_url: str, model_name: str):
        # 修正 Embedding 的端点路径指向 /v1/embeddings
        clean_base = endpoint_url.rstrip("/")
        if clean_base.endswith("/v1/chat/completions"):
            clean_base = clean_base.replace("/v1/chat/completions", "/v1")
        
        if clean_base.endswith("/v1"):
            self.endpoint_url = f"{clean_base}/embeddings"
        elif clean_base.endswith("/embeddings"):
            self.endpoint_url = clean_base
        else:
            self.endpoint_url = f"{clean_base}/v1/embeddings"
            
        self.model_name = model_name

    def embed_query(self, text: str) -> List[float]:
        """向 LiteLLM 閘道器發送 Embedding 請求"""
        payload = {
            "model": self.model_name,
            "input": text
        }
        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(
            self.endpoint_url, 
            data=json.dumps(payload).encode('utf-8'), 
            headers=headers
        )
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode('utf-8'))
                return result['data'][0]['embedding']
        except Exception as e:
            logging.error(f"❌ LiteLLM 閘道器 Embedding 請求失敗 [{self.endpoint_url}]: {e}")
            raise RuntimeError(f"LiteLLM Gateway Embedding Error: {e}")
        
# =====================================================================
# 2. 應用層解耦工廠 (完全基於環境變數與 LiteLLM 閘道器通訊)
# =====================================================================
class VLLMModelFactory:
    """
    輕量級應用層模型工廠
    遵循架構解耦原則：不讀取底層 YAML，統一透過 LITELLM_ENDPOINT 導向網關
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(VLLMModelFactory, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, '_initialized', False):
            return
        
        load_dotenv()
        # 🌟 核心變更：統一透過環境變數獲取 LiteLLM 閘道器位址
        self.default_endpoint = os.getenv("LITELLM_ENDPOINT", "http://172.17.0.1:4000")
        
        logging.info(f"🌐 應用層模型工廠初始化完成，統一網關路由: {self.default_endpoint}")
        self._initialized = True

    def get_llm_client(self, model_name: str = "qwen3-4b") -> VLLMTextClient:
        """獲取純文本 LLM Client (直接指向 LiteLLM 閘道器)"""
        return VLLMTextClient(endpoint_url=self.default_endpoint, model_name=model_name)

    def get_vlm_client(self, model_name: str = "qwen3-vl-4b") -> VLLMVisionClient:
        """獲取多模態 VLM Client (直接指向 LiteLLM 閘道器)"""
        return VLLMVisionClient(endpoint_url=self.default_endpoint, model_name=model_name)

    def get_embedding_client(self, model_name: str = "qwen3-embedding-4b") -> VLLMEmbeddingClient:
        """獲取文本向量 Embedding Client (直接指向 LiteLLM 閘道器)"""
        return VLLMEmbeddingClient(endpoint_url=self.default_endpoint, model_name=model_name)

# =====================================================================
# 3. 本地集成驗證入口 (對應 LiteLLM 閘道器架構)
# =====================================================================
# =====================================================================
# 3. 本地集成驗證入口 (修復：多模態測試帶入真實圖片)
# =====================================================================
if __name__ == "__main__":
    print("\n===================================================")
    print("🚀 開始驗證應用層與 LiteLLM 閘道器的連通性...")
    print("===================================================\n")

    factory = VLLMModelFactory()

    # 1. 驗證純文本模型 (對應 LiteLLM 中的 qwen3-4b)
    print("1️⃣ [測試 LiteLLM 閘道器] 正在向 qwen3-4b 發送測試請求...")
    client_text = factory.get_llm_client(model_name="qwen3-4b")
    t0 = time.time()
    response_text = client_text.invoke(prompt="用一句話總結什麼是流程自动化 (Process Automation)。")
    cost_text = time.time() - t0
    print(f"✅ [qwen3-4b 響應成功] (耗時: {cost_text:.2f}s):")
    print(f"└─ 結果: {response_text[:50].strip()}...\n")

    # 2. 驗證視覺多模態模型 (對應 LiteLLM 中的 qwen3-vl-4b，必須帶圖)
    print("2️⃣ [測試 LiteLLM 閘道器] 正在向 qwen3-vl-4b 發送多模態測試請求...")
    client_vision = factory.get_vlm_client(model_name="qwen3-vl-4b")
    
    # 建立一張簡單的測試圖片 (用於驗證 VLM 圖像傳輸通道)
    test_img = Image.new('RGB', (200, 100), color=(73, 109, 137))
    
    t1 = time.time()
    response_vision = client_vision.invoke(
        prompt="請描述這張圖片的背景顏色，並說明架構解耦的核心價值。", 
        images=[test_img]
    )
    cost_vision = time.time() - t1
    print(f"✅ [qwen3-vl-4b 響應成功] (耗時: {cost_vision:.2f}s):")
    print(f"└─ 結果: {response_vision[:50].strip()}\n")

    # 3. 驗證文本向量模型 (對應 LiteLLM 中的 qwen3-embedding-4b)
    print("3️⃣ [測試 LiteLLM 閘道器] 正在向 qwen3-embedding-4b 發送 Embedding 請求...")
    # 假設你的 factory 提供了獲取 embedding 客戶端的方法，或者使用 LangChain 的 OpenAIEmbeddings 對接 LiteLLM 代理
    client_embedding = factory.get_embedding_client(model_name="qwen3-embedding-4b")
    
    t2 = time.time()
    # 執行向量化測試
    test_text_to_embed = "流程自動化與架構解耦能顯著減少重複勞動。"
    embedding_result = client_embedding.embed_query(test_text_to_embed)
    cost_embedding = time.time() - t2
    
    print(f"✅ [qwen3-embedding-4b 響應成功] (耗時: {cost_embedding:.2f}s):")
    print(f"└─ 向量維度 (Dimension): {len(embedding_result)}")
    print(f"└─ 向量預覽 (前 5 維): {embedding_result[:5]}...\n")
    
    print("===================================================")
    print("🎉 應用層與底層網關解耦驗證通過！")
    print("===================================================\n")
