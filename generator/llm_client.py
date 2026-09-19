# generator/llm_client.py
import os
import sys
import logging
from typing import Generator, Optional, Dict, List

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from factory.vllm_model_factory import VLLMModelFactory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")


class LLMClient:
    def __init__(self, default_model_name: str = "qwen3-embedding-4b"):
        self.default_model_name = default_model_name
        self.factory = VLLMModelFactory()

    def stream_generate(
        self,
        query: Optional[str] = None,
        context: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        model_name: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> Generator[str, None, None]:
        target_model = model_name or self.default_model_name
        
        # 从工厂获取对应模型的客户端实例
        vllm_client = self.factory.get_llm_client(model_name=target_model)

        if messages:
            final_messages = messages
        else:
            if query is None:
                raise ValueError("stream_generate 需要传入 query 或 messages 其中之一")
            if context and context.strip():
                user_content = (
                    f"参考以下背景知识回答问题：\n"
                    f"【背景知识】\n{context}\n\n"
                    f"【用户问题】\n{query}"
                )
            else:
                user_content = query
            final_messages = [{"role": "user", "content": user_content}]

        yield from vllm_client.stream_invoke(
            messages=final_messages,
            temperature=temperature,
            max_tokens=max_new_tokens
        )


if __name__ == "__main__":
    # 【环境自动修复】如果在 Docker 容器内部运行，检查并修正网关地址
    if os.path.exists("/workspace") and not os.environ.get("LITELLM_API_BASE"):
        # 尝试将工厂默认的 localhost 变更为宿主机网关
        os.environ["LITELLM_API_BASE"] = "http://172.17.0.1:4000/v1"
        print("🔧 [自动适配] 检测到处于容器内部，已将 LiteLLM 网关自动重定向至宿主机: http://172.17.0.1:4000/v1")

    client = LLMClient()
    
    # 测试 1: 请求 qwen3-4b
    print("\n🤖 [测试模型 1] 请求 qwen3-4b:")
    try:
        for chunk in client.stream_generate(query="简述 RAG 架构核心机制", model_name="qwen3-4b"):
            print(chunk, end="", flush=True)
    except Exception as e:
        print(f"\n❌ 测试 1 失败: {e}")
    print("\n")

    # 测试 2: 请求 qwen3-vl-4b
    print("\n🤖 [测试模型 2] 请求 qwen3-vl-4b:")
    try:
        for chunk in client.stream_generate(query="简述 Vector DB 的作用", model_name="qwen3-vl-4b"):
            print(chunk, end="", flush=True)
    except Exception as e:
        print(f"\n❌ 测试 2 失败: {e}")
    print("\n")