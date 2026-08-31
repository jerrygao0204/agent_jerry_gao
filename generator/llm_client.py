# generator/llm_client.py
import os
import sys
import logging
from typing import Generator, Optional, Dict, List
from threading import Thread

# 📂 动态计算项目根目录，并将其强行注入系统路径中
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 从独立的全局工厂引入枢纽
from factory import ModelFactory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

try:
    from transformers import TextIteratorStreamer
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
    from transformers import TextIteratorStreamer

class FineBILLMClient:
    def __init__(self, model_short_name: str = "Qwen/Qwen3-32B", prompt_hub_path: str = "prompt_hub.yaml", cuda_device: str = "0"):
        """
        面向业务的高性能大模型推理客户端
        """
        # 初始化中央模型工厂并绑定显卡
        self.factory = ModelFactory(prompt_hub_path=prompt_hub_path)
        self.factory.setup_cuda_device(cuda_device)
        self.model_short_name = model_short_name

    def stream_generate(
        self,
        query: Optional[str] = None,
        context: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs
    ) -> Generator[str, None, None]:
        """
        标准的工业级流式文本问答生成函数
        :param query: 用户提问（单轮模式使用，若传了 messages 则忽略）
        :param context: RAG 检索召回并重排后的上下文文本（单轮模式使用）
        :param messages: 多轮对话消息列表 [{"role": "user"/"assistant", "content": ...}, ...]
                        传入时走标准 chat-template 多轮模式，query/context 会被忽略
        :param max_new_tokens: 最大生成 Token 数
        :param temperature: 采样温度
        """
        # 1. 每次生成请求时，向工厂索取模型与 Tokenizer 句柄
        model, tokenizer = self.factory.get_llm_model(self.model_short_name)

        # 2. 组装 messages：优先使用调用方直接传入的多轮列表
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

        text = tokenizer.apply_chat_template(final_messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        generation_kwargs = dict(
            model_inputs,
            streamer=streamer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature
        )

        # 3. 线程并行启动异步推理
        thread = Thread(target=model.generate, kwargs=generation_kwargs)
        thread.start()

        for new_text in streamer:
            yield new_text

    def close(self):
        """彻底回收纯文本显存"""
        self.factory.destroy_llm_model()

if __name__ == "__main__":
    # 本地链路实战验证
    client = FineBILLMClient(model_short_name="Qwen/Qwen3-32B", cuda_device="0")
    
    test_query = "独立模型工厂架构有什么深远的工程设计优势？"
    print("\n🤖 Qwen 文本大模型正在流式作答：")
    for chunk in client.stream_generate(query=test_query, context="背景资料：模型工厂实现了显存单例模式"):
        print(chunk, end="", flush=True)
    print("\n")