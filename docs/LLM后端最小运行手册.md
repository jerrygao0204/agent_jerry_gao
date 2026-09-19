# LLM 后端最小运行手册

适用范围：仅验证 ModelFactory 的 LLM 后端切换（local / vllm / multi_vllm），不改业务代码。

## 0. 进入目录

cd /workspace/hf-conda/RAG/问答机器人

## 1. 准备环境变量

1) 首次使用可从模板复制：

cp .env.example .env

2) 核心变量（.env）：

MODEL_BACKEND=local
VLLM_ENDPOINTS_CONFIG=/workspace/hf-conda/RAG/问答机器人/config/vllm_endpoints.yaml

说明：
- MODEL_BACKEND 支持 local / vllm / multi_vllm
- vllm 与 multi_vllm 共用同一配置文件，差异只在 profiles 映射

## 2. 检查端点配置

当前配置文件：config/vllm_endpoints.yaml

最小要求：
- profiles.vllm.llm.base_url + model
- profiles.multi_vllm.llm.base_url + model

可选角色（已示例）：
- vlm
- embedding

## 3. 启动并探活 vLLM（远端后端时）

示例（单实例，端口 8000）：

python3 /workspace/test_vllm.py 8000

示例（多实例，按配置分别探活）：

python3 /workspace/test_vllm.py 8000
python3 /workspace/test_vllm.py 8001
python3 /workspace/test_vllm.py 8002

如果探活失败：
- 先确认对应 vllm serve 已启动
- 再确认 base_url 端口与 config/vllm_endpoints.yaml 一致

## 4. 工厂级最小验证（不走业务层）

执行下面命令可直接验证 get_llm_model() 兼容接口仍返回 (model, tokenizer)：

/usr/bin/python - <<'PY'
import os
from factory.model_factory import ModelFactory

backend = os.getenv("MODEL_BACKEND", "local")
f = ModelFactory(prompt_hub_path="prompt_hub.yaml")
model, tokenizer = f.get_llm_model("Qwen/Qwen3-4B")
print("backend=", backend)
print("model_has_generate=", hasattr(model, "generate"))
print("tokenizer_has_apply_chat_template=", hasattr(tokenizer, "apply_chat_template"))
PY

期望输出：
- model_has_generate= True
- tokenizer_has_apply_chat_template= True

## 5. 三种后端最小切换

A) local：

export MODEL_BACKEND=local
/usr/bin/python - <<'PY'
from factory.model_factory import ModelFactory
f = ModelFactory(prompt_hub_path="prompt_hub.yaml")
m, t = f.get_llm_model("Qwen/Qwen3-4B")
print(type(m).__name__, type(t).__name__)
PY

B) vllm：

export MODEL_BACKEND=vllm
export VLLM_ENDPOINTS_CONFIG=/workspace/hf-conda/RAG/问答机器人/config/vllm_endpoints.yaml
/usr/bin/python - <<'PY'
from factory.model_factory import ModelFactory
f = ModelFactory(prompt_hub_path="prompt_hub.yaml")
m, t = f.get_llm_model("Qwen/Qwen3-8B")
print(type(m).__name__, type(t).__name__)
PY

C) multi_vllm：

export MODEL_BACKEND=multi_vllm
export VLLM_ENDPOINTS_CONFIG=/workspace/hf-conda/RAG/问答机器人/config/vllm_endpoints.yaml
/usr/bin/python - <<'PY'
from factory.model_factory import ModelFactory
f = ModelFactory(prompt_hub_path="prompt_hub.yaml")
m, t = f.get_llm_model("Qwen/Qwen3-4B")
print(type(m).__name__, type(t).__name__)
PY

## 6. 资源释放

调用：

/usr/bin/python - <<'PY'
from factory.model_factory import ModelFactory
ModelFactory.destroy_llm_model()
print("done")
PY

说明：
- local 会执行显存回收逻辑
- remote 仅关闭客户端连接，不会停止 vllm serve 进程
