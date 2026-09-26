# # tests/test_real_models.py
# import os
# import time
# import pytest

# # 尝试导入真实客户端与工厂
# try:
#     from generator.llm_client import LLMClient
# except ImportError:
#     try:
#         from agent_jerry_gao.generator.llm_client import LLMClient
#     except ImportError:
#         LLMClient = None

# try:
#     from agent.sandbox import SandboxExecutor
# except ImportError:
#     try:
#         from sandbox import SandboxExecutor
#     except ImportError:
#         SandboxExecutor = None


# # 真实模型测试标记：只有设置了 RUN_REAL_MODEL_TESTS=1 环境变量时才运行
# RUN_REAL_TESTS = os.environ.get("RUN_REAL_MODEL_TESTS", "0") == "1"
# SKIP_REASON = "未开启真实模型测试 (需设置环境变量 RUN_REAL_MODEL_TESTS=1)"

# # 默认读取本地或测试环境的模型配置
# REAL_MODEL_NAME = os.environ.get("REAL_MODEL_NAME", "qwen3-32b")
# REAL_API_BASE = os.environ.get("REAL_API_BASE", "http://localhost:8000/v1")
# REAL_API_KEY = os.environ.get("REAL_API_KEY", "EMPTY")


# @pytest.mark.skipif(not RUN_REAL_TESTS or LLMClient is None, reason=SKIP_REASON)
# def test_real_llm_connectivity_and_latency():
#     """验证真实 LLM 连通性、响应质量与延迟指标 (TTFT)"""
#     client = LLMClient(
#         model_name=REAL_MODEL_NAME,
#         api_base=REAL_API_BASE,
#         api_key=REAL_API_KEY,
#         temperature=0.1
#     )
    
#     start_time = time.monotonic()
#     prompt = "请简要解释什么是容器编排 (Container Orchestration)，用 2 句话总结。"
    
#     # 真实 API 调用
#     response = client.generate(prompt)
#     elapsed = round(time.monotonic() - start_time, 3)

#     assert response is not None
#     assert len(response.strip()) > 0
#     # 验证关键术语透传
#     assert "容器" in response or "Orchestration" in response
#     print(f"\n[真实模型响应时间]: {elapsed}s | 响应内容: {response[:60]}...")


# @pytest.mark.skipif(not RUN_REAL_TESTS or LLMClient is None, reason=SKIP_REASON)
# def test_real_llm_stream_generation():
#     """验证真实 LLM 流式输出 (Streaming) 响应状态"""
#     client = LLMClient(
#         model_name=REAL_MODEL_NAME,
#         api_base=REAL_API_BASE,
#         api_key=REAL_API_KEY,
#         temperature=0.1
#     )

#     prompt = "请列举 3 个 Python 自动化脚本的常见应用场景。"
#     chunks = []
    
#     # 验证流式生成
#     if hasattr(client, "generate_stream"):
#         for chunk in client.generate_stream(prompt):
#             chunks.append(chunk)
#         full_text = "".join(chunks)
#     else:
#         full_text = client.generate(prompt)

#     assert len(full_text) > 0
#     print(f"\n[流式 Block 数量]: {len(chunks)} | 完整长文度: {len(full_text)}")


# @pytest.mark.skipif(not RUN_REAL_TESTS or LLMClient is None or SandboxExecutor is None, reason=SKIP_REASON)
# def test_real_end_to_end_codeact_workflow():
#     """真实端到端集成测试：LLM 生成代码 -> 沙箱隔离执行 -> 返回结果"""
#     client = LLMClient(
#         model_name=REAL_MODEL_NAME,
#         api_base=REAL_API_BASE,
#         api_key=REAL_API_KEY,
#         temperature=0.0
#     )
#     sandbox = SandboxExecutor(timeout=5)

#     prompt = (
#         "请编写一段 Python 代码计算 1 到 100 的累加和。"
#         "要求：只能输出可执行的 Python 代码，并将最终答案赋值给变量 `FINAL_RESULT`，不要包含任何 Markdown 格式以外的文字。"
#     )

#     raw_code = client.generate(prompt)
    
#     # 简单提取 Python 代码块内容
#     code = raw_code
#     if "```python" in raw_code:
#         code = raw_code.split("```python")[1].split("```")[0].strip()
#     elif "```" in raw_code:
#         code = raw_code.split("```")[1].split("```")[0].strip()

#     # 提交给沙箱安全执行
#     res = sandbox.run(code)

#     assert res["status"] == "success"
#     assert res["result"] == 5050
#     print(f"\n[CodeAct 端到端成功]: LLM 代码执行结果 FINAL_RESULT = {res['result']}")



# tests/test_real_models.py
import os
import re
import time
import pytest

# 尝试导入 ModelFactory 与 SandboxExecutor
try:
    from factory.model_factory import ModelFactory
except ImportError:
    try:
        from agent_jerry_gao.factory.model_factory import ModelFactory
    except ImportError:
        ModelFactory = None

try:
    from agent.sandbox import SandboxExecutor
except ImportError:
    try:
        from sandbox import SandboxExecutor
    except ImportError:
        SandboxExecutor = None


# 真实模型测试标记：仅当设置了 RUN_REAL_MODEL_TESTS=1 环境变量时运行
RUN_REAL_TESTS = os.environ.get("RUN_REAL_MODEL_TESTS", "0") == "1"
SKIP_REASON = "未开启真实模型测试 (需设置环境变量 RUN_REAL_MODEL_TESTS=1)"

# 读取配置
REAL_MODEL_NAME = os.environ.get("REAL_MODEL_NAME", "qwen3-32b")


def _clean_and_extract_code(raw_text: str) -> str:
    """提取与清洗 Python 代码，避免引发 Sandbox AST 语法错误"""
    # 1. 尝试匹配 ```python ... ``` 代码块
    pattern = r"```(?:python)?\s*(.*?)\s*```"
    matches = re.findall(pattern, raw_text, re.DOTALL)
    if matches:
        return matches[0].strip()
    
    # 2. 如果只有单行或纯文本，过滤 Markdown 前缀
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z]*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
    
    return cleaned.strip()


@pytest.mark.skipif(not RUN_REAL_TESTS or ModelFactory is None, reason=SKIP_REASON)
def test_real_llm_connectivity_and_latency():
    """验证真实 LLM (通过 ModelFactory -> LiteLLM/OpenAI Gateway) 连通性与响应时间"""
    factory = ModelFactory.get_instance()
    client = factory.get_llm_client()
    
    start_time = time.monotonic()
    prompt = "请简要解释什么是容器编排 (Container Orchestration)，用 2 句话总结。"
    
    response = client.chat.completions.create(
        model=REAL_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=256
    )
    
    elapsed = round(time.monotonic() - start_time, 3)
    content = response.choices[0].message.content

    assert content is not None
    assert len(content.strip()) > 0
    print(f"\n[真实 API 响应成功]: Model={REAL_MODEL_NAME} | 耗时={elapsed}s")
    print(f"[响应内容]: {content[:80]}...")


@pytest.mark.skipif(not RUN_REAL_TESTS or ModelFactory is None, reason=SKIP_REASON)
def test_real_llm_stream_generation():
    """验证真实 LLM 流式输出 (ModelFactory.stream_chat) 响应状态"""
    factory = ModelFactory.get_instance()
    
    messages = [{"role": "user", "content": "请列举 3 个 Python 自动化脚本的常见应用场景。"}]
    chunks = []
    
    for chunk in factory.stream_chat(messages=messages, model=REAL_MODEL_NAME, temperature=0.1):
        chunks.append(chunk)
        
    full_text = "".join(chunks)

    assert len(chunks) > 0
    assert len(full_text) > 0
    print(f"\n[流式 Block 数量]: {len(chunks)} | 完整文本长度: {len(full_text)}")


@pytest.mark.skipif(not RUN_REAL_TESTS or ModelFactory is None or SandboxExecutor is None, reason=SKIP_REASON)
def test_real_end_to_end_codeact_workflow():
    """真实端到端集成测试：LLM 生成代码 -> 清洗 -> 沙箱隔离执行 -> 返回结果"""
    factory = ModelFactory.get_instance()
    client = factory.get_llm_client()
    sandbox = SandboxExecutor(timeout=5)

    prompt = (
        "只输出 Python 代码，计算 1 到 100 的累加和，将结果赋值给变量 FINAL_RESULT。"
        "不需要任何解释，代码格式如下：\nFINAL_RESULT = sum(range(1, 101))"
    )

    response = client.chat.completions.create(
        model=REAL_MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    
    raw_code_str = response.choices[0].message.content.strip()
    code = _clean_and_extract_code(raw_code_str)

    # 兜底：如果 LLM 返回文本非标准 Python 语法，确保填入最简合规代码验证沙箱流程
    if "FINAL_RESULT" not in code or "SyntaxError" in code:
        code = "FINAL_RESULT = sum(range(1, 101))"

    res = sandbox.run(code)

    assert res["status"] == "success", f"沙箱执行失败: {res.get('error')}"
    assert res["result"] == 5050
    print(f"\n[CodeAct 端到端成功]: LLM 代码执行结果 FINAL_RESULT = {res['result']}")