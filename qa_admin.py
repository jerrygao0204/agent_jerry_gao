# qa_admin.py
import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import gradio as gr
except ImportError:
    install_package("gradio")
    import gradio as gr

import os
import sys
import gc
import json
import uuid
import logging
import time
import torch
import yaml
from typing import Dict, Any, List, Tuple, Generator, Optional
from urllib.parse import unquote

# ==========================================
# 📂 1. 动态注入系统路径与模块导入
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

from factory.model_factory import ModelFactory
from generator.qa_chain import QAChain
from generator.llm_client import FineBILLMClient
from search.retriever import FineBIRetriever
from search.reranker import FineBIReranker
from factory import init_tools, tool_factory
from agent.sandbox import SandboxExecutor
from agent.react_agent_integrated import IntegratedReActAgent
from memory.memory_manager import MemoryManager
from agent.compliance import ComplianceChecker

# ==========================================
# ⚙️ 2. 全局配置与用户凭证加载
# ==========================================
CONFIG_FILE_PATH = os.path.join(SCRIPT_DIR, "qa_config.json")
USERS_AUTH_PATH = os.path.join(SCRIPT_DIR, "config", "users_auth.yaml")

DEFAULT_QA_CONFIG = {
    "prompts_hub_path": os.path.join(SCRIPT_DIR, "config", "prompt_hub.yaml"),
    "milvus_host": "172.17.0.1",
    "milvus_port": "19530",
    "collection_name": "finebi_knowledge_chunks",
    "llm_model_name": "Qwen/Qwen3-4B",
    "vlm_model_name": "Qwen/Qwen3-VL-32B-Instruct",
    "emb_model_name": "Qwen/Qwen3-Embedding-8B",
    "cuda_device": "0",
    "top_k_retrieval": 10,
    "top_k_rerank": 3
}

LLM_OPTIONS = [
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-32B",
    "Qwen/Qwen3-VL-32B-Instruct"
]

global_qa_chain = None
global_compliance_checker = None
# 缓存活跃的 MemoryManager 实例: {user_id: MemoryManager}
user_memory_managers: Dict[str, MemoryManager] = {}

def load_user_credentials() -> Dict[str, str]:
    """从 yaml 文件安全加载用户凭证"""
    if os.path.exists(USERS_AUTH_PATH):
        try:
            with open(USERS_AUTH_PATH, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                return data.get("users", {"admin": "admin123"})
        except Exception as e:
            logging.error(f"加载 users_auth.yaml 失败: {e}")
    return {"admin": "admin123"}

VALID_USERS = load_user_credentials()

def get_or_create_user_memory(username: str, session_id: Optional[str] = None) -> MemoryManager:
    """获取或初始化对应用户的 MemoryManager"""
    if username not in user_memory_managers:
        logging.info(f"🛠️ 为账号 [{username}] 初始化 MemoryManager...")
        user_memory_managers[username] = MemoryManager(user_id=username, session_id=session_id, max_messages=20)
    elif session_id and user_memory_managers[username].session_id != session_id:
        user_memory_managers[username].switch_session(session_id)
        
    return user_memory_managers[username]

def fetch_session_dropdown_choices(username: str) -> List[Tuple[str, str]]:
    """获取指定用户的历史会话列表，用于 Radio/Dropdown 组件展示"""
    mem_mgr = get_or_create_user_memory(username)
    sessions = mem_mgr.get_recent_sessions_list()
    choices = [(f"💬 {s.get('title', '新对话')} ({s.get('updated_at', '')[5:16]})", s['session_id']) for s in sessions]
    return choices

# ==========================================
# 🛠️ 3. 显存管理与安全函数
# ==========================================
def get_gpu_memory_status() -> str:
    if not torch.cuda.is_available():
        return "GPU 不可用 (CPU Mode)"
    try:
        device_id = 0
        allocated = torch.cuda.memory_allocated(device_id) / (1024 ** 3)
        reserved = torch.cuda.memory_reserved(device_id) / (1024 ** 3)
        total = torch.cuda.get_device_properties(device_id).total_memory / (1024 ** 3)
        return f"GPU 0: 已分配 {allocated:.2f} GB | 已预留 {reserved:.2f} GB | 总显存 {total:.2f} GB"
    except Exception as e:
        return f"显存获取异常: {str(e)}"

def emergency_force_cleanup() -> str:
    global global_qa_chain
    logging.warning("🚨 [QA Admin] 触发应急显存回收操作！")
    global_qa_chain = None

    try:
        if hasattr(ModelFactory, "_instance"):
            ModelFactory._instance = None
    except Exception as e:
        logging.error(f"清空 ModelFactory 单例句柄失败: {e}")

    try:
        ModelFactory.destroy_all_models_cls()
    except Exception as e:
        logging.error(f"调用 ModelFactory.destroy_all_models_cls 失败: {e}")

    gc.collect(2)

    if torch.cuda.is_available():
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception as e:
            logging.error(f"CUDA 显存回收异常: {e}")

    logging.info("✨ [QA Admin] 物理显存清理完毕！")
    return get_gpu_memory_status()

def get_compliance_checker() -> ComplianceChecker:
    global global_compliance_checker
    if global_compliance_checker is None:
        global_compliance_checker = ComplianceChecker()
    return global_compliance_checker

def clear_agent_memory(user_state: dict):
    """【修正】物理删除当前 Session，清空内存并更新前端 Radio 选择框"""
    username = user_state.get("username", "default") if user_state else "default"
    mem_mgr = get_or_create_user_memory(username)
    current_session_id = mem_mgr.session_id

    # 1. 物理删除持久化存储中的当前 Session
    if hasattr(mem_mgr, "history_storage") and current_session_id:
        try:
            mem_mgr.history_storage.delete_session(username, current_session_id)
            logging.info(f"🗑️ 已成功从存储中删除用户 [{username}] 的会话 [{current_session_id}]")
        except Exception as e:
            logging.error(f"❌ 删除会话记录失败: {e}")

    # 2. 清空内存对象
    mem_mgr.clear_all()

    # 3. 重新获取最新的会话列表
    choices = fetch_session_dropdown_choices(username)

    # 4. 如果会话已被空，自动生成一个全新的 Session
    if not choices:
        new_sess_id = str(uuid.uuid4())
        mem_mgr.switch_session(new_sess_id)
        if hasattr(mem_mgr, "history_storage"):
            mem_mgr.history_storage.create_session(username, new_sess_id, title="新对话")
        choices = fetch_session_dropdown_choices(username)
        new_choice = new_sess_id
    else:
        new_choice = choices[0][1]
        mem_mgr.switch_session(new_choice)

    return [], "*等待启动诊断...*", "✅ 已成功清空对话与记忆", "", gr.update(choices=choices, value=new_choice)

# def clear_agent_memory(user_state: dict):
#     username = user_state.get("username", "default")
#     mem_mgr = get_or_create_user_memory(username)
#     mem_mgr.clear_all()
#     choices = fetch_session_dropdown_choices(username)
#     new_choice = choices[0][1] if choices else None
#     return [], "*等待启动诊断...*", "已清空对话与记忆", "", gr.update(choices=choices, value=new_choice)

# ==========================================
# 🤖 4. 在线 QA 与 Agent 推理逻辑
# ==========================================
current_llm_choice = None

def get_qa_chain(llm_choice: str, top_k_ret: int, top_k_rerank: int):
    global global_qa_chain, current_llm_choice
    top_k_ret, top_k_rerank = int(top_k_ret), int(top_k_rerank)

    if global_qa_chain is None or current_llm_choice != llm_choice:
        logging.info(f"🚀 [QA Admin] 初始化/更新全局 QAChain 句柄, 模型: {llm_choice}")
        global_qa_chain = QAChain(
            cuda_device="0",
            prompt_hub_path=DEFAULT_QA_CONFIG["prompts_hub_path"],
            top_k_retrieval=top_k_ret,
            top_k_rerank=top_k_rerank,
            llm_short_name=llm_choice,
        )
        current_llm_choice = llm_choice
    else:
        global_qa_chain.top_k_retrieval = top_k_ret
        global_qa_chain.top_k_rerank = top_k_rerank

    return global_qa_chain

def format_sources_log(llm_choice: str, top_k_ret: int, top_k_rerank: int, filter_pattern: str, chunks: List[Dict[str, Any]]) -> str:
    log_lines = [f"🔍 **检索与重排配置**: LLM={llm_choice} | Retrieval Top-K={top_k_ret} | Rerank Top-K={top_k_rerank}"]
    log_lines.append(f"🎯 **Filter 表达式**: `{filter_pattern}`" if filter_pattern else "ℹ️ **Filter 表达式**: 无（纯语义+BM25混合召回）")
    log_lines.append("─" * 50)

    if not chunks:
        log_lines.append("⚠️ **提示**: 重排过滤后无满足得分阈值的有效切片，已阻断回答。")
        return "\n".join(log_lines) + "\n"

    for idx, chunk in enumerate(chunks, 1):
        raw_score = chunk.get("rerank_score", chunk.get("score", 0.0))
        try:
            score = float(raw_score) if raw_score is not None else 0.0
        except (ValueError, TypeError):
            score = 0.0

        doc_name = chunk.get("doc_name") or chunk.get("biz_summary") or "未知文档"
        content = chunk.get("content") or chunk.get("base_content") or ""
        content_preview = str(content)[:300].replace("\n", " ")

        log_lines.append(f"**[{idx}] {doc_name}** (Rerank 得分: **{score:.4f}**)")
        log_lines.append(f"```text\n{content_preview}...\n```")

    return "\n".join(log_lines) + "\n"

# Tab 1: 向量库快速检索与 RAG 流式生成（无持久化保存、无 LLM 降级）
def qa_stream_predict(user_message: str, history: List[Dict[str, str]], llm_choice: str, top_k_ret: int, top_k_rerank: int, filter_expr: str, user_state: dict):
    clean_message = user_message.strip()

    if not clean_message:
        # 返回 6 个值，最后两个使用 gr.skip() 保持前端组件状态不变
        yield history, "", "⚠️ 请输入有效内容！", get_gpu_memory_status(), gr.skip(), gr.skip()
        return

    # 1. 安全拦截检查
    checker = get_compliance_checker()
    is_safe, risk_level, hit_rule = checker.check_static_rules(clean_message)
    if not is_safe:
        fallback_msg = checker.fallback_responses.get(risk_level, "⚠️ [安全审计拦截] 您的请求包含高风险操作，已阻止。")
        history.append({"role": "user", "content": clean_message})
        history.append({"role": "assistant", "content": fallback_msg})
        yield history, f"🛡️ 静态规则拦截 [{hit_rule}]", f"🛡️ 静态规则拦截 [{hit_rule}]", get_gpu_memory_status(), gr.skip(), gr.skip()
        return

    # 仅使用当前页面临时传入的上下文，不加载任何持久化 Memory
    past_history = [{"role": msg.get("role"), "content": msg.get("content")} for msg in history if msg.get("content")]
    
    chain = get_qa_chain(llm_choice, top_k_ret, top_k_rerank)
    filter_pattern = filter_expr.strip() if (filter_expr and filter_expr.strip()) else None

    history.append({"role": "user", "content": clean_message})
    history.append({"role": "assistant", "content": ""})

    try:
        # 🎯 2. 底层向量库混合检索 + Rerank 重排
        raw_chunks = chain.retriever.hybrid_search(
            query=clean_message,
            top_k=top_k_ret,
            filter_expr=filter_pattern
        )

        reranked_chunks = chain.reranker.rerank(
            query=clean_message,
            documents=raw_chunks,
            top_n=top_k_rerank
        )

        sources_log = format_sources_log(llm_choice, top_k_ret, top_k_rerank, filter_pattern, reranked_chunks)

        # ❌ 3. 无匹配切片时直接中断提示
        if not reranked_chunks:
            no_rag_msg = "未在知识库中检索到有效参考内容。"
            history[-1]["content"] = no_rag_msg
            yield history, sources_log, "⚠️ 未在知识库检索到内容", get_gpu_memory_status(), gr.skip(), gr.skip()
            return

        # ⚡ 4. 流式生成回答
        stream_gen = chain.stream_answer(
            query=clean_message, 
            history=past_history, 
            filter_expr=filter_pattern, 
            pre_retrieved_chunks=reranked_chunks
        )
        
        accumulated_raw_text = ""
        for token_pkg in stream_gen:
            if isinstance(token_pkg, dict):
                if token_pkg.get("type") == "sources":
                    continue
                token_val = token_pkg.get("data", "")
            else:
                token_val = str(token_pkg)

            token_str = "".join([str(item) for item in token_val]) if isinstance(token_val, list) else str(token_val)
            accumulated_raw_text += token_str
            
            sanitized_display = checker.sanitize_text(accumulated_raw_text.split("</think>")[-1].strip())
            history[-1]["content"] = sanitized_display
            
            # 返回 6 个值，通过 gr.skip() 忽略第 5 和第 6 个下拉框组件的更新
            yield history, sources_log, "⚡ 正在生成回答...", get_gpu_memory_status(), gr.skip(), gr.skip()

    except Exception as e:
        logging.error(f"Tab 1 向量库检索异常: {e}", exc_info=True)
        err_msg = f"❌ 检索失败: {str(e)}"
        history[-1]["content"] = err_msg
        yield history, f"⚠️ 推理异常: {str(e)}", f"❌ 推理错误: {str(e)}", get_gpu_memory_status(), gr.skip(), gr.skip()
        
# Tab 3: ReAct Agent 推理
def agent_stream_predict(user_message, history, llm_model, top_k_ret, top_k_rerank, filter_input, user_state: dict):
    clean_message = user_message.strip() if user_message else ""
    username = user_state.get("username", "default")
    user_mem_mgr = get_or_create_user_memory(username)

    if not clean_message:
        choices = fetch_session_dropdown_choices(username)
        yield history, "*请输入有效指令*", "就绪", get_gpu_memory_status(), gr.update(choices=choices)
        return

   

    history.append({"role": "user", "content": clean_message})
    history.append({"role": "assistant", "content": "🤖 *Agent 正在规划并执行任务...*"})

    agent = IntegratedReActAgent(
        model_name=llm_model,
        top_k_ret=top_k_ret,
        top_k_rerank=top_k_rerank,
        filter_str=filter_input,
        memory_mgr=user_mem_mgr,
        max_steps=5,
        sandbox_timeout=2
    )

    inspector_log = f"🚀 **Agent 任务启动 (用户: {username} | 会话: {user_mem_mgr.session_id[:8]}...)**: `{clean_message}`\n\n---\n"
    choices = fetch_session_dropdown_choices(username)
    yield history, inspector_log, "🤖 推理中...", get_gpu_memory_status(), gr.update(choices=choices, value=user_mem_mgr.session_id)
    final_reply = ""
    for step in agent.run_stream(clean_message):
        stage = step.get("stage")
        content = step.get("content", "")
        inspector_log += f"{content}\n\n"

        if stage == "final_answer":
            history[-1]["content"] = content
            final_reply = content
        elif stage == "rollback":
            history[-1]["content"] = f"🚨 **任务中断**: \n{content}"
            final_reply = f"🚨 任务中断: {content}"

        choices = fetch_session_dropdown_choices(username)
        yield history, inspector_log, f"🤖 执行: {stage}", get_gpu_memory_status(), gr.update(choices=choices, value=user_mem_mgr.session_id)

    # # 持久化保存 AI 最终回答
    # if final_reply:
    #     user_mem_mgr.process_assistant_output(final_reply)

    choices = fetch_session_dropdown_choices(username)
    yield history, inspector_log, "✅ 任务完成", get_gpu_memory_status(), gr.update(choices=choices, value=user_mem_mgr.session_id)
    

# 新建 Session 切换函数（容错处理防止 Radio 报错）
def create_new_session_event(user_state: dict):
    username = user_state.get("username", "default")
    new_sess_id = str(uuid.uuid4())
    mem_mgr = get_or_create_user_memory(username, session_id=new_sess_id)
    mem_mgr.history_storage.create_session(username, new_sess_id, title="新对话")
    
    choices = fetch_session_dropdown_choices(username)
    if not choices:
        choices = [(f"💬 新对话", new_sess_id)]

    return [], "*新对话已开启*", "已新建会话", gr.update(choices=choices, value=new_sess_id), gr.update(choices=choices, value=new_sess_id)

# 在侧边栏选中历史对话时的切换处理函数
def switch_session_event(selected_session_id: str, user_state: dict):
    username = user_state.get("username", "default")
    if not selected_session_id:
        return [], "*未选择会话*", "就绪"

    mem_mgr = get_or_create_user_memory(username, session_id=selected_session_id)
    raw_msgs = mem_mgr.short_term.get_messages()
    rendered_history = [{"role": m["role"], "content": m["content"]} for m in raw_msgs]
    
    return rendered_history, f"📖 已加载历史会话: [{selected_session_id[:8]}...]", f"已切至会话 {selected_session_id[:8]}"

def test_tool_execution(tool_name, tool_input_json):
    start_time = time.time()
    try:
        params = json.loads(tool_input_json) if tool_input_json.strip() else {}
        tool_instance = tool_factory.get_tool(tool_name) if hasattr(tool_factory, "get_tool") else None
        if not tool_instance:
            return json.dumps({"status": "error", "message": f"未找到工具: {tool_name}"}, ensure_ascii=False, indent=2), "❌ 失败"
        
        result_payload = tool_instance.run(**params) if hasattr(tool_instance, "run") else tool_instance(**params)
        return json.dumps(result_payload, ensure_ascii=False, indent=2, default=str), f"✅ 成功 (耗时: {time.time()-start_time:.2f}s)"
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False, indent=2), f"❌ 失败: {str(e)}"

def stream_agent_sandbox_execution(user_query: str):
    if not user_query.strip():
        yield "⚠️ 请输入有效的测试问题！"
        return
    full_log = ""
    agent = IntegratedReActAgent(max_steps=5, sandbox_timeout=2)
    try:
        for step_data in agent.run_stream(user_query):
            full_log += step_data.get("content", "") + "\n\n"
            yield full_log
    except Exception as e:
        yield full_log + f"\n\n🚨 异常: {str(e)}"

def get_all_registered_tool_names() -> list:
    tools = []
    if hasattr(tool_factory, "_flat_tools") and isinstance(tool_factory._flat_tools, dict):
        tools = list(tool_factory._flat_tools.keys())
    return tools or ["search_knowledge_base"]

# ==========================================
# 🖥️ 5. 构建带“3 个完整 Tab”的 Gradio 应用
# ==========================================
CUSTOM_CSS = """
/* 全局字体设置：微软雅黑 (Microsoft YaHei) */
* {
    font-family: "Microsoft YaHei", "微软雅黑", "Segoe UI", Arial, sans-serif !important;
}

/* 默认 / 1080P 屏幕 (1366px - 1920px) 基准字号 */
html, body, .gradio-container {
    font-size: 15px !important;
}

/* 按钮、输入框、标签等组件字号优化 */
.gr-button, .gr-input, .gr-dropdown, .gr-radio, .gr-form {
    font-size: 14px !important;
}

/* Markdown 标题缩放控制 */
.markdown-text h1 { font-size: 1.8rem !important; }
.markdown-text h2 { font-size: 1.5rem !important; }
.markdown-text h3 { font-size: 1.2rem !important; }

/* 📱 移动端与小屏适配 (<= 768px) */
@media screen and (max-width: 768px) {
    html, body, .gradio-container {
        font-size: 13px !important;
        padding: 4px !important;
    }
    
    /* 1. 强制所有并排的 Row 在手机上变为纵向排列（上下堆叠） */
    .gradio-container .gr-row {
        display: flex !important;
        flex-direction: column !important;
        flex-wrap: nowrap !important;
    }

    /* 2. 让所有 Column 在手机端宽度撑满 100% */
    .gradio-container .gr-column {
        width: 100% !important;
        max-width: 100% !important;
        flex: none !important;
    }

    /* 3. 优化聊天框与输入框在手机上的高度与显示 */
    .gr-chatbot {
        height: 380px !important; /* 手机端适当缩减聊天框高度，避免过长 */
    }

    /* 4. 优化顶部 Tab 导航栏文字大小，防止挤压换行 */
    .gr-tabs button {
        font-size: 12px !important;
        padding: 6px 8px !important;
    }

    /* 💡 5. 补充：顶部标题栏专属优化（解决标题与退出按钮拥挤问题） */
    .header-container {
        flex-direction: column !important;
        align-items: flex-start !important;
        gap: 8px !important;
        padding: 8px 4px !important;
    }

    /* 缩减移动端主标题字号并允许适当排布 */
    .header-container h1, .header-container h2 {
        font-size: 1.1rem !important;
        line-height: 1.3 !important;
        margin: 0 !important;
    }

    /* 将退出按钮移至右上角或自适应靠右 */
    #logout_btn {
        align-self: flex-end !important;
        margin-top: -35px !important;
    }
}

/* 🖥️ 2K / 4K 高分辨率大屏适配 (>= 2560px) */
@media screen and (min-width: 2560px) {
    html, body, .gradio-container {
        font-size: 18px !important;
    }
    .gr-button, .gr-input, .gr-dropdown, .gr-radio {
        font-size: 16px !important;
    }
    .markdown-text h1 { font-size: 2.2rem !important; }
    .markdown-text h2 { font-size: 1.8rem !important; }
    .markdown-text h3 { font-size: 1.4rem !important; }
}

/* ---------------------------------------------------------
   🔒 登录界面绝对居中与高度紧凑样式
   --------------------------------------------------------- */
/* 外层容器占满整个视口，作为绝对定位参照物 */
.login-container {
    position: relative !important;
    min-height: 75vh !important;
    width: 100% !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}

/* 登录卡片强行紧凑并视口绝对居中 */
.login-card {
    position: absolute !important;
    top: 50% !important;
    left: 50% !important;
    transform: translate(-50%, -50%) !important;
    
    /* 尺寸强行收缩 */
    width: 100% !important;
    max-width: 400px !important;
    height: auto !important;
    max-height: fit-content !important;
    
    /* 内部间距调整 */
    padding: 30px 24px !important;
    background: var(--background-fill-primary, #ffffff) !important;
    border: 1px solid var(--border-color-primary, #e5e7eb) !important;
    border-radius: 12px !important;
    box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.08) !important;
}

/* 强行收缩 Gradio 卡片内部各子层级高度 */
.login-card > div,
.login-card .form,
.login-card .block {
    height: auto !important;
    min-height: unset !important;
    flex-grow: 0 !important;
}

/* 居中标题与提示 */
.login-card h2 {
    text-align: center !important;
    margin-bottom: 4px !important;
}

#logout_btn {
    border: none !important;
    background: transparent !important;
    color: #666 !important;
    box-shadow: none !important;
}
#logout_btn:hover {
    color: #ef4444 !important;
    background: #fee2e2 !important;
}
"""

# --- 🔑 JS 辅助脚本：持久化与清除 Cookie ---
# --- 🔑 JS 辅助脚本：直接读取 DOM 写入 Cookie ---
JS_SET_COOKIE = """
() => {
    // 1. 尝试直接从 Gradio 输入框 DOM 元素中提取输入的用户名
    const inputEl = document.querySelector('#login_user_input input') || document.querySelector('#login_user_input textarea');
    if (inputEl && inputEl.value) {
        const username = inputEl.value.trim();
        if (username) {
            // 2. 写入全局 Cookie，设置 path=/ 与 7 天有效期
            document.cookie = `qa_session_user=${encodeURIComponent(username)}; path=/; max-age=604800; SameSite=Lax`;
            console.log(`[Cookie Write] 成功写入 Cookie: qa_session_user=${username}`);
        }
    }
}
"""

JS_CLEAR_COOKIE = """
() => {
    document.cookie = "qa_session_user=; path=/; max-age=0; SameSite=Lax";
    console.log("[Cookie Clear] 成功清空 Cookie");
}
"""



def build_qa_admin_ui(qa_chain: Optional[Any] = None):
    try:
        if qa_chain is not None:
            init_tools(retriever=qa_chain.retriever, reranker=qa_chain.reranker)
        else:
            retriever = FineBIRetriever(milvus_host=DEFAULT_QA_CONFIG["milvus_host"], milvus_port=DEFAULT_QA_CONFIG["milvus_port"], collection_name=DEFAULT_QA_CONFIG["collection_name"], cuda_device=DEFAULT_QA_CONFIG["cuda_device"])
            reranker = FineBIReranker(cuda_device=DEFAULT_QA_CONFIG["cuda_device"])
            init_tools(retriever=retriever, reranker=reranker)
    except Exception as e:
        logging.warning(f"⚠️ 工具初始化说明: {e}")

    registered_tools = get_all_registered_tool_names()


    with gr.Blocks(title="FineBI QA 智能问答与 Agent 调试台", theme=gr.themes.Soft(), css=CUSTOM_CSS) as demo:
        # 用户状态存取 State
        user_state = gr.State(value={"is_logged_in": False, "username": ""})

        # =========================================================
        # 🔑 视图 1：登录界面 (默认显示 visible=True)
        # =========================================================
        with gr.Column(visible=True, elem_classes="login-container") as login_view:
            with gr.Column(elem_classes="login-card"):
                 # --- 子视图 A：标准登录表单 ---
                with gr.Column(visible=True) as login_form_container:
                    gr.Markdown("## 🔒 FineBI QA 智能调试台\n请先登录系统以继续操作")
                    login_user_input = gr.Textbox(label="用户名", placeholder="请输入账号 (例如: admin)", lines=1, elem_id="login_user_input")
                    login_pwd_input = gr.Textbox(label="密码", type="password", placeholder="请输入密码", lines=1)
                    btn_login = gr.Button("🔑 登录系统", variant="primary", size="lg")
                    
                    with gr.Row():
                        btn_goto_change_pwd = gr.Button("修改密码", variant="secondary", size="sm")
                    login_msg = gr.Markdown()

                # --- 子视图 B：修改密码表单 (默认隐藏) ---
                with gr.Column(visible=False) as change_pwd_container:
                    gr.Markdown("## 🔑 修改系统密码\n请输入原密码与新密码进行更新")
                    pwd_user_input = gr.Textbox(label="用户名", placeholder="请输入账号", lines=1)
                    pwd_old_input = gr.Textbox(label="原密码", type="password", placeholder="请输入原密码", lines=1)
                    pwd_new_input = gr.Textbox(label="新密码", type="password", placeholder="请输入新密码", lines=1)
                    pwd_new_confirm = gr.Textbox(label="确认新密码", type="password", placeholder="请再次输入新密码", lines=1)
                    
                    with gr.Row():
                        btn_submit_change_pwd = gr.Button("💾 确认修改", variant="primary", size="lg")
                        btn_back_to_login = gr.Button("返回登录", size="lg")
                    change_pwd_msg = gr.Markdown()

        # =========================================================
        # 🤖 视图 2：标准 QA 与 Agent 交互主界面 (默认隐藏 visible=False)
        # =========================================================
        with gr.Column(visible=False) as main_portal_view:
            with gr.Row(elem_classes="header-container"):
                user_info_banner = gr.Markdown("# 🤖 FineBI QA 智能问答与 Agent 工具链调试台", scale=4)
                # 用空列或设置 scale 撑开间距，将退出按钮推到最右侧
                with gr.Column(scale=1, min_width=10, visible=True):
                    pass
                btn_logout = gr.Button("🚪 退出", size="sm", min_width=80, scale=0, elem_id="logout_btn")
            # 顶部显存与控制区
            with gr.Row():
                gpu_box = gr.Textbox(label="🖥️ GPU 显存实时状态", value=get_gpu_memory_status(), interactive=False, scale=4)
                btn_refresh_gpu = gr.Button("🔄 刷新显存", scale=1)
                btn_cleanup_gpu = gr.Button("🔥 应急回收显存", variant="stop", scale=1)

            btn_refresh_gpu.click(fn=get_gpu_memory_status, outputs=[gpu_box])
            btn_cleanup_gpu.click(fn=emergency_force_cleanup, outputs=[gpu_box])

            # 页面 Tabs 选项卡 (保留完整的 3 个 Tab)
            with gr.Tabs():
                
                # ---------------------------------------------------------
                # Tab 1: 在线对话与 RAG 溯源 (支持多 Session 清单)
                # ---------------------------------------------------------
                with gr.Tab("💬 在线对话与 RAG 溯源"):
                    with gr.Row():
                    # 主对话界面：拓展至全宽度
                        with gr.Column(scale=7):
                            chatbot = gr.Chatbot(label="FineBI 智能问答助理", height=500)
                            msg_input = gr.Textbox(label="请输入你的问题", placeholder="例如：怎么创建预警用户？", lines=2)
                            with gr.Row():
                                btn_send = gr.Button("🚀 发送 (Send)", variant="primary")
                                btn_clear = gr.Button("🗑️ 清空对话")
                            status_box = gr.Textbox(label="运行状态", value="就绪", interactive=False)

                        with gr.Column(scale=5):
                            gr.Markdown("### 🎛️ 推理与检索参数控制")
                            llm_dropdown = gr.Dropdown(choices=LLM_OPTIONS, value=DEFAULT_QA_CONFIG["llm_model_name"], label="🧠 推理 LLM 模型")
                            slider_top_k_ret = gr.Slider(minimum=1, maximum=30, value=10, step=1, label="初筛 Top-K")
                            slider_top_k_rerank = gr.Slider(minimum=1, maximum=10, value=3, step=1, label="精排 Top-K")
                            filter_input = gr.Textbox(label="🎯 Milvus 元数据过滤表达式", placeholder="target_version == 'V6.0'", lines=2)
                            gr.Markdown("### 🔍 检索召回溯源")
                            sources_display = gr.Markdown(value="*暂无召回数据...*")

                    # Tab 1 事件绑定
                    btn_send.click(
                        fn=qa_stream_predict,
                        inputs=[msg_input, chatbot, llm_dropdown, slider_top_k_ret, slider_top_k_rerank, filter_input, user_state],
                        # outputs=[chatbot, sources_display, status_box, gpu_box, t1_session_radio, t1_session_radio]
                        outputs=[chatbot, sources_display, status_box, gpu_box, gpu_box, gpu_box]
                    ).then(fn=lambda: "", inputs=None, outputs=[msg_input])

                    btn_clear.click(
                        fn=clear_agent_memory,
                        inputs=[user_state],
                        # outputs=[chatbot, sources_display, status_box, msg_input, t1_session_radio, t1_session_radio]
                        outputs=[chatbot, sources_display, status_box, msg_input]
                    )

                # ---------------------------------------------------------
                # Tab 2: 沙盒测试 (无需多 Session 侧边栏，保留纯工具测试)
                # ---------------------------------------------------------
                with gr.Tab("🛠️ Tools & Agent 沙盒测试"):
                    gr.Markdown("### 🧪 1. 原子工具 (Tool) 独立功能测试")
                    with gr.Row():
                        with gr.Column(scale=6):
                            tool_select = gr.Dropdown(choices=registered_tools, value=registered_tools[0] if registered_tools else None, label="🔧 选择工具")
                            tool_json_input = gr.Code(label="📥 输入参数 (JSON)", value='{\n  "query": "定时任务配置失败",\n  "limit": 3\n}', language="json", lines=7)
                            btn_run_tool = gr.Button("⚡ 运行测试", variant="primary")
                        with gr.Column(scale=6):
                            tool_status = gr.Textbox(label="执行状态", value="等待执行...", interactive=False)
                            tool_json_output = gr.Code(label="📤 输出结果 (JSON)", value="{}", language="json", lines=10)

                    gr.Markdown("---")
                    gr.Markdown("### 🤖 2. ReAct Agent 全链路诊断")
                    with gr.Row():
                        with gr.Column(scale=4):
                            test_input = gr.Textbox(label="测试问题输入", placeholder="例如： FineBI V6.0 怎么解决 MySQL 连接报错？", lines=3)
                            run_btn = gr.Button("🚀 启动 Agent 推理", variant="primary")
                        with gr.Column(scale=8):
                            react_log_markdown = gr.Markdown(value="等待启动诊断...")

                    btn_run_tool.click(fn=test_tool_execution, inputs=[tool_select, tool_json_input], outputs=[tool_json_output, tool_status])
                    run_btn.click(fn=stream_agent_sandbox_execution, inputs=[test_input], outputs=[react_log_markdown])

                # ---------------------------------------------------------
                # Tab 3: ReAct Agent 智能助理 (支持多 Session 清单与思维链)
                # ---------------------------------------------------------
                with gr.Tab("🤖 Agent 智能助理 (思维链 & 沙箱)"):
                    with gr.Row():
                        # 左侧边栏
                        with gr.Column(scale=3, min_width=250):
                            gr.Markdown("### 📜 最近对话清单")
                            btn_t3_new_chat = gr.Button("➕ 新建对话", variant="primary")
                            t3_session_radio = gr.Radio(label="历史会话记录", choices=[], value=None, interactive=True)
                            btn_t3_refresh_sess = gr.Button("🔄 刷新会话列表", size="sm")

                        # 右侧主窗口
                        with gr.Column(scale=9):
                            with gr.Row():
                                with gr.Column(scale=7):
                                    agent_chatbot = gr.Chatbot(label="FineBI ReAct Agent", height=500)
                                    agent_msg_input = gr.Textbox(label="输入你的复杂问题或任务指令", placeholder="例如：FineBI V6.0 怎么配置 MySQL 连接？", lines=2)
                                    with gr.Row():
                                        btn_agent_send = gr.Button("🚀 发送 Agent 任务", variant="primary")
                                        btn_agent_clear = gr.Button("🗑️ 清空当前对话")
                                    agent_status_box = gr.Textbox(label="Agent 状态", value="就绪", interactive=False)

                                with gr.Column(scale=5):
                                    gr.Markdown("### 🔬 Agent 运行诊断 (Inspector)")
                                    agent_inspector_display = gr.Markdown(value="*等待启动诊断...*")

                    # Tab 3 事件绑定
                    btn_agent_send.click(
                        fn=agent_stream_predict,
                        inputs=[agent_msg_input, agent_chatbot, llm_dropdown, slider_top_k_ret, slider_top_k_rerank, filter_input, user_state],
                        outputs=[agent_chatbot, agent_inspector_display, agent_status_box, gpu_box, t3_session_radio]
                    ).then(fn=lambda: "", inputs=None, outputs=[agent_msg_input])

                    btn_t3_new_chat.click(
                        fn=create_new_session_event,
                        inputs=[user_state],
                        outputs=[agent_chatbot, agent_inspector_display, agent_status_box, t3_session_radio]
                    )

                    # 【解冻并修复】点击历史会话 Radio 时切换对话
                    t3_session_radio.change(
                        fn=switch_session_event,
                        inputs=[t3_session_radio, user_state],
                        outputs=[agent_chatbot, agent_inspector_display, agent_status_box]
                    )

                    # 【解冻并修复】刷新会话列表按钮事件
                    btn_t3_refresh_sess.click(
                        fn=lambda st: gr.update(choices=fetch_session_dropdown_choices(st.get("username", "default"))),
                        inputs=[user_state],
                        outputs=[t3_session_radio]
                    )

                    # 【清理垃圾字符】清空会话按钮事件
                    btn_agent_clear.click(
                        fn=clear_agent_memory,
                        inputs=[user_state],
                        outputs=[agent_chatbot, agent_inspector_display, agent_status_box, agent_msg_input, t3_session_radio]
                    )
                    # btn_agent_send.click(
                    #     fn=agent_stream_predict,
                    #     inputs=[agent_msg_input, agent_chatbot, llm_dropdown, slider_top_k_ret, slider_top_k_rerank, filter_input, user_state],
                    #     outputs=[agent_chatbot, agent_inspector_display, agent_status_box, gpu_box, t3_session_radio, t3_session_radio]
                    # ).then(fn=lambda: "", inputs=None, outputs=[agent_msg_input])

                    # btn_t3_new_chat.click(
                    #     fn=create_new_session_event,
                    #     inputs=[user_state],
                    #     outputs=[agent_chatbot, agent_inspector_display, agent_status_box, t3_session_radio, t3_session_radio]
                    # )

                    # t3_session_radio.change(
                    #     fn=switch_session_event,
                    #     inputs=[t3_session_radio, user_state],
                    #     outputs=[agent_chatbot, agent_inspector_display, agent_status_box]
                    # )

                    # btn_t3_refresh_sess.click(
                    #     fn=lambda st: gr.update(choices=fetch_session_dropdown_choices(st.get("username", "default"))),
                    #     inputs=[user_state],
                    #     outputs=[t3_session_radio]
                    # )

                    # btn_agent_clear.click(
                    #     fn=clear_agent_memory,
                    #     inputs=[user_state],
                    #     outputs=[agent_chatbot, agent_inspector_display, agent_status_box, agent_msg_input, t3_session_radio, t3_session_radio]
                    # )

        # =========================================================
        # 🔑 登录视图切换逻辑绑定
        # =========================================================
        btn_goto_change_pwd.click(
            fn=lambda: (gr.update(visible=False), gr.update(visible=True), ""),
            inputs=None,
            outputs=[login_form_container, change_pwd_container, login_msg]
        )
        
        btn_back_to_login.click(
            fn=lambda: (gr.update(visible=True), gr.update(visible=False), ""),
            inputs=None,
            outputs=[login_form_container, change_pwd_container, change_pwd_msg]
        )

        # =========================================================
        # 🔑 修改密码提交逻辑
        # =========================================================
        def perform_change_password(username, old_pwd, new_pwd, confirm_pwd):
            u_clean = username.strip() if username else ""
            if not u_clean or not old_pwd or not new_pwd or not confirm_pwd:
                return "❌ 所有字段都必须填写！"
            
            # 1. 校验内存/字典中的用户是否存在
            if u_clean not in VALID_USERS:
                return f"❌ 用户 `{u_clean}` 不存在！"
            
            # 2. 校验原密码
            if VALID_USERS[u_clean] != old_pwd:
                return "❌ 原密码错误！"
                
            # 3. 校验两次新密码一致性
            if new_pwd != confirm_pwd:
                return "❌ 两次输入的新密码不一致！"
                
            # 4. 校验新旧密码是否相同
            if old_pwd == new_pwd:
                return "❌ 新密码不能与原密码相同！"
            
            # 5. YAML 文件持久化落盘逻辑
            yaml_path = "config/users_auth.yaml"
            try:
                # 读取现有 YAML 文件结构保持注释或格式
                if os.path.exists(yaml_path):
                    with open(yaml_path, "r", encoding="utf-8") as f:
                        yaml_data = yaml.safe_load(f) or {}
                else:
                    yaml_data = {"users": {}}

                # 更新对应用户的密码
                if "users" not in yaml_data:
                    yaml_data["users"] = {}
                
                yaml_data["users"][u_clean] = new_pwd

                # 写回 YAML 文件
                with open(yaml_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(yaml_data, f, allow_unicode=True, sort_keys=False)

                # 同步更新当前运行时的内存字典
                VALID_USERS[u_clean] = new_pwd
                
                return "✅ 密码修改成功！YAML 文件已更新，请返回登录。"
                
            except Exception as e:
                logging.error(f"❌ 写入 users_auth.yaml 失败: {e}")
                return f"❌ 密码修改失败（磁盘写入异常）: {str(e)}"
            
        btn_submit_change_pwd.click(
            fn=perform_change_password,
            inputs=[pwd_user_input, pwd_old_input, pwd_new_input, pwd_new_confirm],
            outputs=[change_pwd_msg]
        )
        # =========================================================
        # 🔑 逻辑 1：页面刷新/加载时静默校验 Cookie 登录态
        # =========================================================
        def check_login_status(request: gr.Request):
            # 1. 安全校验 request 对象
            if not request or not hasattr(request, "headers"):
                logging.warning("⚠️ [Cookie 校验] 未获取到有效 request 对象")
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    "# 🤖 FineBI QA 智能问答",
                    {"is_logged_in": False, "username": ""},
                    gr.update(choices=[], value=None)
                )

            # 2. 读取 Cookie Header 并打印日志
            cookie_str = request.headers.get("cookie", "")
            logging.info(f"🔍 [Cookie 诊断] 收到 Header Cookie: {cookie_str}")
            
            found_user = None

            # 3. 解析 Cookie 并执行 unquote 解码
            if "qa_session_user=" in cookie_str:
                try:
                    for item in cookie_str.split(";"):
                        item = item.strip()
                        if item.startswith("qa_session_user="):
                            raw_val = item.split("=", 1)[1].strip()
                            # 关键修改：必须解出真实用户名（防止 %20 等转义）
                            user_val = unquote(raw_val)
                            
                            logging.info(f"🔍 [Cookie 诊断] 解析到用户名: '{user_val}' | VALID_USERS 匹配结果: {user_val in VALID_USERS}")
                            
                            if user_val in VALID_USERS:
                                found_user = user_val
                                break
                except Exception as e:
                    logging.error(f"❌ [Cookie 诊断] 解析异常: {e}")

            # 4. 渲染界面分支
            if found_user:
                mem_mgr = get_or_create_user_memory(found_user)
                new_state = {"is_logged_in": True, "username": found_user}
                banner_text = f"# 🤖 FineBI QA 智能问答与 Agent 调试台 (当前登录用户: `{found_user}`)"
                
                session_choices = fetch_session_dropdown_choices(found_user)
                if not session_choices:
                    default_sess = str(uuid.uuid4())
                    mem_mgr.switch_session(default_sess)
                    mem_mgr.history_storage.create_session(found_user, default_sess, title="新对话")
                    session_choices = fetch_session_dropdown_choices(found_user)
                else:
                    default_sess = mem_mgr.session_id

                valid_values = [c[1] for c in session_choices]
                if default_sess not in valid_values and valid_values:
                    default_sess = valid_values[0]

                logging.info(f"✅ [Cookie 校验] 成功免密自动登录！用户: {found_user}")
                return (
                    gr.update(visible=False),              # login_view
                    gr.update(visible=True),               # main_portal_view
                    banner_text,                           # user_info_banner
                    new_state,                             # user_state
                    gr.update(choices=session_choices, value=default_sess)  # t3_session_radio
                )
            else:
                logging.warning("⚠️ [Cookie 校验] 未找到合法 Cookie，返回登录页")
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    "# 🤖 FineBI QA 智能问答",
                    {"is_logged_in": False, "username": ""},
                    gr.update(choices=[], value=None)
                )

        # 关键修改：绑定 demo.load 时必须包含 Request 隐式/显式传入逻辑
        demo.load(
            fn=check_login_status,
            inputs=None, # Gradio 会自动将 request 对象注入到函数首个 gr.Request 参数中
            outputs=[
                login_view,
                main_portal_view,
                user_info_banner,
                user_state,
                t3_session_radio,
            ],
        )

        # =========================================================
        # 🔑 逻辑 2：主动点击“登录按钮”逻辑响应
        # =========================================================
        def perform_login(username, password):
            username_clean = username.strip() if username else ""
            if username_clean in VALID_USERS and VALID_USERS[username_clean] == password:
                mem_mgr = get_or_create_user_memory(username_clean)
                new_state = {"is_logged_in": True, "username": username_clean}
                banner_text = f"# 🤖 FineBI QA 智能问答与 Agent 调试台 (当前登录用户: `{username_clean}`)"

                session_choices = fetch_session_dropdown_choices(username_clean)
                if not session_choices:
                    default_sess = str(uuid.uuid4())
                    mem_mgr.switch_session(default_sess)
                    mem_mgr.history_storage.create_session(username_clean, default_sess, title="新对话")
                    session_choices = fetch_session_dropdown_choices(username_clean)
                else:
                    default_sess = mem_mgr.session_id

                valid_values = [c[1] for c in session_choices]
                if default_sess not in valid_values and valid_values:
                    default_sess = valid_values[0]

                return (
                    gr.update(visible=False),              # login_view 隐藏
                    gr.update(visible=True),               # main_portal_view 显示
                    banner_text,                           # user_info_banner
                    new_state,                             # user_state
                    "✅ 登录成功！",                       # login_msg
                    gr.update(choices=session_choices, value=default_sess), # t3_session_radio
                )
            else:
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    "# 🤖 FineBI QA 智能问答",
                    {"is_logged_in": False, "username": ""},
                    "❌ 用户名或密码错误，请重试！",
                    gr.update(choices=[], value=None),
                )

        # 简化后的登录事件绑定
        btn_login.click(
            fn=None,
            inputs=None,
            outputs=None,
            js=JS_SET_COOKIE  # 👈 点击瞬间先执行 JS 写入 Cookie 到浏览器
        ).then(
            fn=perform_login,
            inputs=[login_user_input, login_pwd_input],
            outputs=[
                login_view,
                main_portal_view,
                user_info_banner,
                user_state,
                login_msg,
                t3_session_radio,
            ]
        )

        # =========================================================
        # 🔑 逻辑 3：主动点击“退出登录”逻辑响应
        # =========================================================
        def perform_logout():
            return (
                gr.update(visible=True),                # login_view
                gr.update(visible=False),               # main_portal_view
                {"is_logged_in": False, "username": ""},# user_state
                "💡 您已成功退出系统",                 # login_msg
                gr.update(choices=[], value=None)       # t3_session_radio
            )

        btn_logout.click(
            fn=perform_logout,
            inputs=None,
            outputs=[
                login_view,
                main_portal_view,
                user_state,
                login_msg,
                t3_session_radio,
            ],
            js=JS_CLEAR_COOKIE,
        )

    return demo

# ==========================================
# 🚀 6. 应用启动入口
# ==========================================
if __name__ == "__main__":
    import atexit
    
    def on_app_shutdown():
        logging.info("⚡ 关闭 QA 服务，清除 GPU 显存...")
        emergency_force_cleanup()

    atexit.register(on_app_shutdown)

    qa_ui = build_qa_admin_ui()
    
    qa_ui.queue().launch(
        server_name="0.0.0.0",
        server_port=7862,
        root_path="/qa"
        # share=True
    )