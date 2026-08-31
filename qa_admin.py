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
import re
import yaml
from typing import Dict, Any, List, Tuple, Generator, Optional
from urllib.parse import unquote
from api.observability import scan_metrics
# ==========================================
# 📂 1. 动态注入系统路径与模块导入
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

from factory.model_factory import ModelFactory
from factory.tool_factory import tool_factory, load_tools_from_yaml, BaseTool
from generator.qa_chain import QAChain
from generator.llm_client import FineBILLMClient
from search.retriever import FineBIRetriever
from search.reranker import FineBIReranker
from factory import init_tools
from agent.sandbox import SandboxExecutor
from agent.react_agent import ReActAgent 
from memory.memory_manager import MemoryManager
from agent.compliance import ComplianceChecker
from memory.feedback_store import feedback_store

# ==========================================
# ⚙️ 2. 全局配置与用户凭证加载
# ==========================================
CONFIG_FILE_PATH = os.path.join(SCRIPT_DIR, "qa_config.json")
USERS_AUTH_PATH = os.path.join(SCRIPT_DIR, "config", "users_auth.yaml")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

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

import os
import yaml
import logging
from typing import Dict, Any, Tuple

# 📌 1. 通用安全 YAML 解析函数 (带兜底数据)
def safe_load_yaml(file_path: str, default_payload: Dict[str, Any]) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        logging.warning(f"⚠️ 配置文件不存在: [{file_path}]，启用默认配置。")
        return default_payload
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or default_payload
    except Exception as e:
        logging.error(f"❌ 读取配置文件失败 [{file_path}]: {e}")
        return default_payload


# 📌 2. 精简后的 load_user_credentials
def load_user_credentials() -> Tuple[Dict[str, str], Dict[str, str], Dict[str, str]]:
    default_auth = {"users": {"admin": {"password": "123456", "role": "admin"}}}
    data = safe_load_yaml(USERS_AUTH_PATH, default_auth)
    users_dict = data.get("users", {})

    valid_passwords: Dict[str, str] = {}
    user_roles: Dict[str, str] = {}
    raw_key_map: Dict[str, str] = {}

    for username, info in users_dict.items():
        username_str = str(username).strip()
        if isinstance(info, dict):
            pwd = str(info.get("password", ""))
            role = str(info.get("role", "user")).strip()
        else:
            pwd = str(info)
            role = "user"

        valid_passwords[username_str] = pwd
        user_roles[username_str] = role
        raw_key_map[username_str] = f"{username_str}({role})"

    return valid_passwords, user_roles, raw_key_map

VALID_USERS_PWD, USER_ROLES, RAW_KEY_MAP = load_user_credentials()

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

# ==========================================
# 📊 监控可观测性面板逻辑 (Tab 4 专用)
# ==========================================
# def render_tab3_monitor_dashboard() -> str:
#     """直接复用 scan_metrics 读取系统的真实监控指标，仅在 Tab 3 右侧展示"""
#     try:
#         metrics = scan_metrics(data_dir=DATA_DIR)
#         total_users = metrics.get('total_users', 0)
#         total_sessions = metrics.get('total_sessions', 0)
#         total_queries = metrics.get('total_queries', 0)
#         compliance_interceptions = metrics.get('compliance_interceptions', 0)
#         interception_rate = metrics.get('interception_rate', '0.00%')
#     except Exception as e:
#         logging.error(f"读取监控指标失败: {e}")
#         return "*📊 监控指标读取异常*"

#     return f"""
# > **📊 系统实时运行指标**
# - **👥 累计活跃用户**: `{total_users}` 人
# - **💬 累计会话总数**: `{total_sessions}` 个
# - **❓ 累计提问总次**: `{total_queries}` 次

# > **🛡️ 风控安全监控**
# - **🚨 触发安全拦截**: `{compliance_interceptions}` 次
# - **📉 安全拦截比例**: `{interception_rate}`
# """

def render_observability_dashboard():
    """读取并格式化 Observability 核心指标"""
    metrics = scan_metrics(data_dir=DATA_DIR)
    
    summary_md = f"""
### 📈 系统运行核心指标概览

| 📊 监控维度 | 🔢 统计数值 | 💡 描述说明 |
| :--- | :--- | :--- |
| **👥 累计活跃用户** | `{metrics.get('total_users', 0)}` | 系统中已产生会话的独立账号数 |
| **💬 累计会话总数** | `{metrics.get('total_sessions', 0)}` | 创建的历史 Session 文件总数 |
| **❓ 累计查询次数** | `{metrics.get('total_queries', 0)}` | 用户提交的用户问题 (User Role Messages) 总数 |
| **🚨 安全风控拦截** | `{metrics.get('compliance_interceptions', 0)}` | 触发静态/动态 Compliance 规则拦截的次数 |
| **🛡️ 拦截命中比例** | `{metrics.get('interception_rate', '0.00%')}` | 风控拦截次数 / 累计查询总次数 |
"""
    return summary_md, metrics

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
    log_lines = []
    
    # 1. 顶部配置折叠
    log_lines.append("<details><summary>🔧 <b>点击展开 / 折叠检索与重排配置</b></summary>\n")
    log_lines.append(f"- **LLM 模型**: `{llm_choice}`")
    log_lines.append(f"- **检索/重排 Top-K**: `Ret={top_k_ret}` | `Rerank={top_k_rerank}`")
    log_lines.append(f"- **Filter 过滤**: `{filter_pattern}`" if filter_pattern else "- **Filter 过滤**: *无 (混合召回)*")
    log_lines.append("</details>\n")
    log_lines.append("─" * 50)

    if not chunks:
        log_lines.append("⚠️ **提示**: 检索重排后未返回有效切片。")
        return "\n".join(log_lines) + "\n"

    for idx, chunk in enumerate(chunks, 1):
        if not isinstance(chunk, dict):
            log_lines.append(f"### 📦 切片 [{idx}]\n```text\n{str(chunk)}\n```\n")
            continue

        # 直接提取扁平化字段
        chunk_id = chunk.get("chunk_id", f"Chunk_{idx}")
        file_url = chunk.get("file_url", "")
        hierarchy = chunk.get("hierarchy") or chunk.get("section_id", "")
        biz_summary = chunk.get("biz_summary", "")
        
        # 分数提取
        rerank_score = chunk.get("rerank_score", chunk.get("score", 0.0))
        rerank_prob = chunk.get("rerank_prob")
        
        # 内容与上下文
        main_content = chunk.get("content") or chunk.get("base_content", "")
        up_content = chunk.get("up_content")
        down_content = chunk.get("down_content")

        # 📌 Markdown 卡片头
        log_lines.append(f"### 📄 [{idx}] {chunk_id}")
        if hierarchy:
            log_lines.append(f"📌 **章节层级**: `{hierarchy}`")
        
        # 分数与链接展示
        score_info = f"`Score={rerank_score:.4f}`"
        if rerank_prob is not None:
            score_info += f" ｜ `Prob={rerank_prob:.4f}`"
        
        url_display = f" ｜ 🔗 [查看/下载源文件]({file_url})" if file_url else ""
        log_lines.append(f"🎯 **重排得分**: {score_info}{url_display}")

        # 业务摘要折叠框 (如果有)
        if biz_summary:
            log_lines.append(f"> 💡 **业务摘要**: {biz_summary}")

        log_lines.append("") # 换行

        # 上文折叠框
        if up_content and str(up_content).strip() and str(up_content) != "NULL":
            log_lines.append("<details><summary>⬆️ <b>点击展开上文 (up_content)</b></summary>\n")
            log_lines.append(f"```text\n{str(up_content).strip()}\n```")
            log_lines.append("</details>\n")

        # 核心切片内容
        log_lines.append("**📝 当前切片内容:**")
        log_lines.append(f"```text\n{str(main_content).strip()}\n```\n")

        # 下文折叠框
        if down_content and str(down_content).strip() and str(down_content) != "NULL":
            log_lines.append("<details><summary>⬇️ <b>点击展开下文 (down_content)</b></summary>\n")
            log_lines.append(f"```text\n{str(down_content).strip()}\n```")
            log_lines.append("</details>\n")

        log_lines.append("─" * 50)

    return "\n".join(log_lines) + "\n"


# Tab 1: 向量库快速检索与 RAG 流式生成（无持久化保存、无 LLM 降级）
def qa_stream_predict(user_message: str, history: List[Dict[str, str]], llm_choice: str, top_k_ret: int, top_k_rerank: int, filter_expr: str):
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

        for chunk in reranked_chunks:
            meta = chunk.get("metadata", {})
            prev_id = meta.get("prev_chunk_id")
            next_id = meta.get("next_chunk_id")
            
            # 根据你的 retriever 实际查 ID 的方法补全文本（例如 query_by_id 或 get_by_id）
            if prev_id and hasattr(chain.retriever, "get_by_id"):
                prev_doc = chain.retriever.get_by_id(prev_id)
                if prev_doc:
                    chunk["prev_content"] = prev_doc.get("content")
                    
            if next_id and hasattr(chain.retriever, "get_by_id"):
                next_doc = chain.retriever.get_by_id(next_id)
                if next_doc:
                    chunk["next_content"] = next_doc.get("content")

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
        yield history, sources_log, "🟢 就绪", get_gpu_memory_status(), gr.skip(), gr.skip()

        # =========================================================
        # 💡 5. 生成结束，挂载溯源 Accordion 证据链
        # =========================================================
        # final_answer = history[-1]["content"]
        # # 渲染带 [Doc X] 及底层结构（如 section_id / source_file）的证据链
        # final_with_citation = CitationFormatter.render_citation_accordion(final_answer, reranked_chunks)
        
        # history[-1]["content"] = final_with_citation
        # yield history, sources_log, "✅ 生成完成", get_gpu_memory_status(), gr.skip(), gr.skip()



    except Exception as e:
        logging.error(f"Tab 1 向量库检索异常: {e}", exc_info=True)
        err_msg = f"❌ 检索失败: {str(e)}"
        history[-1]["content"] = err_msg
        yield history, f"⚠️ 推理异常: {str(e)}", f"❌ 推理错误: {str(e)}", get_gpu_memory_status(), gr.skip(), gr.skip()
        
# Tab 3: ReAct Agent 推理
def agent_stream_predict(user_message, history, llm_model, top_k_ret, top_k_rerank, filter_input, user_state: dict):
    clean_message = user_message.strip() if user_message else ""
    username = user_state.get("username", "default")
    user_role = user_state.get("role", "user")

    user_mem_mgr = get_or_create_user_memory(username)

    if not clean_message:
        choices = fetch_session_dropdown_choices(username)
        yield history, "*请输入有效指令*", "就绪", get_gpu_memory_status(), gr.update(choices=choices)
        return

    # 1. 安全风控拦截检查
    checker = get_compliance_checker()
    is_safe, risk_level, hit_rule = checker.check_static_rules(clean_message)

    if not is_safe:
        fallback_msg = checker.fallback_responses.get(risk_level, "⚠️ [安全审计拦截] 您的 Agent 指令包含高风险操作，已终止执行。")
        history.append({"role": "user", "content": clean_message})
        history.append({"role": "assistant", "content": f"[COMPLIANCE_BLOCK] {fallback_msg}"})

        # 写入持久化存储以便可观测性识别
        user_mem_mgr.process_assistant_output(f"[COMPLIANCE_BLOCK] {fallback_msg}")

        choices = fetch_session_dropdown_choices(username)
        yield history, f"🛡️ 规则拦截 [{hit_rule}]", f"🛡️ 安全拦截: {hit_rule}", get_gpu_memory_status(), gr.update(choices=choices)
        return

    user_mem_mgr.short_term.add_message("user", clean_message)

    history.append({"role": "user", "content": clean_message})
    history.append({"role": "assistant", "content": "🤖 *Agent 正在规划并执行任务...*"})

    agent = ReActAgent(
            llm_client=None,
            model_name=llm_model,
            top_k_ret=top_k_ret,
            top_k_rerank=top_k_rerank,
            filter_str=filter_input,
            memory_mgr=user_mem_mgr,
            user_role=user_role,
            max_steps=5,
            sandbox_timeout=2,
        )
    raw_user_key = RAW_KEY_MAP.get(username, f"{username}({user_role})")
    history_context = user_mem_mgr.short_term.get_messages()[:-1]  # 排除刚刚加入的当前 prompt

    inspector_log = f"🚀 **Agent 任务启动 (用户: {username}     {raw_user_key} | 会话: {user_mem_mgr.session_id[:8]}...)**: `{clean_message}`\n\n---\n"
    choices = fetch_session_dropdown_choices(username)
    yield history, inspector_log, "🤖 推理中...", get_gpu_memory_status(), gr.update(choices=choices, value=user_mem_mgr.session_id)
    final_reply = ""
    for step in agent.run_stream(clean_message
                                 # ,tools_schema=tool_specs
                                #  ,history_messages=history_context
                                 ):
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

# 新建 用户点赞/点踩及意见反馈组件
def handle_chatbot_like(like_data: gr.LikeData, history: list, user_state: dict):
    """
    处理 gr.Chatbot 逐条消息的点赞 / 点踩事件并写入 feedback_store
    """
    # 1. 提取当前登录的用户 ID
    user_id = user_state.get("username", "default_user") if isinstance(user_state, dict) else "default_user"
    
    # 2. 判断是点赞还是点踩 ('like' | 'dislike')
    rating = "like" if like_data.liked else "dislike"
    response_text = like_data.value or ""
    
    # 3. 解析对应的 User Query
    query_text = ""
    try:
        # like_data.index 可以是整数或列表 (例如 [msg_index, 1])
        msg_idx = like_data.index[0] if isinstance(like_data.index, (list, tuple)) else like_data.index
        
        # 针对 Gradio 4.x+ 的 messages 格式 (list of dicts)
        if history and 0 <= msg_idx < len(history):
            if msg_idx > 0:
                prev_msg = history[msg_idx - 1]
                query_text = prev_msg.get("content", "") if isinstance(prev_msg, dict) else str(prev_msg)
        # 针对旧版 tuple 格式 [(query, response), ...]
        elif history and isinstance(history[0], (tuple, list)):
            query_text = history[msg_idx][0]
    except Exception as e:
        print(f"⚠️ 解析 Query 索引失败: {e}")

    # 4. 调用 feedback_store 写入 JSONL
    success = feedback_store.record_feedback(
        user_id=user_id,
        query=query_text,
        response=response_text,
        rating=rating,
        feedback_text=f"Inline message feedback: {rating}"
    )
    
    if success:
        print(f"✅ [Feedback] 用户 '{user_id}' 针对 QA 记录了 {rating} 反馈")

# 在侧边栏选中历史对话时的切换处理函数
def switch_session_event(selected_session_id: str, user_state: dict):
    username = user_state.get("username", "default")
    if not selected_session_id:
        return [], "*未选择会话*", "就绪"

    mem_mgr = get_or_create_user_memory(username, session_id=selected_session_id)
    raw_msgs = mem_mgr.short_term.get_messages()
    rendered_history = [{"role": m["role"], "content": m["content"]} for m in raw_msgs]
    
    return rendered_history, f"📖 已加载历史会话: [{selected_session_id[:8]}...]", f"已切至会话 {selected_session_id[:8]}"

def test_tool_execution(tool_name, tool_input_json, user_state: dict):
    start_time = time.time()
    user_role = user_state.get("role", "user") if isinstance(user_state, dict) else "user"
    username = user_state.get("username", "unknown") if isinstance(user_state, dict) else "unknown"
    print(f"\n🚨 [DEBUG 3: 沙盒测试运行鉴权]")
    print(f"👉 用户: '{username}' | 识别到的角色 user_role: '{user_role}'")
    print(f"👉 请求调用的工具: '{tool_name}'")
    print(f"👉 user_state 完整字典: {user_state}")
    try:
        params = json.loads(tool_input_json) if tool_input_json.strip() else {}
        # tool_instance = tool_factory.get_tool(tool_name) if hasattr(tool_factory, "get_tool") else None
        tool_instance = tool_factory.get_tool(tool_name, user_role=user_role)
        if not tool_instance:
            return (
                json.dumps({
                    "status": "error", 
                    "message": f"未找到工具 [{tool_name}] 或当前角色 [{user_role}] 无权访问！"
                }, ensure_ascii=False, indent=2), 
                "❌ 鉴权失败/工具不存在"
            )
        print(f'找到工具:{tool_instance}')
        # if not tool_instance:
        #     return json.dumps({"status": "error", "message": f"未找到工具: {tool_name}"}, ensure_ascii=False, indent=2), "❌ 失败"
        
        result_payload = tool_instance.run(**params) if hasattr(tool_instance, "run") else tool_instance(**params)
        return json.dumps(result_payload, ensure_ascii=False, indent=2, default=str), f"✅ 成功 (耗时: {time.time()-start_time:.2f}s)"
    except Exception as e:
        return json.dumps({"status": "error", "error": str(e)}, ensure_ascii=False, indent=2), f"❌ 失败: {str(e)}"
        
def stream_agent_sandbox_execution(
    user_query: str,
    llm_model: str,
    top_k_ret: int,
    top_k_rerank: int,
    filter_input: str,
    user_state: dict,
):
    clean_query = user_query.strip() if user_query else ""
    user_role = user_state.get("role", "user") if user_state else "user"
    if not clean_query:
        yield "⚠️ 请输入有效的测试问题！"
        return

    # ==================== 🔍 DEBUG PRINT 开始 ====================
    print("\n" + "=" * 60)
    print("🐞 [Tab 2 沙盒 Debug] 启动 ReAct Agent 推理诊断")
    print(f"👉 输入 Query: {clean_query} | 当前用户 Role: {user_role}")
    print(f"👉 当前模型 (LLM): {llm_model}")
    print(f"👉 Top-K 检索/重排: {top_k_ret} / {top_k_rerank}")
    print(f"👉 Filter 过滤词: {filter_input}")

    # 1. 排查工厂绑定的全局全量工具 (Factory 索引)
    global_tools = get_all_registered_tool_names()
    print(f"👉 [全局注册工具总量]: {len(global_tools)} | 列表: {global_tools}")

    # 2. 实例化 Agent
    agent = ReActAgent(
        llm_client=None,
        model_name=llm_model,
        top_k_ret=top_k_ret,
        top_k_rerank=top_k_rerank,
        filter_str=filter_input,
        user_role=user_role,
        max_steps=5,
        sandbox_timeout=2,
    )

    # 3. 模拟三级路由诊断 (Level 1 Domain -> Level 2 Package -> Level 3 Tools)
    scoped_tool_names = []
    selected_domains = []
    selected_packages = []
    tool_specs = []

    if hasattr(agent, "_route_domains") and hasattr(agent, "_route_packages"):
        # Level 1 路由：Domain 剪枝
        selected_domains = agent._route_domains(clean_query)
        # Level 2 路由：Package 锁定
        selected_packages = agent._route_packages(clean_query, selected_domains)

        # Level 3 工具 Schema 提取 (同时提取 名称串 与 标准Specs列表)
        # 确保 ToolFactory 返回的是 as_json_string=False 的格式 (即 List[Dict])
        tool_names_str, tool_specs = agent.tool_factory.get_tools_metadata_by_packages(
            selected_packages, 
            as_json_string=False,
            user_role=user_role
        )

        # # Level 3 工具 Schema 提取
        # tool_names_str, _ = agent.tool_factory.get_tools_metadata_by_packages(selected_packages)
        scoped_tool_names = [t.strip() for t in tool_names_str.split(",") if t.strip()]

        print(f"🎯 [Level 1 命中领域 (Domain)]: {selected_domains}")
        print(f"📦 [Level 2 锁定工具包 (Package)]: {[pkg for _, pkg in selected_packages]}")
        print(f"⚡ [Level 3 按角色 ({user_role}) 过滤后的工具]: {scoped_tool_names}")
        print(f"📋 [Level 3 注入 LLM 的 Schema 总数]: {len(tool_specs)}")

        if "web_search" not in scoped_tool_names:
            print(f"ℹ️ 当前 Query 未激活 'web_search' 工具包，成功执行剪枝。")
        else:
            print("✅ 当前 Query 已成功绑定 'web_search' 工具。")
    print("=" * 60 + "\n")
    # ==================== 🔍 DEBUG PRINT 结束 ====================

    full_log = ""
    try:
        for step_data in agent.run_stream(clean_query, tools_schema=tool_specs):
            full_log += step_data.get("content", "") + "\n\n"
            yield full_log
    except Exception as e:
        logging.error(f"Tab 2 沙盒 Agent 执行异常: {e}", exc_info=True)
        yield full_log + f"\n\n🚨 异常: {str(e)}"

# def get_all_registered_tool_names(user_role: Optional[str] = None) -> list:
#     """按角色动态过滤可展示的工具列表"""
#     tools = []
#     if hasattr(tool_factory, "_flat_tools") and isinstance(tool_factory._flat_tools, dict):
#         for name, tool_obj in tool_factory._flat_tools.items():
#             if tool_factory._is_tool_visible(tool_obj, user_role):
#                 tools.append(name)
#     return tools or ["search_knowledge_base"]
def get_all_registered_tool_names(user_role: Optional[str] = None) -> list:
    tools = []
    print(f"\n---------------- [DEBUG 2: 下拉框过滤] ----------------")
    print(f"👉 入参 user_role: '{user_role}'")
    
    if hasattr(tool_factory, "_flat_tools") and isinstance(tool_factory._flat_tools, dict):
        for name, tool_obj in tool_factory._flat_tools.items():
            is_vis = tool_factory._is_tool_visible(tool_obj, user_role)
            print(f"   - 工具 [{name}] 对角色 [{user_role}] 可见性: {is_vis}")
            if is_vis:
                tools.append(name)
    print(f"👉 最终生成下拉框列表: {tools}")
    print(f"-------------------------------------------------------\n")
    return tools or ["search_knowledge_base"]
# def get_all_registered_tool_names() -> list:
#     tools = []
#     if hasattr(tool_factory, "_flat_tools") and isinstance(tool_factory._flat_tools, dict):
#         tools = list(tool_factory._flat_tools.keys())
#     return tools or ["search_knowledge_base"]

# ==========================================
# 🖥️ 5. 构建带“4 个完整 Tab”的 Gradio 应用
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
    # 📌 1. 显式加载 tools.yaml，确保配置文件中的工具与角色白名单全量载入
    yaml_path = os.path.join(SCRIPT_DIR, "config", "tools.yaml")
    print(f"\n================ [DEBUG 1: YAML 路径与解析] ================")
    print(f"👉 读取的 YAML 路径: {yaml_path}")
    if os.path.exists(yaml_path):
        try:
            load_tools_from_yaml(yaml_path, tool_factory)
            logging.info("✅ 成功从 tools.yaml 装载动态工具元数据")
        except Exception as e:
            logging.error(f"❌ 加载 tools.yaml 失败: {e}")
    print(f"===========================================================\n")
    try:
        if qa_chain is not None:
            init_tools(retriever=qa_chain.retriever, reranker=qa_chain.reranker)
        else:
            retriever = FineBIRetriever(
                milvus_host=DEFAULT_QA_CONFIG["milvus_host"], 
                milvus_port=DEFAULT_QA_CONFIG["milvus_port"], 
                collection_name=DEFAULT_QA_CONFIG["collection_name"], 
                cuda_device=DEFAULT_QA_CONFIG["cuda_device"]
            )
            reranker = FineBIReranker(cuda_device=DEFAULT_QA_CONFIG["cuda_device"])
            init_tools(retriever=retriever, reranker=reranker)
    except Exception as e:
        logging.warning(f"⚠️ 工具初始化说明: {e}")

    registered_tools = get_all_registered_tool_names()

# def build_qa_admin_ui(qa_chain: Optional[Any] = None):
#     yaml_path = os.path.join(SCRIPT_DIR, "tools.yaml")
#     try:
#         if qa_chain is not None:
#             init_tools(retriever=qa_chain.retriever, reranker=qa_chain.reranker)
#         else:
#             retriever = FineBIRetriever(milvus_host=DEFAULT_QA_CONFIG["milvus_host"], milvus_port=DEFAULT_QA_CONFIG["milvus_port"], collection_name=DEFAULT_QA_CONFIG["collection_name"], cuda_device=DEFAULT_QA_CONFIG["cuda_device"])
#             reranker = FineBIReranker(cuda_device=DEFAULT_QA_CONFIG["cuda_device"])
#             init_tools(retriever=retriever, reranker=reranker)
#     except Exception as e:
#         logging.warning(f"⚠️ 工具初始化说明: {e}")

#     registered_tools = get_all_registered_tool_names()


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
                        outputs=[chatbot, sources_display, status_box, gpu_box]
                        # outputs=[chatbot, sources_display, status_box, gpu_box, gpu_box, gpu_box]
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
                # 📌 1. 初始化时使用安全角色 ('user') 获取默认列表，防止未登录状态露出高权工具
                safe_tools = get_all_registered_tool_names(user_role="user")

                with gr.Tab("🛠️ Tools & Agent 沙盒测试"):
                    gr.Markdown("### 🧪 1. 原子工具 (Tool) 独立功能测试")
                    with gr.Row():
                        with gr.Column(scale=6):
                            # 📌 将 choices 指向 safe_tools
                            tool_select = gr.Dropdown(
                                choices=safe_tools, 
                                value=safe_tools[0] if safe_tools else None, 
                                label="🔧 选择工具",
                                interactive=True
                            )
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

                    btn_run_tool.click(fn=test_tool_execution, inputs=[tool_select, tool_json_input, user_state], outputs=[tool_json_output, tool_status])
                    run_btn.click(
                        fn=stream_agent_sandbox_execution, 
                        inputs=[test_input, llm_dropdown, slider_top_k_ret, slider_top_k_rerank, filter_input, user_state],
                        outputs=[react_log_markdown]
                    )

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
                    # 用户赞踩反馈机制
                    agent_chatbot.like(
                        fn=handle_chatbot_like,
                        inputs=[agent_chatbot, user_state],
                        outputs=None  # 纯后台异步静默记录，不打扰前端输出
                    )
                    
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

                # ---------------------------------------------------------
                # 🌟 Tab 4: 📊 基础运维与可观测性面板 (新增)
                # ---------------------------------------------------------
                with gr.Tab("📊 基础运维与可观测性"):
                    gr.Markdown("### 🔍 系统轻量级使用统计与风控监控")
                    with gr.Row():
                        btn_refresh_obs = gr.Button("🔄 刷新监控指标", variant="primary", scale=2)
                    
                    with gr.Row():
                        with gr.Column(scale=7):
                            obs_summary_display = gr.Markdown(value="*点击上方刷新按钮同步最新统计数据...*")
                        with gr.Column(scale=5):
                            obs_json_display = gr.JSON(label="📦 原始 Metrics JSON 数据 Payload")

                    # 页面首次加载或点击刷新时更新监控指标
                    btn_refresh_obs.click(
                        fn=render_observability_dashboard,
                        inputs=None,
                        outputs=[obs_summary_display, obs_json_display]
                    )
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
            if u_clean not in VALID_USERS_PWD:
                return f"❌ 用户 `{u_clean}` 不存在！"
            
            # 2. 校验原密码
            if VALID_USERS_PWD[u_clean] != old_pwd:
                return "❌ 原密码错误！"
                
            # 3. 校验两次新密码一致性
            if new_pwd != confirm_pwd:
                return "❌ 两次输入的新密码不一致！"
                
            # 4. 校验新旧密码是否相同
            if old_pwd == new_pwd:
                return "❌ 新密码不能与原密码相同！"
            
            # 5. YAML 文件持久化落盘逻辑
            # yaml_path = "config/users_auth.yaml"
            yaml_path = USERS_AUTH_PATH

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
                VALID_USERS_PWD[u_clean] = new_pwd
                
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
                    gr.update(visible=True),                   # login_view
                    gr.update(visible=False),                  # main_portal_view
                    "# 🤖 FineBI QA 智能问答",                  # user_info_banner
                    {"is_logged_in": False, "username": ""},   # user_state
                    gr.update(choices=[], value=None),         # t3_session_radio
                    gr.update(choices=[], value=None),         # tool_select (新增)
                    "*点击上方刷新按钮同步最新统计数据...*",        # obs_md
                    {}                                         # obs_json
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
                            
                            logging.info(f"🔍 [Cookie 诊断] 解析到用户名: '{user_val}' | VALID_USERS 匹配结果: {user_val in VALID_USERS_PWD}")
                            
                            if user_val in VALID_USERS_PWD:
                                found_user = user_val
                                break
                except Exception as e:
                    logging.error(f"❌ [Cookie 诊断] 解析异常: {e}")

            # 4. 渲染界面分支
            if found_user:
                found_role = USER_ROLES.get(found_user, "user")
                mem_mgr = get_or_create_user_memory(found_user)
                new_state = {"is_logged_in": True, "username": found_user, "role": found_role}
                banner_text = f"# 🤖 FineBI QA 智能问答与 Agent 调试台 (当前登录用户: `{found_user}` | 角色: `{found_role}`)"                
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
                # 📌 关键追加：根据当前用户的 Role 过滤工具列表
                role_tools = get_all_registered_tool_names(user_role=found_role)
                default_tool = role_tools[0] if role_tools else None
                logging.info(f"✅ [Cookie 校验] 成功免密自动登录！用户: {found_user}| 角色: {found_role}")
                obs_md, obs_json = render_observability_dashboard()
                return (
                    gr.update(visible=False),                                 # login_view
                    gr.update(visible=True),                                  # main_portal_view
                    banner_text,                                              # user_info_banner
                    new_state,                                                # user_state
                    gr.update(choices=session_choices, value=default_sess),     # t3_session_radio
                    gr.update(choices=role_tools, value=default_tool),        # 👈 tool_select (动态同步工具)
                    obs_md,                                                   # obs_summary_display
                    obs_json                                                  # obs_json_display
                )
            else:
                logging.warning("⚠️ [Cookie 校验] 未找到合法 Cookie，返回登录页")
                return (
                    gr.update(visible=True),
                    gr.update(visible=False),
                    "# 🤖 FineBI QA 智能问答",
                    {"is_logged_in": False, "username": ""},
                    gr.update(choices=[], value=None),                        # t3_session_radio
                    gr.update(choices=[], value=None),                        # tool_select
                    "*点击上方刷新按钮同步最新统计数据...*",
                    {}
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
                tool_select,
                obs_summary_display,
                obs_json_display
            ],
        )

        # =========================================================
        # 🔑 逻辑 2：主动点击“登录按钮”逻辑响应
        # =========================================================
        def perform_login(username, password):
            username_clean = username.strip() if username else ""
            if username_clean in VALID_USERS_PWD and VALID_USERS_PWD[username_clean] == password:
                mem_mgr = get_or_create_user_memory(username_clean)
                found_role = USER_ROLES.get(username_clean, "user")
                role_tools = get_all_registered_tool_names(user_role=found_role)
                default_tool = role_tools[0] if role_tools else None
                new_state = {"is_logged_in": True, "username": username_clean, "role": found_role}
                banner_text = f"# 🤖 FineBI QA 智能问答与 Agent 调试台 (当前登录用户: `{username_clean}` | 角色: `{found_role}`)"

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

                obs_md, obs_json = render_observability_dashboard()
                return (
                    gr.update(visible=False),              # login_view 隐藏
                    gr.update(visible=True),               # main_portal_view 显示
                    banner_text,                           # user_info_banner
                    new_state,                             # user_state
                    "✅ 登录成功！",                       # login_msg
                    gr.update(choices=session_choices, value=default_sess), # t3_session_radio
                    gr.update(choices=role_tools, value=default_tool),
                    obs_md,
                    obs_json
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
                tool_select,
                obs_summary_display, # 📌 必须补充此项，与 perform_login 返回值对齐
                obs_json_display     # 📌 必须补充此项，与 perform_login 返回值对齐
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
        server_port=7865,
        root_path="/qa"
        # share=True
    )
