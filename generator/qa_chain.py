# generator/qa_chain.py
import os
import sys
import logging
from typing import List, Dict, Any, Generator, Optional

# 📂 动态计算项目根目录，注入系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 统一引入各核心组件（对接 ModelFactory 网关架构）
from factory.model_factory import ModelFactory
from search.retriever import Retriever
from search.reranker import Reranker
from generator.llm_client import LLMClient
# 🛡️ 引入合规与安全检查模块
from agent.compliance import ComplianceChecker
# ⏱️ 引入超时保护模块
from utils.timeout_ctx import timeout, TimeoutException

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")


class QAChain:
    """
    RAG 核心问答链（端到端：混合检索 -> 交叉重排 -> Prompt拼装 -> 流式生成 -> 合规安全拦截与脱敏）
    """
    def __init__(
        self,
        top_k_retrieval: int = 10,
        top_k_rerank: int = 3,
        llm_model_name: str = "qwen3-4b",
        embedding_model_name: str = "qwen3-embedding-4b" # 👈 允许外部传入向量模型名称
    ):
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank
        self.llm_model_name = llm_model_name
        self.embedding_model_name = embedding_model_name

        logging.info("⚙️ 正在初始化 QAChain 全链路问答组件...")

        # 初始化 API/vLLM LLM 客户端（通过网关统一接管）
        self.llm_client = LLMClient(default_model_name=self.llm_model_name)

        # 2. 实例化检索器与重排器（Retriever 内部按需自行持有 ModelFactory 单例）
        self.retriever = Retriever(
            milvus_host="172.17.0.1",
            collection_name="finebi_knowledge_chunks",
            default_model_name=self.embedding_model_name
        )
        self.reranker = Reranker(
            cache_dir="/workspace/hf-conda/hf_cache/hub"
        )

        # 3. 🛡️ 挂载合规安全审计模块 (ComplianceChecker)
        self.compliance_checker = ComplianceChecker()

        logging.info("✅ QAChain 全链路初始化完毕，合规与安全检查模块挂载成功！")

    def _extract_doc_info(self, item: Dict[str, Any]) -> tuple:
        """多种键名兼容解析：提取来源文件名、章节ID以及文本内容"""
        # 兼容 Retriever 返回的扁平字典結構或嵌套 metadata
        source = (
            item.get("file_url") or 
            item.get("source_file") or 
            item.get("file_name") or 
            "未知文档"
        )

        section = (
            item.get("hierarchy") or 
            item.get("section_id") or 
            "常规章节"
        )

        content = item.get("content") or item.get("base_content") or item.get("text") or ""
        score = item.get("rerank_score", item.get("score", 0.0))

        return source, section, content.strip(), score

    def _build_context_str(self, contexts: List[Dict[str, Any]]) -> str:
        """将 Rerank 后的 Top-K Chunk 格式化为 Prompt 输入的 Context 文本"""
        if not contexts:
            return "（未检索到直接相关的参考资料）"

        formatted_chunks = []
        for idx, item in enumerate(contexts, 1):
            source, section, content, _ = self._extract_doc_info(item)
            chunk_text = f"[参考资料 {idx}] (来源: {source} | 章节: {section})\n{content}"
            formatted_chunks.append(chunk_text)

        return "\n\n".join(formatted_chunks)

    def format_prompt(
        self, 
        query: str, 
        contexts: List[Dict[str, Any]], 
        history: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, str]]:
        """
        🌟 拼装多轮对话 Prompt 结构：完美兼容 Gradio / OpenAI 风格的 messages 数组
        """
        context_str = self._build_context_str(contexts)
        
        system_prompt = (
            "你是一个专业的 RAG 企业级架构助手。请严格依据以下提供的参考资料回答用户的问题。\n"
            "如果资料不足以回答问题，请如实告知，切勿捏造答案。回答必须清晰、准确、结构化。"
        )

        user_content = (
            f"参考以下背景知识回答问题：\n"
            f"【背景知识】\n{context_str}\n\n"
            f"【用户问题】\n{query}"
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # 🌟 1. 解析历史消息
        if history:
            for item in history:
                if "role" in item:
                    role = item["role"]
                    raw_content = item.get("content", "")
                    if role == "user" and "【用户问题】:" in raw_content:
                        raw_content = raw_content.split("【用户问题】:")[-1].strip()
                    messages.append({"role": role, "content": raw_content})
                elif "user" in item:
                    clean_q = item["user"].split("【用户问题】:")[-1].strip()
                    messages.append({"role": "user", "content": clean_q})
                    if "assistant" in item:
                        messages.append({"role": "assistant", "content": item["assistant"]})
                
        # 🌟 2. 将当轮最新 Context + 最新 Query 压入消息列表
        messages.append({"role": "user", "content": user_content})

        return messages

    def stream_answer(
        self, 
        query: str, 
        history: Optional[List[Dict[str, Any]]] = None, 
        filter_expr: Optional[str] = None, 
        pre_retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
        enable_compliance_check: bool = True
    ) -> Generator[Dict[str, Any], None, None]:
        """
        流式问答生成入口
        """
        logging.info(f"🚀 开始 QA 链推理，Query: {query} | Filter: {filter_expr}")

        # 🛡️ 1. 输入问题前的防越狱/高危规则预检
        if enable_compliance_check:
            is_safe, risk_level, hit_rule = self.compliance_checker.check_static_rules(query)
            if not is_safe:
                fallback_response = self.compliance_checker.fallback_responses.get(
                    risk_level, 
                    "⚠️ [安全拦截] 您的提问包含不符合安全规范的内容，系统已被阻止处理。"
                )
                yield {"type": "text", "data": fallback_response}
                return
            
        # 2. 执行混合检索 (或直接使用预检索结果)
        if pre_retrieved_chunks is not None:
            logging.info(f"🚀 使用外部传入的预检索切片，数量: {len(pre_retrieved_chunks)}")
            reranked_chunks = pre_retrieved_chunks
        else:
            try:
                # ⏱️ 为 Milvus 混合检索与重排添加 130 秒超时保护
                with timeout(130, label="Milvus 混合检索与交叉重排"):
                    raw_chunks = self.retriever.hybrid_search(
                        query=query, 
                        top_k=self.top_k_retrieval, 
                        filter_expr=filter_expr
                    )

                    # 执行交叉重排
                    reranked_chunks = self.reranker.rerank(
                        query=query, 
                        documents=raw_chunks, 
                        top_n=self.top_k_rerank
                    )
            except TimeoutException as e:
                logging.error(f"⚠️ [qa_chain] 检索阶段超时: {e}")
                reranked_chunks = []
                yield {"type": "sources", "data": []}
                yield {"type": "text", "data": "⚠️ [系统提示] 知识库检索超时，已为您切至无背景知识回答模式。\n\n"}

        # 先吐出 sources 召回来源消息包
        yield {"type": "sources", "data": reranked_chunks}

        # 3. 组装标准 messages 对话结构
        final_messages = self.format_prompt(
            query=query, 
            contexts=reranked_chunks, 
            history=history
        )

        # 4. 流式生成与合规脱敏
        accumulated_text = ""
        for token in self.llm_client.stream_generate(messages=final_messages, model_name=self.llm_model_name):
            accumulated_text += token

            # 如果开启合规校验，实时对当前累加文本进行数据脱敏处理
            if enable_compliance_check:
                sanitized_token = self.compliance_checker.sanitize_text(token)
                yield {"type": "text", "data": sanitized_token}
            else:
                yield {"type": "text", "data": token}

        # 5. 🛠️ 生成结束后的最终安全审计
        if enable_compliance_check:
            audit_result = self.compliance_checker.audit_and_sanitize(accumulated_text)
            if not audit_result["passed"]:
                logging.warning(f"🛡️ [Compliance] QA Chain 生成结果触发合规拦截: {audit_result['blocked_by']}")
                yield {"type": "security_block", "data": audit_result["sanitized_text"]}


# =====================================================================
# 🧪 测试与可视化打印
# =====================================================================
if __name__ == "__main__":
    # 🌍 环境自动适配检查（ModelFactory 读取 OPENAI_BASE_URL）
    if os.path.exists("/workspace") and not os.environ.get("OPENAI_BASE_URL"):
        os.environ["OPENAI_BASE_URL"] = "http://172.17.0.1:4000/v1"
        print("🔧 [自动适配] 检测到处于容器内部，已将 LiteLLM 网关自动重定向至宿主机: http://172.17.0.1:4000/v1")

    # 🌟 从最上游入口显式注入模型代号，完美隔离中间过程
    chain = QAChain(
        llm_model_name="qwen3-4b",
        embedding_model_name="qwen3-embedding-4b"
    )
    
    test_query = "怎么创建预警用户？"

    print("\n" + "=" * 60)
    print(f"❓ 用户提问: {test_query}")
    print("=" * 60 + "\n")

    for response in chain.stream_answer(query=test_query):
        if response["type"] == "sources":
            print("📚 【召回参考来源与匹配文本内容】:")
            for idx, doc in enumerate(response["data"], 1):
                source, section, content, score = chain._extract_doc_info(doc)
                print(f"  📌 [{idx}] 来源文档: {source} | 章节: {section} | 相关度得分: {score:.4f}")
                print(f"     📄 内容片段: {content[:150]}...")
                print("  " + "-" * 56)
            
            print("\n🤖 【LLM 回答】: ", end="", flush=True)
            
        elif response["type"] == "text":
            print(response["data"], end="", flush=True)

        elif response["type"] == "security_block":
            print(f"\n\n🚨 [系统提示]: {response['data']}")

    print("\n\n🎉 QAChain 全链路问答成功跑通！")