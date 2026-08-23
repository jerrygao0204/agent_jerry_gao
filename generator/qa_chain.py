# generator/qa_chain.py
import os
import sys
import logging
from threading import Thread
from typing import List, Dict, Any, Generator, Optional

from transformers import TextIteratorStreamer

# 📂 动态计算项目根目录，注入系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 统一引入各核心组件
from search.retriever import FineBIRetriever
from search.reranker import FineBIReranker
from generator.llm_client import FineBILLMClient
# 🛡️ 引入合规与安全检查模块
from agent.compliance import ComplianceChecker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")


class QAChain:
    """
    RAG 核心问答链（端到端：混合检索 -> 交叉重排 -> Prompt拼装 -> 流式生成 -> 合规安全拦截与脱敏）
    """
    def __init__(
        self,
        cuda_device: str = "0",
        prompt_hub_path: str = "config/prompt_hub.yaml",
        top_k_retrieval: int = 10,
        top_k_rerank: int = 3,
        llm_short_name: str = "Qwen/Qwen3-8B"
    ):
        self.top_k_retrieval = top_k_retrieval
        self.top_k_rerank = top_k_rerank

        logging.info("⚙️ 正在初始化 QAChain 全链路问答组件...")
        
        # 1. 初始化 LLM 客户端（底层创建 ModelFactory）
        self.llm_client = FineBILLMClient(
            model_short_name=llm_short_name,
            prompt_hub_path=prompt_hub_path,
            cuda_device=cuda_device
        )
        # 获取共享的 factory 单例
        self.factory = self.llm_client.factory
        self.model, self.tokenizer = self.factory.get_llm_model(llm_short_name)
        self.prompts = self.factory.prompts

        # 2. 实例化检索器与重排器，共享同一 factory 实例
        self.retriever = FineBIRetriever(
            prompt_hub_path=prompt_hub_path,
            cuda_device=cuda_device
        )
        self.reranker = FineBIReranker(
            cuda_device=cuda_device,
            factory=self.factory
        )

        # 3. 🛡️ 挂载合规安全审计模块 (ComplianceChecker)
        self.compliance_checker = ComplianceChecker()

        logging.info("✅ QAChain 全链路初始化完毕，合规与安全检查模块挂载成功！")

    def _extract_doc_info(self, item: Dict[str, Any]) -> tuple:
        """多种键名兼容解析：提取来源文件名、章节ID以及文本内容"""
        metadata = item.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        # 兼容各种常见的文档名字段
        source = (
            metadata.get("source_file") or 
            metadata.get("file_name") or 
            metadata.get("doc_name") or 
            item.get("source_file") or 
            item.get("file_name") or 
            item.get("source") or 
            "未知文档"
        )

        section = (
            metadata.get("section_id") or 
            metadata.get("heading") or 
            item.get("section_id") or 
            "常规章节"
        )

        content = item.get("content") or item.get("text") or item.get("chunk_text") or ""
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
    ) -> str:
        """
        🌟 拼装 Prompt：完美兼容 Gradio / OpenAI 风格的 history 结构
        """
        context_str = self._build_context_str(contexts)
        
        prompt_template = self.prompts.get(
            "rag_qa",
            "你是一个专业的 FineBI 助手。请严格依据以下提供的参考资料回答用户的问题。\n"
            "如果资料不足以回答问题，请如实告知，切勿捏造答案。\n\n"
            "【参考资料】:\n{context}\n\n"
            "【用户问题】:\n{query}\n\n"
            "请给出清晰、准确、结构化的回答："
        )

        user_content = prompt_template.format(context=context_str, query=query)
        
        messages = []
        
        # 🌟 1. 解析历史消息（兼容 {"role": "...", "content": "..."} 与 {"user": "...", "assistant": "..."}）
        if history:
            for item in history:
                if "role" in item:
                    role = item["role"]
                    raw_content = item.get("content", "")
                    # 如果历史里不幸混入了【用户问题】前缀，清洗剥离出纯原始提问
                    if role == "user" and "【用户问题】:" in raw_content:
                        raw_content = raw_content.split("【用户问题】:")[-1].strip()
                    messages.append({"role": role, "content": raw_content})
                elif "user" in item:
                    # 兼容双键名旧格式
                    clean_q = item["user"].split("【用户问题】:")[-1].strip()
                    messages.append({"role": "user", "content": clean_q})
                    if "assistant" in item:
                        messages.append({"role": "assistant", "content": item["assistant"]})
                
        # 🌟 2. 将【当轮最新 Context + 最新 Query】作为最后一个 user 消息压入
        messages.append({"role": "user", "content": user_content})

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

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
        :param query: 用户提问
        :param history: 对话历史 [{"user": "...", "assistant": "..."}, ...]
        :param filter_expr: Milvus 元数据过滤表达式 (如: target_version == 'V6.0')
        :param pre_retrieved_chunks: 预检索并已重排的文档切片列表
        :param enable_compliance_check: 是否开启合规检查与脱敏
        """
        logging.info(f"🚀 开始 QA 链推理，Query: {query} | Filter: {filter_expr}")

        # 🛡️ 1. 输入问题前的防越狱/高危 SQL 预检
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
            logging.info(f"🚀 开始 QA 链推理，Query: {query} | History 轮数: {len(history) if history else 0}")
            reranked_chunks = pre_retrieved_chunks
        else:
            raw_chunks = self.retriever.hybrid_search(
                query=query, 
                top_k=self.top_k_retrieval, 
                filter_expr=filter_expr
            )

            # 2. 执行交叉重排
            reranked_chunks = self.reranker.rerank(
                query=query, 
                documents=raw_chunks, 
                top_n=self.top_k_rerank
            )

        # 先吐出 sources 召回来源消息包
        yield {"type": "sources", "data": reranked_chunks}

        # 3. 组装 Prompt 模板
        full_prompt = self.format_prompt(
            query=query, 
            contexts=reranked_chunks, 
            history=history
        )

        # 4. 流式生成与合规脱敏
        accumulated_text = ""
        for response in self.llm_client.stream_generate(query=query, context=full_prompt):
            # 获取文本内容
            token = response.get("data", "") if isinstance(response, dict) else str(response)
            accumulated_text += token

            # 如果开启合规校验，实时对当前累加文本进行数据脱敏处理（如掩码手机号）
            if enable_compliance_check:
                sanitized_token = self.compliance_checker.sanitize_text(token)
                yield {"type": "text", "data": sanitized_token}
            else:
                yield {"type": "text", "data": token}

        # 5. 🛠️ 生成结束后的最终安全审计（检查完整的生成文本）
        if enable_compliance_check:
            audit_result = self.compliance_checker.audit_and_sanitize(accumulated_text)
            if not audit_result["passed"]:
                logging.warning(f"🛡️ [Compliance] QA Chain 生成结果触发合规拦截: {audit_result['blocked_by']}")
                # 若触发拦截，替换整个回答输出降级拦截文本
                yield {"type": "security_block", "data": audit_result["sanitized_text"]}


# =====================================================================
# 🧪 测试与可视化打印
# =====================================================================
if __name__ == "__main__":
    chain = QAChain(cuda_device="0")
    test_query = "请问怎么清理 FineBI 的日志文件？是不是用 sudo rm -rf /var/log/finebi 这个命令？"

    print("\n" + "=" * 60)
    print(f"❓ 用户提问: {test_query}")
    print("=" * 60 + "\n")

    for response in chain.stream_answer(query=test_query):
        if response["type"] == "sources":
            print("📚 【召回参考来源与匹配文本内容】:")
            for idx, doc in enumerate(response["data"], 1):
                source, section, content, score = chain._extract_doc_info(doc)
                print(f"  📌 [{idx}] 来源文档: {source} | 章节: {section} | 相关度得分: {score:.4f}")
                print(f"     📄 内容片段: {content}")
                print("  " + "-" * 56)
            
            print("\n🤖 【LLM 回答】: ", end="", flush=True)
            
        elif response["type"] == "text":
            print(response["data"], end="", flush=True)

        elif response["type"] == "security_block":
            print(f"\n\n🚨 [系统提示]: {response['data']}")

    print("\n\n🎉 QAChain 全链路问答成功跑通！")
