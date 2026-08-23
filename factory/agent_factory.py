# factory/agent_factory.py
import json
import logging
from typing import Dict, Any, Generator
from factory.tool_factory import tool_factory
from generator.llm_client import FineBILLMClient

logger = logging.getLogger("AgentFactory")

class RouterAgent:
    """Level 1: 意图识别与业务领域路由 Agent"""
    def __init__(self, llm_client: FineBILLMClient):
        self.llm_client = llm_client

    def route(self, query: str) -> str:
        domains_summary = tool_factory.get_domains_summary()
        prompt = f"""你是一个智能路由助手。请分析用户输入的意图，并从以下可选业务领域中选择最匹配的一个。
            只能输出选中的 domain 名称，不要输出任何其他解释文字。

            可选领域列表：
            {json.dumps(domains_summary, ensure_ascii=False, indent=2)}

            用户问题：{query}
            匹配的 domain："""
        
        # 调用大模型生成决策
        response = self.llm_client.generate(prompt=prompt, max_new_tokens=20)
        selected_domain = response.strip().lower()
        
        # 校验合法性
        valid_domains = [d["domain"] for d in domains_summary]
        if selected_domain not in valid_domains:
            selected_domain = "rag_knowledge" # 默认兜底领域
            
        logger.info(f"🔀 [RouterAgent] 问题意图路由结果: {selected_domain}")
        return selected_domain


class ReActAgent:
    """Level 2 & 3: 结合指定领域工具链的多步推理与工具执行 Agent"""
    def __init__(self, llm_client: FineBILLMClient, domain: str):
        self.llm_client = llm_client
        self.domain = domain
        # 按领域隔离加载工具描述 (节约 Token)
        self.tools_schema = tool_factory.get_tools_schema_by_domains([domain])

    def run_stream(self, query: str) -> Generator[str, None, None]:
        prompt = f"""你可以使用以下工具来回答问题：
            {self.tools_schema}

            请按照以下格式回答：
            Thought: 思考下一步该怎么做
            Action: 要使用的工具名称（必须是上述工具之一）
            Action Input: 工具调用的 JSON 格式参数
            Observation: 工具执行的结果
            ... (重复 Thought/Action/Action Input/Observation 过程)
            Final Answer: 最终给用户的完整回答

            用户问题：{query}
            Thought:"""
        
        yield f"🤔 [Agent 推理 - Domain: {self.domain}]\n"
        # 1. 触发大模型思考与 Action 生成
        response = self.llm_client.generate(prompt=prompt, max_new_tokens=512)
        
        # 2. 简单的解析逻辑（实际工程可使用正则匹配 Action & Action Input）
        if "Action:" in response and "Action Input:" in response:
            try:
                # 解析工具名与参数
                lines = response.split("\n")
                tool_name = ""
                tool_input_str = ""
                for line in lines:
                    if line.startswith("Action:"):
                        tool_name = line.replace("Action:", "").strip()
                    elif line.startswith("Action Input:"):
                        tool_input_str = line.replace("Action Input:", "").strip()

                yield f"🛠️ [调用工具]: `{tool_name}` | 参数: `{tool_input_str}`\n"
                
                # 执行工具
                tool = tool_factory.get_tool(tool_name)
                if tool:
                    kwargs = json.loads(tool_input_str) if tool_input_str else {}
                    obs = tool.run(**kwargs)
                    yield f"👁️ [工具返回]:\n{obs}\n\n"
                    
                    # 总结最终回答
                    final_prompt = f"{prompt}\n{response}\nObservation: {obs}\nFinal Answer:"
                    final_ans = self.llm_client.generate(prompt=final_prompt, max_new_tokens=512)
                    yield f"💬 [最终回答]:\n{final_ans}"
                else:
                    yield f"❌ 未找到工具: {tool_name}"
            except Exception as e:
                yield f"❌ 工具解析/执行异常: {e}"
        else:
            # 无需调用工具，直接给出回答
            yield f"💬 [最终回答]:\n{response}"


class AgentFactory:
    """Agent 工厂入口"""
    @staticmethod
    def create_pipeline(llm_client: FineBILLMClient):
        return RouterAgent(llm_client)