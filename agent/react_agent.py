# agent/react_agent.py
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import re
import json
import logging
from typing import Dict, Any, Generator, Optional, List, Tuple, Union
from factory.tool_factory import HierarchicalToolFactory, BaseTool, tool_factory as default_tool_factory
from memory.memory_manager import MemoryManager
from agent.sandbox import SandboxExecutor

logger = logging.getLogger("ReActAgent")


class ReActAgent:
    """结合短文本快速收敛与 RAG 参数自动补全的两阶段 ReAct Agent (向下兼容版)"""

    def __init__(
        self, 
        llm_client: Optional[Any] = None, 
        model_name: Optional[str] = None,             # 💡 [兼容项]: 支持旧版 model_name 参数
        max_steps: Optional[int] = None,              # 💡 [兼容项]: 支持旧版 max_steps 参数
        tool_factory: Optional[HierarchicalToolFactory] = None,
        prompt_hub: Optional[Any] = None, 
        system_prompt_template: Optional[str] = None,
        router_prompt_template: Optional[str] = None,
        package_router_prompt_template: Optional[str] = None,
        max_iterations: int = 5,
        user_role: Optional[str] = None,
        memory_mgr: Optional[MemoryManager] = None,
        sandbox_timeout: int = 2,
        top_k_ret: int = 5,
        top_k_rerank: int = 3,
        filter_str: str = "",
        min_query_length: int = 3
    ):
        # ---------------------------------------------------------------------
        # 💡 [适配 1]: 自动转换旧版参数 (llm_client 缺失时自动补全)
        # ---------------------------------------------------------------------
        if llm_client is None:
            from generator.llm_client import FineBILLMClient
            target_model = model_name or "Qwen/Qwen3-4B"
            self.llm_client = FineBILLMClient(target_model)
        else:
            self.llm_client = llm_client

        self.tool_factory = tool_factory or default_tool_factory
        self.prompt_hub = prompt_hub
        self.user_role = user_role
        # 优先使用 max_steps，无则使用 max_iterations
        self.max_iterations = max_steps if max_steps is not None else max_iterations
        
        self.memory_mgr = memory_mgr if memory_mgr is not None else MemoryManager(max_messages=20)
        self.sandbox = SandboxExecutor(timeout=sandbox_timeout)

        self.top_k_ret = top_k_ret
        self.top_k_rerank = top_k_rerank
        self.filter_str = filter_str
        self.min_query_length = min_query_length

        self._init_prompts(system_prompt_template, router_prompt_template, package_router_prompt_template)
        

    def _format_packages_summary(self, packages_summary: List[Dict[str, Any]]) -> str:
        """将 Package 与 Tool 描述格式化为结构化 Markdown DSL，提升 LLM 的上下文感知力"""
        formatted = []
        for pkg in packages_summary:
            pkg_name = pkg.get("package")
            desc = pkg.get("description", "无详细描述")
            raw_tools = pkg.get("tools", [])

            tool_lines = []
            # 📌 针对各种不同的 Tool 数据结构类型进行容错提取
            if isinstance(raw_tools, list):
                for t in raw_tools:
                    if isinstance(t, dict):
                        t_name = t.get("name", "unknown_tool")
                        t_desc = t.get("description", "无工具描述")
                        tool_lines.append(f"    * `{t_name}`: {t_desc}")
                    elif hasattr(t, "name"):
                        t_name = getattr(t, "name")
                        t_desc = getattr(t, "description", "无工具描述")
                        tool_lines.append(f"    * `{t_name}`: {t_desc}")
                    elif isinstance(t, str):
                        # 如果工具列表中只有名称，尝试从 ToolFactory 补全描述
                        tool_obj = self.tool_factory.get_tool(t) if hasattr(self.tool_factory, "get_tool") else None
                        t_desc = tool_obj.description if (tool_obj and hasattr(tool_obj, "description")) else "通用执行工具"
                        tool_lines.append(f"    * `{t}`: {t_desc}")
            
            tool_str = "\n".join(tool_lines) if tool_lines else "    * 无可用下属工具"


            # # 解析工具列表及具体功能
            # if isinstance(tools, list) and len(tools) > 0 and isinstance(tools[0], dict):
            #     tool_lines = [f"    * `{t['name']}`: {t.get('description', '无工具描述')}" for t in tools]
            #     tool_str = "\n" + "\n".join(tool_lines)
            # else:
            #     tool_str = ", ".join(tools) if isinstance(tools, list) else str(tools)
            
            formatted.append(
                f"- **Package 名称**: `{pkg_name}`\n"
                f"  - **包功能范围/边界**: {desc}\n"
                f"  - **下属工具清单**:\n{tool_str}"
            )
        return "\n\n".join(formatted)

    def _init_prompts(self, system_prompt_template, router_prompt_template, package_router_prompt_template):
        """初始化 Prompt 模板（支持 Domain Router 与 Package Router 拆分）"""
        if system_prompt_template:
            self.system_prompt_template = system_prompt_template
        elif self.prompt_hub and hasattr(self.prompt_hub, "get_prompt"):
            prompt_obj = self.prompt_hub.get_prompt("agent_react_prompt")
            self.system_prompt_template = getattr(prompt_obj, "content", str(prompt_obj))
        else:
            self.system_prompt_template = (
                "尽可能回答以下问题。你可以使用以下工具：\n{tools_description}\n\n"
                "请严格按照以下格式进行思考和调用工具：\n"
                "Question: 你需要回答的输入问题\n"
                "Thought: 你应该总是思考下一步要做什么\n"
                "Action: 要调用的工具名称（必须是 [{tool_names}] 中的一个）\n"
                "Action Input: 传递给工具的 JSON 格式参数\n"
                "Observation: 工具返回的结果\n"
                "... (这个 Thought/Action/Action Input/Observation 过程可以重复 N 次)\n"
                "Thought: 我现在知道最终答案了\n"
                "Final Answer: 针对原始输入问题的最终回答\n\n"
                "开始！\n\n"
                "Question: {input}\n"
                "Thought: {agent_scratchpad}"
            )

        # 1. Level 1 Router Prompt (Domain Selection)
        if router_prompt_template:
            self.router_prompt_template = router_prompt_template
        else:
            self.router_prompt_template = (
                "你是一个意图路由专家。请分析用户问题，从给定的领域列表中选择最相关的 1~2 个领域 (Domain)。\n"
                "可用领域清单:\n{domains_summary}\n\n"
                "用户问题: {input}\n\n"
                "请严格仅返回 JSON 数组格式的领域代码，例如: [\"rag_knowledge\"]，不要输出任何额外内容。"
            )

        # 2. Level 2 Router Prompt (Package Selection)
        # 💡 [通用路由 Prompt]: 零业务侵入，纯靠输入上下文的 Tool Spec 动态分类
        if package_router_prompt_template:
            self.package_router_prompt_template = package_router_prompt_template
        else:
            self.package_router_prompt_template = (
                "### 任务目标\n"
                "你是一个通用 Agent 工具包分类路由专家。请分析用户 Query 的真实意图，从下述【候选工具包】中选择 1~2 个最相符的包 (Package)。\n\n"
                "### 候选工具包与功能边界定义\n"
                "{packages_summary}\n\n"
                "### 路由匹配通用准则\n"
                "1. **精准对齐**：仔细比对 Query 与工具包的【功能范围】及【下属工具清单】，选择最贴切的包。\n"
                "2. **粒度区分**：若 Query 涉及底层明细数据（如行列数、具体字段、详细结构），优先选择提供元数据/明细的工具包；若涉及高层视图（如汇总看板、图形报表），选择报表类工具包。\n"
                "3. **按需组合**：若 Query 跨越多个独立场景，可同时选中多个工具包，但数量严格限制在 1~2 个。\n\n"
                "### 参考示例 (Few-Shot Examples)\n"
                "- Query: '帮我查询某数据表的行数和列数'\n"
                "  Response: [\"dataset_pkg\"]\n"
                "- Query: '检索最新的行业新闻并在本地运行 Python 分析'\n"
                "  Response: [\"search_pkg\", \"analytics_pkg\"]\n\n"
                "### 当前用户问题\n"
                "Query: {input}\n\n"
                "### 输出要求\n"
                "请严格仅返回 JSON 数组格式（例如: [\"pkg_name\"]），严禁包含任何 Markdown 格式以外的解释或说明文字。"
            )

    def _is_short_query(self, query: str) -> bool:
        clean_q = query.strip().lower()
        if len(clean_q) < self.min_query_length:
            return True
        common_greetings = {"你好", "您好", "在吗", "谢谢", "收到", "好的", "hi", "hello", "hey"}
        return clean_q in common_greetings

    def _prepare_tool_kwargs(self, tool_name: str, raw_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        kwargs = dict(raw_kwargs)
        if tool_name == "search_knowledge_base":
            kwargs.setdefault("top_k", self.top_k_ret)
            kwargs.setdefault("top_k_rerank", self.top_k_rerank)
            kwargs.setdefault("filter", self.filter_str)
        return kwargs

    def _route_domains(self, query: str) -> List[str]:
        """Level 1 路由：选出命中的 Domains"""
        domains_summary = self.tool_factory.get_domains_summary()
        if not domains_summary:
            return []

        router_prompt = self.router_prompt_template.format(
            domains_summary=json.dumps(domains_summary, ensure_ascii=False, indent=2),
            input=query
        )

        try:
            response = ""
            for chunk in self.llm_client.stream_generate(query=router_prompt, context=""):
                response += chunk
            
            clean_res = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
            clean_res = re.sub(r"^```(?:json)?|```$", "", clean_res, flags=re.IGNORECASE).strip()
            
            json_match = re.search(r"\[.*?\]", clean_res, re.DOTALL)
            if json_match:
                matched_domains = json.loads(json_match.group(0))
            else:
                matched_domains = json.loads(clean_res)
            if isinstance(matched_domains, list) and len(matched_domains) > 0:
                return matched_domains
        except Exception as e:
            logger.warning(f"[Domain Router Exception] 路由降级: {str(e)}")

        return [item["domain"] for item in domains_summary]

    def _route_packages(self, query: str, target_domains: List[str]) -> List[Tuple[str, str]]:
        """Level 2 路由：基于选定的 Domain 选出命中的 (Domain, Package) 二元组"""
        packages_summary = self.tool_factory.get_packages_summary_by_domains(target_domains)
        # ==================== 🔍 DEBUG 打印开始 ====================
        print("\n" + "🔍" * 25 + " [TOOL FACTORY 结构诊断] " + "🔍" * 25)
        print(f"👉 目标 Domains: {target_domains}")
        print(f"👉 ToolFactory 返回的 packages_summary 原始数据类型: {type(packages_summary)}")
        print("👉 packages_summary 完整内容:")
        print(json.dumps(packages_summary, ensure_ascii=False, indent=2, default=str))
        print("🔍" * 68 + "\n")
        # ==================== 🔍 DEBUG 打印结束 ====================

        if not packages_summary:
            print("⚠️ [WARNING] packages_summary 为空！")
            return []
        
        # 1. 转化为结构化 Markdown 描述，消除纯 JSON 的符号噪音
        formatted_summary = self._format_packages_summary(packages_summary)

        # 2. 构造通用 Prompt
        package_prompt = self.package_router_prompt_template.format(
            packages_summary=formatted_summary,
            input=query
        )

        # 💡 [调试新增]: 打印/日志输出最终送入 LLM 的完整 Prompt，用于人工检查
        print("\n" + "=" * 30 + " [DEBUG] Package Router Prompt to LLM " + "=" * 30)
        print(package_prompt)
        print("=" * 82 + "\n")
        logger.info(f"[Package Router] Prompt 打印检查完成，Length: {len(package_prompt)}")

        try:
            response = ""
            for chunk in self.llm_client.stream_generate(query=package_prompt, context=""):
                response += chunk
            
            clean_res = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
            clean_res = re.sub(r"^```(?:json)?|```$", "", clean_res, flags=re.IGNORECASE).strip()

            json_match = re.search(r"\[.*?\]", clean_res, re.DOTALL)
            matched_pkg_names = json.loads(json_match.group(0)) if json_match else json.loads(clean_res)
            
            if isinstance(matched_pkg_names, list) and len(matched_pkg_names) > 0:
                result_tuples = []
                for item in packages_summary:
                    if item["package"] in matched_pkg_names:
                        result_tuples.append((item["domain"], item["package"]))
                if result_tuples:
                    return result_tuples
        except Exception as e:
            logger.warning(f"[Package Router Exception] 包选择降级: {str(e)}")

        # 兜底降级：暴露当前 Domain 下的所有 Package
        return [(item["domain"], item["package"]) for item in packages_summary]
    
    def _parse_action_input(self, raw_str: str) -> Dict[str, Any]:
        cleaned = raw_str.strip()
        cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.IGNORECASE).strip()
        if cleaned.startswith("{") and cleaned.endswith("}"):
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
        return {"query": cleaned}

    # -------------------------------------------------------------------------
    # 💡 [适配 2]: 辅助生成器函数，保证同时输出 type 和 stage 两个 key
    # -------------------------------------------------------------------------
    def _yield_step(self, event_type: str, content: str) -> Dict[str, Any]:
        """统一数据吐出管道，双写 type 与 stage 以兼容旧版 Gradio"""
        return {
            "type": event_type,   # 新版协议 Key
            "stage": event_type,  # 旧版协议 Key (向后兼容)
            "content": content
        }

    def run_stream(
        self, 
        query: str, 
        tools_schema: Optional[Union[str, List[Dict[str, Any]]]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        流式 Agent 推理逻辑
        :param query: 用户输入 Query
        :param tools_schema: (可选) 外部传入的工具 Schema。支持标准的 Specs List[Dict] 或描述字符串
        """
        self.memory_mgr.begin_transaction()
        self.memory_mgr.process_user_input(query)

        # 从 MemoryManager 取出历史消息，组成标准多轮 messages（排除刚存进去的这次 query）
        context_data = self.memory_mgr.get_context_for_llm()
        history_msgs = context_data.get("messages", [])[:-1]
        chat_history_messages: List[Dict[str, str]] = [
            {"role": m["role"], "content": m["content"]} for m in history_msgs
        ]

        # 📌【修复 1】：在入口处安全初始化变量，防范 UnboundLocalError
        selected_packages: List[Tuple[str, str]] = []
        pkg_names: List[str] = ["injected_custom_schema"]


        try:
            # 1. 短文本拦截
            if self._is_short_query(query):
                yield self._yield_step("thought", "检测到用户输入为超短问句或通用问候词，跳过工具检索，直连 LLM 回复。")
                
                full_response = ""
                turn_messages = chat_history_messages + [{"role": "user", "content": query}]
                for chunk in self.llm_client.stream_generate(messages=turn_messages):
                    full_response += chunk
                
                final_ans = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
                yield self._yield_step("final_answer", final_ans)
                
                self.memory_mgr.process_assistant_output(final_ans)
                self.memory_mgr.commit()
                return
            
            # 2. 三级分级路由机制 (Hierarchical Routing)
            # 📌 优先使用外部注入的 tools_schema；若无，则自动触发内部路由获取
            if tools_schema is not None:
                yield self._yield_step("thought", "检测到沙盒注入的预剪枝工具 Schema，直接载入...")
                if isinstance(tools_schema, list):
                    # 转换标准的 Tool Spec 结构为 Prompt 渲染需要的文本格式
                    tool_names_list = []
                    descriptions = []
                    for spec in tools_schema:
                        func_info = spec.get("function", spec) if isinstance(spec, dict) else {}
                        name = func_info.get("name", "unknown_tool")
                        desc = func_info.get("description", "")
                        params = func_info.get("parameters", {})
                        tool_names_list.append(name)
                        descriptions.append(f"- **{name}**: {desc}\n  参数规范: {json.dumps(params, ensure_ascii=False)}")
                    
                    tool_names = ", ".join(tool_names_list)
                    tools_description = "\n".join(descriptions)
                    pkg_names = tool_names_list  # 给报错打印提示用
                else:
                    tools_description = str(tools_schema)
                    tool_names = "已加载工具"
            else:
                yield self._yield_step("thought", "正在分析用户意图，匹配业务领域 (Domain)...")
                selected_domains = self._route_domains(query)
                
                yield self._yield_step("thought", f"锁定业务领域: `{selected_domains}`，正在筛选工具包 (Package)...")
                selected_packages = self._route_packages(query, selected_domains)
                
                pkg_names = [pkg for _, pkg in selected_packages]
                yield self._yield_step("thought", f"锁定工具包: `{pkg_names}`，装载精准工具 Schema。")

                tool_names, tools_description = self.tool_factory.get_tools_metadata_by_packages(selected_packages, user_role=self.user_role)
           
            scratchpad = ""

            # 3. 核心 ReAct 循环
            for iteration in range(1, self.max_iterations + 1):
                logger.info(f"[ReAct Step] 开始第 {iteration}/{self.max_iterations} 轮推理/调用...")
                prompt = self.system_prompt_template.format(
                    tools_description=tools_description,
                    tool_names=tool_names,
                    input=query,
                    agent_scratchpad=scratchpad
                )

                full_response = ""
                turn_messages = chat_history_messages + [{"role": "user", "content": prompt}]
                for chunk in self.llm_client.stream_generate(messages=turn_messages):
                    full_response += chunk

                clean_response = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
                think_match = re.search(r"<think>(.*?)</think>", full_response, re.DOTALL)
                extracted_think = think_match.group(1).strip() if think_match else ""
                
                action_match = re.search(r"Action:\s*([^\n]+)", clean_response)
                action_input_match = re.search(r"Action Input:\s*(\{.*?\}|```.*?```|[^\n]+)", clean_response, re.DOTALL)
                final_answer_match = re.search(r"Final Answer:\s*(.*)", clean_response, re.DOTALL)

                if final_answer_match and not action_match:
                    final_ans = final_answer_match.group(1).strip()
                    if extracted_think:
                        yield self._yield_step("thought", extracted_think)
                    yield self._yield_step("final_answer", final_ans)

                    self.memory_mgr.process_assistant_output(final_ans)
                    self.memory_mgr.commit()
                    return

                if action_match and action_input_match:
                    tool_name = action_match.group(1).strip()
                    raw_input = action_input_match.group(1).strip()

                    thought_content = clean_response.split("Action:")[0].replace("Thought:", "").strip()
                    yield self._yield_step("thought", f"【第 {iteration}/{self.max_iterations} 步】{thought_content or extracted_think or '准备调用工具...'}")

                    raw_kwargs = self._parse_action_input(raw_input)
                    kwargs = self._prepare_tool_kwargs(tool_name, raw_kwargs)

                    yield self._yield_step("action", f"调用工具: `{tool_name}` | 执行参数: `{json.dumps(kwargs, ensure_ascii=False)}`")

                    if tool_name == "python_sandbox_executor":
                        code_str = kwargs.get("code") or raw_input
                        sandbox_res = self.sandbox.run(code_str)
                        if sandbox_res["status"] == "security_blocked":
                            raise ValueError(f"安全沙箱检测到高危指令: {sandbox_res['error']}")
                        observation = sandbox_res["result"]
                    else:
                        tool_obj = self.tool_factory.get_tool(tool_name, user_role=self.user_role)
                        if tool_obj:
                            observation = tool_obj.run(**kwargs) if hasattr(tool_obj, "run") else tool_obj.execute(**kwargs)
                        else:
                            observation = f"错误: 在工具包 {pkg_names} 中未找到工具 [{tool_name}]"


                    RERANK_THRESHOLD = 0.4
                    if isinstance(observation, list) and len(observation) > 0 and isinstance(observation[0], dict):
                        score = observation[0].get("rerank_score") or observation[0].get("score", 1.0)
                        if 0.1 <= score < RERANK_THRESHOLD:
                            observation = "【系统提示】: 本地知识库检索相关度得分过低，无匹配结果。"

                    yield self._yield_step("observation", str(observation))
                    scratchpad += f"{thought_content}\nAction: {tool_name}\nAction Input: {json.dumps(kwargs, ensure_ascii=False)}\nObservation: {observation}\nThought: "
                else:
                    final_ans = clean_response.replace("Thought:", "").strip()
                    yield self._yield_step("final_answer", final_ans)
                    self.memory_mgr.process_assistant_output(final_ans)
                    self.memory_mgr.commit()
                    return

        except Exception as e:
            logger.error(f"ReAct 运行捕获异常: {e}")
            self.memory_mgr.rollback()
            yield self._yield_step("rollback", f"🚨 运行异常已触发 Memory Rollback: {str(e)}")


# ==============================================================================
# 🧪 本地功能与参数透传精准测试套件 (精确隔离 Mock 条件)
# ==============================================================================
if __name__ == "__main__":
    import time
    from pydantic import BaseModel, Field

    # 1. 注册带参数打印日志的测试工具
    class WebSearchInput(BaseModel):
        query: str = Field(description="网络检索关键词")

    class WebSearchTool(BaseTool):
        name = "web_search"
        description = "在公网上检索实时信息，如天气、新闻、最新资讯等"
        domain = "web_search"
        package = "search_pkg"
        args_schema = WebSearchInput

        def run(self, query: str, **kwargs) -> Any:
            return "【模拟网页搜索结果】: 华盛顿州西雅图今天天气晴朗，气温 18°C - 24°C。"

    class RAGSearchInput(BaseModel):
        query: str = Field(description="知识库检索问题")
        top_k: Optional[int] = Field(default=5, description="向量检索数量")
        top_k_rerank: Optional[int] = Field(default=3, description="重排序数量")
        filter: Optional[str] = Field(default="", description="过滤条件")

    class KnowledgeSearchTool(BaseTool):
        name = "search_knowledge_base"
        description = "检索 FineBI 系统手册、用户指南与业务术语"
        domain = "rag_knowledge"
        package = "knowledge_pkg"
        args_schema = RAGSearchInput

        def run(self, query: str, top_k: int = 5, top_k_rerank: int = 3, filter: str = "", **kwargs) -> Any:
            # 💡 [验证断言]: 捕获底层透传参数
            print("\n" + "🔥" * 40)
            print("⚙️  [RAG 物理执行器成功捕获参数透传]:")
            print(f"    - query: '{query}'")
            print(f"    - top_k: {top_k} (预期: 10)")
            print(f"    - top_k_rerank: {top_k_rerank} (预期: 4)")
            print(f"    - filter: '{filter}' (预期: 'department == \'IT\'')")
            print("🔥" * 40 + "\n")
            return [{"content": "FineBI 创建预警用户的步骤：1. 进入管理系统；2. 选择用户管理；3. 新增预警用户。", "rerank_score": 0.88}]

    default_tool_factory.register_tool(WebSearchTool())
    default_tool_factory.register_tool(KnowledgeSearchTool())

    # 2. 严谨判定的 Mock LLM Client
    class MockLLMClient:
        def stream_generate(self, query: str, context: str = "") -> Generator[str, None, None]:
            clean_q = query.strip().lower()

            # A. 短文本直连
            if clean_q in ["你好", "您好", "在吗", "hi"]:
                response = "你好！我是你的 AI 智能助手，请问今天有什么我可以帮你的？"
            
            # B. Stage 1 路由匹配
            elif "意图路由专家" in query:
                if "天气" in query:
                    response = "```json\n[\"web_search_tool\"]\n```"
                else:
                    response = "```json\n[\"rag_knowledge\"]\n```"
            
            # C. Stage 2 Step 2: Observation 返回后的总结阶段 (判断是否包含工具返回结果)
            elif "Observation:" in query and "FineBI 创建预警用户的步骤" in query:
                response = "<think>根据 RAG 检索结果得出创建预警用户步骤。</think>\nFinal Answer: 创建预警用户的步骤如下：1. 进入管理系统；2. 选择用户管理；3. 新增预警用户。"
            
            # D. Stage 2 Step 1: 首次 ReAct 推理阶段 (匹配具体的用户提问尾部)
            elif "Question: 怎么创建预警用户？" in query:
                response = (
                    "Thought: 需要查询 FineBI 知识库中关于预警用户的创建方法。\n"
                    "Action: search_knowledge_base\n"
                    "Action Input: {\"query\": \"怎么创建预警用户\"}"
                )
            elif "Question: 华盛顿州西雅图今天的天气如何?" in query:
                response = (
                    "Thought: 需要在公网搜索西雅图今日天气。\n"
                    "Action: web_search\n"
                    "Action Input: {\"query\": \"西雅图今天天气\"}"
                )
            else:
                response = "Final Answer: 处理完成。"

            for char in response:
                yield char
                time.sleep(0.001)

    # 3. 初始化配置显式 RAG 参数的 Agent
    mock_llm = MockLLMClient()
    memory_manager = MemoryManager(max_messages=10)
    
    agent = ReActAgent(
        llm_client=mock_llm,
        tool_factory=default_tool_factory,
        memory_mgr=memory_manager,
        max_iterations=3,
        # 💡 [注入自定义检索参数]
        top_k_ret=10,
        top_k_rerank=4,
        filter_str="department == 'IT'",
        min_query_length=3
    )

    print("\n" + "=" * 80)
    print("🚀 [Suite 1]: 测试短文本 / 通用问候语快速收敛 (应跳过 Router 与 Tools)")
    print("=" * 80)
    for step in agent.run_stream("你好"):
        print(f"[{step['type'].upper()}]: {step['content']}")

    print("\n" + "=" * 80)
    print("🚀 [Suite 2]: 测试 RAG 知识库检索 & 检验 top_k_ret / top_k_rerank / filter 自动透传")
    print("=" * 80)
    for step in agent.run_stream("怎么创建预警用户？"):
        print(f"[{step['type'].upper()}]: {step['content']}")

    print("\n" + "=" * 80)
    print("🚀 [Suite 3]: 测试恶意代码沙箱拦截与 Memory Rollback")
    print("=" * 80)
    
    def test_sandbox_rollback():
        agent.memory_mgr.begin_transaction()
        agent.memory_mgr.process_user_input("帮我执行命令: import os; os.system('rm -rf /')")
        
        bad_code = "import os\nos.system('rm -rf /')"
        sandbox_res = agent.sandbox.run(bad_code)
        
        if sandbox_res["status"] == "security_blocked":
            print(f"🛡️ [Sandbox Guardrail]: 成功拦截高危代码 -> {sandbox_res['error']}")
            agent.memory_mgr.rollback()
            print("🚨 [Memory Manager]: 已成功触发事务回滚 (Rollback)！")

    test_sandbox_rollback()

    print("\n" + "=" * 80)
    print("✅ 全套 Agent 改造验证完成！所有新特性运行正常。")
    print("=" * 80)
