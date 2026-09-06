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
from agent.tool_transport import ToolDispatcher

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
        self.tool_dispatcher = ToolDispatcher(
            tool_factory=self.tool_factory,
            user_role=self.user_role,
            kwargs_preprocessor=self._prepare_tool_kwargs,
            result_postprocessor=self._postprocess_tool_result,
        )

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
                "你是一个 CodeAct 智能体：通过编写并执行 Python 代码来完成任务。\n\n"
                "你可以在代码里直接调用以下函数（就像调用普通 Python 函数一样，不需要 import）：\n"
                "{tools_description}\n\n"
                "### 执行规范与纠错指令：\n"
                "1. **输出结构**：严格先输出 `<reflection>自检反思</reflection>`，再输出 ` ```python ` 代码块。\n"
                "2. **关键变量**：必须将关键计算结果或工具返回赋值给 `FINAL_RESULT` 变量，并使用 `print()` 打印关键中间过程。\n"
                "3. **自动纠错 (Self-Correction)**：若上一轮 Observation 包含 `[执行异常]` 或 `runtime_error`，你**必须**在 `<reflection>` 中分析报错原因（如函数名拼错、参数缺失等），并在本轮修正代码。**严禁连续生成完全相同的无效代码！**\n"
                "4. **终止条件**：如果通过之前的 Observation 已经获得完整答案，直接输出 `Final Answer: ...`，不要再生成代码块。\n\n"
                "示例：\n"
                "<reflection>上一轮提示 search_knowledge 未定义，检查工具列表发现正确名称为 search_knowledge_base，现予以更正。</reflection>\n"
                "```python\n"
                "res = search_knowledge_base(query=\"怎么创建预警用户\")\n"
                "print(res)\n"
                "FINAL_RESULT = res\n"
                "```\n\n"
                "开始！\n\n"
                "Question: {input}\n"
                "{agent_scratchpad}"
            )
            # self.system_prompt_template = (
            #     "你是一个 CodeAct 智能体：通过编写并执行 Python 代码来完成任务，"
            #     "而不是使用固定格式的 Action/Action Input。\n\n"
            #     "你可以在代码里直接调用以下函数（就像调用普通 Python 函数一样，不需要 import）：\n"
            #     "{tools_description}\n\n"
            #     "请严格按照以下格式输出：\n"
            #     "1. 先用 <reflection></reflection> 标签做简短自检：这一步要做什么？"
            #     "代码里是否包含 print() 打印关键中间结果？逻辑是否已经能回答问题？\n"
            #     "2. 然后输出一个 ```python 代码块，这段代码会被安全沙箱执行，"
            #     "执行时的 print() 输出和最终的 FINAL_RESULT 变量会作为 Observation 返回给你。\n"
            #     "3. 如果已经得到最终答案，不要再输出代码块，直接输出 Final Answer。\n\n"
            #     "示例：\n"
            #     "<reflection>需要检索知识库获取创建预警用户的步骤，并打印结果方便确认。</reflection>\n"
            #     "```python\n"
            #     "res = search_knowledge_base(query=\"怎么创建预警用户\")\n"
            #     "print(res)\n"
            #     "FINAL_RESULT = res\n"
            #     "```\n\n"
            #     "开始！\n\n"
            #     "Question: {input}\n"
            #     "{agent_scratchpad}"
            # )

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

    def _postprocess_tool_result(self, tool_name: str, result: Any) -> Any:
        """工具结果的统一后处理：RAG 检索相关度过低时替换为提示文案（原逻辑迁移，语义不变）"""
        RERANK_THRESHOLD = 0.4
        if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
            score = result[0].get("rerank_score") or result[0].get("score", 1.0)
            if 0.1 <= score < RERANK_THRESHOLD:
                return "【系统提示】: 本地知识库检索相关度得分过低，无匹配结果。"
        return result

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

        if not packages_summary:
            logger.warning("⚠️ [WARNING] packages_summary 为空！")
            return []
        
        # 1. 转化为结构化 Markdown 描述，消除纯 JSON 的符号噪音
        formatted_summary = self._format_packages_summary(packages_summary)

        # 2. 构造通用 Prompt
        package_prompt = self.package_router_prompt_template.format(
            packages_summary=formatted_summary,
            input=query
        )

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

    def _extract_reflection_and_code(self, response_text: str) -> Tuple[str, Optional[str]]:
        """从 LLM 输出中提取 <reflection> 内容和 ```python 代码块"""
        reflection_match = re.search(r"<reflection>(.*?)</reflection>", response_text, re.DOTALL)
        reflection = reflection_match.group(1).strip() if reflection_match else ""

        code_match = re.search(r"```python\s*(.*?)```", response_text, re.DOTALL)
        code = code_match.group(1).strip() if code_match else None
        return reflection, code

    def _build_observation(self, sandbox_res: Dict[str, Any]) -> str:
        """把沙箱执行结果（stdout/result/error/warnings）组装成喂给下一轮 LLM 的 Observation 文本"""
        parts = []
        if sandbox_res.get("stdout"):
            parts.append(f"[stdout]\n{sandbox_res['stdout'].strip()}")
        if sandbox_res["status"] == "success":
            parts.append(f"[执行结果 FINAL_RESULT] {sandbox_res.get('result')}")
        else:
            parts.append(f"[执行异常] status={sandbox_res['status']} error={sandbox_res.get('error')}")
        if sandbox_res.get("warnings"):
            parts.append(f"[提示] {'; '.join(sandbox_res['warnings'])}")
        return "\n".join(parts) if parts else "(无输出)"

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
        available_tool_names: List[str] = []

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
                    
                    # tool_names = ", ".join(tool_names_list)
                    tools_description = "\n".join(descriptions)
                    available_tool_names = tool_names_list  # 💡 修复：正确填充沙箱白名单
                    pkg_names = tool_names_list

                    pkg_names = tool_names_list  # 给报错打印提示用
                else:
                    tools_description = str(tools_schema)
                    # tool_names = "已加载工具"
                    available_tool_names = []  # 若为纯字符串描述，由沙箱自行处理或默认放行

            else:
                yield self._yield_step("thought", "正在分析用户意图，匹配业务领域 (Domain)...")
                selected_domains = self._route_domains(query)
                
                yield self._yield_step("thought", f"锁定业务领域: `{selected_domains}`，正在筛选工具包 (Package)...")
                selected_packages = self._route_packages(query, selected_domains)
                
                pkg_names = [pkg for _, pkg in selected_packages]
                yield self._yield_step("thought", f"锁定工具包: `{pkg_names}`，装载精准工具 Schema。")

                tool_names, tools_description = self.tool_factory.get_tools_metadata_by_packages(selected_packages, user_role=self.user_role)
                available_tool_names = [t.strip() for t in tool_names.split(",") if t.strip()]
           
            scratchpad = ""

            # 3. 核心 CodeAct 循环：LLM 生成代码 -> AST 预检 -> 沙箱执行 -> Observation 喂回
            for iteration in range(1, self.max_iterations + 1):
                logger.info(f"[CodeAct Step] 开始第 {iteration}/{self.max_iterations} 轮推理/执行...")
                prompt = self.system_prompt_template.format(
                    tools_description=tools_description,
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

                reflection, code = self._extract_reflection_and_code(clean_response)
                final_answer_match = re.search(r"Final Answer:\s*(.*)", clean_response, re.DOTALL)

                if final_answer_match and not code:
                    final_ans = final_answer_match.group(1).strip()
                    if extracted_think:
                        yield self._yield_step("thought", extracted_think)
                    yield self._yield_step("final_answer", final_ans)

                    self.memory_mgr.process_assistant_output(final_ans)
                    self.memory_mgr.commit()
                    return

                if code:
                    yield self._yield_step("thought", f"【第 {iteration}/{self.max_iterations} 步】{reflection or extracted_think or '准备执行代码...'}")
                    yield self._yield_step("code", f"```python\n{code}\n```")

                    sandbox_res = self.sandbox.run(
                        code_str=code,
                        tool_names=available_tool_names,
                        tool_dispatcher=self.tool_dispatcher,
                    )
                    if sandbox_res["status"] == "security_blocked":
                        raise ValueError(f"安全沙箱检测到高危指令: {sandbox_res['error']}")

                    observation = self._build_observation(sandbox_res)
                    yield self._yield_step("observation", observation)

                    # 💡 [收敛机制 1]: 纯变量计算自动收敛
                    is_pure_calc = sandbox_res["status"] == "success" and not any(t in code for t in available_tool_names)
                    if is_pure_calc and sandbox_res.get("result") is not None:
                        final_ans = f"计算完成，执行结果为: {sandbox_res['result']}"
                        yield self._yield_step("final_answer", final_ans)
                        self.memory_mgr.process_assistant_output(final_ans)
                        self.memory_mgr.commit()
                        return

                    # 💡 [收敛机制 2]: 工具已成功返回结果且无异常，若 LLM 代码未包含后续操作逻辑，标记可收敛上下文
                    if sandbox_res["status"] == "success" and sandbox_res.get("result") is not None:
                        logger.info(f"[CodeAct] 工具执行成功并获取结果: {sandbox_res['result']}")

                    scratchpad += (
                        f"<reflection>{reflection}</reflection>\n"
                        f"```python\n{code}\n```\n"
                        f"Observation: {observation}\n\n"
                    )

                    # # 💡 [自动终止判断]: 若没有工具调用且非报错执行（例如纯变量计算），直接收敛输出，无需死循环
                    # is_pure_calc = sandbox_res["status"] == "success" and not any(t in code for t in available_tool_names)
                    # if is_pure_calc and sandbox_res.get("result") is not None:
                    #     final_ans = f"计算完成，执行结果为: {sandbox_res['result']}"
                    #     yield self._yield_step("final_answer", final_ans)
                    #     self.memory_mgr.process_assistant_output(final_ans)
                    #     self.memory_mgr.commit()
                    #     return

                    # # 记录轨迹供下一轮 Self-Correction 纠错


                    # scratchpad += (
                    #     f"<reflection>{reflection}</reflection>\n"
                    #     f"```python\n{code}\n```\n"
                    #     f"Observation: {observation}\n\n"
                    # )
                else:
                    final_ans = clean_response.strip()
                    yield self._yield_step("final_answer", final_ans)
                    self.memory_mgr.process_assistant_output(final_ans)
                    self.memory_mgr.commit()
                    return

        except Exception as e:
            logger.error(f"CodeAct 运行捕获异常: {e}")
            self.memory_mgr.rollback()
            yield self._yield_step("rollback", f"🚨 运行异常已触发 Memory Rollback: {str(e)}")


# =============================================================================
# 单元测试桩
# =============================================================================
if __name__ == "__main__":
    from typing import Generator, List, Dict, Any, Optional, Tuple

    class DynamicCodeActMockLLM:
        def __init__(self):
            self.codeact_step = 0

        def reset_step(self):
            self.codeact_step = 0

        def stream_generate(self, query: str = None, messages: List[Dict[str, str]] = None, context: str = "") -> Generator[str, None, None]:
            prompt_str = query if query else (messages[-1]["content"] if messages else "")

            # 1. 路由阶段拦截 (Domain Router & Package Router)
            if "你是一个意图路由专家" in prompt_str:
                yield '["rag_domain"]'
                return
            if "通用 Agent 工具包分类路由专家" in prompt_str:
                yield '["rag_pkg"]'
                return

            # 2. 通用问候直连拦截
            if "你好" in prompt_str and "Question:" not in prompt_str:
                yield "你好！我是 CodeAct 智能助手，请问有什么可以帮您？"
                return

            # 3. 场景 4: 变量计算 Mock
            if "a = 10" in prompt_str or "计算两个数的和" in prompt_str or "缺失变量" in prompt_str:
                yield (
                    "<reflection>计算两个数的和。</reflection>\n"
                    "```python\n"
                    "a = 10\n"
                    "b = 20\n"
                    "FINAL_RESULT = a + b\n"
                    "```"
                )
                return

            # 4. 场景 3: 代码报错自纠错 Mock (同步兼容 "触发自纠错" 与 "报错自纠错测试")
            if "触发自纠错" in prompt_str or "报错自纠错测试" in prompt_str:
                self.codeact_step += 1
                
                # 第 1 步: 故意写错函数名，触发 NameError 异常
                if self.codeact_step == 1:
                    yield (
                        "<reflection>尝试调用知识库检索，故意写错函数名以测试自纠错机制。</reflection>\n"
                        "```python\n"
                        "res = search_knowledge_error_name(query='报错自纠错测试')\n"
                        "```"
                    )
                    return
                # 第 2 步: 捕获异常后纠错，输出正确的函数名
                elif self.codeact_step == 2:
                    yield (
                        "<reflection>上一轮函数名写错了，观察到报错 NameError，现修正为 search_knowledge_base。</reflection>\n"
                        "```python\n"
                        "res = search_knowledge_base(query='报错自纠错测试')\n"
                        "print(res)\n"
                        "FINAL_RESULT = res\n"
                        "```"
                    )
                    return
                # 第 3 步: 获取正确结果，吐出 Final Answer 闭环
                else:
                    yield "Final Answer: 已成功通过纠错后的接口检索到内容。"
                    return

            # 5. 场景 2: CodeAct 多轮工具调用 Mock
            if "Observation:" in prompt_str or "FINAL_RESULT" in prompt_str:
                yield "Final Answer: 根据知识库检索结果，创建预警用户的步骤为：1. 打开系统设置；2. 点击新增预警用户。"
            else:
                yield (
                    "<reflection>需要调用 search_knowledge_base 函数检索创建预警用户的步骤，并打印结果。</reflection>\n"
                    "```python\n"
                    "res = search_knowledge_base(query='怎么创建预警用户')\n"
                    "print(res)\n"
                    "FINAL_RESULT = res\n"
                    "```"
                )
    # 2. Mock 工具工厂 (修复 BaseTool 抽象类实例化问题)
    class MockTool:
        """使用鸭子类型 Mock 工具，实现 name, description 和 run 方法，无需继承 BaseTool"""
        def __init__(self, name: str = "search_knowledge_base", description: str = "检索知识库内容"):
            self.name = name
            self.description = description

        def run(self, query: str = "", **kwargs) -> Any:
            return f"【知识库文档】: 步骤1. 打开系统设置；步骤2. 点击新增预警用户。(query={query})"

        def _run(self, query: str = "", **kwargs) -> Any:
            return self.run(query=query, **kwargs)

    class MockToolFactory(HierarchicalToolFactory):
        def __init__(self):
            mock_tool = MockTool()
            self._tools = {"search_knowledge_base": mock_tool}
            self._flat_tools = self._tools  # 补全私有属性防范 fallback 崩溃

        def get_tool(self, name: str, user_role: Optional[str] = None) -> Optional[Any]:
            return self._tools.get(name)

        def get_domains_summary(self) -> List[Dict[str, Any]]:
            return [{"domain": "rag_domain", "description": "知识库问答领域"}]

        def get_packages_summary_by_domains(self, domains: List[str]) -> List[Dict[str, Any]]:
            return [{
                "package": "rag_pkg",
                "domain": "rag_domain",
                "description": "知识库检索工具包",
                "tools": [{"name": "search_knowledge_base", "description": "检索知识库内容"}]
            }]

        def get_tools_metadata_by_packages(self, packages: List[Tuple[str, str]], user_role: Optional[str] = None) -> Tuple[str, str]:
            return "search_knowledge_base", "- `search_knowledge_base(query: str)`: 检索知识库内容。"

        def execute_tool(self, tool_name: str, kwargs: Dict[str, Any], user_role: Optional[str] = None) -> Any:
            tool = self.get_tool(tool_name)
            if tool:
                return tool.run(**kwargs)
            return f"Unknown tool: {tool_name}"

    mock_llm = DynamicCodeActMockLLM()
    mock_factory = MockToolFactory()

    agent = ReActAgent(
        llm_client=mock_llm,
        tool_factory=mock_factory,
        max_iterations=3,
        sandbox_timeout=3
    )

    print("\n=================== 场景 1: 短文本/通用问候拦截测试 ===================")
    for step in agent.run_stream("你好"):
        print(f"[{step['type'].upper()}]: {step['content']}")

    print("\n=================== 场景 2: CodeAct 多轮工具调用 + Final Answer 闭环 ===================")
    for step in agent.run_stream("请问怎么创建预警用户？"):
        print(f"[{step['type'].upper()}] ({step['stage']}):\n{step['content']}\n" + "-"*50)

    print("\n=================== 场景 3: 代码报错 (Runtime Error) 自动纠错测试 ===================")
    for step in agent.run_stream("触发自纠错"):
        print(f"[{step['type'].upper()}]:\n{step['content']}\n" + "-"*50)

    print("\n=================== 场景 4: 变量计算自动终止测试 ===================")
    for step in agent.run_stream("缺失变量"):
        print(f"[{step['type'].upper()}]:\n{step['content']}\n" + "-"*50)