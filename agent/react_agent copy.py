# # agent/react_agent.py
# import re
# import json
# import logging
# from typing import Dict, Any, Generator, Optional, List
# from factory.tool_factory import HierarchicalToolFactory, tool_factory as default_tool_factory

# logger = logging.getLogger("ReActAgent")


# class ReActAgent:
#     """支持两阶段路由与 PromptHub 动态引用的 ReAct Agent"""

#     def __init__(
#         self, 
#         llm_client: Any, 
#         tool_factory: Optional[HierarchicalToolFactory] = None,
#         prompt_hub: Optional[Any] = None, 
#         system_prompt_template: Optional[str] = None, 
#         router_prompt_template: Optional[str] = None,
#         max_iterations: int = 3
#     ):
#         self.llm_client = llm_client
#         self.tool_factory = tool_factory or default_tool_factory
#         self.prompt_hub = prompt_hub
#         self.max_iterations = max_iterations

#         # 🌟 1. 动态加载主 ReAct System Prompt：优先传入參數 -> 其次 prompt_hub -> 兜底默认值
#         if system_prompt_template:
#             self.system_prompt_template = system_prompt_template
#         elif prompt_hub and hasattr(prompt_hub, "get_prompt"):
#             prompt_obj = prompt_hub.get_prompt("agent_react_prompt")
#             self.system_prompt_template = getattr(prompt_obj, "content", str(prompt_obj))
#         else:
#             self.system_prompt_template = (
#                 "尽可能回答以下问题。你可以使用以下工具：\n{tools_description}\n\n"
#                 "请严格按照以下格式进行思考和调用工具：\n"
#                 "Question: 你需要回答的输入问题\n"
#                 "Thought: 你应该总是思考下一步要做什么\n"
#                 "Action: 要调用的工具名称（必须是 [{tool_names}] 中的一个）\n"
#                 "Action Input: 传递给工具的 JSON 格式参数\n"
#                 "Observation: 工具返回的结果\n"
#                 "... (这个 Thought/Action/Action Input/Observation 过程可以重复 N 次)\n"
#                 "Thought: 我现在知道最终答案了\n"
#                 "Final Answer: 针对原始输入问题的最终回答\n\n"
#                 "开始！\n\n"
#                 "Question: {input}\n"
#                 "Thought: {agent_scratchpad}"
#             )

#         # 🌟 2. 动态加载 Router Prompt (Level 1 领域路由用)
#         if router_prompt_template:
#             self.router_prompt_template = router_prompt_template
#         elif prompt_hub and hasattr(prompt_hub, "get_prompt"):
#             router_obj = prompt_hub.get_prompt("agent_router_prompt")
#             self.router_prompt_template = getattr(router_obj, "content", str(router_obj))
#         else:
#             self.router_prompt_template = (
#                 "你是一个意图路由专家。请分析用户问题，从给定的领域列表中选择最相关的 1~2 个领域。\n"
#                 "可用领域清单:\n{domains_summary}\n\n"
#                 "用户问题: {input}\n\n"
#                 "请严格仅返回 JSON 数组格式的领域代码，例如: [\"rag_knowledge\"]，不要输出任何额外内容。"
#             )

#     def _get_default_router_template(self) -> str:
#         return (
#             "你是一个意图路由专家。请分析用户问题，从给定的领域列表中选择最相关的 1~2 个领域。\n"
#             "可用领域清单:\n{domains_summary}\n\n"
#             "用户问题: {input}\n\n"
#             "请严格仅返回 JSON 数组格式的领域代码，例如: [\"rag_knowledge\"]，不要输出任何额外内容。"
#         )

#     def _route_domains(self, query: str) -> List[str]:
#         """第一级路由：选择目标 Domain"""
#         domains_summary = self.tool_factory.get_domains_summary()
#         if not domains_summary:
#             return []

#         router_prompt = self.router_prompt_template.format(
#             domains_summary=json.dumps(domains_summary, ensure_ascii=False, indent=2),
#             input=query
#         )

#         try:
#             response = ""
#             for chunk in self.llm_client.stream_generate(query=router_prompt, context=""):
#                 response += chunk
            
#             clean_res = re.sub(r"^```(?:json)?|```$", "", response.strip(), flags=re.IGNORECASE).strip()
#             matched_domains = json.loads(clean_res)
#             if isinstance(matched_domains, list) and len(matched_domains) > 0:
#                 logger.info(f"🎯 [Level 1 Router] 成功命中领域: {matched_domains}")
#                 return matched_domains
#         except Exception as e:
#             logger.warning(f"⚠️ [Level 1 Router] 路由失败 ({str(e)})，降级全量。")

#         return [item["domain"] for item in domains_summary]

#     def _parse_action_input(self, raw_str: str) -> Dict[str, Any]:
#         cleaned = raw_str.strip()
#         cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.IGNORECASE).strip()
#         if cleaned.startswith("{") and cleaned.endswith("}"):
#             try:
#                 return json.loads(cleaned)
#             except json.JSONDecodeError:
#                 pass
#         return {"query": cleaned}

#     def run_stream(self, query: str) -> Generator[Dict[str, Any], None, None]:
#         # Step 1: 路由过滤第一级领域
#         yield {"type": "thought", "content": "正在分析意图，匹配对应业务领域..."}
#         selected_domains = self._route_domains(query)
#         yield {"type": "thought", "content": f"目标领域已锁定: `{selected_domains}`，已加载相关工具。"}

#         # Step 2: 获取局部领域的工具元数据
#         tool_names, tools_description = self.tool_factory.get_tools_metadata_by_domains(selected_domains)
#         scratchpad = ""

#         # Step 3: 基于提取到的 self.system_prompt_template 格式化推理
#         for iteration in range(1, self.max_iterations + 1):
#             logger.info(f"🔄 [ReAct Agent] 第 {iteration}/{self.max_iterations} 轮推理...")

#             # 动态渲染 YAML / 传入的 System Prompt
#             prompt = self.system_prompt_template.format(
#                 tools_description=tools_description,
#                 tool_names=tool_names,
#                 input=query,
#                 agent_scratchpad=scratchpad
#             )

#             full_response = ""
#             for chunk in self.llm_client.stream_generate(query=prompt, context=""):
#                 full_response += chunk

#             clean_response = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
#             think_match = re.search(r"<think>(.*?)</think>", full_response, re.DOTALL)
#             extracted_think = think_match.group(1).strip() if think_match else ""

#             final_answer_match = re.search(r"Final Answer:\s*(.*)", clean_response, re.DOTALL)
#             action_match = re.search(r"Action:\s*([^\n]+)", clean_response)
#             action_input_match = re.search(r"Action Input:\s*(\{.*?\}|```.*?```|[^\n]+)", clean_response, re.DOTALL)

#             if final_answer_match and not action_match:
#                 final_ans = final_answer_match.group(1).strip()
#                 if extracted_think:
#                     yield {"type": "thought", "content": extracted_think}
#                 yield {"type": "final_answer", "content": final_ans}
#                 return

#             if action_match and action_input_match:
#                 tool_name = action_match.group(1).strip()
#                 raw_input = action_input_match.group(1).strip()
                
#                 thought_content = clean_response.split("Action:")[0].replace("Thought:", "").strip()
#                 if extracted_think and not thought_content:
#                     thought_content = extracted_think

#                 yield {"type": "thought", "content": thought_content or "正在准备调用工具..."}
#                 yield {"type": "action", "content": f"调用工具: {tool_name} | 参数: {raw_input}"}

#                 kwargs = self._parse_action_input(raw_input)
#                 tool_obj = self.tool_factory.get_tool(tool_name)
                
#                 if tool_obj:
#                     try:
#                         if hasattr(tool_obj, "run"):
#                             observation = tool_obj.run(**kwargs)
#                         elif hasattr(tool_obj, "execute"):
#                             observation = tool_obj.execute(**kwargs)
#                     except Exception as e:
#                         logger.error(f"❌ 工具 [{tool_name}] 执行异常: {str(e)}", exc_info=True)
#                         observation = f"工具执行报错: {str(e)}"
#                 else:
#                     observation = f"错误: 在目标领域 {selected_domains} 中找不到工具 [{tool_name}]，可选工具为: [{tool_names}]"

#                 yield {"type": "observation", "content": str(observation)}
#                 scratchpad += f"{thought_content}\nAction: {tool_name}\nAction Input: {raw_input}\nObservation: {observation}\nThought: "
#             else:
#                 yield {"type": "thought", "content": extracted_think or "大模型选择直接回答。"}
#                 yield {"type": "final_answer", "content": clean_response.replace("Thought:", "").strip()}
#                 return

#         yield {"type": "final_answer", "content": "⚠️ 已达到最大推理轮数限制。"}

# agent/react_agent.py
import re
import json
import logging
from typing import Dict, Any, Generator, Optional, List
from factory.tool_factory import HierarchicalToolFactory, tool_factory as default_tool_factory

logger = logging.getLogger("ReActAgent")


class ReActAgent:
    """两阶段分级路由与动态 Prompt 绑定的 ReAct Agent (含详细调试打印)"""

    def __init__(
        self, 
        llm_client: Any, 
        tool_factory: Optional[HierarchicalToolFactory] = None,
        prompt_hub: Optional[Any] = None, 
        system_prompt_template: Optional[str] = None,
        router_prompt_template: Optional[str] = None,
        max_iterations: int = 3
    ):
        self.llm_client = llm_client
        self.tool_factory = tool_factory or default_tool_factory
        self.prompt_hub = prompt_hub
        self.max_iterations = max_iterations

        # 🌟 1. 加载 agent_react_prompt
        if system_prompt_template:
            self.system_prompt_template = system_prompt_template
        elif prompt_hub and hasattr(prompt_hub, "get_prompt"):
            prompt_obj = prompt_hub.get_prompt("agent_react_prompt")
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

        # 🌟 2. 加载 agent_router_prompt
        if router_prompt_template:
            self.router_prompt_template = router_prompt_template
        elif prompt_hub and hasattr(prompt_hub, "get_prompt"):
            router_obj = prompt_hub.get_prompt("agent_router_prompt")
            self.router_prompt_template = getattr(router_obj, "content", str(router_obj))
        else:
            self.router_prompt_template = (
                "你是一个意图路由专家。请分析用户问题，从给定的领域列表中选择最相关的 1~2 个领域。\n"
                "可用领域清单:\n{domains_summary}\n\n"
                "用户问题: {input}\n\n"
                "请严格仅返回 JSON 数组格式的领域代码，例如: [\"rag_knowledge\"]，不要输出任何额外内容。"
            )

    def _route_domains(self, query: str) -> List[str]:
        """第一级路由：从 PromptHub 提取 agent_router_prompt 进行领域匹配"""
        domains_summary = self.tool_factory.get_domains_summary()
        if not domains_summary:
            print("\n⚠️ [Debug Router] 当前没有任何注册的 Domain，降级处理。")
            return []

        # 渲染动态读取到的 agent_router_prompt
        router_prompt = self.router_prompt_template.format(
            domains_summary=json.dumps(domains_summary, ensure_ascii=False, indent=2),
            input=query
        )

        print("\n" + "="*80)
        print("🔹 [Stage 1: 意图路由与领域筛选]")
        print("="*80)
        print("📥 【输入到 Router LLM 的完整 Prompt】:")
        print(router_prompt)
        print("-" * 50)

        try:
            response = ""
            for chunk in self.llm_client.stream_generate(query=router_prompt, context=""):
                response += chunk
            
            print(f"📤 【Router LLM 原始输出】:\n{response}")
            
            # 🌟 1. 过滤掉 <think>...</think> 思考链内容
            clean_res = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL).strip()
            # 🌟 2. 剥离 ```json 代码块外壳
            clean_res = re.sub(r"^```(?:json)?|```$", "", clean_res, flags=re.IGNORECASE).strip()
            
            # 🌟 3. 核心修复：强行用正则抓取文本中符合 [...] 格式的 JSON 数组部分
            json_match = re.search(r"\[.*?\]", clean_res, re.DOTALL)
            if json_match:
                matched_domains = json.loads(json_match.group(0))
            else:
                # 兜底直接尝试解析
                matched_domains = json.loads(clean_res)
            if isinstance(matched_domains, list) and len(matched_domains) > 0:
                print(f"🎯 【Router 最终判定领域】: {matched_domains}")
                print("="*80 + "\n")
                return matched_domains
        except Exception as e:
            print(f"❌ [Router Error] 路由解析失败 ({str(e)})，降级使用全量领域。")

        fallback_domains = [item["domain"] for item in domains_summary]
        print(f"🎯 【降级全量领域】: {fallback_domains}")
        print("="*80 + "\n")
        return fallback_domains

    def _parse_action_input(self, raw_str: str) -> Dict[str, Any]:
        """解析 Action Input 的 JSON 容错逻辑"""
        cleaned = raw_str.strip()
        cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.IGNORECASE).strip()
        if cleaned.startswith("{") and cleaned.endswith("}"):
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                pass
        return {"query": cleaned}

    def run_stream(self, query: str) -> Generator[Dict[str, Any], None, None]:
        """流式分级 ReAct 执行逻辑"""
        # Step 1: 执行 Level 1 路由过滤
        yield {"type": "thought", "content": "正在分析用户意图与匹配业务领域..."}
        selected_domains = self._route_domains(query)
        yield {"type": "thought", "content": f"锁定领域范围: `{selected_domains}`，按需装载相关工具 Schema。"}

        # Step 2: 仅加载过滤出的工具元数据，打印查看装载的工具 Schema
        tool_names, tools_description = self.tool_factory.get_tools_metadata_by_domains(selected_domains)
        
        print("\n" + "="*80)
        print("🔹 [Stage 2: 动态工具装载 (Tool Schema Loading)]")
        print("="*80)
        print(f"🛠️  【可选工具名称列表 (tool_names)】: [{tool_names}]")
        print("📋 【注入 ReAct 上下文的工具详细说明 (tools_description)】:")
        print(tools_description)
        print("="*80 + "\n")

        scratchpad = ""

        # Step 3: ReAct 思维链循环
        for iteration in range(1, self.max_iterations + 1):
            print("\n" + "="*80)
            print(f"🔹 [Stage 3: ReAct 推理循环 - 第 {iteration}/{self.max_iterations} 轮]")
            print("="*80)

            prompt = self.system_prompt_template.format(
                tools_description=tools_description,
                tool_names=tool_names,
                input=query,
                agent_scratchpad=scratchpad
            )

            print("📥 【输入到 ReAct LLM 的 Prompt (包含历史 Scratchpad)】:")
            print(prompt)
            print("-" * 50)

            full_response = ""
            for chunk in self.llm_client.stream_generate(query=prompt, context=""):
                full_response += chunk

            print(f"📤 【ReAct LLM 原始输出】:\n{full_response}")

            clean_response = re.sub(r"<think>.*?</think>", "", full_response, flags=re.DOTALL).strip()
            think_match = re.search(r"<think>(.*?)</think>", full_response, re.DOTALL)
            extracted_think = think_match.group(1).strip() if think_match else ""
            
            action_match = re.search(r"Action:\s*([^\n]+)", clean_response)
            action_input_match = re.search(r"Action Input:\s*(\{.*?\}|```.*?```|[^\n]+)", clean_response, re.DOTALL)
            final_answer_match = re.search(r"Final Answer:\s*(.*)", clean_response, re.DOTALL)

            # 命中 Final Answer
            if final_answer_match and not action_match:
                final_ans = final_answer_match.group(1).strip()
                print("🎉 【LLM 决策】: 给出最终回答 (Final Answer)，无需继续调用工具。")
                print("="*80 + "\n")
                if extracted_think:
                    yield {"type": "thought", "content": extracted_think}
                yield {"type": "final_answer", "content": final_ans}
                return

            # 命中 Action 调用
            if action_match and action_input_match:
                # ------------------------------------------------------------------
                # 🌟【新增核心修剪逻辑】: 强制拦截并丢弃 Action 之后的任何虚假 Final Answer 文本
                # ------------------------------------------------------------------
                action_pos = clean_response.find("Action:")
                if action_pos != -1:
                    # 强行截断，只保留 Action 之前（包含 Thought）以及 Action/Action Input 本身
                    # 扔掉 Action Input 后面误输出的 "Thought: 我知道答案了 Final Answer: ..."
                    clean_response_truncated = clean_response[:action_pos]
                else:
                    clean_response_truncated = clean_response
                    
                tool_name = action_match.group(1).strip()
                raw_input = action_input_match.group(1).strip()
                
                print(f"👉 【LLM 决策工具选择】: Action -> [{tool_name}]")
                print(f"👉 【LLM 决策工具参数】: Action Input -> {raw_input}")

                thought_content = clean_response.split("Action:")[0].replace("Thought:", "").strip()
                if extracted_think and not thought_content:
                    thought_content = extracted_think

                yield {"type": "thought", "content": thought_content or "正在准备调用工具..."}
                yield {"type": "action", "content": f"调用工具: {tool_name} | 参数: {raw_input}"}

                kwargs = self._parse_action_input(raw_input)
                tool_obj = self.tool_factory.get_tool(tool_name)

                if tool_obj:
                    try:
                        print(f"⚙️ 【开始物理执行工具】: {tool_name} ...")
                        if hasattr(tool_obj, "run"):
                            observation = tool_obj.run(**kwargs)
                        elif hasattr(tool_obj, "execute"):
                            observation = tool_obj.execute(**kwargs)
                        print(f"👁️ 【工具返回结果 (Observation)】:\n{observation}")
                        RERANK_THRESHOLD = 0.4
                        
                        if isinstance(observation, list) and len(observation) > 0:
                            first_item = observation[0]
                            if isinstance(first_item, dict):
                                explicit_rerank_score = first_item.get("rerank_score")
                                fallback_score = first_item.get("score")

                                # 1. 优先使用显式的 rerank_score 做拦截
                                if explicit_rerank_score is not None and isinstance(explicit_rerank_score, (int, float)):
                                    print(f"📊 【Rerank Top-1 Score】: {explicit_rerank_score}")
                                    if explicit_rerank_score < RERANK_THRESHOLD:
                                        print(f"⚠️ 【低于重排序阈值 {RERANK_THRESHOLD}】: 触发知识库缺失兜底。")
                                        observation = (
                                            "【系统提示】: 本地知识库中未检索到相关度达标的文档（最高相关度得分未达到阈值），"
                                            "请直接告知用户知识库缺失此信息，并基于通用知识解答。"
                                        )
                                # 2. 如果只有基础 score 字段，判断其是否为 RRF 得分（RRF 得分数值一般小于 0.1）
                                elif fallback_score is not None and isinstance(fallback_score, (int, float)):
                                    print(f"📊 【检索组件 Top-1 Score】: {fallback_score}")
                                    # 如果是普通相似度得分且小于阈值，拦截；若小于 0.1 认定为 RRF 分值，放行给 ReAct Agent 自行研判
                                    if 0.1 <= fallback_score < RERANK_THRESHOLD:
                                        print(f"⚠️ 【低于相似度阈值 {RERANK_THRESHOLD}】: 触发知识库缺失兜底。")
                                        observation = (
                                            "【系统提示】: 本地知识库中未检索到相关度达标的文档（最高相关度得分未达到阈值），"
                                            "请直接告知用户知识库缺失此信息，并基于通用知识解答。"
                                        )
                                    elif fallback_score < 0.1:
                                        print("ℹ️ 【识别为 RRF 融合得分】: 跳过 0.4 阈值拦截，直接交付 LLM 推理。")

                        if observation is None or observation == "" or (isinstance(observation, list) and len(observation) == 0):
                            observation = "【系统提示】: 本地知识库中未检索到相关文档，请告知用户缺失该信息。"
                    except Exception as e:
                        print(f"❌ 【工具物理执行失败】: {str(e)}")
                        observation = f"工具执行报错: {str(e)}"
                else:
                    observation = f"错误: 在目标领域 {selected_domains} 中找不到工具 [{tool_name}]，可选工具为: [{tool_names}]"
                    print(f"❌ 【工具未找到】: {observation}")

                print("="*80 + "\n")
                yield {"type": "observation", "content": str(observation)}
                scratchpad += f"{thought_content}\nAction: {tool_name}\nAction Input: {raw_input}\nObservation: {observation}\nThought: "
            else:
                print("🎉 【LLM 决策】: 未检测到 Action 标识，直接输出文本内容。")
                print("="*80 + "\n")
                yield {"type": "thought", "content": extracted_think or "大模型选择直接回答。"}
                yield {"type": "final_answer", "content": clean_response.replace("Thought:", "").strip()}
                return

        print("⚠️ 【警告】: 达到最大推理轮数限制。")
        print("="*80 + "\n")
        yield {"type": "final_answer", "content": "⚠️ 已达到最大推理轮数限制，强制收敛输出。"}