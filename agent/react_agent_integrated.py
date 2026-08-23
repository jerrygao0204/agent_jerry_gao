# agent/react_agent_integrated.py
import os
import sys
import json
import logging
from typing import Generator, Dict, Any, Optional

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from agent.sandbox import SandboxExecutor
from memory.memory_manager import MemoryManager  # 👈 直接使用你现有的 MemoryManager
from factory.tool_factory import tool_factory
from generator.llm_client import FineBILLMClient

logger = logging.getLogger("IntegratedReActAgent")


class IntegratedReActAgent:
    """接入 SandboxExecutor 与已有 MemoryManager 事务控制的强约束 ReAct Agent"""

    def __init__(
        self, 
        model_name: str = "Qwen/Qwen3-32B", 
        top_k_ret: int = 5, 
        top_k_rerank: int = 3, 
        filter_str: str = "",
        max_steps: int = 5, 
        sandbox_timeout: int = 2,
        memory_mgr: Optional[MemoryManager] = None  # 🛠️ 1. 支持外部传入单例 MemoryManager
    ):
        self.max_steps = max_steps
        self.sandbox = SandboxExecutor(timeout=sandbox_timeout)
        
        # 🛠️ 复用传入的 memory_mgr 实例，避免每次调用重新初始化
        self.memory_mgr = memory_mgr if memory_mgr is not None else MemoryManager(max_messages=20)
        self.llm_client = FineBILLMClient(model_short_name=model_name)
        
        self.top_k_ret = top_k_ret
        self.top_k_rerank = top_k_rerank
        self.filter_str = filter_str

    def _generate_llm_final_answer(self, query: str, obs_result: Any) -> str:
        """解析 Observation 数据，结合现有的 ShortTermMemory 与 EntityMemory 生成解答"""
        # 1. 格式化知识库检索到的 context
        context_str = ""
        if isinstance(obs_result, list):
            for idx, item in enumerate(obs_result, 1):
                if isinstance(item, dict):
                    content = item.get("content") or item.get("base_content") or str(item)
                    context_str += f"【资料 {idx}】:\n{content}\n\n"
                else:
                    context_str += f"【资料 {idx}】:\n{str(item)}\n\n"
        else:
            context_str = str(obs_result)

        # 🛠️ 2. 从已有的 MemoryManager 中提取历史对话和实体
        context_data = self.memory_mgr.get_context_for_llm()
        messages_history = context_data.get("messages", [])  # 来自 ShortTermMemory
        entities = context_data.get("entities", {})          # 来自 EntityMemory

        # 🛠️ 3. 组装记忆 Prompt（将实体和近几轮历史对话拼接给 LLM）
        prompt_parts = []
        if entities:
            prompt_parts.append(f"【已知业务实体/用户属性】:\n{json.dumps(entities, ensure_ascii=False)}")

        # 格式化近期的多轮对话记录（排除刚添加的最后一条重复 query）
        if len(messages_history) > 1:
            recent_chats = messages_history[:-1]  # 取之前的对话
            formatted_history = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_chats])
            prompt_parts.append(f"【前文对话历史】:\n{formatted_history}")

        prompt_parts.append(f"【当前用户提问】:\n{query}")
        
        # 最终组合出来的 Full Query
        full_query = "\n\n".join(prompt_parts)

        try:
            full_response = ""
            for chunk in self.llm_client.stream_generate(query=full_query, context=context_str):
                full_response += chunk

            # 过滤推理链标签
            if "</think>" in full_response:
                final_text = full_response.split("</think>")[-1].strip()
            else:
                final_text = full_response.strip()

            return final_text if final_text else f"从知识库检索成功，原始参考内容如下：\n{context_str}"

        except Exception as e:
            logger.error(f"调用 LLM 汇总生成 Final Answer 失败: {e}", exc_info=True)
            return f"从知识库检索成功，原始参考内容如下：\n{context_str}"

        
    def run_stream(self, query: str) -> Generator[Dict[str, Any], None, None]:
        """流式运行 Agent 推理链（结合你写的 MemoryManager 事务管理）"""

        # ----------------------------------------------------
        # 📸 Step A: 开启 Memory 事务快照，并追加 User 输入 & 抽取实体
        # ----------------------------------------------------
        self.memory_mgr.begin_transaction()
        self.memory_mgr.process_user_input(query)  # 会自动触发 ShortTerm 保存和 Entity 规则提取

        yield {
            "stage": "header",
            "content": f"🚀 **开始全链路 Agent 运行**: *\"{query}\"*\n",
        }

        # 从 EntityMemory 展示提取到的实体
        context_data = self.memory_mgr.get_context_for_llm()
        entities = context_data.get("entities", {})
        if entities:
            yield {
                "stage": "entity_memory",
                "content": f"🧠 **[EntityMemory]** 当前匹配/已存实体: `{json.dumps(entities, ensure_ascii=False)}`\n",
            }

        try:
            # ----------------------------------------------------
            # 🧠 Step B: ReAct 推理循环
            # ----------------------------------------------------
            for step_idx in range(1, self.max_steps + 1):
                thought_msg = f"🧠 **[Thought - Step {step_idx}]**: 分析用户问题与上下文实体，决定调用工具..."
                yield {"stage": "thought", "content": thought_msg}

                # 判定 Action
                if "计算" in query or "执行" in query or "代码" in query:
                    action_payload = {
                        "tool_name": "python_sandbox_executor",
                        "code": "import math\na = 10\nb = 20\nFINAL_RESULT = math.sqrt(a + b)",
                    }
                else:
                    action_payload = {
                        "tool_name": "search_knowledge_base",
                        "args": {
                            "query": query, 
                            "top_k": self.top_k_ret,
                            "top_k_rerank": self.top_k_rerank,
                            "filter": self.filter_str
                        },
                    }

                action_msg = f"🔧 **[Action]**: 执行工具 `{action_payload['tool_name']}`"
                yield {"stage": "action", "content": action_msg}

                # ----------------------------------------------------
                # 🛡️ Step C: 沙箱检测与执行
                # ----------------------------------------------------
                if action_payload["tool_name"] == "python_sandbox_executor":
                    sandbox_res = self.sandbox.run(action_payload["code"])

                    if sandbox_res["status"] == "security_blocked":
                        obs_msg = f"🛡️ **[Observation - 沙箱拦截]**: {sandbox_res['error']}"
                        yield {"stage": "observation", "content": obs_msg}
                        raise ValueError(f"安全沙箱检测到高危代码: {sandbox_res['error']}")
                    else:
                        obs_result = sandbox_res['result']
                        obs_msg = f"👁️ **[Observation]**: {obs_result}"
                        yield {"stage": "observation", "content": obs_msg}
                        dynamic_answer = f"计算执行成功，最终结果为: `{obs_result}`"
                else:
                    tool_inst = tool_factory.get_tool(action_payload["tool_name"])
                    obs_result = tool_inst.run(**action_payload["args"]) if tool_inst else [{'content': '无'}]

                    obs_msg = f"👁️ **[Observation]** 工具返回:\n```text\n{obs_result}\n```\n"
                    yield {"stage": "observation", "content": obs_msg}

                    dynamic_answer = self._generate_llm_final_answer(query, obs_result)

                # ----------------------------------------------------
                # 💬 Step D: 记录 Assistant 输出并提交事务
                # ----------------------------------------------------
                final_answer = f"💬 **[Final Answer]**:\n{dynamic_answer}"
                yield {"stage": "final_answer", "content": final_answer}

                # 记录到 ShortTermMemory 中
                self.memory_mgr.process_assistant_output(dynamic_answer)

                # 提交 Memory 事务
                self.memory_mgr.commit()
                yield {
                    "stage": "commit",
                    "content": "\n✅ **[Memory Transaction]** 事务成功提交 (Commit)，记忆与实体已持久化！",
                }
                return

        except Exception as e:
            # ----------------------------------------------------
            # 🚨 Step E: 出现异常触发你写好的 rollback()
            # ----------------------------------------------------
            logger.error(f"ReAct 过程发生错误: {e}")
            self.memory_mgr.rollback()  # 恢复快照
            yield {
                "stage": "rollback",
                "content": f"\n🚨 **[Memory Transaction]** 捕获异常: *{str(e)}*\n已触发 **Memory rollback()** 恢复状态！",
            }

# ----------------------------------------------------
# 🧪 本地运行演示测试
# ----------------------------------------------------
if __name__ == "__main__":
    agent = IntegratedReActAgent()

    print("==================================================")
    print("🧪 测试场景 1: 正常查询流程 (正常 Commit)")
    print("==================================================")
    for step in agent.run_stream("怎么创建预警用户？"):
        print(step["content"])

    print("\n==================================================")
    print("🧪 测试场景 2: 注入恶意代码 triggering 沙箱拦截 & Rollback")
    print("==================================================")
    agent_bad = IntegratedReActAgent()

    def bad_run_stream():
        agent_bad.memory_mgr.begin_transaction()
        agent_bad.memory_mgr.process_user_input("帮我执行计算: import os")
        res = agent_bad.sandbox.run("import os; os.system('rm -rf /')")
        if res["status"] == "security_blocked":
            print(f"🛡️ 静态 AST 安全沙箱拦截报告: {res['error']}")
            agent_bad.memory_mgr.rollback()
            print("🚨 MemoryManager 事务已成功回滚！")

    bad_run_stream()
