# memory_growth/extractor.py
"""
extractor.py - 成长型语境系统的事实抽取管道 (QAChain 适配 + 单文件元数据版)

架构特点：
1. 元数据合并：提取水位线 last_run_at 直接写在 facts.json 的 metadata 字段中，无需独立的 state JSON 文件。
2. 数据格式兼容：完全适配 session_*.json 的消息数据结构。
3. 零云端依赖：集成 QAChainLLMAdapter 适配器，支持思考过程 (<think>) 的过滤。
"""

import sys
from pathlib import Path

# 获取当前文件所在的上一级目录（即：/workspace/hf-conda/RAG/问答机器人/）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ----------------- 以下是原来的 import 语句 -----------------
from generator.qa_chain import QAChain
from path_config import UserMemoryPathConfig
from atomic_io import file_lock_for, atomic_dump_json

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional, Type

from pydantic import BaseModel, Field
from path_config import UserMemoryPathConfig


TIMESTAMP_FMT = "%Y-%m-%d %H:%M:%S"


# ==========================================
# 1. QAChainLLMAdapter 适配器
# ==========================================


class QAChainLLMAdapter:
    """针对 QAChain (llm_client.stream_generate) 的零污染适配器"""

    def __init__(self, qa_chain: Any, clean_think: bool = True):
        self.qa_chain = qa_chain
        self.clean_think = clean_think
        self.tokenizer = getattr(qa_chain, "tokenizer", None) or getattr(
            getattr(qa_chain, "llm", None), "tokenizer", None
        )

    def _clean_think_tags(self, text: str) -> str:
        """多重正则清理思考过程"""
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)
        return cleaned.strip()

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        target_schema: Optional[Type[BaseModel]] = None,
        temperature: float = 0.1,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # 1. 构造 Prompt
        if self.tokenizer and hasattr(self.tokenizer, "apply_chat_template"):
            full_prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            full_prompt = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"

        # 2. 调用 QAChain 流式接口
        response_text = ""
        llm_client = getattr(
            self.qa_chain,
            "llm_client",
            getattr(self.qa_chain, "llm", self.qa_chain),
        )

        for resp in llm_client.stream_generate(query=prompt, context=full_prompt):
            token = resp.get("data", "") if isinstance(resp, dict) else str(resp)
            response_text += token

        # 3. 过滤思考标签
        if self.clean_think:
            response_text = self._clean_think_tags(response_text)

        return response_text.strip()

    def generate_structured(
        self,
        prompt: str,
        system_prompt: str,
        target_schema: Type[BaseModel],
        temperature: float = 0.1,
    ) -> str:
        schema_dict = (
            target_schema.model_json_schema()
            if hasattr(target_schema, "model_json_schema")
            else target_schema.schema()
        )
        schema_json = json.dumps(schema_dict, ensure_ascii=False)

        enhanced_system_prompt = (
            f"{system_prompt}\n\n"
            f"【硬性输出约束】：\n"
            f"1. 严禁输出 <think> 思考过程，直接返回 JSON 内容！\n"
            f"2. 必须且只能输出符合以下 JSON Schema 的合法 JSON，严禁 Markdown 代码块包裹或解释！\n"
            f"JSON Schema: {schema_json}"
        )

        return self.generate(
            prompt=prompt,
            system_prompt=enhanced_system_prompt,
            target_schema=target_schema,
            temperature=temperature,
        )


# ==========================================
# 2. Pydantic 结构化 Schema 定义
# ==========================================


class FactItem(BaseModel):
    category: Literal[
        "explicit_preference",
        "explicit_decision",
        "explicit_constraint",
        "technical_fact",
        "identity_fact"
    ] = Field(description="事实分类")
    fact: str = Field(description="一句话客观事实")
    source_quote: str = Field(description="用户原话片段")
    timestamp: str = Field(description="消息的原始 timestamp")


class ExtractedFactsResponse(BaseModel):
    facts: list[FactItem] = Field(default_factory=list, description="提取出的事实列表")


# ==========================================
# 3. FactExtractor 核心抽取类
# ==========================================

EXTRACTION_SYSTEM_PROMPT  = """你是一个事实抽取器 (Fact Extractor)，负责从用户与助手的对话片段中，
抽取可验证的、客观的事实，用于构建长期语境记忆。

严格规则：
1. 只抽取用户明确说过的、可验证的陈述。禁止抽取推断、情绪状态、模型自己的概括。
   - 允许："用户姓名是高铮"（用户原话表达过）
   - 允许："用户偏好结构化拆解"（用户原话表达过）
   - 禁止："用户处于探索期"（这是推断）
   - 禁止："用户对这个方向很兴奋"（主观情绪标签）

2. 每条事实必须归入以下五类之一：
   - explicit_preference: 用户直接表达的偏好（如“我喜欢用 Python”、“输出要用中文”）
   - explicit_decision: 用户做出的选择或决定（如“决定放弃 Marker，改用 VLM”）
   - explicit_constraint: 用户给出的限制或约束条件（如“不能使用云端 API”、“显存只有 16G”）
   - technical_fact: 客观技术事实（如工具版本、架构设计、部署环境等）
   - identity_fact: 用户的个人身份与基础静态档案（如姓名、职业定位、物理位置、居住地等）

3. 每条事实必须包含 source_quote 字段，即用户原话中支持该事实的片段（可以是概括性的原话片段，不要求逐字，但必须忠实于原意，不得脑补）。

4. 严格按 JSON 格式输出。如果这段对话中没有任何符合条件的事实，返回 {"facts": []}。

JSON Schema:
{
  "facts": [
    {
      "category": "上述 5 种类别之一",
      "fact": "一句话概括的客观事实",
      "source_quote": "用户原话片段"
    }
  ]
}
"""

class FactExtractor:
    def __init__(
        self,
        data_dir: str,
        llm_adapter: QAChainLLMAdapter,
    ):
        self.data_dir = Path(data_dir)
        self.adapter = llm_adapter

    # ---------- 状态与元数据读取 (包含输出文件原子化读取) ----------

    def _load_facts_and_metadata(
        self, output_file: Path
    ) -> tuple[dict[str, Any], list[dict]]:
        """从输出文件一次性读取元数据与已有事实列表"""
        if not output_file.exists():
            return {"last_run_at": None}, []

        try:
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return data.get("metadata", {}), data.get("facts", [])
                elif isinstance(data, list):
                    # 兼容老版本纯数组文件
                    return {"last_run_at": None}, data
        except Exception as e:
            print(f"[WARN] 读取历史 facts.json 失败: {e}")

        return {"last_run_at": None}, []

    # ---------- 数据读取 ----------

    def _load_sessions_index(self, user_id: str) -> dict[str, Any]:
        # 如果 self.data_dir 结尾本身就是 user_id 目录，做一次兼容判断
        if self.data_dir.name == user_id:
            index_path = self.data_dir / "sessions_index.json"
        else:
            index_path = self.data_dir / user_id / "sessions_index.json"

        if not index_path.exists():
            return {}
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_session_messages(
        self, user_id: str, session_id: str
    ) -> list[dict]:
        if self.data_dir.name == user_id:
            session_path = self.data_dir / f"session_{session_id}.json"
        else:
            session_path = self.data_dir / user_id / f"session_{session_id}.json"

        if not session_path.exists():
            return []
        with open(session_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def _parse_ts(ts: str) -> datetime:
        return datetime.strptime(ts, TIMESTAMP_FMT)

    # ---------- 候选筛选 ----------

    def _candidate_sessions(
        self, user_id: str, since: datetime | None
    ) -> list[str]:
        index = self._load_sessions_index(user_id)
        if since is None:
            return list(index.keys())
        candidates = []
        for session_id, meta in index.items():
            updated_at = self._parse_ts(meta["updated_at"])
            if updated_at > since:
                candidates.append(session_id)
        return candidates

    def _new_messages(
        self, user_id: str, session_id: str, since: datetime | None
    ) -> list[dict]:
        messages = self._load_session_messages(user_id, session_id)
        if since is None:
            return messages
        return [m for m in messages if self._parse_ts(m["timestamp"]) > since]

    # ---------- 抽取与结构化解析 ----------

    def _build_transcript(self, messages: list[dict]) -> str:
        lines = []
        for m in messages:
            role = "用户" if m["role"] == "user" else "助手"
            lines.append(f"[{m['timestamp']}] {role}: {m['content']}")
        return "\n".join(lines)

    def _clean_json_string(self, raw_str: str) -> str:
        text = raw_str.strip()
        if "```" in text:
            match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
            if match:
                return match.group(1).strip()
        return text

    def _extract_from_messages(self, messages: list[dict]) -> list[dict]:
        if not messages:
            return []

        transcript = self._build_transcript(messages)

        raw_output = self.adapter.generate_structured(
            prompt=transcript,
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            target_schema=ExtractedFactsResponse,
            temperature=0.1,
        )

        cleaned_output = self._clean_json_string(raw_output)

        try:
            parsed_data = json.loads(cleaned_output)
            if isinstance(parsed_data, dict):
                facts = parsed_data.get("facts", [])
            elif isinstance(parsed_data, list):
                facts = parsed_data
            else:
                facts = []
            return facts
        except json.JSONDecodeError:
            print(
                f"[WARN] 结构化输出解析失败，已跳过该批次。原始文本：\n{cleaned_output[:200]}"
            )
            return []

    # ---------- 去重 ----------

    @staticmethod
    def _dedup(facts: list[dict], existing: list[dict]) -> list[dict]:
        existing_keys = {(f["category"], f["fact"]) for f in existing}
        deduped = []
        seen = set()
        for fact in facts:
            key = (fact.get("category"), fact.get("fact"))
            if key in existing_keys or key in seen:
                continue
            seen.add(key)
            deduped.append(fact)
        return deduped

    # ---------- 主流程 ----------

    def run(self, user_id: str, output_path: str) -> list[dict]:
        run_started_at = datetime.now().strftime(TIMESTAMP_FMT)
        output_file = Path(output_path)
        # 把"读取旧数据 -> 抽取 -> 合并去重 -> 写入"整个流程包在同一把锁里，
        # 避免两次并发运行各自基于同一份旧 facts.json 算出不同结果、
        # 后写入的把先写入的覆盖掉
        with file_lock_for(output_file):
            # 1. 直接从输出文件中获取水位线 last_run_at 和已有事实
            metadata, existing_facts = self._load_facts_and_metadata(output_file)
            since_str = metadata.get("last_run_at")
            since = self._parse_ts(since_str) if since_str else None

            # 2. 检索并分析增量会话
            candidate_sessions = self._candidate_sessions(user_id, since)

            all_new_facts: list[dict] = []
            for session_id in candidate_sessions:
                new_messages = self._new_messages(user_id, session_id, since)
                if not new_messages:
                    continue
                facts = self._extract_from_messages(new_messages)
                all_new_facts.extend(facts)

            # 3. 数据合并与去重
            new_unique_facts = self._dedup(all_new_facts, existing_facts)
            merged_facts = existing_facts + new_unique_facts

            # 4. 原子写入：临时文件 + os.replace，避免写一半崩溃损坏文件
            save_payload = {
                "metadata": {
                    "last_run_at": run_started_at,
                    "updated_at": datetime.now(timezone.utc).strftime(TIMESTAMP_FMT),
                },
                "facts": merged_facts,
            }

            atomic_dump_json(output_file, save_payload)
        return new_unique_facts


# ==========================================
# 4. 执行入口示例
# ==========================================
if __name__ == "__main__":
    # 实例化已有的 qa_chain 对象
    from generator.qa_chain import QAChain
    qa_chain = QAChain(cuda_device="0")

    # 1. 创建适配器实例
    adapter = QAChainLLMAdapter(qa_chain=qa_chain, clean_think=True)
    
    # 2. 获取 data 根目录路径
    data_root = Path("/workspace/hf-conda/RAG/问答机器人/data")
    # 3. 自动遍历 data 目录下的所有用户文件夹
    user_dirs = [d for d in data_root.iterdir() if d.is_dir()]

    print(
        f"🔍 扫描到 {len(user_dirs)} 个用户账号: {[d.name for d in user_dirs]}"
    )

    for user_dir in user_dirs:
        user_id = user_dir.name
        print(f"\n🚀 开始抽取用户 [{user_id}] 的增量事实...")

        # 动态实例化路径
        paths = UserMemoryPathConfig(user_id=user_id)

        # 运行抽取器
        extractor = FactExtractor(
            data_dir=str(paths.data_dir),
            llm_adapter=adapter,
        )

        extractor.run(
            user_id=paths.user_id,
            output_path=str(paths.facts_path),
        )