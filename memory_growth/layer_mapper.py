# memory_growth/layer_mapper.py
"""
layer_mapper.py - 把 FactExtractor 产出的扁平事实(facts.json)
映射进标准的三层语境 JSON 结构 (QAChain 适配版)。

主要修改点：
1. 适配 QAChainLLMAdapter 适配器，移除 Anthropic API 依赖。
2. 兼容 facts.json 中包含 metadata 和 facts 的新数据格式。
3. 支持 layered_context.json 文件的增量合并与去重。
4. 增强分类的鲁棒性（清理思考标签与正则表达式匹配）。
"""
"""
layer_mapper.py - 把 FactExtractor 产出的扁平事实(facts.json)
映射进标准的三层语境 + 单层静态画像 结构。
"""

import sys
from pathlib import Path

# 获取当前文件所在的上一级目录（即：/workspace/hf-conda/RAG/问答机器人/）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ----------------- 以下是原来的 import 语句 -----------------
from generator.qa_chain import QAChain
import json
import re
from pathlib import Path
from typing import Any
from path_config import UserMemoryPathConfig

# ==========================================
# 1. 扩充的 Schema (新增 user_profile)
# ==========================================

STANDARD_SCHEMA: dict[str, Any] = {
    "user_profile": {
        "identity_facts": []  # 静态画像存储区，直接扁平化存放
    },
    "stable_context": {
        "long_term_goals": [],
        "ability_tree": [],
        "co_reasoning_protocol": [],
    },
    "dynamic_context": {
        "short_term_goals": [],
        "current_blockers": [],
        "preference_trajectory": [],
        "reasoning_patterns": [],
        "technical_migration": [],
        "reflection_points": [],
    },
    "growth_context": {
        "growth_trajectory": [],
        "user_growth_points": [],
        "model_growth_points": [],
    },
}

# 动态/成长类 category -> 允许映射的候选 field
CATEGORY_FIELD_CANDIDATES: dict[str, list[str]] = {
    "explicit_preference": ["preference_trajectory", "reasoning_patterns"],
    "explicit_decision": ["short_term_goals", "growth_trajectory"],
    "explicit_constraint": ["current_blockers", "co_reasoning_protocol"],
    "technical_fact": ["technical_migration", "ability_tree", "long_term_goals"],
}

# field -> 所属 layer 反查表 (排除 user_profile)
FIELD_TO_LAYER: dict[str, str] = {
    field: layer
    for layer, fields in STANDARD_SCHEMA.items()
    if layer != "user_profile"
    for field in fields
}

CLASSIFY_SYSTEM_PROMPT = """你是一个语境归类器。给你一条事实（包含 category 和原话），
请从给定的候选 field 列表中，选出唯一一个最合适的 field。

判断标准：
- long_term_goals: 长期、跨项目的方向性目标
- ability_tree: 用户已掌握或正在建立的技能/能力
- co_reasoning_protocol: 用户对"如何和模型协作/推理"的规则性要求
- short_term_goals: 近期、有时效性的目标或计划
- growth_trajectory: 标志性的成长节点/里程碑式的决定
- technical_migration: 一次性的技术栈切换/迁移事件
- current_blockers: 当前卡点
- preference_trajectory: 偏好本身（不一定是长期能力）
- reasoning_patterns: 用户的思维/推理习惯

注意：严禁输出 <think> 思考过程，只输出一个候选 field 名称本身，不要带标点或解释！
"""

# ... [保留原本的 QAChainLLMAdapter 代码不变] ...
class QAChainLLMAdapter:
    def __init__(self, qa_chain: Any, clean_think: bool = True):
        self.qa_chain = qa_chain
        self.clean_think = clean_think
        self.tokenizer = getattr(qa_chain, "tokenizer", None) or getattr(
            getattr(qa_chain, "llm", None), "tokenizer", None
        )

    def _clean_think_tags(self, text: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)
        return cleaned.strip()

    def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.1) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if self.tokenizer and hasattr(self.tokenizer, "apply_chat_template"):
            full_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            full_prompt = f"System: {system_prompt}\nUser: {prompt}\nAssistant:"

        response_text = ""
        llm_client = getattr(self.qa_chain, "llm_client", getattr(self.qa_chain, "llm", self.qa_chain))
        for resp in llm_client.stream_generate(query=prompt, context=full_prompt):
            token = resp.get("data", "") if isinstance(resp, dict) else str(resp)
            response_text += token

        if self.clean_think:
            response_text = self._clean_think_tags(response_text)
        return response_text.strip()


class LayerMapper:
    def __init__(self, llm_adapter: QAChainLLMAdapter):
        self.adapter = llm_adapter

    def _classify_field(self, fact: dict[str, Any]) -> str:
        category = fact.get("category", "")
        candidates = CATEGORY_FIELD_CANDIDATES.get(category)
        if not candidates:
            raise ValueError(f"未知 category: {category}")
        if len(candidates) == 1:
            return candidates[0]

        prompt = (
            f"事实：{fact['fact']}\n"
            f"原话：{fact.get('source_quote', '')}\n"
            f"候选 field：{', '.join(candidates)}\n"
            f"请从候选 field 中选择最贴切的一个选择输出："
        )

        raw_output = self.adapter.generate(prompt=prompt, system_prompt=CLASSIFY_SYSTEM_PROMPT, temperature=0.1)
        clean_target = raw_output.strip().strip("`").strip('"').strip("'")

        matched_candidate = None
        for candidate in candidates:
            if candidate == clean_target or re.search(rf"\b{candidate}\b", clean_target):
                matched_candidate = candidate
                break

        if matched_candidate:
            return matched_candidate

        print(f"[WARN] 分类输出 '{raw_output}' 无法精准匹配候选集合 {candidates}，降级回退为 '{candidates[0]}'")
        return candidates[0]

    def _load_existing_context(self, output_path: Path) -> dict[str, dict[str, list]]:
        if not output_path.exists():
            return json.loads(json.dumps(STANDARD_SCHEMA))
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                base = json.loads(json.dumps(STANDARD_SCHEMA))
                # 兼容旧版本，保留所有数据
                for layer, fields in base.items():
                    if layer in data and isinstance(data[layer], dict):
                        for field in fields:
                            if field in data[layer] and isinstance(data[layer][field], list):
                                fields[field] = data[layer][field]
                return base
        except Exception as e:
            print(f"[WARN] 读取历史 {output_path.name} 失败，重新初始化: {e}")
            return json.loads(json.dumps(STANDARD_SCHEMA))

    def map_facts(self, facts: list[dict[str, Any]], existing_context: dict[str, dict[str, list]]) -> dict[str, dict[str, list]]:
        result = json.loads(json.dumps(existing_context))
        existing_keys = set()
        
        # 构建现有数据索引以去重
        for layer, fields in result.items():
            for field, items in fields.items():
                for item in items:
                    existing_keys.add((item.get("category"), item.get("fact"), field))

        for fact in facts:
            category = fact.get("category")
            
            # 核心拦截逻辑：静态画像直接入库，不走 LLM 分类
            if category == "identity_fact":
                field = "identity_facts"
                layer = "user_profile"
            else:
                field = self._classify_field(fact)
                layer = FIELD_TO_LAYER[field]

            key = (category, fact.get("fact"), field)
            if key in existing_keys:
                continue

            entry = {**fact, "field": field}
            result[layer][field].append(entry)
            existing_keys.add(key)

        return result

    def run(self, facts_path: str, output_path: str) -> dict[str, dict[str, list]]:
        facts_file = Path(facts_path)
        if not facts_file.exists():
            return json.loads(json.dumps(STANDARD_SCHEMA))

        with open(facts_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        facts = raw_data.get("facts", []) if isinstance(raw_data, dict) else (raw_data if isinstance(raw_data, list) else [])

        output_file = Path(output_path)
        existing_context = self._load_existing_context(output_file)
        layered = self.map_facts(facts, existing_context)

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(layered, f, ensure_ascii=False, indent=2)

        return layered

# # ==========================================
# # 4. 执行入口示例
# # ==========================================

# if __name__ == "__main__":
#     from generator.qa_chain import QAChain

#     qa_chain = QAChain(cuda_device="0")
#     adapter = QAChainLLMAdapter(qa_chain=qa_chain, clean_think=True)

#     current_user = "gaozheng"
#     paths = UserMemoryPathConfig(user_id=current_user)

#     mapper = LayerMapper(llm_adapter=adapter)

#     mapper.run(
#         facts_path=str(
#             paths.facts_path
#         ),  # 读取: .../memory_growth/context/users/gaozheng/facts.json
#         output_path=str(
#             paths.layered_context_path
#         ),  # 写入: .../memory_growth/context/users/gaozheng/layered_context.json
#     )

# ==========================================
# 4. 执行入口示例（批量自动处理所有用户）
# ==========================================
if __name__ == "__main__":

    # 1. 实例化模型与适配器
    qa_chain = QAChain(cuda_device="0")
    adapter = QAChainLLMAdapter(qa_chain=qa_chain, clean_think=True)

    # 2. 自动获取所有用户目录
    data_root = Path("/workspace/hf-conda/RAG/问答机器人/data")
    user_dirs = [d for d in data_root.iterdir() if d.is_dir()]

    for user_dir in user_dirs:
        user_id = user_dir.name
        print(f"\n🗺️ 开始映射用户 [{user_id}] 的三层语境...")

        paths = UserMemoryPathConfig(user_id=user_id)
        mapper = LayerMapper(llm_adapter=adapter)
        print(paths.facts_path)
        print(paths.layered_context_path)
        mapper.run(
            facts_path=str(paths.facts_path),
            output_path=str(paths.layered_context_path),
        )

    print('修改完毕')