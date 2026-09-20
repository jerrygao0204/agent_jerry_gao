# memory_growth/2_layer_mapper.py

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
try:
    from llm_guard import LLMGuard
except ImportError:
    from .llm_guard import LLMGuard

logger = logging.getLogger(__name__)


# ==========================================
# 1. LLM 網關適配器
# ==========================================

class LLMClientAdapter:
    """LiteLLM 網關流式適配器 (支持控制台實時 Token 打印與 <think> 清洗)"""
    def __init__(self, llm_client, model_name: str = "qwen3-32b", clean_think: bool = True):
        self.llm_client = llm_client
        self.model_name = model_name
        self.clean_think = clean_think

    def chat_completion(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        stream = self.llm_client.stream_generate(
            messages=messages,
            model_name=self.model_name,
            temperature=0.1
        )

        full_chunks = []
        for chunk in stream:
            print(chunk, end="", flush=True)
            full_chunks.append(chunk)
        print()

        content = "".join(full_chunks)

        if self.clean_think and "</think>" in content:
            content = content.split("</think>")[-1].strip()

        return content


# ==========================================
# 2. 三層語境映射器
# ==========================================

class LayerMapper:
    """事實規範化與 Profile / Memory / State 三層語境映射器 (集成 LLMGuard 防護)"""

    def __init__(self, llm_adapter: LLMClientAdapter):
        self.llm_adapter = llm_adapter
        # 裝配 LLMGuard 防護層，底層共用 llm_adapter 的 llm_client
        self.guard = LLMGuard(
            llm_client=self.llm_adapter.llm_client,
            default_model=self.llm_adapter.model_name,
            max_retries=5
        )

    def _load_json(self, file_path: str) -> Any:
        """安全載入 JSON 文件"""
        path = Path(file_path)
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"⚠️ 讀取 JSON 失敗 ({file_path}): {e}")
            return None

    def _save_json(self, file_path: str, data: Any):
        """安全保存 JSON 文件"""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _clean_and_canonicalize_facts(self, raw_facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """【Pass 1】事實碎片規範化清理與去重 (使用 LLMGuard 防護)"""
        if not raw_facts:
            return []

        system_prompt = (
            "你是一個數據清洗與規範化專家。請分析以下抽取出的原始事實碎片 (Raw Facts)：\n"
            "1. 去除重複或高相似度的無效事實。\n"
            "2. 修正模稜兩可的描述，使其符合標準語法結構。\n"
            "3. 確保包含 category, content, confidence 欄位。\n"
            "請嚴格輸出合法 JSON 格式，根節點必須包含 'cleaned_facts' 列表。"
        )
        user_prompt = f"【待處理的 Raw Facts】:\n{json.dumps(raw_facts, ensure_ascii=False, indent=2)}"

        schema_desc = (
            "必須返回包含根鍵 'cleaned_facts' 的 JSON 物件。\n"
            "'cleaned_facts' 為列表，列表中每個對象必須包含：\n"
            "- category (string): 類別 (如 identity, preference, constraint, decision, goal)\n"
            "- content (string): 清洗後的精準事實描述\n"
            "- confidence (string): 置信度 ('high', 'medium', 'low')"
        )

        try:
            result_json = self.guard.generate_guaranteed_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema_desc=schema_desc,
                model_name=self.llm_adapter.model_name
            )

            if isinstance(result_json, dict):
                return result_json.get("cleaned_facts", [])
            return []
        except Exception as e:
            logger.error(f"❌ [Pass 1] 事實清洗過程發生異常: {e}")
            return raw_facts  # 降級使用原始事實

    def _map_layered_context(self, cleaned_facts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """【Pass 2】三層語境映射 (Profile / Memory / State，使用 LLMGuard 防護)"""
        
        system_prompt = (
            "你是一個 AI 記憶系統三層語境映射專家。\n"
            "請將整理後的事實碎片，映射至 Profile、Memory、State 三層結構中：\n"
            "- Profile (靜態畫像): identity (身份/姓名/角色), preferences (偏好/習慣), constraints (硬性限制/禁忌)\n"
            "- Memory (動態記憶鏈與關係圖譜): user_growth_chains (個人成長演進鏈), social_and_attitude_graph (人際關係與態度圖譜)\n"
            "- State (當前狀態): current_goals (當前短期目標), active_focus (當前關注焦點/任務)\n"
            "請嚴格輸出合法 JSON 格式。"
        )
        user_prompt = f"【待映射的事實數據】:\n{json.dumps(cleaned_facts, ensure_ascii=False, indent=2)}"

        schema_desc = (
            "必須返回包含三個根鍵 'profile', 'memory', 'state' 的 JSON 物件：\n"
            "1. 'profile' (dict): 包含 'identity' (list), 'preferences' (list), 'constraints' (list)\n"
            "2. 'memory' (dict): 包含 'user_growth_chains' (list), 'social_and_attitude_graph' (list)\n"
            "3. 'state' (dict): 包含 'current_goals' (list), 'active_focus' (list)"
        )

        default_fallback = {
            "profile": {"identity": [], "preferences": [], "constraints": []},
            "memory": {"user_growth_chains": [], "social_and_attitude_graph": []},
            "state": {"current_goals": [], "active_focus": []}
        }

        try:
            result_json = self.guard.generate_guaranteed_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema_desc=schema_desc,
                model_name=self.llm_adapter.model_name
            )

            if isinstance(result_json, dict) and "profile" in result_json:
                return result_json
            return default_fallback
        except Exception as e:
            logger.error(f"❌ [Pass 2] 三層語境映射過程發生異常: {e}")
            return default_fallback

    def run(
        self,
        facts_path: str,
        output_path: str
    ) -> Dict[str, Any]:
        """執行 Phase 2 三層語境映射流水線"""
        
        # 1. 載入 Raw Facts
        raw_facts = self._load_json(facts_path) or []
        if not raw_facts:
            logger.info("ℹ️ 沒有可用的 Raw Facts，跳過 Phase 2。")
            return {}

        logger.info(f"🧹 [Pass 1] 開始規範化清洗 {len(raw_facts)} 條 Raw Facts...")
        cleaned_facts = self._clean_and_canonicalize_facts(raw_facts)
        logger.info(f"✅ [Pass 1] 清洗完成，保留 Cleaned Facts: {len(cleaned_facts)} 條")

        logger.info("🧩 [Pass 2] 開始進行三層語境映射 (Profile / Memory / State)...")
        layered_context = self._map_layered_context(cleaned_facts)

        # 2. 保存 layered_context.json
        self._save_json(output_path, layered_context)
        logger.info(f"✅ Phase 2 映射完成！數據已保存至: {output_path}")

        return layered_context


# ==========================================
# 3. 執行入口（基於 LLMClient 網關）
# ==========================================
if __name__ == "__main__":
    from path_config import UserMemoryPathConfig
    from generator.llm_client import LLMClient

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

    test_user = "admin"
    paths = UserMemoryPathConfig(user_id=test_user)

    print(f"--- 測試 2_layer_mapper.py (Qwen3-32B + LLMGuard 護航模式) [{test_user}] ---")

    # 1. 初始化網關與適配器
    client = LLMClient(default_model_name="qwen3-32b")
    adapter = LLMClientAdapter(llm_client=client, model_name="qwen3-32b")

    # 2. 初始化並執行 Phase 2 映射
    mapper = LayerMapper(llm_adapter=adapter)
    layered_data = mapper.run(
        facts_path=str(paths.facts_path),
        output_path=str(paths.layered_context_path)
    )

    print(f"\n✅ 映射完成，Profile 項目數: {len(layered_data.get('profile', {}).get('identity', []))}")
    print(f"📄 三層語境已保存至: {paths.layered_context_path}")