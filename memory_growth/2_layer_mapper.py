# memory_growth/2_layer_mapper.py
import json
import logging
import yaml
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

    def __init__(self, 
                 llm_adapter: LLMClientAdapter,
                 prompt_hub_path: str = ""):
        self.llm_adapter = llm_adapter
        # 裝配 LLMGuard 防護層，底層共用 llm_adapter 的 llm_client
        self.guard = LLMGuard(
            llm_client=self.llm_adapter.llm_client,
            default_model=self.llm_adapter.model_name,
            max_retries=5
        )
        self.prompt_hub_path = prompt_hub_path
        self.prompt_hub = self._load_prompt_hub(prompt_hub_path)

    def _load_prompt_hub(self, prompt_hub_path: str) -> Dict[str, Any]:
        """從 YAML 加載 Prompt Hub 映射表 (按 name 建立索引)"""
        hub_file = Path(prompt_hub_path)
        if not hub_file.exists():
            logger.error(f"❌ Prompt HUB 配置文件不存在: {prompt_hub_path}")
            return {}
        try:
            with open(hub_file, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                prompts_list = data.get("prompts", [])
                return {item["name"]: item["content"] for item in prompts_list if "name" in item}
        except Exception as e:
            logger.error(f"⚠️ 加載 Prompt HUB 失敗 ({prompt_hub_path}): {e}")
            return {}
        
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

        system_prompt = self.prompt_hub.get(
            "layer_mapper_clean_and_canonicalize_facts_system_prompt")

        user_prompt = f"【待處理的 Raw Facts】:\n{json.dumps(raw_facts, ensure_ascii=False, indent=2)}"

        schema_desc = self.prompt_hub.get(
            "layer_mapper_clean_and_canonicalize_facts_schema_prompt")


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

        system_prompt = self.prompt_hub.get(
            "layer_mapper_map_layered_context_system_prompt"
        )
        user_prompt = f"【待映射的事實數據】:\n{json.dumps(cleaned_facts, ensure_ascii=False, indent=2)}"

        schema_desc = self.prompt_hub.get(
            "layer_mapper_map_layered_context_schema_prompt"
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
    prompt_hub_path = getattr(paths, "prompt_hub_path", "")

    print(f"--- 測試 2_layer_mapper.py (Qwen3-32B + LLMGuard 護航模式) [{test_user}] ---")
    print(f"📜 Prompt Hub Path: {prompt_hub_path}")

    # 1. 初始化網關與適配器
    client = LLMClient(default_model_name="qwen3-32b")
    adapter = LLMClientAdapter(llm_client=client, model_name="qwen3-32b")

    # 2. 初始化並執行 Phase 2 映射 (傳入 prompt_hub_path)
    mapper = LayerMapper(
        llm_adapter=adapter,
        prompt_hub_path=str(prompt_hub_path)
    )
    layered_data = mapper.run(
        facts_path=str(paths.facts_path),
        output_path=str(paths.layered_context_path)
    )

    print(f"\n✅ 映射完成，Profile 項目數: {len(layered_data.get('profile', {}).get('identity', []))}")
    print(f"📄 三層語境已保存至: {paths.layered_context_path}")