# memory_growth/1_extractor.py
import json
import logging
from pathlib import Path
import yaml
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
# 2. 事實與圖譜關係抽取器
# ==========================================

class FactExtractor:
    """高召回率事實與關係圖譜抽取器 (支持增量/全量與 Hash 狀態追蹤)"""

    def __init__(self, 
                 data_dir: str, 
                 llm_adapter: LLMClientAdapter,
                 prompt_hub_path: str = ""
                 ):
        self.data_dir = Path(data_dir)
        self.llm_adapter = llm_adapter
        self.guard = LLMGuard(
            llm_client=self.llm_adapter.llm_client, 
            default_model=self.llm_adapter.model_name,
            max_retries=5,
            
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

    def _load_user_messages(self, user_id: str) -> List[Dict[str, Any]]:
        """載入指定用戶的所有對話數據 (包含 Message Hash 與 Timestamp)"""
        # 1. 優先嘗試讀取 merged_chats.json
        merged_file = self.data_dir / "merged_chats.json"
        if merged_file.exists():
            data = self._load_json(str(merged_file))
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return data.get("messages", [])

        # 2. 備用讀取方案：讀取 self.data_dir 目錄下所有 JSON 對話文件
        if not self.data_dir.exists():
            logger.error(f"❌ 用戶數據目錄不存在: {self.data_dir}")
            return []

        all_messages = []
        for p in sorted(self.data_dir.glob("*.json")):
            if p.name in ["processed_state.json", "facts.json", "layered_context.json"]:
                continue
            content = self._load_json(str(p))
            if isinstance(content, list):
                all_messages.extend(content)
            elif isinstance(content, dict) and "messages" in content:
                all_messages.extend(content["messages"])

        return all_messages

    def _extract_chunk(self, chunk_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """針對窗口內的消息塊調用 LLM 抽取事實"""
        if not chunk_messages:
            return []

        system_prompt = self.prompt_hub.get("extractor_extract_chunk_system_prompt")
            
        user_prompt = f"【待分析對話記錄】:\n{json.dumps(chunk_messages, ensure_ascii=False, indent=2)}"

        schema_desc = self.prompt_hub.get("extractor_extract_chunk_schema_prompt")
        
        try:
            result_json = self.guard.generate_guaranteed_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                schema_desc=schema_desc,
                model_name=self.llm_adapter.model_name
            )
            
            # 1. 防禦性檢查：空回應直接返回
            if isinstance(result_json, dict):
                return result_json.get("raw_facts", [])
            return []
        except Exception as e:
            logger.error(f"❌ 窗口事實抽取過程發生異常: {e}")
            return []
        
    def run(
        self,
        user_id: str,
        state_path: str,
        output_path: str,
        window_size: int = 5,
        overlap_size: int = 1,
        is_full_run: bool = False
    ) -> List[Dict[str, Any]]:
        """執行事實與圖譜抽取流水線 (帶重疊窗口與 Hash 狀態控管)"""
        
        # 1. 載入歷史處理狀態
        state_data = self._load_json(state_path) or {}
        processed_hashes = set(state_data.get("processed_hashes", []))

        # 2. 載入對話訊息
        user_messages = self._load_user_messages(user_id)

        # 3. 根據 is_full_run 標籤進行差集/全量選擇
        if is_full_run:
            logger.info(f"🔄 [全量模式] 清空歷史 Hash 記錄，將重新處理所有 {len(user_messages)} 條訊息")
            pending_messages = user_messages
            processed_hashes = set()
        else:
            pending_messages = [msg for msg in user_messages if msg.get("hash") not in processed_hashes]
            logger.info(f"⚡ [增量模式] 未處理訊息數量: {len(pending_messages)} / 總量: {len(user_messages)}")

        if not pending_messages:
            logger.info("ℹ️ 沒有需要抽取的訊息，跳過 Phase 1。")
            existing_facts = self._load_json(output_path) or []
            return existing_facts

        # 4. 重疊滑動窗口抽取
        new_facts = []
        step = max(1, window_size - overlap_size)
        
        for i in range(0, len(pending_messages), step):
            chunk = pending_messages[i : i + window_size]
            if not chunk:
                break

            logger.info(f"🔍 正在處理窗口 Chunk [{i} : {i + len(chunk)}]")
            extracted = self._extract_chunk(chunk)
            new_facts.extend(extracted)

            for msg in chunk:
                if "hash" in msg:
                    processed_hashes.add(msg["hash"])

            if i + window_size >= len(pending_messages):
                break

        # 5. 更新 processed_state.json
        self._save_json(state_path, {"processed_hashes": list(processed_hashes)})

        # 6. 合併並保存 facts.json
        if is_full_run:
            all_facts = new_facts
        else:
            existing_facts = self._load_json(output_path) or []
            all_facts = existing_facts + new_facts

        self._save_json(output_path, all_facts)
        logger.info(f"✅ Phase 1 抽取完成！產出 Raw Facts 數量: {len(new_facts)}，總累計: {len(all_facts)}")
        return new_facts


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

    print(f"--- 測試 1_extractor.py (Qwen3-32B + 圖譜/態度/重疊窗口模式) [{test_user}] ---")
    print(f"📜 Prompt Hub Path: {prompt_hub_path}")
    
    # 1. 初始化網關與適配器
    client = LLMClient(default_model_name="qwen3-32b")
    adapter = LLMClientAdapter(llm_client=client, model_name="qwen3-32b")

    # 2. 初始化並執行 Phase 1 抽取 (傳入 prompt_hub_path)
    extractor = FactExtractor(
        data_dir=str(paths.data_dir), 
        llm_adapter=adapter,
        prompt_hub_path=str(prompt_hub_path)
    )
    extracted_facts = extractor.run(
        user_id=paths.user_id,
        state_path=str(paths.processed_state_path),
        output_path=str(paths.facts_path),
        window_size=5,
        overlap_size=1,
        is_full_run=False
    )

    print(f"\n✅ 本次運行抽取的 Raw Facts 數量: {len(extracted_facts)}")
    print(f"📄 Raw Facts 已更新保存至: {paths.facts_path}")
    print(f"📌 訊息 Hash 狀態已更新保存至: {paths.processed_state_path}")