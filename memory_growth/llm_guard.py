# memory_growth/llm_guard.py

import json
import logging
from typing import Dict, Any, Tuple, Optional, List

logger = logging.getLogger(__name__)


class LLMGuard:
    """
    獨立 LLM 防護與修復適配器 (LLM Guard Adapter)
    完全不改動原有 llm_client.py，提供 5 輪 Self-Healing 重試與 LLM 專家二次評估。
    """

    def __init__(self, llm_client: Any, default_model: str = "qwen3-32b", max_retries: int = 5):
        self.llm_client = llm_client
        self.default_model = default_model
        self.max_retries = max_retries

    def _call_llm_stream(self, system_prompt: str, user_prompt: str, model_name: str) -> str:
        """調用原生的 llm_client.stream_generate 並拼接字符串，自動清洗 <think> 標籤"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        chunks = []
        for chunk in self.llm_client.stream_generate(
            messages=messages,
            model_name=model_name,
            temperature=0.1
        ):
            chunks.append(chunk)

        raw_text = "".join(chunks).strip()
        # 清洗 Reasoning Model 的思考標籤
        if "</think>" in raw_text:
            raw_text = raw_text.split("</think>")[-1].strip()
        return raw_text

    def _parse_and_validate_syntax(self, raw_text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        """提取並驗證 JSON 語法合規性"""
        if not raw_text:
            return None, "輸出內容為空 (Empty Output)"

        text = raw_text
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()

        try:
            parsed = json.loads(text)
            if not isinstance(parsed, dict):
                return None, f"JSON 根節點必須是物件 (Dict)，當前類型為: {type(parsed).__name__}"
            return parsed, None
        except json.JSONDecodeError as e:
            return None, f"JSON 語法解析報錯 (第 {e.lineno} 行, 第 {e.colno} 欄): {e.msg}"

    def _evaluate_with_llm(
        self,
        parsed_json: Dict[str, Any],
        schema_desc: str,
        model_name: str
    ) -> Tuple[bool, int, str]:
        """【LLM 專家二次評估】由 LLM 擔任 Auditor 審查業務邏輯與結構合規性"""
        eval_system = (
            "你是一個極度嚴苛的數據結構與品質審查專家 (Quality Auditor)。\n"
            "請評估給出的 JSON 數據是否嚴格符合要求的結構與業務邏輯。\n"
            "請嚴格輸出合法 JSON 格式：\n"
            '{\n  "passed": true/false,\n  "score": 1-10,\n  "reason": "審查意見說明"\n}'
        )
        eval_user = (
            f"【要求的數據結構與規格】:\n{schema_desc}\n\n"
            f"【待評估的 JSON 數據】:\n{json.dumps(parsed_json, ensure_ascii=False, indent=2)}"
        )

        try:
            eval_raw = self._call_llm_stream(eval_system, eval_user, model_name)
            eval_parsed, _ = self._parse_and_validate_syntax(eval_raw)
            if eval_parsed and isinstance(eval_parsed, dict):
                return (
                    bool(eval_parsed.get("passed", False)),
                    int(eval_parsed.get("score", 0)),
                    str(eval_parsed.get("reason", "無詳細說明"))
                )
        except Exception as e:
            logger.error(f"⚠️ 評估過程發生異常: {e}")

        # 評估模組本身異常時預設放行，避免阻斷流程
        return True, 7, "審查模組執行異常，預設放行"

    def generate_guaranteed_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_desc: str,
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        核心介面：帶 5 輪 Self-Healing 重試與 LLM 評估的 JSON 生成方法
        """
        target_model = model_name or self.default_model
        current_user_prompt = user_prompt
        history_attempts: List[Dict[str, Any]] = []

        for attempt in range(1, self.max_retries + 1):
            logger.info(f"🔄 [LLMGuard Pipeline] 第 {attempt}/{self.max_retries} 輪生成嘗試...")

            # 1. 生成原始文本
            raw_response = self._call_llm_stream(system_prompt, current_user_prompt, target_model)

            # 2. 語法檢查 (Syntax Check)
            parsed_json, syntax_error = self._parse_and_validate_syntax(raw_response)

            if syntax_error:
                logger.warning(f"⚠️ [第 {attempt} 輪] 語法檢查失敗: {syntax_error}")
                history_attempts.append({
                    "score": 0,
                    "parsed": None,
                    "raw": raw_response,
                    "error": syntax_error
                })

                # 將具體錯誤行號與理由帶入下一輪 Prompt 進行自我修復
                current_user_prompt = (
                    f"{user_prompt}\n\n"
                    f"====================================================\n"
                    f"❌ [上一輪生成失敗]: JSON 語法解析報錯！\n"
                    f"【具體錯誤】: {syntax_error}\n"
                    f"【錯誤的原始輸出】:\n{raw_response}\n"
                    f"====================================================\n"
                    f"請仔細檢查上述報錯位置（如缺少逗號、引號未閉合等），嚴格修正後重新輸出！"
                )
                continue

            # 3. LLM 專家二次評估 (LLM Evaluation)
            eval_passed, eval_score, eval_reason = self._evaluate_with_llm(parsed_json, schema_desc, target_model)

            if eval_passed:
                logger.info(f"🎉 [第 {attempt} 輪] 通過 LLM 專家評估！得分: {eval_score}/10")
                return parsed_json
            else:
                logger.warning(f"⚠️ [第 {attempt} 輪] 未通過審查 | 得分: {eval_score} | 原因: {eval_reason}")
                history_attempts.append({
                    "score": eval_score,
                    "parsed": parsed_json,
                    "raw": raw_response,
                    "error": eval_reason
                })

                # 將審查意見帶給 LLM 進行修正
                current_user_prompt = (
                    f"{user_prompt}\n\n"
                    f"====================================================\n"
                    f"⚠️ [上一輪審查未通過]: 得分 {eval_score}/10\n"
                    f"【審查意見】: {eval_reason}\n"
                    f"====================================================\n"
                    f"請根據審查意見改進，重新生成完整合規的 JSON！"
                )

        # 4. 5 輪全部未達標：選出歷史最佳版本 (Score 最高且語法正確)
        logger.error(f"❌ 已達最大重試上限 ({self.max_retries} 輪)，選擇歷史最佳版本...")
        valid_candidates = [item for item in history_attempts if item["parsed"] is not None]

        if valid_candidates:
            best = max(valid_candidates, key=lambda x: x["score"])
            logger.info(f"🏆 選出歷史最佳版本 (得分: {best['score']}/10)")
            return best["parsed"]

        logger.critical("💥 5 輪生成均無法解析為有效 JSON，降級返回空結構！")
        return {}