# memory_growth/3_context_builder.py

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


# ==========================================
# 1. Context Builder 核心類別
# ==========================================

class ContextBuilder:
    """語境構建器：將 layered_context.json 渲染為結構化 Markdown 系統語境"""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def _load_json(self, file_path: Path) -> Dict[str, Any]:
        if not file_path.exists():
            logger.warning(f"⚠️ JSON 檔案不存在: {file_path}")
            return {}
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"❌ 讀取 JSON 失敗 ({file_path}): {e}")
            return {}

    def _save_text(self, file_path: Path, content: str):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)

    def _render_profile(self, profile: Dict[str, Any]) -> str:
        identities = profile.get("identity", [])
        preferences = profile.get("preferences", [])
        constraints = profile.get("constraints", [])

        # 模組 01: 核心身份与定位
        identities_text = "\n".join([f"- **{i}**" for i in identities]) if identities else "- 未提供"
        mod_01 = f"### [模組 01: 核心身份與定位 (Identity & Role)]\n{identities_text}\n"

        # 模組 02: 工作習慣與偏好
        prefs_text = "\n".join([f"- {p}" for p in preferences]) if preferences else "- 無特殊偏好"
        mod_02 = f"### [模組 02: 技術與工作偏好 (Preferences)]\n{prefs_text}\n"

        # 模組 08: 硬性約束與禁忌
        constraints_text = "\n".join([f"- ⚠️ **{c}**" for c in constraints]) if constraints else "- 無硬性限制"
        mod_08 = f"### [模組 08: 硬性約束與禁忌 (Hard Constraints)]\n{constraints_text}\n"

        return f"{mod_01}\n{mod_02}\n{mod_08}"

    def _render_memory(self, memory: Dict[str, Any]) -> str:
        growth_chains = memory.get("user_growth_chains", [])
        social_graph = memory.get("social_and_attitude_graph", [])

        # 模組 10: 用戶成長軌跡 (User Growth Chains)
        growth_lines = []
        for item in growth_chains:
            step = item.get("step", 0)
            phase = item.get("phase", "未知階段")
            detail = item.get("detail", "")
            ts = item.get("timestamp", "")
            status = item.get("status", "completed")
            growth_lines.append(f"  {step}. **[{ts}] {phase}** (`{status}`)\n     └─ {detail}")

        growth_text = "\n".join(growth_lines) if growth_lines else "- 尚無紀錄"
        mod_10 = f"### [模組 10: 用戶能力與專案演進軌跡 (User Growth Chains)]\n{growth_text}\n"

        # 模組 12: 人際關係與態度演進 (Social & Attitude Graph)
        graph_lines = []
        for node in social_graph:
            entity = node.get("entity", "未知實體")
            relation = node.get("relation", "ASSOCIATE")
            curr_att = node.get("current_attitude", {})
            hist_atts = node.get("historical_attitudes", [])

            curr_text = f"**{curr_att.get('stance', 'NEUTRAL')}** ({curr_att.get('sentiment', 'neutral')}): \"{curr_att.get('impression', '')}\" [更新於: {curr_att.get('last_updated', '未知')}]"
            
            hist_text_list = []
            for h in hist_atts:
                hist_text_list.append(f"└─ [{h.get('timestamp', '過去')}] 曾為 **{h.get('stance', '')}**: \"{h.get('impression', '')}\"")
            hist_rendered = ("\n     " + "\n     ".join(hist_text_list)) if hist_text_list else ""

            graph_lines.append(f"- **{entity}** (`{relation}`)\n  - **當前態度**: {curr_text}{hist_rendered}")

        graph_text = "\n".join(graph_lines) if graph_lines else "- 尚無社交關係與態度數據"
        mod_12 = f"### [模組 12: 人際關係網絡與態度時間演進 (Social & Attitude Graph)]\n{graph_text}\n"

        return f"{mod_10}\n{mod_12}"

    def _render_state(self, state: Dict[str, Any]) -> str:
        current_goals = state.get("current_goals", [])
        active_focus = state.get("active_focus", [])

        # 模組 05: 當前階段目標與重點
        goals_text = "\n".join([f"- **{g}**" for g in current_goals]) if current_goals else "- 無"
        focus_text = "\n".join([f"- {f}" for f in active_focus]) if active_focus else "- 無"

        mod_05 = f"### [模組 05: 當前階段目標與技術焦點 (Current Goals & Active Focus)]\n**核心目標**:\n{goals_text}\n\n**當前研發焦點**:\n{focus_text}\n"

        return mod_05

    def build(self, layered_context_path: Path, output_text_path: Path) -> str:
        """執行渲染並導出文本"""
        ctx = self._load_json(layered_context_path)

        profile_sec = self._render_profile(ctx.get("profile", {}))
        memory_sec = self._render_memory(ctx.get("memory", {}))
        state_sec = self._render_state(ctx.get("state", {}))

        # 組合 11 個標準模組範本結構
        rendered_doc = (
            "# SYSTEM CONTEXT: USER COGNITIVE PROFILE & MEMORY\n"
            "====================================================\n"
            "以下內容為用戶的個人化認知圖譜、語境記憶與當前狀態。請在對話中嚴格遵循此 context，提供精準、個性化且符合硬性約束的回應。\n\n"
            f"{profile_sec}\n"
            f"{state_sec}\n"
            f"{memory_sec}\n"
            "====================================================\n"
            "END OF SYSTEM CONTEXT\n"
        )

        self._save_text(output_text_path, rendered_doc)
        logger.info(f"✅ 成功渲染並導出 Context 文本至: {output_text_path}")
        return rendered_doc


# ==========================================
# 2. 執行入口
# ==========================================
if __name__ == "__main__":
    from path_config import UserMemoryPathConfig

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

    test_user = "admin"
    paths = UserMemoryPathConfig(user_id=test_user)

    print(f"--- 測試 3_context_builder.py (模組化渲染) [{test_user}] ---")

    builder = ContextBuilder(data_dir=str(paths.data_dir))
    context_text = builder.build(
        layered_context_path=paths.layered_context_path,
        output_text_path=paths.user_prompt_context_path
    )

    print("\n📄 生成的 user_prompt_context.txt 預覽：\n")
    print(context_text[:1200] + "\n...\n(剩餘內容已省略)")