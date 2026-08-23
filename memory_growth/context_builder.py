"""
context_builder.py - 成长型语境 Prompt 构建器

架构特点：
1. 项目根目录自动定位：防范 ModuleNotFoundError 路径报错。
2. 批量扫描多租户：自动处理 data/ 下的所有用户目录。
3. 11 模块防御性渲染：对空字段、字典列表和字符串列表进行统一结构化渲染。
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

# 1. 自动定位项目根目录 (/workspace/hf-conda/RAG/问答机器人/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from path_config import UserMemoryPathConfig


class ContextBuilder:
    """语境构建器：将 layered_context.json 渲染为 11 模块的 System Context"""

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.paths = UserMemoryPathConfig(user_id=user_id)

    def load_layered_context(self) -> Dict[str, Any]:
        """读取 layered_context.json 文件"""
        file_path = self.paths.layered_context_path
        if not file_path.exists():
            print(f"[WARN] 文件不存在: {file_path}，将采用空结构渲染")
            return {}

        try:
            if file_path.stat().st_size == 0:
                print(f"[WARN] 发现空文件: {file_path.name}，重置为空结构")
                return {}

            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[ERROR] 读取 {file_path.name} 失败: {e}")
            return {}

    def _format_items(self, items: Any) -> str:
        """统一渲染列表或字典子项，保证缩进与可读性"""
        if not items:
            return "- 暂无相关记录"

        if isinstance(items, dict):
            items = [items]

        if not isinstance(items, list):
            return f"- {str(items)}"

        lines = []
        for item in items:
            if isinstance(item, dict):
                # 兼容格式：优先提取 fact/content，否则抽取字段组合
                fact_val = item.get("fact") or item.get("content")
                if fact_val:
                    lines.append(f"- {fact_val}")
                else:
                    kv_pairs = [
                        f"{k}: {v}"
                        for k, v in item.items()
                        if k not in ["category", "field", "timestamp"]
                    ]
                    lines.append(
                        f"- {', '.join(kv_pairs)}"
                        if kv_pairs
                        else f"- {str(item)}"
                    )
            else:
                lines.append(f"- {str(item)}")

        return "\n".join(lines) if lines else "- 暂无相关记录"

    def build_11_modules_prompt(self, data: Dict[str, Any]) -> str:
        """解析三层语境并填充 11 模块"""
        user_profile = data.get("user_profile", {})
        stable_ctx = data.get("stable_context", {})
        dynamic_ctx = data.get("dynamic_context", {})
        growth_ctx = data.get("growth_context", {})

        # 身份细节处理
        identity_facts = user_profile.get("identity_facts", [])
        identity_text = self._format_items(identity_facts)

        rendered_text = f"""==================================================
【用户长期记忆与成长语境】 (User ID: {self.user_id})
==================================================

[模块 1: 用户静态基础档案 (Identity Fact)]
{identity_text}

[模块 2: 用户硬性偏好 (Explicit Preferences)]
{self._format_items(dynamic_ctx.get("preference_trajectory"))}

[模块 3: 用户技术与业务约束 (Explicit Constraints)]
{self._format_items(dynamic_ctx.get("current_blockers"))}

[模块 4: 已明确决策历史与技术迁移 (Explicit Decisions & Migration)]
{self._format_items(dynamic_ctx.get("technical_migration"))}

[模块 5: 长期认知与能力协议 (Co-Reasoning Protocol)]
{self._format_items(stable_ctx.get("co_reasoning_protocol"))}

[模块 6: 基础能力现状 (Layer 1: Ability Tree Anchor)]
{self._format_items(stable_ctx.get("ability_tree"))}

[模块 7: 动态实践与思维模式 (Layer 2: Reasoning Patterns)]
{self._format_items(dynamic_ctx.get("reasoning_patterns"))}

[模块 8: 演进目标与长远规划 (Layer 3: Long-Term Goals)]
{self._format_items(stable_ctx.get("long_term_goals"))}

[模块 9: 反思与避坑清单 (Reflection Points & Avoidance)]
{self._format_items(dynamic_ctx.get("reflection_points"))}

[模块 10: 用户成长轨迹 (User Growth Points)]
{self._format_items(growth_ctx.get("user_growth_points"))}

[模块 11: 模型协作演進点 (Model Growth Points)]
{self._format_items(growth_ctx.get("model_growth_points"))}
==================================================
"""
        return rendered_text

    def run(self) -> str:
        """执行构建并写入 user_prompt_context.txt"""
        context_data = self.load_layered_context()
        prompt_text = self.build_11_modules_prompt(context_data)

        output_file = self.paths.user_prompt_context_path
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(prompt_text)

        print(f"✅ 用户 [{self.user_id}] 的 11 模块 Prompt 构建完成！")
        print(f"   路径: {output_file.resolve()}\n")
        return prompt_text


# ==========================================
# 执行入口（批量扫描所有用户）
# ==========================================
if __name__ == "__main__":
    data_root = Path("/workspace/hf-conda/RAG/问答机器人/data")

    if not data_root.exists():
        print(f"[ERROR] 未找到数据根目录: {data_root}")
        sys.exit(1)

    user_dirs = [d for d in data_root.iterdir() if d.is_dir()]
    print(
        f"🔍 扫描到 {len(user_dirs)} 个用户账号: {[d.name for d in user_dirs]}\n"
    )

    for user_dir in user_dirs:
        user_id = user_dir.name
        print(f"🛠️ 开始构建用户 [{user_id}] 的 Context Prompt...")
        builder = ContextBuilder(user_id=user_id)
        builder.run()