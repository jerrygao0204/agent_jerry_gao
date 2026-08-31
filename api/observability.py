# api/observability.py
import glob
import json
import os
from typing import Dict, Any

def scan_metrics(data_dir: str = "data") -> Dict[str, Any]:
    """
    轻量级可观测性扫描器：递归扫描 data 目录下所有 session_*.json
    """
    # 转换为绝对路径，避免绝对/相对路径解析异常
    abs_data_dir = os.path.abspath(data_dir)
    pattern = os.path.join(abs_data_dir, "**", "session_*.json")
    session_files = glob.glob(pattern, recursive=True)

    total_sessions = len(session_files)
    total_queries = 0
    compliance_interceptions = 0
    active_users = set()

    for file_path in session_files:
        # 稳健提取 userId (适配 /workspace/.../data/{userId}/session_*.json)
        rel_path = os.path.relpath(file_path, abs_data_dir)
        parts = rel_path.split(os.sep)
        if len(parts) >= 2:
            active_users.add(parts[0])  # 第一层即为 userId (如 admin)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                history = json.load(f)

            if isinstance(history, list):
                for msg in history:
                    if msg.get("role") == "user":
                        total_queries += 1

                    content = msg.get("content", "")
                    meta = msg.get("metadata", {})
                    if meta.get("blocked") or "触发安全策略拦截" in content or "[COMPLIANCE_BLOCK]" in content:
                        compliance_interceptions += 1
        except Exception:
            continue

    return {
        "total_users": len(active_users),
        "total_sessions": total_sessions,
        "total_queries": total_queries,
        "compliance_interceptions": compliance_interceptions,
        "interception_rate": f"{(compliance_interceptions / total_queries * 100):.2f}%" if total_queries > 0 else "0.00%"
    }