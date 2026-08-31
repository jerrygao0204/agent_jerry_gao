# tests/conftest.py

import sys
from pathlib import Path

# 确保 pytest 运行时能正确导入项目根目录（agent_jerry_gao/）下的所有模块
# 例如 agent.compliance / search.retriever / factory.tool_factory 等
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
