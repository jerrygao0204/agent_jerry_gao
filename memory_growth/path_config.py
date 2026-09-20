# memory_growth/path_config.py
"""
path_config.py - 統一管理多用戶的輸入、輸出與增量狀態路徑
"""

import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config_loader import config_loader


class UserMemoryPathConfig:
    """自動生成並管理指定用戶的聊天歷史路徑、記憶存儲路徑與處理狀態記錄"""

    def __init__(
        self,
        user_id: str,
        data_root: Optional[str] = None,
        memory_root: Optional[str] = None,
    ):
        self.user_id = user_id

        config_dict = config_loader.get_config_dict()
        if data_root is None:
            data_root = config_dict["data_root"]
        if memory_root is None:
            memory_root = config_dict["memory_root"]

        # 1. 輸入路徑：原始聊天記錄目錄 (data/gaozheng, data/admin 等)
        self.data_dir = Path(data_root) / user_id

        # 2. 輸出路徑：成長記憶存儲目錄
        self.memory_dir = Path(memory_root) / user_id

        # 3. 各階段產出的具體檔案路徑
        self.processed_state_path = self.memory_dir / "processed_state.json"  # 新增：Message Hash 記錄表
        self.facts_path = self.memory_dir / "facts.json"
        self.layered_context_path = self.memory_dir / "layered_context.json"
        self.user_prompt_context_path = (
            self.memory_dir / "user_prompt_context.txt"
        )

        # 自動創建目錄
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)