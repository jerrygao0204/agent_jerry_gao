# memory_growth/path_config.py
"""
path_config.py - 统一管理多用户的输入与输出路径
"""

import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.config_loader import config_loader


class UserMemoryPathConfig:
    """自动生成并管理指定用户的聊天历史路径与记忆存储路径"""

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

        # 1. 输入路径：该用户的原始聊天记录目录 (data/gaozheng, data/jiyun 等)
        self.data_dir = Path(data_root) / user_id

        # 2. 输出路径：该用户的成长记忆存储目录 (memory_growth/.../users/gaozheng 等)
        self.memory_dir = Path(memory_root) / user_id

        # 3. 各阶段产出的具体文件路径
        self.facts_path = self.memory_dir / "facts.json"
        self.layered_context_path = self.memory_dir / "layered_context.json"
        self.user_prompt_context_path = (
            self.memory_dir / "user_prompt_context.txt"
        )

        # 自动创建输入/输出目录（如果不存在会自动创建，防止报错）
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
