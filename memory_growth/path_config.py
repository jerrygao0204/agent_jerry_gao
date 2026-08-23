"""
path_config.py - 统一管理多用户的输入与输出路径
"""

from pathlib import Path


class UserMemoryPathConfig:
    """自动生成并管理指定用户的聊天历史路径与记忆存储路径"""

    def __init__(
        self,
        user_id: str,
        data_root: str = "/workspace/hf-conda/RAG/问答机器人/data",
        memory_root: str = "/workspace/hf-conda/RAG/问答机器人/memory_growth/context/users",
    ):
        self.user_id = user_id

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