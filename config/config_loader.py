# config/config_labder.py
import os
import yaml
import logging
from typing import Dict, Any, Optional

class ConfigLoader:
    """集中式配置文件读取助手 (Absolute Path Guaranteed & List-structure Adaptive)"""

    def __init__(self, config_dir: Optional[str] = None):
        if config_dir is None:
            # 动态计算项目根目录 (Project Root)
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_dir = os.path.join(project_root, "config")

        self.config_dir = os.path.abspath(config_dir)
        self.prompt_hub_path = os.path.join(self.config_dir, "prompt_hub.yaml")
        self.patterns_path = os.path.join(self.config_dir, "patterns.yaml")
        # 新增：数据根目录。优先读环境变量 DATA_ROOT；
        # 未设置时回退为 <项目根目录>/data —— 这跟你现在硬编码的
        # /workspace/hf-conda/RAG/问答机器人/data 是同一个目录，只是换成动态计算，
        # 不设环境变量的情况下行为完全不变，换机器部署时只需要设一个环境变量。
        project_root = os.path.dirname(self.config_dir)
        self.data_root = os.environ.get("DATA_ROOT", os.path.join(project_root, "data"))

    def load_prompts_raw(self) -> Dict[str, Any]:
        """加载原始 YAML 数据结构 (保持 list/dict 原貌)"""
        if not os.path.exists(self.prompt_hub_path):
            logging.warning(f"⚠️ 配置文件不存在: {self.prompt_hub_path}")
            return {}
        with open(self.prompt_hub_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def load_prompts(self) -> Dict[str, str]:
        """解析 prompt_hub.yaml 并转换为 {prompt_name: content} 映射字典"""
        raw_data = self.load_prompts_raw()
        prompt_map = {}

        # 兼容列表结构的 prompts: [{name: ..., content: ...}]
        if "prompts" in raw_data and isinstance(raw_data["prompts"], list):
            for item in raw_data["prompts"]:
                if isinstance(item, dict) and "name" in item:
                    prompt_map[item["name"]] = item.get("content", "")
        # 兼容扁平键值对结构的 prompts: {key: content}
        elif isinstance(raw_data, dict):
            for k, v in raw_data.items():
                if isinstance(v, str):
                    prompt_map[k] = v
                elif isinstance(v, dict) and "content" in v:
                    prompt_map[k] = v.get("content", "")

        return prompt_map

    def load_patterns(self) -> Dict[str, Any]:
        """加载 patterns.yaml"""
        if not os.path.exists(self.patterns_path):
            logging.warning(f"⚠️ 配置文件不存在: {self.patterns_path}")
            return {}
        with open(self.patterns_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def get_prompt(self, key: str, default: str = "") -> str:
        """根据 name 提取具体的 Prompt 模板 content"""
        prompts_map = self.load_prompts()
        content = prompts_map.get(key)
        if content is None:
            logging.error(f"❌ 未能在 {self.prompt_hub_path} 中找到 name='{key}' 的 Prompt 配置！")
            return default
        return content

    def get_pattern(self, key: str) -> Any:
        """获取具体的 Pattern 规则或白名单/枚举"""
        patterns = self.load_patterns()
        return patterns.get(key, None)

    def get_config_dict(self) -> Dict[str, Any]:
        """导出全局绝对路径配置项 (Export Absolute Paths Config)"""
        return {
            "prompts_hub_path": self.prompt_hub_path,
            "patterns_path": self.patterns_path,
            "config_dir": self.config_dir,
            "data_root": self.data_root, 
        }

# 全局单例对象供快捷调用
config_loader = ConfigLoader()

# 全局默认配置对象
DEFAULT_CONFIG = config_loader.get_config_dict()