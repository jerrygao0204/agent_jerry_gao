# memory/entity_memory.py

import os
import re
import yaml
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("EntityMemory")

class EntityMemory:
    def __init__(self, user_id: str = "default", config_path: Optional[str] = None):
        """
        修正路径计算逻辑：根据 config_path 自动推导全局与个人规则路径
        """
        self.user_id = user_id
        
        # 1. 确定全局 patterns.yaml 绝对路径
        if config_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            config_path = os.path.join(project_root, "config", "patterns.yaml")

        self.global_config_path = config_path
        
        # 2. 根据 global_config_path 的所在目录推导出 users 文件夹及 账号_patterns.yaml 路径
        config_dir = os.path.dirname(self.global_config_path)
        self.user_config_path = os.path.join(config_dir, "users", f"{self.user_id}_patterns.yaml")

        self.patterns: Dict[str, Any] = {}
        self.entities: Dict[str, Any] = {}
        
        # 3. 触发加载与拼接
        self._load_and_merge_patterns()

    def _load_and_merge_patterns(self):
        """核心逻辑：加载 patterns.yaml 并叠加 账号_patterns.yaml"""
        merged_patterns = {}

        # 1. 读取全局公共规则 (patterns.yaml)
        if os.path.exists(self.global_config_path):
            try:
                with open(self.global_config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                merged_patterns.update(cfg.get("entity_patterns", {}))
                logger.info(f"🌐 [EntityMemory] 已成功加载全局规则: {self.global_config_path}")
            except Exception as e:
                logger.error(f"⚠️ 读取全局 patterns.yaml 失败: {e}")

        # 2. 读取个人独立规则 (账号_patterns.yaml)
        if os.path.exists(self.user_config_path):
            try:
                with open(self.user_config_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                user_patterns = cfg.get("entity_patterns", {})
                merged_patterns.update(user_patterns)
                logger.info(f"🔒 [EntityMemory] 已成功叠加用户 [{self.user_id}] 规则: {self.user_config_path}")
            except Exception as e:
                logger.error(f"⚠️ 读取用户 [{self.user_id}] 规则文件失败: {e}")

        self.patterns = merged_patterns

    def extract_and_update(self, text: str) -> Dict[str, Any]:
        """使用拼接后的规则匹配并提炼实体"""
        extracted = {}
        for key, conf in self.patterns.items():
            if "pattern" in conf:
                match = re.search(conf["pattern"], text)
                if match:
                    extracted[key] = match.group(conf.get("group", 1))
            elif "keywords" in conf:
                found_kw = [kw for kw in conf["keywords"] if kw.lower() in text.lower()]
                if found_kw:
                    extracted[key] = found_kw if len(found_kw) > 1 else found_kw[0]

        if extracted:
            self.entities.update(extracted)
        return extracted

    def save_custom_user_pattern(self, entity_key: str, pattern_conf: Dict[str, Any]):
        """写回账号专属的 账号_patterns.yaml"""
        user_dir = os.path.dirname(self.user_config_path)
        os.makedirs(user_dir, exist_ok=True)

        user_cfg = {"entity_patterns": {}}
        if os.path.exists(self.user_config_path):
            with open(self.user_config_path, "r", encoding="utf-8") as f:
                user_cfg = yaml.safe_load(f) or {"entity_patterns": {}}

        user_cfg["entity_patterns"][entity_key] = pattern_conf

        with open(self.user_config_path, "w", encoding="utf-8") as f:
            yaml.dump(user_cfg, f, allow_unicode=True, sort_keys=False)

        # 重新加载生效
        self._load_and_merge_patterns()

    def get_entities(self) -> Dict[str, Any]:
        return self.entities.copy()

    def set_entities(self, entities: Dict[str, Any]):
        self.entities = entities.copy()

    def clear(self):
        self.entities.clear()