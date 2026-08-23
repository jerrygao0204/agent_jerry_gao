# agent/compliance.py

import re
import os
import yaml
import logging
from typing import Dict, Any, Tuple, List, Optional

logger = logging.getLogger("ComplianceChecker")

class ComplianceChecker:
    """
    合规与安全审计引擎 (Compliance Checker)
    提供双重审计机制：
    1. 静态正则过滤与隐私数据脱敏 (Static Regex & Sanitization)
    2. LLM 语义合规二次审查 (LLM-based Compliance Audit)
    """

    def __init__(self, patterns_yaml_path: Optional[str] = None):
        if patterns_yaml_path is None:
            # 默认指向 config/patterns.yaml
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            patterns_yaml_path = os.path.join(base_dir, "config", "patterns.yaml")

        self.patterns_path = patterns_yaml_path
        self.sensitive_patterns: List[Dict[str, Any]] = []
        self.masking_patterns: List[Dict[str, Any]] = []
        self.fallback_responses: Dict[str, str] = {}
        
        self.load_config()

    def load_config(self):
        """加载 patterns.yaml 中的合规规则"""
        if not os.path.exists(self.patterns_path):
            logger.warning(f"⚠️ [Compliance] 未找到合规配置文件: {self.patterns_path}，将采用默认空规则。")
            return

        try:
            with open(self.patterns_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f) or {}
                rules = data.get("compliance_rules", {})
                
                self.sensitive_patterns = rules.get("sensitive_patterns", [])
                self.masking_patterns = rules.get("masking_patterns", [])
                self.fallback_responses = rules.get("fallback_responses", {
                    "HIGH": "⚠️ [安全拦截] 该请求触发高风险安全策略，已停止输出。",
                    "MEDIUM": "⚠️ [合规拦截] 该请求包含敏感信息，已停止输出。"
                })
            logger.info(f"✅ [Compliance] 成功装载 {len(self.sensitive_patterns)} 条敏感正则与 {len(self.masking_patterns)} 条脱敏规则。")
        except Exception as e:
            logger.error(f"❌ [Compliance] 加载配置失败: {e}")

    # -------------------------------------------------------------------------
    # 第一层：静态正则黑名单扫描与数据脱敏
    # -------------------------------------------------------------------------
    def check_static_rules(self, text: str) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        扫描文本是否触发敏感正则规则
        Returns:
            (is_safe: bool, risk_level: Optional[str], hit_rule_name: Optional[str])
        """
        if not text:
            return True, None, None

        for item in self.sensitive_patterns:
            rule_name = item.get("name", "unknown")
            pattern = item.get("pattern", "")
            risk_level = item.get("risk_level", "HIGH")

            if re.search(pattern, text):
                logger.warning(f"🛡️ [Compliance] 静态规则拦截! 触发规则: [{rule_name}] 风险等级: [{risk_level}]")
                return False, risk_level, rule_name

        return True, None, None

    def sanitize_text(self, text: str) -> str:
        """对文本进行数据脱敏（如手机号、邮箱等隐私掩码处理）"""
        if not text:
            return text

        sanitized_text = text
        for mask_rule in self.masking_patterns:
            pattern = mask_rule.get("pattern", "")
            replace_with = mask_rule.get("replace_with", "****")
            sanitized_text = re.sub(pattern, replace_with, sanitized_text)

        return sanitized_text

    # -------------------------------------------------------------------------
    # 第二层：LLM 语义合规二次审查 (选配/高安全等级场景)
    # -------------------------------------------------------------------------
    def check_llm_compliance(self, text: str, llm_client: Any) -> Tuple[bool, str]:
        """
        基于大模型判断生成的回复是否包含非法/破坏性/隐私泄露语义
        """
        audit_prompt = f"""你是一名严格的企业数据安全与合规审计员。
            请评估以下回答是否存在以下违规风险：
            1. 包含可直接执行的破坏性系统命令或数据库删除指令。
            2. 泄露了系统底层数据库的明文账号密码、密钥、敏感 IP 地址。
            3. 包含不符合企业道德规范或违法的内容。

            [待审计文本]:
            {text}

            请严格按 JSON 格式回答：
            {{"safe": true/false, "reason": "若不安全，简要说明原因"}}
            仅输出 JSON 字符串，不要包含任何 markdown 块或多余解释。"""

        try:
            response = llm_client.generate(audit_prompt)
            # 提取 JSON 内容
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(0))
                is_safe = result.get("safe", True)
                reason = result.get("reason", "")
                return is_safe, reason
        except Exception as e:
            logger.warning(f"⚠️ [Compliance] LLM 合规审查调用异常，默认放行: {e}")

        return True, ""

    # -------------------------------------------------------------------------
    # 统一管道入口：审计与降级处理
    # -------------------------------------------------------------------------
    def audit_and_sanitize(
        self, 
        text: str, 
        llm_client: Optional[Any] = None, 
        enable_llm_audit: bool = False
    ) -> Dict[str, Any]:
        """
        全流程合规检查：静态拦截 -> LLM 审计 -> 数据脱敏 -> 降级替换
        """
        # 1. 静态规则审计
        is_safe, risk_level, hit_rule = self.check_static_rules(text)
        if not is_safe:
            fallback_msg = self.fallback_responses.get(
                risk_level, 
                "⚠️ [安全拦截] 您的请求或结果包含高风险内容，已被系统过滤。"
            )
            return {
                "passed": False,
                "sanitized_text": fallback_msg,
                "blocked_by": f"static_rule:{hit_rule}",
                "risk_level": risk_level
            }

        # 2. LLM 深度审计 (可选)
        if enable_llm_audit and llm_client is not None:
            llm_safe, reason = self.check_llm_compliance(text, llm_client)
            if not llm_safe:
                logger.warning(f"🛡️ [Compliance] LLM 拦截! 原因: {reason}")
                return {
                    "passed": False,
                    "sanitized_text": f"⚠️ [合规审计拦截] 经系统深度审查，该回复包含潜在不合规内容 ({reason})。",
                    "blocked_by": "llm_compliance_audit",
                    "risk_level": "HIGH"
                }

        # 3. 数据脱敏处理
        sanitized_output = self.sanitize_text(text)

        return {
            "passed": True,
            "sanitized_text": sanitized_output,
            "blocked_by": None,
            "risk_level": "SAFE"
        }