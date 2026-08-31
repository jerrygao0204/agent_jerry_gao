# factory/tool_factory.py
import sys
from pathlib import Path

# 📌 自动将当前文件所在的上一级目录（即项目根目录）加入 sys.path
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import logging
from abc import ABC, abstractmethod
import importlib
import os
import yaml
from typing import Dict, Any, Type, List, Optional, Tuple, Union
from pydantic import BaseModel, Field

logger = logging.getLogger("ToolFactory")

# ==========================================
# 1. 工具基类设计 (Level 3 Atom Tool)
# ==========================================
class BaseTool(ABC):
    """Agent 原子工具抽象基类 (Level 3)"""
    name: str = ""
    description: str = ""
    domain: str = "general"        # Level 1: 业务领域 (如: rag_knowledge, finebi_system, data_analytics)
    package: str = "default_pkg"   # Level 2: 工具包分类 (如: metadata_pkg, search_pkg)
    args_schema: Optional[Type[BaseModel]] = None
    is_read_only: bool = True      # 默认只读安全保障
    role_whitelist: Optional[List[str]] = None  # None 表示继承 Domain，[] 表示全员可访问

    def __init__(self):
        # 强制确保实例属性同步类属性
        self.name = getattr(self, "name", self.__class__.name)
        self.description = getattr(self, "description", self.__class__.description)
        self.domain = getattr(self, "domain", self.__class__.domain)
        self.package = getattr(self, "package", self.__class__.package)
        self.role_whitelist = getattr(self, "role_whitelist", self.__class__.role_whitelist)
        
    @abstractmethod
    def run(self, **kwargs) -> Any:
        """工具的核心执行逻辑"""
        pass

    
    def get_json_schema(self) -> Dict[str, Any]:
        """获取干净的标准 JSON Schema 元数据"""
        if not self.args_schema:
            return {"type": "object", "properties": {}}

        if hasattr(self.args_schema, "model_json_schema"):
            schema = self.args_schema.model_json_schema()
        elif hasattr(self.args_schema, "schema"):
            schema = self.args_schema.schema()
        else:
            schema = {"type": "object", "properties": {}}

        # 1. 移除顶级 title
        schema.pop("title", None)

        # 2. 递归清理字段内部不必要的 title (进一步节约 Token)
        if "properties" in schema:
            for prop in schema["properties"].values():
                if isinstance(prop, dict):
                    prop.pop("title", None)

        return schema


# ==========================================
# 2. 三级分级工具工厂 (Hierarchical Tool Factory)
# ==========================================
class HierarchicalToolFactory:
    """三级分级工具工厂：支持按 Domain -> Package -> Tool 三级按需加载与路由"""

    def __init__(self):
        # 内部三级索引结构: {domain: {package: {tool_name: BaseTool}}}
        self._hierarchy: Dict[str, Dict[str, Dict[str, BaseTool]]] = {}
        self._flat_tools: Dict[str, BaseTool] = {}
        

        # Level 1 动态元数据定义 (Domain Description) -> 置空
        # self._domain_metadata: Dict[str, str] = {}
        self._domain_metadata: Dict[str, Dict[str, Any]] = {}

        # Level 2 动态元数据定义 (Package Description) -> 置空
        self._package_metadata: Dict[str, Dict[str, str]] = {}
        

    def _is_tool_visible(self, tool: BaseTool, user_role: Optional[str]) -> bool:
        """
        判断指定角色对工具的可见性
        1. 若未指定 user_role（内部调用），放行
        2. 若工具配置了 role_whitelist (不为 None)：
           - 为 [] 说明公开；包含 user_role 则放行，否则拒绝
        3. 若工具未配置 role_whitelist (为 None)：继承 Domain 级 role_whitelist
           - 若 Domain 配置了白名单，必须包含 user_role
        4. 均未配置，默认放行
        """
        if not user_role or user_role in ["None", "null"]:
            return True

        # 1. 优先校验：工具级白名单 (显式配置)
        if tool.role_whitelist is not None:
            if not tool.role_whitelist:  # [] 空列表代表全员开放
                return True
            is_allowed = user_role in tool.role_whitelist
            # logging.info(f"🔍 [工具级校验] 工具:{tool.name} | 白名单:{tool.role_whitelist} | 当前角色:{user_role} -> 结果:{is_allowed}")
            return is_allowed

        # 2. 次要校验：继承 Domain 级白名单
        domain_meta = self._domain_metadata.get(tool.domain, {})
        domain_roles = domain_meta.get("role_whitelist", [])
        if domain_roles:
            is_allowed = user_role in domain_roles
            # logging.info(f"🔍 [Domain级校验] 工具:{tool.name} | Domain:{tool.domain} | 白名单:{domain_roles} | 当前角色:{user_role} -> 结果:{is_allowed}")
            return is_allowed

        # 3. 均未配置，默认放行
        return True

    def register_tool(self, tool: BaseTool) -> None:
        """注册工具并建立索引 (带 RBAC 防覆盖保护)"""
        if not tool.name:
            raise ValueError("Tool 必须包含非空的 name 属性")

        domain = tool.domain
        package = tool.package

        # 📌 核心修复：检查是否已有被 YAML 加载并锁定白名单的同名工具
        existing_tool = self._flat_tools.get(tool.name)
        if existing_tool and existing_tool.role_whitelist is not None:
            # 如果新注册的工具 role_whitelist 为 None，说明是原生 Python 代码二次注册
            # 必须继承并保留 YAML 已经锁定的 role_whitelist！
            if tool.role_whitelist is None:
                tool.role_whitelist = existing_tool.role_whitelist
                logger.info(f"🛡️ [RBAC 阻断保护] 工具 [{tool.name}] 被二次注册，已强制继承 YAML 白名单: {tool.role_whitelist}")

        self._hierarchy.setdefault(domain, {}).setdefault(package, {})[tool.name] = tool
        self._flat_tools[tool.name] = tool

        # 仅当没有从配置文件/显式 API 注册过元数据时，才写入兜底描述
        if domain not in self._domain_metadata:
            self._domain_metadata[domain] = {
                "description": f"处理与 {domain} 相关的业务操作",
                "role_whitelist": []
            }
        self._package_metadata.setdefault(domain, {}).setdefault(package, f"{package} 工具包")

        logger.info(f"🛠️ [ToolFactory] 注册成功: [{domain} -> {package} -> {tool.name}]")
        
    def register_domain_meta(
        self, 
        domain: str, 
        description: str, 
        role_whitelist: Optional[List[str]] = None
    ) -> None:
        """动态配置 Level 1 领域描述与白名单 (Domain Metadata)"""
        self._domain_metadata[domain] = {
            "description": description,
            "role_whitelist": role_whitelist or []
        }

    def register_package_meta(self, domain: str, package: str, description: str) -> None:
        """动态配置 Level 2 工具包描述 (Package Metadata)"""
        self._package_metadata.setdefault(domain, {})[package] = description

    # -------------------------------------------------------------
    # Level 1 API: 获取全局 Domain 清单 (Domain Level)
    # -------------------------------------------------------------
    def get_domains_summary(self,user_role: Optional[str] = None) -> List[Dict[str, str]]:
        """获取 Level 1 领域元数据，供第一级 Router 决策"""
        visible_domains = []
        
        for domain, packages in self._hierarchy.items():
            domain_meta = self._domain_metadata.get(domain, {})
            domain_roles = domain_meta.get("role_whitelist", [])

            # 检查 Domain 级别权限
            if user_role and domain_roles and user_role not in domain_roles:
                continue

            # 进一步检查 Domain 下是否有至少一个工具对此角色可见
            has_visible_tool = False
            for pkg, tools in packages.items():
                for tool in tools.values():
                    if self._is_tool_visible(tool, user_role):
                        has_visible_tool = True
                        break
                if has_visible_tool:
                    break

            if has_visible_tool:
                visible_domains.append({
                    "domain": domain,
                    "description": domain_meta.get("description", "通用未定义领域")
                })

        return visible_domains
    
    def get_packages_summary_by_domains(
        self, 
        target_domains: List[str], 
        user_role: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """获取指定 Domain 下的 Package 元数据，带角色鉴权与动态裁减 (供 Level 2 Router 决策)"""
        packages_summary = []
        
        for dom in target_domains:
            if dom not in self._hierarchy:
                continue
                
            for pkg, tools in self._hierarchy[dom].items():
                desc = self._package_metadata.get(dom, {}).get(pkg, f"{pkg} 工具包")
                tools_summary = []
                
                # 兼容处理：支持 tools 为字典 {name: obj} 或列表/集合 [obj_or_name]
                tool_items = tools.items() if isinstance(tools, dict) else [(getattr(t, "name", str(t)), t) for t in tools]
                
                for tool_name, tool_val in tool_items:
                    # 1. 解析获得真实的 BaseTool 对象
                    tool_obj = tool_val if isinstance(tool_val, BaseTool) else self.get_tool(tool_name)
                    
                    # 2. 🔐 核心鉴权逻辑：基于 user_role 校验工具可见性
                    if tool_obj and not self._is_tool_visible(tool_obj, user_role):
                        continue
                    
                    # 3. 提取工具描述
                    tool_desc = "通用工具"
                    if tool_obj and hasattr(tool_obj, "description"):
                        tool_desc = tool_obj.description
                    elif hasattr(tool_val, "description"):
                        tool_desc = tool_val.description
                        
                    tools_summary.append({
                        "name": tool_name,
                        "description": tool_desc
                    })

                # 4. 🧹 动态裁减：仅当该 Package 下存在当前角色可见的工具时才暴露该 Package
                if tools_summary:
                    packages_summary.append({
                        "domain": dom,
                        "package": pkg,
                        "description": desc,
                        "tools": tools_summary
                    })
                    
        return packages_summary
    
    # -------------------------------------------------------------
    # Level 3 API: 根据命中的 Packages 抽取原子工具集 (Tool Level)
    # -------------------------------------------------------------
    def get_tools_by_packages(
        self, 
        target_packages: List[Tuple[str, str]], 
        user_role: Optional[str] = None  # 📌 修改参数名: role -> user_role
    ) -> Dict[str, BaseTool]:
        """按 (domain, package) 二元组定位，并严格基于 user_role 过滤 Tool 集合"""
        scoped_tools: Dict[str, BaseTool] = {}
        for dom, pkg in target_packages:
            if dom in self._hierarchy and pkg in self._hierarchy[dom]:
                for tool_name, tool_obj in self._hierarchy[dom][pkg].items():
                    if self._is_tool_visible(tool_obj, user_role):
                        scoped_tools[tool_name] = tool_obj
        return scoped_tools

    def get_tools_metadata_by_packages(
        self, 
        target_packages: List[Tuple[str, str]], 
        as_json_string: bool = False,
        user_role: Optional[str] = None  # 📌 修改参数名: role -> user_role
    ) -> Tuple[str, Union[str, List[Dict[str, Any]]]]:
        """
        导出符合 OpenAI Function Call 标准规范的工具 Schema 描述
        """
        tools = self.get_tools_by_packages(target_packages, user_role=user_role)
        if not tools:
            logger.warning(f"⚠️ 角色 [{user_role}] 请求包 [{target_packages}] 未命中或权限不足，返回空工具列表。")
            return "", json.dumps([]) if as_json_string else []

        tool_names = ", ".join(tools.keys())
        formatted_tool_specs = []

        for name, tool in tools.items():
            schema_dict = tool.get_json_schema()
            tool_spec = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": schema_dict
                }
            }
            formatted_tool_specs.append(tool_spec)

        if as_json_string:
            return tool_names, json.dumps(formatted_tool_specs, ensure_ascii=False, indent=2)
        return tool_names, formatted_tool_specs

    def get_tool(self, name: str, user_role: Optional[str] = None) -> Optional[BaseTool]: # 📌 修改参数名: role -> user_role
        tool = self._flat_tools.get(name)
        if tool and self._is_tool_visible(tool, user_role):
            return tool
        return None


# 全局三级工具工厂单例 (Global Tool Factory Singleton)
tool_factory = HierarchicalToolFactory()

def load_tools_from_yaml(yaml_path: str = "", factory: HierarchicalToolFactory = tool_factory) -> None:
    """自动解析最新的 tools.yaml 并动态注册 Domains, Packages 和 Tools"""
    if not os.path.exists(yaml_path):
        logger.error(f"❌ 配置文件不存在: {yaml_path}")
        return

    with open(yaml_path, "r", encoding="utf-8") as f:
        config: Dict[str, Any] = yaml.safe_load(f) or {}

    # 1. 解析 domains 及嵌套 packages
    domains_config = config.get("domains", {})
    for domain_name, domain_meta in domains_config.items():
        factory.register_domain_meta(
            domain=domain_name,
            description=domain_meta.get("description", ""),
            role_whitelist=domain_meta.get("role_whitelist", [])
        )
        
        # 提取嵌套 packages
        packages = domain_meta.get("packages", {})
        for pkg_name, pkg_desc in packages.items():
            factory.register_package_meta(
                domain=domain_name,
                package=pkg_name,
                description=pkg_desc
            )

    # 2. 动态反射加载 tools 列表
    tools_list = config.get("tools", [])
    for tool_cfg in tools_list:
        if not tool_cfg.get("enabled", True):
            logger.info(f"⏭️ 工具 [{tool_cfg.get('name')}] 已禁用 (enabled=false)，跳过加载。")
            continue

        module_path = tool_cfg.get("module")
        class_name = tool_cfg.get("class")

        try:
            module = importlib.import_module(module_path)
            tool_class = getattr(module, class_name)
            tool_instance: BaseTool = tool_class()

            # 📌 明确重写 YAML 中配置的元数据
            tool_instance.name = tool_cfg.get("name", tool_instance.name)
            tool_instance.domain = tool_cfg.get("domain", tool_instance.domain)
            tool_instance.package = tool_cfg.get("package", tool_instance.package)

            # 📌 核心修复点：显式覆盖 role_whitelist，若 YAML 未配置工具级则赋值为 None
            if "role_whitelist" in tool_cfg:
                tool_instance.role_whitelist = tool_cfg["role_whitelist"]
            else:
                tool_instance.role_whitelist = None

            # 注册到工厂
            factory.register_tool(tool_instance)

            # 📌 再次二次绑定，防止 register_tool 内部重置属性
            if "role_whitelist" in tool_cfg:
                factory._flat_tools[tool_instance.name].role_whitelist = tool_cfg["role_whitelist"]
                
            logger.info(f"🔒 [RBAC 装载] 工具 [{tool_instance.name}] 白名单已锁定为: {factory._flat_tools[tool_instance.name].role_whitelist}")

        except Exception as e:
            logger.error(f"❌ 动态加载工具失败 [{module_path}.{class_name}]: {str(e)}")

if __name__ == "__main__":
    # 配置日志输出格式
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # 执行从 tools.yaml 动态装载
    try:
        yaml_path = Path(__file__).resolve().parent.parent /"config"/ "tools.yaml"
        load_tools_from_yaml(str(yaml_path), tool_factory)
        
        
        print("\n========== 1. [普通用户 user] Level 1 可见 Domain ==========")
        print(tool_factory.get_domains_summary(user_role="user"))

        print("\n========== 2. [数据分析师 analyst] Level 2 可见 Packages ==========")
        print(tool_factory.get_packages_summary_by_domains(["finebi_system", "rag_knowledge"], user_role="analyst"))

        print("\n========== 3. [管理员 admin] Level 3 工具 Schema 导出 ==========")
        names, _ = tool_factory.get_tools_metadata_by_packages([("finebi_system", "metadata_pkg")], user_role="admin")
        print("Admin 检索到工具:", names)

        print("\n========== 4. [数据分析师 analyst] Level 3 工具 Schema 导出 ==========")
        names, _ = tool_factory.get_tools_metadata_by_packages([("finebi_system", "metadata_pkg")], user_role="analyst")
        print("Analyst 检索到工具:", names)

    except FileNotFoundError:
        logger.warning("未找到 tools.yaml 文件，请确保运行目录下存在配置文件。")
