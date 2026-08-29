# factory/tool_factory.py
import json
import logging
from abc import ABC, abstractmethod
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
        self._domain_metadata: Dict[str, str] = {}

        # Level 2 动态元数据定义 (Package Description) -> 置空
        self._package_metadata: Dict[str, Dict[str, str]] = {}

    def register_tool(self, tool: BaseTool) -> None:
        """注册工具并建立索引 (Register Tool)"""
        if not tool.name:
            raise ValueError("Tool 必须包含非空的 name 属性")

        domain = tool.domain
        package = tool.package

        self._hierarchy.setdefault(domain, {}).setdefault(package, {})[tool.name] = tool
        self._flat_tools[tool.name] = tool

        # 仅当没有从配置文件/显式 API 注册过元数据时，才写入兜底描述
        self._domain_metadata.setdefault(domain, f"处理与 {domain} 相关的业务操作")
        self._package_metadata.setdefault(domain, {}).setdefault(package, f"{package} 工具包")

        logger.info(f"🛠️ [ToolFactory] 注册成功: [{domain} -> {package} -> {tool.name}]")

    def register_domain_meta(self, domain: str, description: str) -> None:
        """动态配置 Level 1 领域描述 (Domain Metadata)"""
        self._domain_metadata[domain] = description

    def register_package_meta(self, domain: str, package: str, description: str) -> None:
        """动态配置 Level 2 工具包描述 (Package Metadata)"""
        self._package_metadata.setdefault(domain, {})[package] = description

    # -------------------------------------------------------------
    # Level 1 API: 获取全局 Domain 清单 (Domain Level)
    # -------------------------------------------------------------
    def get_domains_summary(self) -> List[Dict[str, str]]:
        """获取 Level 1 领域元数据，供第一级 Router 决策"""
        return [
            {
                "domain": domain,
                "description": self._domain_metadata.get(domain, "通用未定义领域")
            }
            for domain in self._hierarchy.keys()
        ]

    def get_packages_summary_by_domains(self, target_domains: List[str]) -> List[Dict[str, Any]]:
        """获取指定 Domain 下的所有 Package 元数据，供第二级 Router 决策"""
        packages_summary = []
        for dom in target_domains:
            if dom in self._hierarchy:
                for pkg, tools in self._hierarchy[dom].items():
                    desc = self._package_metadata.get(dom, {}).get(pkg, f"{pkg} 工具包")
                    
                    # 📌 补全工具列表解析
                    tools_summary = []
                    # 支持 tools 为列表（[tool_name1, tool_name2]）或字典（{tool_name: tool_obj}）
                    tool_items = tools.items() if isinstance(tools, dict) else [(t, None) for t in tools]
                    
                    for tool_name, tool_val in tool_items:
                        # 尝试从对象、注册表或默认获取描述
                        tool_desc = "通用工具"
                        if hasattr(tool_val, "description"):
                            tool_desc = tool_val.description
                        elif hasattr(self, "get_tool") and callable(getattr(self, "get_tool")):
                            tool_obj = self.get_tool(tool_name)
                            if tool_obj and hasattr(tool_obj, "description"):
                                tool_desc = tool_obj.description
                        
                        tools_summary.append({
                            "name": tool_name,
                            "description": tool_desc
                        })

                    packages_summary.append({
                        "domain": dom,
                        "package": pkg,
                        "description": desc,
                        "tools": tools_summary  # 📌 取消注释并注入解析好的工具列表
                    })
        return packages_summary
    
    # -------------------------------------------------------------
    # Level 3 API: 根据命中的 Packages 抽取原子工具集 (Tool Level)
    # -------------------------------------------------------------
    def get_tools_by_packages(self, target_packages: List[Tuple[str, str]]) -> Dict[str, BaseTool]:
        """按 (domain, package) 二元组精确定位并提取 Tool 集合"""
        scoped_tools: Dict[str, BaseTool] = {}
        for dom, pkg in target_packages:
            if dom in self._hierarchy and pkg in self._hierarchy[dom]:
                scoped_tools.update(self._hierarchy[dom][pkg])
        return scoped_tools

    def get_tools_metadata_by_packages(
        self, 
        target_packages: List[Tuple[str, str]], 
        as_json_string: bool = False
    ) -> Tuple[str, Union[str, List[Dict[str, Any]]]]:
        """
        导出符合 OpenAI Function Call 标准规范的工具 Schema 描述
        """
        tools = self.get_tools_by_packages(target_packages)
        if not tools:
            logger.warning(f"⚠️ 未命中任何有效包 [{target_packages}]，降级全量查找。")
            tools = self._flat_tools

        tool_names = ", ".join(tools.keys())
        formatted_tool_specs = []

        for name, tool in tools.items():
            # 获取 Pydantic 导出的 parameters schema
            schema_dict = tool.get_json_schema()

            # 📌 严格按照 OpenAI 标准格式封装
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

    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self._flat_tools.get(name)


# 全局三级工具工厂单例 (Global Tool Factory Singleton)
tool_factory = HierarchicalToolFactory()


if __name__ == "__main__":
    class DatasetSummaryInput(BaseModel):
        dataset_name: str = Field(description="目标 FineBI 数据集的精确表名称，例如 'sales_2026'")

    class DatasetSummaryTool(BaseTool):
        name = "get_dataset_summary"
        description = "获取指定数据集的行数、列数及字段数据摘要信息"
        domain = "data_analytics"
        package = "analytics_pkg"
        args_schema = DatasetSummaryInput

        def run(self, dataset_name: str, **kwargs):
            return f"数据集 [{dataset_name}] 摘要信息..."

    factory = HierarchicalToolFactory()
    factory.register_tool(DatasetSummaryTool())

    print("\n========== 1. Level 1: LLM 获取 Domain 概要 ==========")
    print(factory.get_domains_summary())

    print("\n========== 2. Level 2: LLM 获取 Package 概要 ==========")
    print(factory.get_packages_summary_by_domains(["data_analytics"]))

    print("\n========== 3. Level 3: LLM 提取最终标准 Tool Schema ==========")
    names, desc = factory.get_tools_metadata_by_packages([("data_analytics", "analytics_pkg")])
    print("工具名称列表:", names)
    print(desc)

