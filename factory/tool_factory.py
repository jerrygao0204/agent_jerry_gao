# factory/tool_factory.py
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Type, List, Optional, Tuple
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

        # Level 1 静态元数据定义 (Domain Description)
        self._domain_metadata: Dict[str, str] = {
            "rag_knowledge": "处理用户手册、排错指南、FAQ 等非结构化文档检索",
            "finebi_system": "查询 FineBI 系统仪表板、数据集元数据、权限及定时任务状态",
            "data_analytics": "对业务数据进行二次统计分析、指标计算与报表生成",
            "web_search": "进行网络搜索与公网实时信息查询"
        }

        # Level 2 静态元数据定义 (Package Description)
        self._package_metadata: Dict[str, Dict[str, str]] = {
            "data_analytics": {
                "analytics_pkg": "针对数据集元数据、行数、列数及结构信息的统计分析工具包",
                "chart_pkg": "针对报表渲染、图表生成的可视化工具包"
            }
        }

    def register_tool(self, tool: BaseTool) -> None:
        """注册工具并建立索引 (Register Tool)"""
        if not tool.name:
            raise ValueError("Tool 必须包含非空的 name 属性")

        domain = tool.domain
        package = tool.package

        self._hierarchy.setdefault(domain, {}).setdefault(package, {})[tool.name] = tool
        self._flat_tools[tool.name] = tool

        if domain not in self._domain_metadata:
            self._domain_metadata[domain] = f"处理与 {domain} 相关的业务操作"

        if domain not in self._package_metadata or package not in self._package_metadata[domain]:
            self._package_metadata.setdefault(domain, {})[package] = f"{package} 相关工具包"

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

    # -------------------------------------------------------------
    # Level 2 API: 根据命中 Domain 获取其下属 Package 清单 (Package Level)
    # -------------------------------------------------------------
    def get_packages_summary_by_domains(self, target_domains: List[str]) -> List[Dict[str, str]]:
        """获取指定 Domain 下的所有 Package 元数据，供第二级 Router 决策"""
        packages_summary = []
        for dom in target_domains:
            if dom in self._hierarchy:
                for pkg in self._hierarchy[dom].keys():
                    desc = self._package_metadata.get(dom, {}).get(pkg, f"{pkg} 工具包")
                    packages_summary.append({
                        "domain": dom,
                        "package": pkg,
                        "description": desc
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

    def get_tools_metadata_by_packages(self, target_packages: List[Tuple[str, str]]) -> Tuple[str, str]:
        """导出精确定位后的工具标准 JSON Schema 描述"""
        tools = self.get_tools_by_packages(target_packages)
        if not tools:
            logger.warning(f"⚠️ 未命中任何有效包 [{target_packages}]，降级全量查找。")
            tools = self._flat_tools

        tool_names = ", ".join(tools.keys())
        descriptions = []

        for name, tool in tools.items():
            schema_dict = tool.get_json_schema()
            tool_spec = {
                "name": tool.name,
                "description": tool.description,
                "parameters": schema_dict
            }
            descriptions.append(json.dumps(tool_spec, ensure_ascii=False, indent=2))

        return tool_names, "\n\n".join(descriptions)

    def get_openai_tools_schema_by_packages(
        self, target_packages: Optional[List[Tuple[str, str]]] = None
    ) -> List[Dict[str, Any]]:
        """专供 OpenAI / DeepSeek 原生 Function Calling 使用的标准 API 结构"""
        tools = self.get_tools_by_packages(target_packages) if target_packages else self._flat_tools
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.get_json_schema()
                }
            }
            for tool in tools.values()
        ]

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