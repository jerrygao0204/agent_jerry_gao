# factory/tool_factory.py
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Type, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("ToolFactory")

# ==========================================
# 1. 工具基类设计
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


# ==========================================
# 2. 三级分级工具工厂
# ==========================================
class HierarchicalToolFactory:
    """三级分级工具工厂：支持按 Domain / Package 按需加载与路由"""

    def __init__(self):
        # 内部索引结构: {domain: {package: {tool_name: BaseTool}}}
        self._hierarchy: Dict[str, Dict[str, Dict[str, BaseTool]]] = {}
        self._flat_tools: Dict[str, BaseTool] = {}

        # Level 1 领域的静态描述元数据 (用于第一级 Router 判断)
        self._domain_metadata: Dict[str, str] = {
            "rag_knowledge": "处理 FineBI 用户手册、排错指南、FAQ 等非结构化文档检索,創建預警用戶等",
            "finebi_system": "查询 FineBI 系统仪表板、数据集元数据、权限及定时任务状态",
            "data_analytics": "对业务数据进行二次统计分析、指标计算与报表生成"
        }

    def register_tool(self, tool: BaseTool) -> None:
        """注册工具并建立索引 (Register Tool)"""
        if not tool.name:
            raise ValueError("Tool 必须包含非空的 name 属性")

        domain = tool.domain
        package = tool.package

        self._hierarchy.setdefault(domain, {}).setdefault(package, {})[tool.name] = tool

        self._flat_tools[tool.name] = tool
        
        logger.info(f"🛠️ [ToolFactory] 注册成功: [{domain} -> {package} -> {tool.name}]")

    def register_domain_meta(self, domain: str, description: str) -> None:
        """动态配置 Level 1 领域描述 (Register Domain Metadata)"""
        self._domain_metadata[domain] = description

    # -------------------------------------------------------------
    # 第一级过滤：获取全局领域清单 (极低 Token，仅供 Router 使用)
    # -------------------------------------------------------------
    def get_domains_summary(self) -> List[Dict[str, str]]:
        """获取所有可用 Level 1 领域及其语义描述 (Domain Summary)"""
        return [
            {
                "domain": domain,
                "description": self._domain_metadata.get(domain, "通用未定义领域")
            }
            for domain in self._hierarchy.keys()
        ]

    # -------------------------------------------------------------
    # 第二/三级按需抽取：根据命中的 Domains 抽取子集工具
    # -------------------------------------------------------------
    def get_tools_by_domains(self, target_domains: List[str]) -> Dict[str, BaseTool]:
        """依据选定的 Domain 列表筛选出对应的 Tool 子集 (Scoped Tools)"""
        scoped_tools: Dict[str, BaseTool] = {}
        for dom in target_domains:
            if dom in self._hierarchy:
                for pkg in self._hierarchy[dom].values():
                    scoped_tools.update(pkg)
        return scoped_tools

    # -------------------------------------------------------------
    # 供 ReAct Agent 使用：导出文本格式的 Tools Description
    # -------------------------------------------------------------
    def get_tools_metadata_by_domains(self, target_domains: List[str]) -> tuple[str, str]:
        """获取筛选后工具的 Name 列表与 Schema 描述文本 (Scoped Metadata)"""
        tools = self.get_tools_by_domains(target_domains)
        
        # 若未命中任何领域，降级容错暴露全局工具或空列表
        if not tools:
            logger.warning(f"⚠️ 未命中任何有效领域 [{target_domains}]，降级全量查找。")
            tools = self._flat_tools

        tool_names = ", ".join(tools.keys())
        descriptions = []
        
        for name, tool in tools.items():
            schema_str = "{}"
            if tool.args_schema:
                if hasattr(tool.args_schema, "model_json_schema"):
                    schema_str = json.dumps(tool.args_schema.model_json_schema(), ensure_ascii=False)
                elif hasattr(tool.args_schema, "schema"):
                    schema_str = json.dumps(tool.args_schema.schema(), ensure_ascii=False)
            
            descriptions.append(f"- {name}: [{tool.domain}/{tool.package}] {tool.description}\n  参数 Schema: {schema_str}")
            
        return tool_names, "\n".join(descriptions)

    # def get_tool(self, name: str) -> Optional[BaseTool]:
    #     """按名称获取工具实例 (Get Tool Instance)"""
    #     return self._flat_tools.get(name)

    # -------------------------------------------------------------
    # 补全保留：供 MCP / Function Calling / API 对接使用 (JSON Schema)
    # -------------------------------------------------------------
    def get_mcp_tools_schema_by_domains(self, target_domains: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        按照标准 MCP / OpenAI Tools 格式导出 JSON Schema。
        如果 target_domains 为空，则全量导出，否则按领域导出。
        """
        tools = self.get_tools_by_domains(target_domains) if target_domains else self._flat_tools
        mcp_tools = []
        
        for tool in tools.values():
            parameters = {}
            if tool.args_schema:
                if hasattr(tool.args_schema, "model_json_schema"):
                    parameters = tool.args_schema.model_json_schema()
                elif hasattr(tool.args_schema, "schema"):
                    parameters = tool.args_schema.schema()
            
            mcp_tools.append({
                "name": tool.name,
                "description": tool.description,
                "domain": tool.domain,
                "package": tool.package,
                "is_read_only": tool.is_read_only,
                "inputSchema": parameters  # 符合 MCP Tool 标注规范
            })
            
        return mcp_tools

    def get_tool(self, name: str) -> Optional[BaseTool]:
        return self._flat_tools.get(name)

# 全局三级工具工厂单例
tool_factory = HierarchicalToolFactory()

if __name__ == "__main__":
    # 1. 模拟定义 2 个不同领域的工具
    class RAGSearchInput(BaseModel):
        query: str = Field(description="查询列表")

    class RAGSearchTool(BaseTool):
        name = "search_docs"
        description = "检索 FineBI 文档"
        domain = "rag_knowledge"         # Level 1
        package = "knowledge_search_pkg" # Level 2
        args_schema = RAGSearchInput

        def run(self, query: str):
            return "文档检索结果"

    class FineBIDashboardsTool(BaseTool):
        name = "get_dashboards"
        description = "获取仪表板列表"
        domain = "finebi_system"         # Level 1
        package = "metadata_pkg"         # Level 2
        
        def run(self):
            return ["仪表板1", "仪表板2"]

    # 注册工具
    factory = HierarchicalToolFactory()
    factory.register_tool(RAGSearchTool())
    factory.register_tool(FineBIDashboardsTool())

    print("\n========== [Step 1: Router Agent 看到的全局领域清单 (耗时<1ms, 极省 Token)] ==========")
    print(json.dumps(factory.get_domains_summary(), ensure_ascii=False, indent=2))

    print("\n========== [Step 2: 当判定意图为 'rag_knowledge' 时，只导出该领域的 Tool Schema] ==========")
    # 此时只加载 search_docs，不加载 get_dashboards！
    print(factory.get_tools_schema_by_domains(["rag_knowledge"]))
