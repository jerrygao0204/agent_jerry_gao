# factory/tools/api_tool.py
import logging
import json
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from factory.tool_factory import BaseTool

logger = logging.getLogger("APITool")

class DashboardQueryInput(BaseModel):
    keyword: Optional[str] = Field(default="", description="仪表板名称或关键词，为空则查询全部")

class FineBIDashboardTool(BaseTool):
    name: str = "get_finebi_dashboards"
    description: str = "获取 FineBI 系统中已发布的仪表板列表及状态元数据"
    domain: str = "finebi_system"
    package: str = "metadata_pkg"
    args_schema = DashboardQueryInput
    is_read_only: bool = True

    def run(self, keyword: str = "", **kwargs) -> Any:
        logger.info(f"📊 [APITool] 查询 FineBI 仪表板列表，关键词: '{keyword}'")
        # 此处连接 FineBI 开放 API 或后台数据库 (示例模拟数据)
        mock_dashboards = [
            {"id": "dash_01", "name": "集团销售月报仪表板", "status": "active", "owner": "admin"},
            {"id": "dash_02", "name": "供应链库存预警监控", "status": "active", "owner": "supply_team"},
            {"id": "dash_03", "name": "财务应收账款分析", "status": "maintenance", "owner": "finance"}
        ]
        if keyword:
            filtered = [d for d in mock_dashboards if keyword.lower() in d["name"].lower()]
            return json.dumps(filtered, ensure_ascii=False)
        return json.dumps(mock_dashboards, ensure_ascii=False)