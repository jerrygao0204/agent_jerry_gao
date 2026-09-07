from pydantic import BaseModel, Field
from factory.tool_factory import BaseTool

# 1. 定义入参 Schema
class DatasetSummaryInput(BaseModel):
    dataset_name: str = Field(description="需要查询的数据集名称")
    query: str = Field(description="具體查詢的內容")

# 2. 定义工具类 (Level 3)
class DatasetSummaryTool(BaseTool):
    name: str = "get_dataset_summary"
    description: str = "查询指定数据集/数据表的元数据信息，如行数、列数、字段列表、表结构摘要。适用于获取数据表规模及明细结构。"
    domain: str = "finebi_system"  # ⚠️ Level 1: 必须与 ToolFactory 的 Key 一致
    package: str = "metadata_pkg"   # Level 2: 工具包
    args_schema = DatasetSummaryInput
    is_read_only: bool = True

    def run(self, dataset_name: str,query: str, **kwargs) -> str:
        # 模拟数据分析执行逻辑
        return f"数据集 [{dataset_name}] 摘要: 內容{query}共 10,000 行，15 列，更新时间: 2026-08-25"