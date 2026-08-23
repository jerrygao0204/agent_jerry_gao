# mcp_server.py
import subprocess
import sys
import os

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
install_package("mcp")
install_package("fastmcp")

# 注入根目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import asyncio
import logging
from mcp.server.fastmcp import FastMCP
from factory import init_tools, tool_factory





# 创建 MCP Server 实例
mcp = FastMCP("FineBI-Agent-MCP-Server")

# 初始化本地工具链
init_tools()

# 将 ToolFactory 中的工具暴露为 MCP Tools
@mcp.tool(name="search_knowledge_base", description="检索 FineBI 用户手册及 FAQ")
def mcp_search_knowledge(query: str, top_k: int = 3) -> str:
    tool = tool_factory.get_tool("search_knowledge_base")
    if tool:
        return tool.run(query=query, top_k=top_k)
    return "Tool Not Found"

@mcp.tool(name="get_finebi_dashboards", description="查询 FineBI 系统仪表板元数据")
def mcp_get_dashboards(keyword: str = "") -> str:
    tool = tool_factory.get_tool("get_finebi_dashboards")
    if tool:
        return tool.run(keyword=keyword)
    return "Tool Not Found"

if __name__ == "__main__":
    # 使用 stdio 或 sse 运行 MCP 服务，供其他客户端（如 Claude Desktop / 外部 Agent）连接
    logging.info("🚀 启动 FineBI MCP Server...")
    mcp.run(transport="stdio")