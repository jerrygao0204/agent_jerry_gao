# factory/tools/web_search_tool.py
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import logging
import requests
from typing import Any
from pydantic import BaseModel, Field
from factory.tool_factory import BaseTool


logger = logging.getLogger("WebSearchTool")

class WebSearchInput(BaseModel):
    query: str = Field(description="需要联网查询的问题或关键词")
    top_k: int = Field(default=5, description="返回结果条数")

class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = "联网查询实时信息，用于回答知识库中没有覆盖的、时效性强的问题"
    domain: str = "web_search"        # 新增一个 Level 1 领域
    package: str = "search_pkg"
    args_schema = WebSearchInput
    is_read_only: bool = True

    def run(self, query: str, top_k: int = 5, **kwargs) -> Any:
        logger.info(f"🌐 [WebSearchTool] 查询: {query}")
        
        # 从环境变量读取 API key，避免硬编码
        api_key = os.getenv("SEARCH_API_KEY")
        if not api_key:
            raise ValueError("SEARCH_API_KEY 未设置，请在环境变量中配置。")
        print(f"query={query}")
        # 使用 POST 请求，传递 data 参数
        resp = requests.post(
            'https://serpapi.talordata.net/serp/v1/request',
            headers={'Authorization': f'Bearer {api_key}'},
            data={
                'engine': 'google',
                'q': query,
                'device': 'mobile',
                'location': 'Singapore',
                'gl': 'sg',
                'hl': 'zh-cn',
                'render_js': 'false',
                'uule': 'w+CAIQICIU2luZ2Fwb3Jl',
                'json': '1',
                'count': top_k
            },
            timeout=60,
        )

        resp.raise_for_status()
        return resp.json()

if __name__ == "__main__":
    # 简单测试
    tool = WebSearchTool()
    try:
        result = tool.run("华盛顿的天气", top_k=3)
        print("搜索结果：")
        print(result)
    except Exception as e:
        print(f"❌ 出错: {e}")