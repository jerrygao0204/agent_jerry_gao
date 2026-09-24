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
    domain: str = "web_search"
    package: str = "search_pkg"
    args_schema = WebSearchInput
    is_read_only: bool = True

    def run(self, query: str, top_k: int = 5, **kwargs) -> Any:
        logger.info(f"🌐 [WebSearchTool] 查询: {query}")
        
        api_key = os.getenv("SEARCH_API_KEY")
        if not api_key:
            return "[网络搜索失败]: SEARCH_API_KEY 未设置，请在环境变量中配置。"

        url = 'https://serpapi.talordata.net/serp/v1/request'
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        # 推荐使用 json 格式提交 payload，保持结构一致性
        payload = {
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
        }

        try:
            # connect timeout: 3.0s (建立 TCP/SSL 连接的上限，超长说明节点不可达)
            # read timeout:    10.0s (等待服务器返回数据的上限，适应慢速网络)
            resp = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=(3.0, 10.0), 
                verify=False
            )
            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.ConnectTimeout:
            return "[网络搜索失败]: 无法连接到搜索服务器(TCP/SSL超时)，请直接基于自带知识库回答。"
        except requests.exceptions.ReadTimeout:
            return "[网络搜索失败]: 搜索服务器响应过慢(读取超时)，请直接基于自带知识库回答。"
        except Exception as e:
            return f"[网络搜索失败]: {str(e)}"

        except requests.exceptions.SSLError as e:
            logger.error(f"❌ [WebSearchTool] SSL 握手失败: {e}")
            return f"[网络搜索失败]: SSL/TLS 协议握手失败 ({str(e)})。请放弃联网，直接依据自带知识库回答用户问题。"
            
        except requests.exceptions.Timeout:
            logger.error(f"❌ [WebSearchTool] 网络请求超时")
            return "[网络搜索失败]: SerpAPI 服务响应超时。请放弃联网，直接依据自带知识库回答用户问题。"
            
        except requests.exceptions.HTTPError as e:
            logger.error(f"❌ [WebSearchTool] HTTP 错误: {e}")
            return f"[网络搜索失败]: API 返回 HTTP 状态码错误 ({resp.status_code})。"
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ [WebSearchTool] 网络请求异常: {e}")
            return f"[网络搜索失败]: 无法连接至搜索服务 ({str(e)})。"

if __name__ == "__main__":
    tool = WebSearchTool()
    try:
        result = tool.run("华盛顿的天气", top_k=3)
        print("搜索结果：")
        print(result)
    except Exception as e:
        print(f"❌ 出错: {e}")