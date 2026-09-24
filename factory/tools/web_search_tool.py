# factory/tools/web_search_tool.py
import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
import time
from typing import Any

from factory.tool_factory import BaseTool
from pydantic import BaseModel, Field
import requests
from requests.adapters import HTTPAdapter
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger("WebSearchTool")


# --- 初始化全局共享的 Session，避免重复 DNS 查找与 TLS 握手 ---
def _create_search_session() -> requests.Session:
    session = requests.Session()
    # 配置连接池大小，防止 React Agent 并发调用时卡顿
    adapter = HTTPAdapter(
        pool_connections=20, pool_maxsize=20, max_retries=1
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


GLOBAL_SEARCH_SESSION = _create_search_session()


class WebSearchInput(BaseModel):
    query: str = Field(description="需要联网查询的问题或关键词")
    top_k: int = Field(default=5, description="返回结果条数")


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "联网查询实时信息，用于回答知识库中没有覆盖的、时效性强的问题"
    )
    domain: str = "web_search"
    package: str = "search_pkg"
    args_schema = WebSearchInput
    is_read_only: bool = True

    def _parse_and_clean_results(self, payload: dict, top_k: int) -> str:
        organic_results = payload.get("organic") or payload.get(
            "organic_results", []
        )
        # if not organic_results:
        #     return "未找到相关搜索结果。"

        formatted_results = []
        for idx, item in enumerate(organic_results[:top_k], 1):
            title = item.get("title", "无标题")
            snippet = item.get("snippet") or item.get("description", "无摘要")
            link = item.get("link", "")

            formatted_results.append(
                f"[{idx}] {title}\n摘要: {snippet}\n链接: {link}"
            )

        cleaned_text = "\n\n".join(formatted_results)
        max_chars = 2000
        if len(cleaned_text) > max_chars:
            cleaned_text = cleaned_text[:max_chars] + "...[内容已截断]"

        return cleaned_text

    def run(self, query: str, top_k: int = 5, **kwargs) -> Any:
        logger.info(f"🌐 [WebSearchTool] 开始查询: {query}")

        api_key = os.getenv("SEARCH_API_KEY")
        if not api_key:
            return (
                "[网络搜索失败]: SEARCH_API_KEY"
                " 未设置，请在环境变量中配置。"
            )

        url = "https://serpapi.talordata.net/serp/v1/request"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Connection": "keep-alive",
        }
        payload = {
            "engine": "google",
            "q": query,
            "device": "mobile",
            "location": "Singapore",
            "gl": "sg",
            "hl": "zh-cn",
            "render_js": "false",
            "uule": "w+CAIQICIU2luZ2Fwb3Jl",
            "json": "1",
            "count": top_k,
        }

        start_time = time.perf_counter()

        try:
            # 使用复用的 GLOBAL_SEARCH_SESSION 代替 requests.post
            resp = GLOBAL_SEARCH_SESSION.post(
                url,
                headers=headers,
                data=payload,
                timeout=(5.0, 25.0),  # 适当放宽 Read Timeout 到 25s
                verify=False,
            )
            elapsed_time = time.perf_counter() - start_time
            logger.info(
                f"⏱️ [WebSearchTool] 请求完成，耗时: {elapsed_time:.2f}s"
            )

            resp.raise_for_status()
            res_json = resp.json()

            if not isinstance(res_json, dict):
                return "[网络搜索失败]: 响应格式非法 (非 JSON 对象)。"

            # 网关状态码校验
            gateway_code = res_json.get("code")
            if gateway_code is not None and gateway_code != 0:
                err_msg = res_json.get("msg") or res_json.get("data", "网关异常")
                logger.error(
                    f"❌ [WebSearchTool] TalorData 网关报错 [{gateway_code}]:"
                    f" {err_msg}"
                )
                return f"[网络搜索失败]: 搜索网关错误 ({err_msg})。"

            # 数据提取与 SERP 状态校验
            search_payload = res_json.get("data", res_json)
            payload_code = search_payload.get("code")
            if payload_code is not None and payload_code != 200:
                err_msg = search_payload.get("error", "搜索采集失败")
                logger.error(
                    f"❌ [WebSearchTool] SERP 采集失败 [{payload_code}]:"
                    f" {err_msg}"
                )
                return f"[网络搜索失败]: SERP 采集错误 ({err_msg})。"

            return self._parse_and_clean_results(search_payload, top_k)

        except requests.exceptions.ReadTimeout:
            elapsed_time = time.perf_counter() - start_time
            logger.error(
                f"❌ [WebSearchTool] 读取超时 (耗时: {elapsed_time:.2f}s)"
            )
            return (
                "[网络搜索失败]: 搜索服务器响应过慢"
                f" (读取超时，已等待 {elapsed_time:.2f}s)。"
            )
        except requests.exceptions.RequestException as e:
            elapsed_time = time.perf_counter() - start_time
            logger.error(
                f"❌ [WebSearchTool] 网络请求异常 (耗时: {elapsed_time:.2f}s):"
                f" {e}"
            )
            return f"[网络搜索失败]: 无法连接至搜索服务 ({str(e)})。"
        except Exception as e:
            elapsed_time = time.perf_counter() - start_time
            logger.error(
                f"❌ [WebSearchTool] 未知异常 (耗时: {elapsed_time:.2f}s): {e}"
            )
            return f"[网络搜索失败]: {str(e)}"


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    tool = WebSearchTool()
    print(tool.run("电车难题 网上讨论", top_k=3))