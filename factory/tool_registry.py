# factory/tool_registry.py
import os
import sys

# ==========================================
# 0. 动态修复 Python 模块搜索路径 (sys.path)
# ==========================================
# 获取当前文件所在目录的上一级目录（即项目根目录）
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
import yaml
from factory.tool_factory import tool_factory
import importlib
from typing import Any, Dict, Optional
import inspect

# 🌟 从具体实现模块导入定义，避免重复定义
# from factory.tools.rag_tool import RAGKnowledgeSearchTool
# from factory.tools.api_tool import FineBIDashboardTool
# from factory.tools.web_search_tool import WebSearchTool
# from factory.tools.dataset_summary import DatasetSummaryTool


logger = logging.getLogger("ToolsModule")


# def init_tools(retriever: Optional[Any] = None, reranker: Optional[Any] = None) -> None:
#     """集中注册所有的原子工具到全局单例 tool_factory"""
#     try:
#         # 1. 注册带 Reranker 能力的 RAG 工具
#         rag_tool = RAGKnowledgeSearchTool(retriever=retriever, reranker=reranker)
#         tool_factory.register_tool(rag_tool)

#         # 2. 注册 API 工具
#         bi_tool = FineBIDashboardTool()
#         tool_factory.register_tool(bi_tool)

#         # 3. 注册 web 工具
#         web_tool = WebSearchTool()
#         tool_factory.register_tool(web_tool)

#         # 💡 4. 注册 数据集摘要工具 (Dataset Summary Tool)
#         tool_factory.register_tool(DatasetSummaryTool())

#         logger.info("✅ [init_tools] 所有工具类注册完毕。")
#     except Exception as e:
#         logger.error(f"❌ 工具初始化失败: {e}")
#         raise e


def init_tools(retriever: Optional[Any] = None, reranker: Optional[Any] = None) -> None:
    """初始化工具工厂：加载配置并动态注册工具"""
    yaml_path = os.path.join(project_root, "config", "tools.yaml")

    if not os.path.exists(yaml_path):
        logger.error(f"❌ 找不到配置文件: {yaml_path}")
        return

    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 1. 先载入 YAML 定义的 Domain 与 Package 元数据
    for domain, dmeta in config.get("domains", {}).items():
        tool_factory.register_domain_meta(domain, dmeta.get("description", ""))
        for pkg, desc in dmeta.get("packages", {}).items():
            tool_factory.register_package_meta(domain, pkg, desc)

    # 2. 构造依赖注入上下文
    injection_context: Dict[str, Any] = {
        "retriever": retriever,
        "reranker": reranker
    }

    # 3. 动态实例化并注册工具
    for entry in config.get("tools", []):
        if not entry.get("enabled", True):
            continue
            
        module = importlib.import_module(entry["module"])
        cls = getattr(module, entry["class"])

        # 动态检测构造函数参数，仅注入需要的依赖（消除硬编码判断）
        sig = inspect.signature(cls.__init__)
        tool_kwargs = {
            k: v for k, v in injection_context.items() 
            if k in sig.parameters and v is not None
        }

        tool_instance = cls(**tool_kwargs)
        tool_factory.register_tool(tool_instance)



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 📌 使用基於 project_root 的動態路徑進行驗證
    target_yaml_path = os.path.join(project_root, "config", "tools.yaml")

    print("\n" + "=" * 60)
    print("🔍 [YAML 讀取測試] 開始驗證...")
    print(f"📌 解析出的專案根目錄: {project_root}")
    print(f"📌 目標 YAML 絕對路徑: {target_yaml_path}")
    print("=" * 60)

    if not os.path.exists(target_yaml_path):
        print(f"❌ [錯誤] 檔案不存在，請檢查目錄結構: {target_yaml_path}")
    else:
        print(f"✅ [成功] 成功定位檔案！")
        try:
            init_tools()
            tools = tool_factory.get_openai_tools_schema_by_packages()
            print(f"🎉 [成功] 成功加載 YAML 並註冊 {len(tools)} 個工具！")
        except Exception as e:
            print(f"❌ [錯誤] 初始化失敗: {e}")

    print("=" * 60 + "\n")