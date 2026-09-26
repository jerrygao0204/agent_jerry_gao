# import os
# import sys
# from unittest.mock import MagicMock, patch


# # ============================================================================
# # 0. 核心防爆层：在一切 import 发生前，用 Mock 彻底接管动态库入口
# # ============================================================================
# c_extensions_to_block = [
#     "torch",
#     "torch.cuda",
#     "torch.nn",
#     "transformers",
#     "accelerate",
#     "factory.model_factory",
# ]

# for mod in c_extensions_to_block:
#     sys.modules[mod] = MagicMock()

# os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

# # ============================================================================
# # 1. 显式安全导入待测类与函数（避免属性链查找覆盖）
# # ============================================================================
# import pytest
# from pydantic import BaseModel, Field

# from factory.tool_factory import (
#     BaseTool,
#     HierarchicalToolFactory,
#     load_tools_from_yaml,
#     tool_factory,
# )
# from factory.tool_registry import init_tools


# # ============================================================================
# # 2. ToolRegistry (init_tools) 单元测试
# # ============================================================================

# class TestToolRegistry:
#     """验证 init_tools 函数逻辑"""

#     @patch("factory.tool_registry.os.path.exists")
#     def test_init_tools_missing_yaml_config(self, mock_exists):
#         """测试 1: YAML 配置文件不存在时的异常捕获与日志处理"""
#         mock_exists.return_value = False
#         init_tools()

#     @patch("factory.tool_registry.importlib.import_module")
#     @patch("factory.tool_registry.os.path.exists")
#     @patch("builtins.open")
#     @patch("factory.tool_registry.yaml.safe_load")
#     def test_init_tools_successful_registration(
#         self, mock_yaml_load, mock_open, mock_exists, mock_import_module
#     ):
#         """测试 2: 完整流程解析 YAML、过滤 disabled 工具并精准依赖注入"""
#         mock_exists.return_value = True

#         mock_yaml_load.return_value = {
#             "domains": {
#                 "rag_domain": {
#                     "description": "RAG Domain Description",
#                     "packages": {"search_pkg": "Search Package Description"}
#                 }
#             },
#             "tools": [
#                 {
#                     "name": "mock_search_tool",
#                     "module": "mock_module",
#                     "class": "MockSearchTool",
#                     "enabled": True
#                 },
#                 {
#                     "name": "disabled_tool",
#                     "module": "mock_module",
#                     "class": "DisabledTool",
#                     "enabled": False
#                 }
#             ]
#         }

#         class DummySearchTool:
#             def __init__(self, retriever):
#                 self.retriever = retriever
#                 self.domain = "rag_domain"
#                 self.package = "search_pkg"
#                 self.name = "mock_search_tool"
#                 self.role_whitelist = ["admin"]
#                 self.description = "Mock Search Tool"

#         mock_module = MagicMock()
#         mock_module.MockSearchTool = DummySearchTool
#         mock_import_module.return_value = mock_module

#         mock_retriever = MagicMock()
#         init_tools(retriever=mock_retriever, reranker=None)

#         fetched_tool = tool_factory.get_tool("mock_search_tool", user_role="admin")
#         assert fetched_tool is not None
#         assert fetched_tool.retriever == mock_retriever


# # ============================================================================
# # 3. tool_factory 实例单元测试
# # ============================================================================

# class TestToolFactoryInstance:
#     """验证 tool_factory 实例的角色鉴权与元数据导出"""

#     def test_factory_register_and_get_tool(self):
#         """测试 1: 工具注册与角色权限鉴权"""
#         mock_tool = MagicMock()
#         mock_tool.name = "demo_tool"
#         mock_tool.domain = "demo_domain"
#         mock_tool.package = "demo_pkg"
#         mock_tool.role_whitelist = ["admin", "developer"]

#         tool_factory.register_tool(mock_tool)

#         assert tool_factory.get_tool("demo_tool", user_role="admin") == mock_tool
#         assert tool_factory.get_tool("demo_tool", user_role="guest") is None

#     def test_get_tools_metadata_by_packages(self):
#         """测试 2: 层级元数据导出与 Package 筛选"""
#         tool_factory.register_domain_meta("meta_domain", "Meta Domain Desc")
#         tool_factory.register_package_meta("meta_domain", "meta_pkg", "Meta Pkg Desc")

#         mock_tool = MagicMock()
#         mock_tool.name = "meta_tool"
#         mock_tool.domain = "meta_domain"
#         mock_tool.package = "meta_pkg"
#         mock_tool.role_whitelist = ["admin"]
#         mock_tool.description = "Tool Description"

#         tool_factory.register_tool(mock_tool)

#         tool_names, formatted_tool_specs = tool_factory.get_tools_metadata_by_packages(
#             [("meta_domain", "meta_pkg")], user_role="admin"
#         )

#         assert "meta_tool" in tool_names
#         assert len(formatted_tool_specs) >= 1
#         assert any(
#             spec.get("function", {}).get("name") == "meta_tool"
#             for spec in formatted_tool_specs
#         )


# # ============================================================================
# # 4. 覆盖率提升补充测试（使用显式导入的类）
# # ============================================================================

# class DummyArgsSchema(BaseModel):
#     query: str = Field(description="搜索关键词")
#     top_k: int = Field(default=5, description="返回数量")


# class TestToolFactoryExtended:

#     @patch("factory.tool_factory.importlib.import_module")
#     def test_load_tools_from_yaml(self, mock_import_module, tmp_path):
#         """测试 load_tools_from_yaml 反射装载逻辑（使用真实临时 YAML 文件，避免 mock 文件 I/O 的不确定性）"""

#         # 1. 构造真实的临时 YAML 配置文件
#         config_content = """
# domains:
#   yaml_dom:
#     description: "YAML Dom"
#     packages:
#       yaml_pkg: "YAML Pkg"

# tools:
#   - name: y_tool
#     module: m_path
#     class: YClass
#     enabled: true
#     domain: yaml_dom
#     package: yaml_pkg
#     role_whitelist:
#       - admin
# """
#         config_file = tmp_path / "tools.yaml"
#         config_file.write_text(config_content, encoding="utf-8")

#         # 2. 构造具名 BaseTool 派生类
#         class ConcreteMockTool(BaseTool):
#             name = "y_tool"
#             domain = "yaml_dom"
#             package = "yaml_pkg"
#             role_whitelist = ["admin"]

#             def run(self, **kwargs):
#                 return "running..."

#         mock_mod = MagicMock()
#         mock_mod.YClass = ConcreteMockTool
#         mock_import_module.return_value = mock_mod

#         # 3. 实例化隔离的工厂并加载真实临时文件（真实路径，真实存在）
#         tf = HierarchicalToolFactory()
#         load_tools_from_yaml(str(config_file), factory=tf)

#         # 4. 断言
#         loaded_tool = tf.get_tool("y_tool", user_role="admin")
#         assert loaded_tool is not None, "工具未成功注册至工厂实例"
#         assert loaded_tool.name == "y_tool"
#         assert loaded_tool.domain == "yaml_dom"
#         assert loaded_tool.package == "yaml_pkg"



# import os
# import sys
# from unittest.mock import MagicMock, patch

# # ============================================================================
# # 0. 核心防爆层：在一切 import 发生前，用 Mock 彻底接管动态库入口
# # ============================================================================
# c_extensions_to_block = [
#     "torch",
#     "torch.cuda",
#     "torch.nn",
#     "transformers",
#     "accelerate",
#     "factory.model_factory",
# ]

# for mod in c_extensions_to_block:
#     sys.modules[mod] = MagicMock()

# os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

# # ============================================================================
# # 1. 显式安全导入待测类与函数
# # ============================================================================
# import pytest
# from pydantic import BaseModel, Field

# from factory.tool_factory import (
#     BaseTool,
#     HierarchicalToolFactory,
#     load_tools_from_yaml,
#     tool_factory,
# )
# from factory.tool_registry import init_tools


# # ============================================================================
# # 2. ToolRegistry (init_tools) 单元测试
# # ============================================================================

# class TestToolRegistry:
#     """验证 init_tools 函数逻辑"""

#     @patch("factory.tool_registry.os.path.exists")
#     def test_init_tools_missing_yaml_config(self, mock_exists):
#         """测试 1: YAML 配置文件不存在时的异常捕获与日志处理"""
#         mock_exists.return_value = False
#         init_tools()

#     @patch("factory.tool_registry.importlib.import_module")
#     @patch("factory.tool_registry.os.path.exists")
#     @patch("builtins.open")
#     @patch("factory.tool_registry.yaml.safe_load")
#     def test_init_tools_successful_registration(
#         self, mock_yaml_load, mock_open, mock_exists, mock_import_module
#     ):
#         """测试 2: 完整流程解析 YAML、过滤 disabled 工具并精准依赖注入"""
#         mock_exists.return_value = True

#         mock_yaml_load.return_value = {
#             "domains": {
#                 "rag_domain": {
#                     "description": "RAG Domain Description",
#                     "packages": {"search_pkg": "Search Package Description"}
#                 }
#             },
#             "tools": [
#                 {
#                     "name": "mock_search_tool",
#                     "module": "mock_module",
#                     "class": "MockSearchTool",
#                     "enabled": True
#                 },
#                 {
#                     "name": "disabled_tool",
#                     "module": "mock_module",
#                     "class": "DisabledTool",
#                     "enabled": False
#                 }
#             ]
#         }

#         class DummySearchTool:
#             def __init__(self, retriever):
#                 self.retriever = retriever
#                 self.domain = "rag_domain"
#                 self.package = "search_pkg"
#                 self.name = "mock_search_tool"
#                 self.role_whitelist = ["admin"]
#                 self.description = "Mock Search Tool"

#         mock_module = MagicMock()
#         mock_module.MockSearchTool = DummySearchTool
#         mock_import_module.return_value = mock_module

#         mock_retriever = MagicMock()
#         init_tools(retriever=mock_retriever, reranker=None)

#         fetched_tool = tool_factory.get_tool("mock_search_tool", user_role="admin")
#         assert fetched_tool is not None
#         assert fetched_tool.retriever == mock_retriever

#     @patch("factory.tool_registry.importlib.import_module")
#     @patch("factory.tool_registry.os.path.exists")
#     @patch("builtins.open")
#     @patch("factory.tool_registry.yaml.safe_load")
#     def test_init_tools_with_reranker_and_all_dependencies(
#         self, mock_yaml_load, mock_open, mock_exists, mock_import_module
#     ):
#         """覆盖 init_tools 中 reranker/retriever 参数自动匹配与实例化注册分支 (Lines 66-93)"""
#         mock_exists.return_value = True
#         mock_yaml_load.return_value = {
#             "domains": {
#                 "rag_domain": {
#                     "description": "RAG Domain",
#                     "packages": {"rag_pkg": "RAG Package"}
#                 }
#             },
#             "tools": [
#                 {
#                     "name": "full_dep_tool",
#                     "module": "mock_module",
#                     "class": "FullDepTool",
#                     "enabled": True,
#                     "domain": "rag_domain",
#                     "package": "rag_pkg",
#                     "role_whitelist": ["admin"]
#                 }
#             ]
#         }

#         # 构造正常接受 retriever 和 reranker 的工具类，派生自 BaseTool
#         class FullDepTool(BaseTool):
#             name = "full_dep_tool"
#             domain = "rag_domain"
#             package = "rag_pkg"
#             role_whitelist = ["admin"]

#             def __init__(self, retriever=None, reranker=None):
#                 self.retriever = retriever
#                 self.reranker = reranker

#             def run(self, **kwargs):
#                 return "ok"

#         mock_module = MagicMock()
#         mock_module.FullDepTool = FullDepTool
#         mock_import_module.return_value = mock_module

#         mock_retriever = MagicMock()
#         mock_reranker = MagicMock()

#         # 执行注册
#         init_tools(retriever=mock_retriever, reranker=mock_reranker)

#         # 断言依赖正常注入与注册
#         tool = tool_factory.get_tool("full_dep_tool", user_role="admin")
#         assert tool is not None
#         assert tool.retriever == mock_retriever
#         assert tool.reranker == mock_reranker
        

# # ============================================================================
# # 3. tool_factory 实例单元测试
# # ============================================================================

# class TestToolFactoryInstance:
#     """验证 tool_factory 实例的角色鉴权与元数据导出"""

#     def test_factory_register_and_get_tool(self):
#         """测试 1: 工具注册与角色权限鉴权"""
#         mock_tool = MagicMock()
#         mock_tool.name = "demo_tool"
#         mock_tool.domain = "demo_domain"
#         mock_tool.package = "demo_pkg"
#         mock_tool.role_whitelist = ["admin", "developer"]

#         tool_factory.register_tool(mock_tool)

#         assert tool_factory.get_tool("demo_tool", user_role="admin") == mock_tool
#         assert tool_factory.get_tool("demo_tool", user_role="guest") is None

#     def test_get_tools_metadata_by_packages(self):
#         """测试 2: 层级元数据导出与 Package 筛选"""
#         tool_factory.register_domain_meta("meta_domain", "Meta Domain Desc")
#         tool_factory.register_package_meta("meta_domain", "meta_pkg", "Meta Pkg Desc")

#         mock_tool = MagicMock()
#         mock_tool.name = "meta_tool"
#         mock_tool.domain = "meta_domain"
#         mock_tool.package = "meta_pkg"
#         mock_tool.role_whitelist = ["admin"]
#         mock_tool.description = "Tool Description"

#         tool_factory.register_tool(mock_tool)

#         tool_names, formatted_tool_specs = tool_factory.get_tools_metadata_by_packages(
#             [("meta_domain", "meta_pkg")], user_role="admin"
#         )

#         assert "meta_tool" in tool_names
#         assert len(formatted_tool_specs) >= 1
#         assert any(
#             spec.get("function", {}).get("name") == "meta_tool"
#             for spec in formatted_tool_specs
#         )


# # ============================================================================
# # 4. 覆盖率提升补充测试（追加用例）
# # ============================================================================

# class DummyArgsSchema(BaseModel):
#     query: str = Field(description="搜索关键词")
#     top_k: int = Field(default=5, description="返回数量")


# class TestToolFactoryExtended:

#     def test_base_tool_json_schema(self):
#         """测试 BaseTool.get_json_schema 的标准生成与 title 剥离逻辑"""
#         class SchemaTool(BaseTool):
#             name = "schema_tool"
#             args_schema = DummyArgsSchema

#             def run(self, **kwargs):
#                 return "ok"

#         tool = SchemaTool()
#         schema = tool.get_json_schema()

#         assert "title" not in schema
#         assert "properties" in schema
#         assert "query" in schema["properties"]
#         assert "top_k" in schema["properties"]
#         assert "title" not in schema["properties"]["query"]

#         class NoSchemaTool(BaseTool):
#             name = "no_schema_tool"

#             def run(self, **kwargs):
#                 return "ok"

#         tool_no_schema = NoSchemaTool()
#         assert tool_no_schema.get_json_schema() == {"type": "object", "properties": {}}

#     def test_get_domains_and_packages_summary(self):
#         """测试 Level 1 Domain 与 Level 2 Package 级别的 RBAC 动态裁减策略"""
#         tf = HierarchicalToolFactory()
#         tf.register_domain_meta("sec_dom", "Security Domain", role_whitelist=["admin"])
#         tf.register_package_meta("sec_dom", "sec_pkg", "Security Package")

#         class SecretTool(BaseTool):
#             name = "secret_tool"
#             domain = "sec_dom"
#             package = "sec_pkg"
#             role_whitelist = ["admin"]

#             def run(self, **kwargs):
#                 return "secret"

#         tf.register_tool(SecretTool())

#         # guest 角色应该无法被搜寻到任何 Domain 与 Package 摘要
#         guest_domains = tf.get_domains_summary(user_role="guest")
#         assert len(guest_domains) == 0

#         guest_packages = tf.get_packages_summary_by_domains(["sec_dom"], user_role="guest")
#         assert len(guest_packages) == 0

#         # admin 角色可以正常搜寻
#         admin_domains = tf.get_domains_summary(user_role="admin")
#         assert len(admin_domains) == 1
#         assert admin_domains[0]["domain"] == "sec_dom"

#         admin_packages = tf.get_packages_summary_by_domains(["sec_dom"], user_role="admin")
#         assert len(admin_packages) == 1
#         assert admin_packages[0]["package"] == "sec_pkg"

#     @patch("factory.tool_factory.importlib.import_module")
#     def test_load_tools_from_yaml(self, mock_import_module, tmp_path):
#         """测试 load_tools_from_yaml 反射装载逻辑（使用真实临时 YAML 文件，避免 mock 文件 I/O 的不确定性）"""

#         config_content = """
# domains:
#   yaml_dom:
#     description: "YAML Dom"
#     packages:
#       yaml_pkg: "YAML Pkg"

# tools:
#   - name: y_tool
#     module: m_path
#     class: YClass
#     enabled: true
#     domain: yaml_dom
#     package: yaml_pkg
#     role_whitelist:
#       - admin
# """
#         config_file = tmp_path / "tools.yaml"
#         config_file.write_text(config_content, encoding="utf-8")

#         class ConcreteMockTool(BaseTool):
#             name = "y_tool"
#             domain = "yaml_dom"
#             package = "yaml_pkg"
#             role_whitelist = ["admin"]

#             def run(self, **kwargs):
#                 return "running..."

#         mock_mod = MagicMock()
#         mock_mod.YClass = ConcreteMockTool
#         mock_import_module.return_value = mock_mod

#         tf = HierarchicalToolFactory()
#         load_tools_from_yaml(str(config_file), factory=tf)

#         loaded_tool = tf.get_tool("y_tool", user_role="admin")
#         assert loaded_tool is not None, "工具未成功注册至工厂实例"
#         assert loaded_tool.name == "y_tool"
#         assert loaded_tool.domain == "yaml_dom"
#         assert loaded_tool.package == "yaml_pkg"

#     @patch("factory.tool_factory.importlib.import_module")
#     def test_load_tools_from_yaml_exception_handling(self, mock_import_module, tmp_path):
#         """测试 load_tools_from_yaml 在导入失败时的捕获机制"""
#         config_content = """
# tools:
#   - name: broken_tool
#     module: non_exist_module
#     class: NonExistClass
#     enabled: true
# """
#         config_file = tmp_path / "broken_tools.yaml"
#         config_file.write_text(config_content, encoding="utf-8")

#         mock_import_module.side_effect = ImportError("Module not found")

#         tf = HierarchicalToolFactory()
#         # 确保抛出异常时代码不崩溃，静默日志并记录
#         load_tools_from_yaml(str(config_file), factory=tf)
#         assert tf.get_tool("broken_tool") is None



import os
import sys
from unittest.mock import MagicMock, patch

# ============================================================================
# 0. 核心防爆层：在一切 import 发生前，用 Mock 彻底接管动态库入口
# ============================================================================
c_extensions_to_block = [
    "torch",
    "torch.cuda",
    "torch.nn",
    "transformers",
    "accelerate",
    "factory.model_factory",
]

for mod in c_extensions_to_block:
    sys.modules[mod] = MagicMock()

os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"

# ============================================================================
# 1. 显式安全导入待测类与函数
# ============================================================================
import pytest
from pydantic import BaseModel, Field

from factory.tool_factory import (
    BaseTool,
    HierarchicalToolFactory,
    load_tools_from_yaml,
    tool_factory,
)
from factory.tool_registry import init_tools


# ============================================================================
# 2. ToolRegistry (init_tools) 单元测试
# ============================================================================

class TestToolRegistry:
    """验证 init_tools 函数逻辑"""

    @patch("factory.tool_registry.os.path.exists")
    def test_init_tools_missing_yaml_config(self, mock_exists):
        """测试 1: YAML 配置文件不存在时的异常捕获与日志处理"""
        mock_exists.return_value = False
        init_tools()

    @patch("factory.tool_registry.importlib.import_module")
    @patch("factory.tool_registry.os.path.exists")
    @patch("builtins.open")
    @patch("factory.tool_registry.yaml.safe_load")
    def test_init_tools_successful_registration(
        self, mock_yaml_load, mock_open, mock_exists, mock_import_module
    ):
        """测试 2: 完整流程解析 YAML、过滤 disabled 工具并精准依赖注入"""
        mock_exists.return_value = True

        mock_yaml_load.return_value = {
            "domains": {
                "rag_domain": {
                    "description": "RAG Domain Description",
                    "packages": {"search_pkg": "Search Package Description"}
                }
            },
            "tools": [
                {
                    "name": "mock_search_tool",
                    "module": "mock_module",
                    "class": "MockSearchTool",
                    "enabled": True
                },
                {
                    "name": "disabled_tool",
                    "module": "mock_module",
                    "class": "DisabledTool",
                    "enabled": False
                }
            ]
        }

        class DummySearchTool:
            def __init__(self, retriever):
                self.retriever = retriever
                self.domain = "rag_domain"
                self.package = "search_pkg"
                self.name = "mock_search_tool"
                self.role_whitelist = ["admin"]
                self.description = "Mock Search Tool"

        mock_module = MagicMock()
        mock_module.MockSearchTool = DummySearchTool
        mock_import_module.return_value = mock_module

        mock_retriever = MagicMock()
        init_tools(retriever=mock_retriever, reranker=None)

        fetched_tool = tool_factory.get_tool("mock_search_tool", user_role="admin")
        assert fetched_tool is not None
        assert fetched_tool.retriever == mock_retriever

    @patch("factory.tool_registry.importlib.import_module")
    @patch("factory.tool_registry.os.path.exists")
    @patch("builtins.open")
    @patch("factory.tool_registry.yaml.safe_load")
    def test_init_tools_with_reranker_and_all_dependencies(
        self, mock_yaml_load, mock_open, mock_exists, mock_import_module
    ):
        """覆盖 init_tools 中 reranker/retriever 参数自动匹配与实例化注册分支"""
        mock_exists.return_value = True
        mock_yaml_load.return_value = {
            "domains": {
                "rag_domain": {
                    "description": "RAG Domain",
                    "packages": {"rag_pkg": "RAG Package"}
                }
            },
            "tools": [
                {
                    "name": "full_dep_tool",
                    "module": "mock_module",
                    "class": "FullDepTool",
                    "enabled": True,
                    "domain": "rag_domain",
                    "package": "rag_pkg",
                    "role_whitelist": ["admin"]
                }
            ]
        }

        class FullDepTool(BaseTool):
            name = "full_dep_tool"
            domain = "rag_domain"
            package = "rag_pkg"
            role_whitelist = ["admin"]

            def __init__(self, retriever=None, reranker=None):
                self.retriever = retriever
                self.reranker = reranker

            def run(self, **kwargs):
                return "ok"

        mock_module = MagicMock()
        mock_module.FullDepTool = FullDepTool
        mock_import_module.return_value = mock_module

        mock_retriever = MagicMock()
        mock_reranker = MagicMock()

        init_tools(retriever=mock_retriever, reranker=mock_reranker)

        tool = tool_factory.get_tool("full_dep_tool", user_role="admin")
        assert tool is not None
        assert tool.retriever == mock_retriever
        assert tool.reranker == mock_reranker


# ============================================================================
# 3. tool_factory 实例单元测试
# ============================================================================

class TestToolFactoryInstance:
    """验证 tool_factory 实例的角色鉴权与元数据导出"""

    def test_factory_register_and_get_tool(self):
        mock_tool = MagicMock()
        mock_tool.name = "demo_tool"
        mock_tool.domain = "demo_domain"
        mock_tool.package = "demo_pkg"
        mock_tool.role_whitelist = ["admin", "developer"]

        tool_factory.register_tool(mock_tool)

        assert tool_factory.get_tool("demo_tool", user_role="admin") == mock_tool
        assert tool_factory.get_tool("demo_tool", user_role="guest") is None

    def test_get_tools_metadata_by_packages(self):
        tool_factory.register_domain_meta("meta_domain", "Meta Domain Desc")
        tool_factory.register_package_meta("meta_domain", "meta_pkg", "Meta Pkg Desc")

        mock_tool = MagicMock()
        mock_tool.name = "meta_tool"
        mock_tool.domain = "meta_domain"
        mock_tool.package = "meta_pkg"
        mock_tool.role_whitelist = ["admin"]
        mock_tool.description = "Tool Description"

        tool_factory.register_tool(mock_tool)

        tool_names, formatted_tool_specs = tool_factory.get_tools_metadata_by_packages(
            [("meta_domain", "meta_pkg")], user_role="admin"
        )

        assert "meta_tool" in tool_names
        assert len(formatted_tool_specs) >= 1
        assert any(
            spec.get("function", {}).get("name") == "meta_tool"
            for spec in formatted_tool_specs
        )


# ============================================================================
# 4. 原有覆盖率补充测试
# ============================================================================

class DummyArgsSchema(BaseModel):
    query: str = Field(description="搜索关键词")
    top_k: int = Field(default=5, description="返回数量")


class TestToolFactoryExtended:

    def test_base_tool_json_schema(self):
        class SchemaTool(BaseTool):
            name = "schema_tool"
            args_schema = DummyArgsSchema

            def run(self, **kwargs):
                return "ok"

        tool = SchemaTool()
        schema = tool.get_json_schema()

        assert "title" not in schema
        assert "properties" in schema
        assert "query" in schema["properties"]
        assert "top_k" in schema["properties"]
        assert "title" not in schema["properties"]["query"]

        class NoSchemaTool(BaseTool):
            name = "no_schema_tool"

            def run(self, **kwargs):
                return "ok"

        tool_no_schema = NoSchemaTool()
        assert tool_no_schema.get_json_schema() == {"type": "object", "properties": {}}

    def test_get_domains_and_packages_summary(self):
        tf = HierarchicalToolFactory()
        tf.register_domain_meta("sec_dom", "Security Domain", role_whitelist=["admin"])
        tf.register_package_meta("sec_dom", "sec_pkg", "Security Package")

        class SecretTool(BaseTool):
            name = "secret_tool"
            domain = "sec_dom"
            package = "sec_pkg"
            role_whitelist = ["admin"]

            def run(self, **kwargs):
                return "secret"

        tf.register_tool(SecretTool())

        guest_domains = tf.get_domains_summary(user_role="guest")
        assert len(guest_domains) == 0

        guest_packages = tf.get_packages_summary_by_domains(["sec_dom"], user_role="guest")
        assert len(guest_packages) == 0

        admin_domains = tf.get_domains_summary(user_role="admin")
        assert len(admin_domains) == 1
        assert admin_domains[0]["domain"] == "sec_dom"

        admin_packages = tf.get_packages_summary_by_domains(["sec_dom"], user_role="admin")
        assert len(admin_packages) == 1
        assert admin_packages[0]["package"] == "sec_pkg"

    @patch("factory.tool_factory.importlib.import_module")
    def test_load_tools_from_yaml(self, mock_import_module, tmp_path):
        config_content = """
domains:
  yaml_dom:
    description: "YAML Dom"
    packages:
      yaml_pkg: "YAML Pkg"

tools:
  - name: y_tool
    module: m_path
    class: YClass
    enabled: true
    domain: yaml_dom
    package: yaml_pkg
    role_whitelist:
      - admin
"""
        config_file = tmp_path / "tools.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        class ConcreteMockTool(BaseTool):
            name = "y_tool"
            domain = "yaml_dom"
            package = "yaml_pkg"
            role_whitelist = ["admin"]

            def run(self, **kwargs):
                return "running..."

        mock_mod = MagicMock()
        mock_mod.YClass = ConcreteMockTool
        mock_import_module.return_value = mock_mod

        tf = HierarchicalToolFactory()
        load_tools_from_yaml(str(config_file), factory=tf)

        loaded_tool = tf.get_tool("y_tool", user_role="admin")
        assert loaded_tool is not None
        assert loaded_tool.name == "y_tool"
        assert loaded_tool.domain == "yaml_dom"
        assert loaded_tool.package == "yaml_pkg"

    @patch("factory.tool_factory.importlib.import_module")
    def test_load_tools_from_yaml_exception_handling(self, mock_import_module, tmp_path):
        config_content = """
tools:
  - name: broken_tool
    module: non_exist_module
    class: NonExistClass
    enabled: true
"""
        config_file = tmp_path / "broken_tools.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        mock_import_module.side_effect = ImportError("Module not found")

        tf = HierarchicalToolFactory()
        load_tools_from_yaml(str(config_file), factory=tf)
        assert tf.get_tool("broken_tool") is None


# ============================================================================
# 5. 新补充：真正缺口测试（RBAC 继承 / 防覆盖保护 / 兜底分支等）
# ============================================================================

class LegacySchema:
    """模拟 Pydantic v1 风格，只有 .schema() 没有 .model_json_schema()"""
    @classmethod
    def schema(cls):
        return {"title": "Legacy", "properties": {"x": {"title": "X", "type": "string"}}}


class NeitherSchema:
    """既没有 model_json_schema 也没有 schema 方法"""
    pass


class TestGetJsonSchemaFallbacks:

    def test_legacy_pydantic_v1_branch(self):
        class LegacyTool(BaseTool):
            name = "legacy_tool"
            args_schema = LegacySchema

            def run(self, **kwargs):
                return "ok"

        schema = LegacyTool().get_json_schema()
        assert schema["properties"]["x"] == {"type": "string"}  # title 已剥离

    def test_neither_schema_type_branch(self):
        class WeirdTool(BaseTool):
            name = "weird_tool"
            args_schema = NeitherSchema

            def run(self, **kwargs):
                return "ok"

        schema = WeirdTool().get_json_schema()
        assert schema == {"type": "object", "properties": {}}


class TestRoleVisibilityInheritance:
    """覆盖 _is_tool_visible 的三条分支：工具级公开 / Domain 级继承 / 全放行"""

    def test_tool_level_public_whitelist_allows_anyone(self):
        tf = HierarchicalToolFactory()

        class PublicTool(BaseTool):
            name = "public_tool"
            role_whitelist = []  # 显式公开

            def run(self, **kwargs):
                return "ok"

        tf.register_tool(PublicTool())
        assert tf.get_tool("public_tool", user_role="literally_anyone") is not None

    def test_domain_level_whitelist_inheritance_allowed_and_denied(self):
        tf = HierarchicalToolFactory()
        tf.register_domain_meta("inherit_dom", "desc", role_whitelist=["manager"])

        class InheritingTool(BaseTool):
            name = "inheriting_tool"
            domain = "inherit_dom"
            role_whitelist = None  # 显式继承 Domain

            def run(self, **kwargs):
                return "ok"

        tf.register_tool(InheritingTool())
        assert tf.get_tool("inheriting_tool", user_role="manager") is not None
        assert tf.get_tool("inheriting_tool", user_role="intern") is None

    def test_no_whitelist_anywhere_defaults_to_allow(self):
        tf = HierarchicalToolFactory()

        class OpenTool(BaseTool):
            name = "open_tool"
            domain = "no_meta_dom"
            role_whitelist = None

            def run(self, **kwargs):
                return "ok"

        tf.register_tool(OpenTool())
        assert tf.get_tool("open_tool", user_role="anyone") is not None


class TestRegisterToolGuards:

    def test_register_tool_without_name_raises(self):
        tf = HierarchicalToolFactory()

        class NamelessTool(BaseTool):
            name = ""

            def run(self, **kwargs):
                return "ok"

        with pytest.raises(ValueError):
            tf.register_tool(NamelessTool())

    def test_reregister_inherits_locked_whitelist(self):
        """核心 RBAC 防覆盖保护：YAML 锁定过白名单后，原生 Python 二次注册不能清空它"""
        tf = HierarchicalToolFactory()

        class LockedTool(BaseTool):
            name = "locked_tool"
            role_whitelist = ["admin"]  # 模拟 YAML 加载后锁定的白名单

            def run(self, **kwargs):
                return "v1"

        tf.register_tool(LockedTool())

        class ReregisteredTool(BaseTool):
            name = "locked_tool"
            role_whitelist = None  # 原生代码二次注册，未显式声明

            def run(self, **kwargs):
                return "v2"

        second = ReregisteredTool()
        tf.register_tool(second)

        assert second.role_whitelist == ["admin"]
        assert tf.get_tool("locked_tool", user_role="admin") is not None
        assert tf.get_tool("locked_tool", user_role="guest") is None


class TestPackagesSummaryEdgeCases:

    def test_skips_unknown_domain(self):
        tf = HierarchicalToolFactory()
        result = tf.get_packages_summary_by_domains(["nonexistent_domain"], user_role="admin")
        assert result == []

    def test_raw_object_description_fallback(self):
        """覆盖 tool_val 不是 BaseTool 实例、且未注册到 _flat_tools 时的兜底描述提取"""
        tf = HierarchicalToolFactory()
        tf.register_domain_meta("raw_dom", "desc")
        tf.register_package_meta("raw_dom", "raw_pkg", "pkg desc")

        raw_obj = MagicMock(spec=["description"])
        raw_obj.description = "raw description"

        tf._hierarchy.setdefault("raw_dom", {}).setdefault("raw_pkg", {})["unregistered_name"] = raw_obj

        packages = tf.get_packages_summary_by_domains(["raw_dom"], user_role="admin")
        assert len(packages) == 1
        assert packages[0]["tools"][0]["description"] == "raw description"


class TestToolsMetadataEdgeCases:

    def test_empty_result_branch(self):
        tf = HierarchicalToolFactory()
        tool_names, specs = tf.get_tools_metadata_by_packages([("no_dom", "no_pkg")], user_role="admin")
        assert tool_names == ""
        assert specs == []

        tool_names_json, specs_json = tf.get_tools_metadata_by_packages(
            [("no_dom", "no_pkg")], as_json_string=True, user_role="admin"
        )
        assert tool_names_json == ""
        assert specs_json == "[]"

    def test_as_json_string_true_branch(self):
        tf = HierarchicalToolFactory()

        class JsonTool(BaseTool):
            name = "json_tool"
            domain = "json_dom"
            package = "json_pkg"
            role_whitelist = []

            def run(self, **kwargs):
                return "ok"

        tf.register_tool(JsonTool())
        tool_names, specs_json = tf.get_tools_metadata_by_packages(
            [("json_dom", "json_pkg")], as_json_string=True, user_role="admin"
        )
        assert "json_tool" in tool_names
        assert isinstance(specs_json, str)
        assert "json_tool" in specs_json


class TestLoadToolsFromYamlEdgeCases:

    def test_path_does_not_exist(self, tmp_path):
        tf = HierarchicalToolFactory()
        missing_path = str(tmp_path / "does_not_exist.yaml")
        load_tools_from_yaml(missing_path, factory=tf)  # 应静默记录日志并返回，不抛异常
        assert tf.get_tool("anything") is None

    @patch("factory.tool_factory.importlib.import_module")
    def test_disabled_tool_skipped_and_missing_whitelist_defaults_to_none(self, mock_import_module, tmp_path):
        config_content = """
domains:
  gap_dom:
    description: "Gap Dom"
    packages:
      gap_pkg: "Gap Pkg"

tools:
  - name: disabled_tool
    module: fake_mod
    class: FakeClass
    enabled: false
  - name: no_whitelist_tool
    module: fake_mod
    class: FakeClass
    enabled: true
    domain: gap_dom
    package: gap_pkg
"""
        config_file = tmp_path / "gap_tools.yaml"
        config_file.write_text(config_content, encoding="utf-8")

        class FakeClass(BaseTool):
            name = "placeholder"

            def run(self, **kwargs):
                return "ok"

        fake_module = MagicMock()
        fake_module.FakeClass = FakeClass
        mock_import_module.return_value = fake_module

        tf = HierarchicalToolFactory()
        load_tools_from_yaml(str(config_file), factory=tf)

        # 被禁用的工具绝不能被注册
        assert tf.get_tool("disabled_tool") is None

        # YAML 未配置 role_whitelist -> 显式置 None；且 Domain 未配置白名单 -> 默认放行
        tool = tf.get_tool("no_whitelist_tool", user_role="whoever")
        assert tool is not None
        assert tool.role_whitelist is None