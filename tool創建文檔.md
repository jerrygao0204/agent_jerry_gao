# Agent 三级分级工具系统开发与注册指南 (SOP)

本文档旨在说明 **三级分级工具 (Hierarchical Tool)** 的架构设计、YAML 配置文件与 Python 实现类的字段映射关系、YAML 详细字段解析与校验规范，以及新增工具的标准开发流程。

---

## 一、 架构设计与三级路由机制

本 Agent 架构采用 **Domain -> Package -> Tool** 三级路由机制，旨在降低 LLM 在庞大工具库中的检索注意力消耗（Token 节约与决策准确率提升）：

* **Level 1 (Domain)**：业务领域（如 `rag_knowledge`, `finebi_system`, `data_analytics`），用于一级 Router 快速匹配高层业务场景，过滤无关领域。
* **Level 2 (Package)**：工具包功能分组（如 `metadata_pkg`, `dashboard_pkg`, `chart_pkg`），用于二级 Router 在命中领域下聚焦候选功能包。
* **Level 3 (Atom Tool)**：原子工具层。经过 Level 1/2 路由剪枝后，工厂**仅提取命中 Package 下的工具标准 JSON Schema（Specs List）**注入给 Agent，实现工具上下文的物理隔离与精准调用。

---

## 二、 YAML 配置与 Python 代码映射全景

### 1. 映射全景图

```text
config/tools.yaml                                factory/tools/xxx_tool.py
┌──────────────────────────────────────────┐     ┌──────────────────────────────────────────┐
│ domains:                                 │     │ class CustomTool(BaseTool):              │
│   <domain_id>: ──────────────────────────┼────>│     domain = "<domain_id>"               │
│     description: "..."                   │     │                                          │
│     packages:                            │     │                                          │
│       <package_id>: ─────────────────────┼────>│     package = "<package_id>"             │
│         description: "..."               │     │                                          │
│                                          │     │                                          │
│ tools:                                   │     │                                          │
│   - name: <tool_name> ───────────────────┼────>│     name = "<tool_name>"                 │
│     module: factory.tools.<file_name> ───┼────>│     # 文件名: <file_name>.py              │
│     class: <Class_Name> ─────────────────┼────>│     class <Class_Name>(BaseTool):        │
│     enabled: true                        │     │                                          │
└──────────────────────────────────────────┘     └──────────────────────────────────────────┘

```

---

### 2. `tools.yaml` 配置文件字段深度解析与约束规范

`tools.yaml` 是整个 Agent 三级路由与动态加载的核心大脑。配置文件分为两大部分：**`domains`（领域与工具包元数据定义）** 和 **`tools`（具体原子工具注册清单）**。

#### 2.1 第一块：`domains` 节点解析 (Level 1 & Level 2 路由索引)

`domains` 节点用于声明系统支持的业务领域（Domain）以及领域下辖的工具包（Package）。**这一块的文本直接作为 Prompt 喂给 Level 1 和 Level 2 Router，影响路由匹配精度。**

```yaml
domains:
  rag_knowledge:
    description: "处理用户手册、排错指南、FAQ 等非结构化文档检索"

  finebi_system:
    description: "查询 FineBI 系统仪表板、数据集元数据、表结构及行数信息"
    packages:
      metadata_pkg: "查询数据集元数据、表结构、行数及字段明细"
      dashboard_pkg: "查询系统仪表板、看板、报表组件列表"

  data_analytics:
    description: "对业务数据进行二次统计分析、指标计算与报表生成"
    packages:
      chart_pkg: "针对报表渲染、图表生成的可视化工具包"

  web_search:
    description: "进行网络搜索与公网实时信息查询"

```

* **`domains.<domain_id>` (Level 1 Key)**：
* **含义**：一级业务领域的唯一标识符（如 `finebi_system`）。
* **要求**：全局唯一，采用 `snake_case`（小写字母+下划线）。


* **`description` (Domain 描述)**：
* **含义**：对该业务领域的自然语言总结，供 Level 1 Router 判断意图。
* **要求**：**必填**。避免模糊表述，须精确写明**触发场景**与**数据能力**。


* **`packages` (Level 2 集合)**：
* **含义**：定义该领域下属的工具包字典。若领域较简单，可不定义或仅定义 1 个默认包；若较复杂，需按逻辑拆分多个包。


* **`packages.<package_id>` (Level 2 Key & Value)**：
* **Key (`package_id`)**：工具包的唯一标识（如 `metadata_pkg`）。
* **Value (Package 描述)**：**必填**。简短精准说明该 Package 包含的原子能力，供二级 Router 选择。



#### 2.2 第二块：`tools` 节点解析 (Level 3 原子工具注册)

`tools` 节点是一个列表，每个元素代表一个要自动加载到工厂（`ToolFactory`）中的 Python 原子工具。

```yaml
tools:
  - name: search_knowledge_base
    module: factory.tools.rag_tool
    class: RAGKnowledgeSearchTool
    domain: rag_knowledge
    package: knowledge_search_pkg
    enabled: true

  - name: get_finebi_dashboards
    module: factory.tools.api_tool
    class: FineBIDashboardTool
    domain: finebi_system
    package: dashboard_pkg
    enabled: true

  - name: get_dataset_summary
    module: factory.tools.dataset_summary
    class: DatasetSummaryTool
    domain: finebi_system
    package: metadata_pkg
    enabled: true

  - name: web_search
    module: factory.tools.web_search_tool
    class: WebSearchTool
    domain: web_search
    package: search_pkg
    enabled: true

```

| 字段名称 (Key) | 数据类型 | 是否必填 | 含义与要求描述 | 校验规则 / 避坑指南 |
| --- | --- | --- | --- | --- |
| **`name`** | String | **是** | **工具唯一标识**。LLM 生成 `Action: <tool_name>` 时使用的识别名称。 | 必须与 Python 工具类中的 `name` 属性**完全一致**。全局唯一，推荐使用 `snake_case`。 |
| **`module`** | String | **是** | **Python 模块导入路径**。底层利用 `importlib.import_module()` 动态加载。 | 对应磁盘文件路径。必须以 `factory.tools.` 开头，**严禁带 `.py` 后缀**。Linux 系统严格**区分大小写**！ |
| **`class`** | String | **是** | **模块内的 Python 类名**。实例化时通过 `getattr(module, class_name)` 调用。 | 必须与 Python 文件中继承 `BaseTool` 的 Class 名称**严格一致（区分大小写）**。 |
| **`domain`** | String | **是** | **归属的 Level 1 业务领域**。用于路由关系映射与归类。 | **必须与 YAML 上方 `domains:` 节点下定义的 Key 绝对匹配**。填错会导致路由寻找失败。 |
| **`package`** | String | **是** | **归属的 Level 2 工具包**。用于实现按包（Package）剪枝。 | **必须与上方 `domains.<domain_id>.packages` 中定义的 Key 匹配**。实现 `(domain, package)` 严格二元绑定。 |
| **`enabled`** | Boolean | **是** | **工具上线/下线热开关**。`true` 表示注册加载，`false` 表示忽略。 | 调试新工具或临时下线故障工具时设为 `false`，**无需删除代码，无侵入式下线**。 |

#### 2.3 YAML 三大核心校验规则

在新增或修改 `tools.yaml` 时，请务必遵循以下校验规则：

```text
【规则 1：Domain 存在性校验】 
tools[i].domain 必须真实存在于 domains 字典的 Key 中。

【规则 2：Package 归属校验】 
tools[i].package 必须存在于 domains[tools[i].domain].packages 字典的 Key 中。

【规则 3：名称与路径一致性校验】 
tools[i].module 对应的 .py 文件必须真实存在于 factory/tools/ 目录下，且类名 tools[i].class 存在于该文件中。

```

---

## 三、 工具开发三步法 (SOP)

```text
[步骤 1] 编写工具类 (.py)  ──>  [步骤 2] 配置 YAML 路由 (.yaml)  ──>  [步骤 3] 运行自动化测试验证

```

### 步骤 1：编写工具实现文件

在 `factory/tools/` 目录下新建 Python 模块文件，继承 `BaseTool` 并定义 `Pydantic` 入参模型。

* **文件路径**：`factory/tools/demo_calculator_tool.py`

```python
import logging
from typing import Type, Optional
from pydantic import BaseModel, Field
from factory.tools.base_tool import BaseTool  # 📌 统一使用基础工具类

logger = logging.getLogger("DemoCalculatorTool")

# 1. 定义入参 Schema (严格标注 Field 描述，利于 LLM 正确提取参数)
class CalculatorInput(BaseModel):
    expression: str = Field(description="需要计算的数学表达式，例如: '12 * (3 + 4)'")
    precision: int = Field(default=2, description="保留的小数位数，默认 2 位")

# 2. 定义工具核心实现类
class DemoCalculatorTool(BaseTool):
    name: str = "demo_calculator"
    description: str = "用于执行基础数学计算与数值表达式求值的计算器工具"
    domain: str = "data_analytics"     # Level 1 业务领域
    package: str = "analytics_pkg"     # Level 2 工具包分类
    args_schema: Optional[Type[BaseModel]] = CalculatorInput
    is_read_only: bool = True          # 安全控制：只读操作设为 True

    def run(self, expression: str, precision: int = 2, **kwargs) -> str:
        """
        核心业务逻辑入口 (必须自带 try-except 优雅降级)
        """
        try:
            # 安全计算逻辑
            result = round(eval(expression), precision)
            logger.info(f"🧮 [DemoCalculatorTool] 计算成功: {expression} = {result}")
            return f"计算结果: {result}"
        except Exception as e:
            logger.error(f"❌ [DemoCalculatorTool] 计算异常: {str(e)}")
            # 📌 捕获异常并返回友好字符串，允许 LLM 在下一步 Thought 中根据 Observation 进行自我纠错
            return f"工具执行失败，原因: {str(e)}"

```

#### 💡 工具开发 4 大规范 (Best Practices)

1. **字段描述强约束**：`Pydantic` 的 `Field(description="...")` 极为关键，LLM 会直接根据该描述生成 Action Input，**必须清晰说明输入格式与示例**。
2. **防守型参数设计**：对可选参数必须设置默认值（如 `precision: int = 2`），且在 `run()` 方法签名中统一包含 `**kwargs` 吸收 LLM 可能多传的无关参数。
3. **输出异常捕获机制**：`run()` 方法内部**切勿抛出未捕获的 Unhandled Exception**，应捕获后返回明确的错误提示字符串，以便 Agent 在 ReAct 循环中自动纠错。
4. **只读状态声明**：涉及写入、删除、高危命令（如执行 Python 代码、数据库 Update）的工具，必须显式声明 `is_read_only = False`，以供沙箱安全机制识别。

---

### 步骤 2：配置 YAML 动态注册

打开 `config/tools.yaml` 配置文件，追加新工具的声明。**无需修改 Python 注册代码**。

```yaml
# Level 1 & Level 2 领域描述
domains:
  data_analytics:
    description: "对业务数据进行二次统计分析、指标计算与报表生成"
    packages:
      analytics_pkg: "针对数据集元数据、行数、列数及结构信息的统计分析工具包"

# Level 3 工具注册清单
tools:
  - name: demo_calculator
    module: factory.tools.demo_calculator_tool   # 📌 必须与 Python 文件名绝对一致 (区分大小写)
    class: DemoCalculatorTool                    # 📌 必须与 Python class 类名严格一致
    domain: data_analytics
    package: analytics_pkg
    enabled: true                                 # 热开关：false 表示临时下线

```

---

### 步骤 3：运行自动化验证

在项目根目录下直接执行注册单体测试脚本：

```bash
python factory/tool_registry.py

```

**预期成功输出**：

```text
============================================================
🔍 [YAML 读取测试] 开始验证...
📌 解析出的项目根目录: /workspace/hf-conda/RAG/问答机器人
📌 目标 YAML 绝对路径: /workspace/hf-conda/RAG/问答机器人/config/tools.yaml
============================================================
✅ [成功] 成功定位文件！
2026-08-29 00:45:10,120 - [INFO] - 🛠️ [ToolFactory] 注册成功: [data_analytics -> analytics_pkg -> demo_calculator]
🎉 [成功] 成功加载 YAML 并注册工具！
============================================================

```

---

## 四、 常见报错与避坑指南

| 现象 / 报错信息 | 根本原因 (Root Cause) | 解决办法 (Solution) |
| --- | --- | --- |
| **`UnboundLocalError: local variable 'selected_packages' referenced before assignment`** | 在 `run_stream` 中外部注入了 `tools_schema`（如沙盒模式），跳过了路由逻辑，但后续代码或异常处理中仍引用了未初始化的 `selected_packages` / `pkg_names` | 在 `run_stream` 方法**最入口处**统一显式初始化 `selected_packages = []` 和 `pkg_names = []` 兜底值。 |
| **`ModuleNotFoundError: No module named 'factory'`** | 未包含根路径，或直接在子目录下执行 `.py` 脚本 | 在入口头部注入 `sys.path.insert(0, project_root)`，或使用 `python -m factory.tool_registry` 运行。 |
| **`FileNotFoundError: /workspace/config/tools.yaml`** | 使用了相对路径 `config/tools.yaml`，导致基于终端执行目录查找失败 | 改用 `os.path.join(project_root, "config", "tools.yaml")` 绝对路径拼接。 |
| **`No module named 'factory.tools.xxx'`** | Linux 系统对文件名大小写敏感，YAML 中 `module` 写错大小写或误加/漏加了前缀 | 检查 YAML 中 `module` 与磁盘上的 `.py` 实际文件名，确保大小写与字符完全一致。 |
| **`ValidationError` / LLM 传入参数校验失败** | LLM 生成的 JSON 参数不符合 Pydantic 定义（例如将字符串传给了 list，或漏传必需字段） | 在工具类的 `run` 函数参数中尽量设置默认值（如 `Optional[str] = None`），并在 `args_schema` 中将描述写得更具体。 |

```