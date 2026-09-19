# 📄 DGX SPARK vLLM 與 LiteLLM 自動化部署腳本技術文件

> **架構設計核心聲明** ： 本部署腳本 `start.sh` 專為 DGX SPARK NVIDIA NGC 環境設計。遵循**「配置即契約 (Configuration as a Contract)」** 與**「運維/用戶權限分離」** 原則：由運維人員在啟動時精準掌控硬體資源分配與服務清單，確保後端 vLLM 實體進程與 LiteLLM 網關路由嚴格對齊，杜絕終端用戶越權幹預或自動 Fallback 帶來的結果不可控。 

## 1\. 核心架構與設計背景 (Background & Architecture)

本脚本旨在解决在单节点多模型并发加载时容易发生的显存溢出 (OOM)、端口冲突及路径转义失效等痛点。

**核心方法 (Method)** ：采用宿主机-容器分离控制与模型串行加载 (Serial Loading) 策略。通过前置配置解析与严格的 Fail-Fast 机制，结合带重试机制的 vllm serve 进程挂载，实现多模型平稳串行初始化。

**技术栈 (Technology Stack)** ：

  * **容器化引擎** ：Docker (`nvcr.io/nvidia/vllm:26.07-py3`)
  * **推理框架** ：vLLM (Virtual Large Language Model) (基于 FLASH_ATTN 后端)
  * **网关代理** ：LiteLLM Proxy (统一管理多模型 OpenAI 兼容接口，端口 4000)
  * **向量检索** ：Milvus (Docker Compose 编排)



## 2\. 脚本模块设计與流程解析 (Workflow & Modules)

脚本整体划分为多个核心执行阶段，各模块协同保障自动化流水线的健壮性：

模块阶段 | 对应执行逻辑 | 核心技术点与优化说明  
---|---|---  
**0\. 端口净化** | `fuser -k` 清理 | 自动清理宿主机 4000（LiteLLM）及 8000-8003（vLLM）端口，防止残留进程占用。  
**1\. 目录初始化与配置** | `mkdir`, `cat << 'EOF'` | 确保宿主机 hf-conda/litellm 目录真实落盘，自动生成或加载包含 qwen3-4b、qwen3-vl-4b 等的 `litellm_config.yaml`。  
**2\. Milvus 数据库重置** | `docker compose up -d` | 自动检查并重启 Milvus 向量检索服务，确保在保留现有持久化数据的前提下完成服务刷新。  
**3\. LiteLLM 网关拉起** | `docker run litellm_proxy` | 在宿主机后台启动统一代理网关（端口 4000），将多路 vLLM 后端路由聚合为标准 OpenAI API 格式。  
**4\. NGC 容器环境自检** | `docker run dgx_dev_session` | 挂载本地缓存及工作目录，通过离线环境变量（`HF_HUB_OFFLINE=1`）强行锁死内网离线运行模式。  
**5\. 虚拟环境与依赖构建** | `/workspace/hf-conda/venv312` | 在容器内部局域网隔离环境中配置 Python 3.12 虚拟环境，并静默安装 sentence-transformers 等辅助工具。  
**6\. 串行多模型动态加载** | `/usr/local/bin/run-multi-vllm` | 核心交互器：提供单模型、双模型串行、三模型串行三种模式，带显存动态清理与健康检查重试机制。  
  
## 3\. 關鍵代碼段對比與技術亮點 (Technical Highlights)

### 3.1 規避 Shell 變數轉義 Bug 的 Heredoc 機制

在向容器注入多行脚本时，原代码采用了严格的单引号定界符 (`<< 'HOST_EOF_VENV' / << 'HOST_EOF_WRITER'`)。

  * **优势** ：强制宿主机 Bash 不展开内部的 `$` 变量与双引号,保证所有嵌套的动态解析脚本能够原封不动、无转义污染地写入容器内部，徹底杜绝了历史版本中因变量置空导致的 `--port ''` 崩溃事故。



### 3.2 智能 YAML 解析與路徑對齊

脚本内置了稳健的解析逻辑：

  * **精准定位** ：透过解析 LiteLLM 配置中的 `api_base` 端口，并利用字符串匹配在 `hf_cache/hub` 的 `snapshots` 目录中自动抓取模型绝对路径。
  * **硬核兜底 (Hardcoded Fallback)** ：若因动态索引失败导致路径为空，脚本会自动针对 vl、embedding 或默认模型分流至已知的稳定快照绝对路径，保障系统绝对不会因路径断裂而中止。



### 3.3 顯存動態清理與重試閉環 (launch_model_with_retry)

针对大模型连续启动容易导致的显存碎片与峰值溢出问题，实现了工程化闭环：

  * 每成功启动一个模型，都会强制执行 `python3 -c "import torch; torch.cuda.empty_cache()"` 释放显存暂存。
  * 引入 3 次重试机制 (Max Retries = 3)，当某端口健康检查 (`/health`) 超时或崩溃时，自动强杀残留的 vllm 进程并等待冷却重试，提升复杂生产环境下的容灾能力。



## 4\. 資源讀取與寫入路徑清單 (Files & Paths)

以下是該 `start.sh` 腳本在運行過程中讀取 (Read) 和生成/寫入 (Write/Generate) 的所有文件與路徑清單：

### 一、 讀取的文件與路徑清單 (Read)

腳本在運行時需要依賴並讀取以下主機或容器內的外部資源：

  * **LiteLLM 配置文件**   
路徑：`/home/gaozheng/venv/hf-conda/litellm/litellm_config.yaml`（或容器內掛載的 `/workspace/litellm/litellm_config.yaml`）   
作用：定義可調用的模型清單、真實模型名稱、`api_base` 端口以及代理參數。 
  * **Hugging Face 模型快照與權重檔案**   
路徑：`/workspace/hf-conda/hf_cache/hub/` 及其子目錄（如 `models--Qwen--Qwen3-4B`、`models--Qwen--Qwen3-VL-4B-Instruct` 等的 `snapshots/` 目錄）   
作用：vLLM 啟動時讀取本地的模型權重與 `generation_config.json`。 
  * **Pip 離線安裝包**   
路徑：`/workspace/hf-conda/pip_wheels/`   
作用：在虛擬環境初始化時提供離線 Python 依賴包。 



### 二、 生成與寫入的文件清單 (Write / Generate)

腳本在執行過程中自動建立或寫入以下配置、日誌與執行檔：

  * **LiteLLM 默認配置文件**   
路徑：`/home/gaozheng/venv/hf-conda/litellm/litellm_config.yaml`   
作用：若宿主機缺少該文件時，腳本會自動寫入包含 qwen3-4b、qwen3-vl-4b 和 qwen3-embedding-4b 的默認 YAML 模板。 
  * **Python 虛擬環境**   
路徑：`/workspace/hf-conda/venv312/`   
作用：在容器內部獨立建立的 Python 3.12 運行環境。 
  * **多模型串行啟動工具**   
路徑：`/usr/local/bin/run-multi-vllm`   
作用：動態寫入並賦予執行權限的容器內交互式管理腳本，用於執行串行加載、顯存清理與健康檢查。 
  * **vLLM 模型運行日誌文件**   
路徑：   
`/workspace/vllm_model1.log`   
`/workspace/vllm_model2.log`   
`/workspace/vllm_model3.log`   
作用：分別記錄各個串行啟動的 vLLM 服務實例標準輸出與錯誤日誌，便於異常排查。 
  * **Torch 編譯與 AOT 緩存**   
路徑：`/root/.cache/vllm/torch_compile_cache/`   
作用：vLLM 運行時自動生成的模型計算圖編譯緩存，用於加速後續推理。 



## 5\. 總結與操作指南 (Summary & Operations)

  * **運行方式** ：直接在宿主机拥有执行权限的终端下运行： 
        
        bash start.sh

  * **交互选择** ：脚本最后会自动连接容器终端并唤起多模型启动器 (`run-multi-vllm`)，支持用户根据显存余量自主选择 [1] 单模型、[2] 双模型串行 或 [3] 三模型串行 模式。


