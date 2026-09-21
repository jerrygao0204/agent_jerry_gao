# memory_growth/app.py
import argparse
import logging
import sys
import time
from pathlib import Path

# 📌 關鍵修正：將當前文件所在目錄加入 sys.path，防止模組找不到同級依賴
current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

from path_config import UserMemoryPathConfig
from generator.llm_client import LLMClient

# 動態匯入 Phase 1, Phase 2, Phase 3 核心模組
try:
    import importlib
    extractor_mod = importlib.import_module("1_extractor")
    mapper_mod = importlib.import_module("2_layer_mapper")
    builder_mod = importlib.import_module("3_context_builder")

    LLMClientAdapter = extractor_mod.LLMClientAdapter
    FactExtractor = extractor_mod.FactExtractor
    LayerMapper = mapper_mod.LayerMapper
    ContextBuilder = builder_mod.ContextBuilder
except ImportError as e:
    logging.error(f"❌ 模組載入失敗，請確認 1_extractor.py, 2_layer_mapper.py, 3_context_builder.py 及 llm_guard.py 檔案均在同級或 PythonPath 中: {e}")
    sys.exit(1)


# ==========================================
# 流水線控制器 (Memory Growth Pipeline)
# ==========================================

class MemoryGrowthPipeline:
    """Memory Growth 記憶成長流水線主控制器"""

    def __init__(self, user_id: str, 
                 model_name: str = "qwen3-32b",
                 prompt_hub_path: str = "/workspace/hf-conda/RAG/agent_jerry_gao/config/prompt_hub.yaml"
                 ):
        self.user_id = user_id
        self.model_name = model_name
        self.prompt_hub_path = prompt_hub_path
        self.paths = UserMemoryPathConfig(user_id=user_id)

        # 初始化 LLM 網關與適配器
        logger.info(f"🚀 初始化 LLM 網關 (指定模型: {self.model_name})...")
        self.llm_client = LLMClient(default_model_name=self.model_name)
        self.llm_adapter = LLMClientAdapter(
            llm_client=self.llm_client,
            model_name=self.model_name,
            clean_think=True
        )

    def run_pipeline(
        self,
        is_full_run: bool = False,
        window_size: int = 5,
        overlap_size: int = 1
    ):
        """執行完整 Phase 1 -> Phase 2 -> Phase 3 流水線"""
        start_time = time.time()
        print("\n" + "=" * 65)
        print(f"🧠 [Memory Growth Pipeline] 啟動用戶記憶成長流水線 [{self.user_id}]")
        print(f"📌 模式: {'🔄 全量重新抽取 (Full Run)' if is_full_run else '⚡ 增量抽取 (Incremental Run)'}")
        print(f"📁 數據目錄: {self.paths.data_dir}")
        print(f"📜 Prompt Hub: {self.prompt_hub_path}")
        print("=" * 65 + "\n")

        # ----------------------------------------------------
        # Phase 1: 高召回率事實與關係圖譜抽取 (含 LLMGuard 護航)
        # ----------------------------------------------------
        logger.info("=== 📍 Phase 1: 執行事實與圖譜抽取 (1_extractor.py) ===")
        p1_start = time.time()
        extractor = FactExtractor(
            data_dir=str(self.paths.data_dir), 
            llm_adapter=self.llm_adapter,
            prompt_hub_path=self.prompt_hub_path
        )
        extracted_facts = extractor.run(
            user_id=self.paths.user_id,
            state_path=str(self.paths.processed_state_path),
            output_path=str(self.paths.facts_path),
            window_size=window_size,
            overlap_size=overlap_size,
            is_full_run=is_full_run
        )
        p1_cost = time.time() - p1_start
        logger.info(f"✅ Phase 1 完成 | 耗時: {p1_cost:.2f}s | 本次產出/處理 Raw Facts: {len(extracted_facts)} 條\n")

        # ----------------------------------------------------
        # Phase 2: 規範化清洗與三層語境映射 (含 LLMGuard 護航)
        # ----------------------------------------------------
        logger.info("=== 📍 Phase 2: 執行規範化清洗與三層語境映射 (2_layer_mapper.py) ===")
        p2_start = time.time()
        mapper = LayerMapper(
            llm_adapter=self.llm_adapter,
            prompt_hub_path=self.prompt_hub_path
        )
        layered_context = mapper.run(
            facts_path=str(self.paths.facts_path),
            output_path=str(self.paths.layered_context_path)
        )
        p2_cost = time.time() - p2_start
        logger.info(f"✅ Phase 2 完成 | 耗時: {p2_cost:.2f}s | 頂層語境 Key: {list(layered_context.keys())}\n")

        # ----------------------------------------------------
        # Phase 3: 模組化 Prompt 渲染與 System Context 導出
        # ----------------------------------------------------
        logger.info("=== 📍 Phase 3: 執行系統語境模組化渲染 (3_context_builder.py) ===")
        p3_start = time.time()
        builder = ContextBuilder(
            data_dir=str(self.paths.data_dir),
            prompt_hub_path=self.prompt_hub_path
        )
        context_text = builder.build(
            layered_context_path=self.paths.layered_context_path,
            output_text_path=self.paths.user_prompt_context_path
        )
        p3_cost = time.time() - p3_start
        logger.info(f"✅ Phase 3 完成 | 耗時: {p3_cost:.2f}s | 生成 Context 長度: {len(context_text)} 字元\n")

        # ----------------------------------------------------
        # 執行總結
        # ----------------------------------------------------
        total_cost = time.time() - start_time
        print("=" * 65)
        print("🎉 [Memory Growth Pipeline] 流水線全線貫通執行成功！")
        print(f"⏱️ 總耗時: {total_cost:.2f}s (Phase1: {p1_cost:.1f}s, Phase2: {p2_cost:.1f}s, Phase3: {p3_cost:.1f}s)")
        print("📄 最終系統語境 (System Context) 已生成至:")
        print(f"👉 {self.paths.user_prompt_context_path}")
        print("=" * 65 + "\n")


# ==========================================
# CLI 入口
# ==========================================

def main():
    parser = argparse.ArgumentParser(description="Memory Growth 自動化流水線調度器 (app.py)")
    parser.add_argument("--user", type=str, default="admin", help="指定用戶 ID (預設: admin)")
    parser.add_argument("--model", type=str, default="qwen3-32b", help="指定 LLM 模型 (預設: qwen3-32b)")
    parser.add_argument("--prompt-hub-path", type=str, 
                        default="/workspace/hf-conda/RAG/agent_jerry_gao/config/prompt_hub.yaml", 
                        help="指定 Prompt Hub 配置文件路徑")
    parser.add_argument("--full-run", action="store_true", help="是否強制執行全量重新抽取 (覆蓋歷史 Hash)")
    parser.add_argument("--window-size", type=int, default=5, help="Phase 1 滑動窗口大小 (預設: 5)")
    parser.add_argument("--overlap-size", type=int, default=1, help="Phase 1 滑動窗口重疊大小 (預設: 1)")
    parser.add_argument("--debug", action="store_true", help="開啟 Debug 級別日誌")

    args = parser.parse_args()

    # 設置日誌格式
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    global logger
    logger = logging.getLogger(__name__)

    # 執行流水線
    pipeline = MemoryGrowthPipeline(
        user_id=args.user, 
        model_name=args.model,
        prompt_hub_path=args.prompt_hub_path
    )
    pipeline.run_pipeline(
        is_full_run=args.full_run,
        window_size=args.window_size,
        overlap_size=args.overlap_size
    )


# ==========================================
# 常用命令行操作說明與測試選擇
# ==========================================

if __name__ == "__main__":
    # 📌 定義本地開發/調試時使用的標準 Prompt Hub 路徑

    # 1. 獲取當前 app.py 的絕對路徑
    # 当前文件路径: /workspace/hf-conda/RAG/agent_jerry_gao/memory_growth/app.py
    current_file_path = Path(__file__).resolve()

    # 2. 定義標準絕對路徑 (硬編碼備用)
    STANDARD_PROMPT_HUB_PATH = Path("/workspace/hf-conda/RAG/agent_jerry_gao/config/prompt_hub.yaml")

    # 3. 動態相對轉換: app.py (memory_growth) -> 父目錄 (agent_jerry_gao) -> config/prompt_hub.yaml
    dynamic_prompt_hub_path = current_file_path.parent.parent / "config" / "prompt_hub.yaml"

    # 4. 判斷邏輯：優先使用動態計算且存在的路徑，否則回退至標準絕對路徑
    if dynamic_prompt_hub_path.exists():
        DEFAULT_PROMPT_HUB = str(dynamic_prompt_hub_path)
    elif STANDARD_PROMPT_HUB_PATH.exists():
        DEFAULT_PROMPT_HUB = str(STANDARD_PROMPT_HUB_PATH)
    else:
        # 兩者皆不存在時的兜底 (指向動態路徑)
        DEFAULT_PROMPT_HUB = str(dynamic_prompt_hub_path)

    # -------------------------------------------------------------------------
    # 根據測試情境，取消註解 (Uncomment) 下方對應的情境組合進行快速測試：
    # -------------------------------------------------------------------------

    # 【情境 1】：預設增量運行 (只處理新對話)
    # sys.argv = ["app.py", "--user", "admin", "--prompt-hub-path", DEFAULT_PROMPT_HUB]

    # 【情境 2】：全量重洗運行 (清空舊 Hash 重新抽取全量)
    sys.argv = ["app.py", "--user", "admin", "--full-run", "--prompt-hub-path", DEFAULT_PROMPT_HUB]

    # 【情境 3】：指定模型与調優窗口參數
    # sys.argv = ["app.py", "--user", "gao", "--model", "qwen3-32b", "--prompt-hub-path", DEFAULT_PROMPT_HUB, "--window-size", "8", "--overlap-size", "2"]

    # 【情境 4】：開啟 Debug 日誌模式
    # sys.argv = ["app.py", "--user", "admin", "--debug", "--prompt-hub-path", DEFAULT_PROMPT_HUB]

    main()