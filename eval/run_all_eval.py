# eval/run_all_eval.py
import os
import sys
import json
import logging

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from eval.eval_retriever import RetrieverEvaluator
from eval.eval_generator import GeneratorEvaluator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

def run_pipeline():
    dataset_path = os.path.join(CURRENT_DIR, "eval_dataset.json")
    output_dir = os.path.join(CURRENT_DIR, "reports")
    
    # Step 1: 執行檢索評測
    logging.info("=== 阶段 1/2: 開始執行 Retriever 評測 ===")
    ret_evaluator = RetrieverEvaluator()
    ret_summary, _ = ret_evaluator.run_evaluation(dataset_path=dataset_path, output_dir=output_dir)
    
    # Step 2: 執行生成質量評測
    logging.info("=== 阶段 2/2: 開始執行 Generator 評測 ===")
    gen_evaluator = GeneratorEvaluator()
    gen_summary, _ = gen_evaluator.run_evaluation(dataset_path=dataset_path, output_dir=output_dir)
    
    # 匯總全鏈路 Metrics
    full_report = {
        "retriever_metrics": ret_summary,
        "generator_metrics": gen_summary
    }
    
    summary_path = os.path.join(output_dir, "full_eval_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, ensure_ascii=False, indent=2)
        
    logging.info(f"🎉 全鏈路自動化評測完成！綜合報告已存至: {summary_path}")

if __name__ == "__main__":
    run_pipeline()