import os
import sys
import json
import logging
import time
import pandas as pd
from typing import List, Dict, Any, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from search.retriever import FineBIRetriever
from search.reranker import FineBIReranker
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

class RetrieverEvaluator:
    def __init__(
        self,
        milvus_host: str = None,
        milvus_port: str = None,
        collection_name: str = None,
        cuda_device: str = "0"
    ):
        self.host = milvus_host or os.getenv("MILVUS_HOST", "172.17.0.1")
        self.port = milvus_port or os.getenv("MILVUS_PORT", "19530")
        self.collection = collection_name or os.getenv("MILVUS_COLLECTION", "finebi_knowledge_chunks")
        
        logging.info("🚀 初始化 Evaluator: 載入 FineBIRetriever 與 FineBIReranker...")
        self.retriever = FineBIRetriever(
            milvus_host=self.host,
            milvus_port=self.port,
            collection_name=self.collection,
            cuda_device=cuda_device
        )
        self.reranker = FineBIReranker(cuda_device=cuda_device, min_prob=0.25)
        
        self._warmup()

    def _warmup(self):
        logging.info("🔥 正在執行端到端檢索預熱 (Dummy Search Warm-up)...")
        try:
            t_start = time.perf_counter()
            dummy_chunks = self.retriever.hybrid_search(query="数据预警", top_k=1)
            dummy_doc = dummy_chunks if dummy_chunks else [{"content": "数据预警", "chunk_id": "warmup_0"}]
            _ = self.reranker.rerank(query="数据预警", documents=dummy_doc, top_n=1, disable_filtering=True)
            
            warmup_ms = (time.perf_counter() - t_start) * 1000
            logging.info(f"✅ 端到端檢索與重排預熱完成，耗時: {warmup_ms:.2f} ms")
        except Exception as e:
            logging.warning(f"⚠️ 預熱階段捕獲異常 (可忽略): {e}")

    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"❌ 評測集檔案不存在: {dataset_path}")
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        logging.info(f"📚 成功載入評測集，共 {len(data)} 筆測試 Sample。")
        return data

    def evaluate_sample(
        self, 
        sample: Dict[str, Any], 
        top_k_ret: int = 10, 
        top_k_rerank: int = 3
    ) -> Dict[str, Any]:
        query = sample.get("query", "")
        expected_ids = set(sample.get("expected_chunk_ids", []))
        filter_expr = sample.get("filter_expr", None)

        t_total_start = time.perf_counter()
        
        # 1. 初篩 (Hybrid Search)
        t_ret_start = time.perf_counter()
        raw_chunks = self.retriever.hybrid_search(
            query=query, 
            top_k=top_k_ret, 
            filter_expr=filter_expr
        )
        retrieval_ms = (time.perf_counter() - t_ret_start) * 1000

        # 2. 精排 (Rerank) - 評測模式傳入 disable_filtering=True
        t_rerank_start = time.perf_counter()
        reranked_chunks = self.reranker.rerank(
            query=query, 
            documents=raw_chunks, 
            top_n=top_k_rerank,
            disable_filtering=True
        )
        rerank_ms = (time.perf_counter() - t_rerank_start) * 1000
        
        total_ms = (time.perf_counter() - t_total_start) * 1000

        sample_id = sample.get("query_id", sample.get("id", "N/A"))
        logging.info(
            f"⏱️ Sample [{sample_id}] | "
            f"Hybrid Search: {retrieval_ms:.2f} ms ({len(raw_chunks)} Chunks) | "
            f"Rerank: {rerank_ms:.2f} ms ({len(reranked_chunks)} Chunks) | "
            f"Total: {total_ms:.2f} ms"
        )

        retrieved_ids = [c.get("chunk_id") for c in reranked_chunks if isinstance(c, dict) and "chunk_id" in c]

        hit = any(cid in expected_ids for cid in retrieved_ids)
        mrr = 0.0
        for rank, cid in enumerate(retrieved_ids, 1):
            if cid in expected_ids:
                mrr = 1.0 / rank
                break

        return {
            "sample_id": sample_id,
            "query": query,
            "expected_ids": list(expected_ids),
            "retrieved_ids": retrieved_ids,
            "hit": 1 if hit else 0,
            "mrr": round(mrr, 4),
            "retrieval_ms": round(retrieval_ms, 2),
            "rerank_ms": round(rerank_ms, 2),
            "latency_ms": round(total_ms, 2)
        }

    def run_evaluation(
        self, 
        dataset_path: str, 
        top_k_ret: int = 10, 
        top_k_rerank: int = 3,
        output_dir: str = None
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        dataset = self.load_dataset(dataset_path)
        results = []

        logging.info(f"⚡ 開始執行檢索評測 (Top-K Ret={top_k_ret}, Rerank={top_k_rerank})...")
        for sample in dataset:
            res = self.evaluate_sample(sample, top_k_ret=top_k_ret, top_k_rerank=top_k_rerank)
            results.append(res)

        df = pd.DataFrame(results)
        
        total_samples = len(df)
        avg_hit_rate = df["hit"].mean() if total_samples > 0 else 0.0
        avg_mrr = df["mrr"].mean() if total_samples > 0 else 0.0
        avg_retrieval_ms = df["retrieval_ms"].mean() if total_samples > 0 else 0.0
        avg_rerank_ms = df["rerank_ms"].mean() if total_samples > 0 else 0.0
        avg_latency = df["latency_ms"].mean() if total_samples > 0 else 0.0

        summary = {
            "total_samples": total_samples,
            "top_k_retrieval": top_k_ret,
            "top_k_rerank": top_k_rerank,
            "hit_rate": round(avg_hit_rate, 4),
            "mrr": round(avg_mrr, 4),
            "avg_retrieval_ms": round(avg_retrieval_ms, 2),
            "avg_rerank_ms": round(avg_rerank_ms, 2),
            "avg_latency_ms": round(avg_latency, 2)
        }

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            report_json_path = os.path.join(output_dir, "retriever_eval_report.json")
            report_md_path = os.path.join(output_dir, "retriever_eval_report.md")

            with open(report_json_path, "w", encoding="utf-8") as f:
                json.dump({"summary": summary, "details": results}, f, ensure_ascii=False, indent=2)

            md_content = f"""# 📊 RAG 檢索器評測報告 (Retriever Evaluation Report)

### 📈 核心效能指標
- **總測試 Sample 數**: `{summary['total_samples']}`
- **初篩 Top-K**: `{summary['top_k_retrieval']}`
- **精排 Top-K**: `{summary['top_k_rerank']}`
- **🎯 平均命中率 (Hit Rate)**: `{summary['hit_rate'] * 100:.2f}%`
- **🥇 平均倒數排名 (MRR)**: `{summary['mrr']:.4f}`

### ⏱️ 耗時拆解
- **平均初篩耗時 (Hybrid Search)**: `{summary['avg_retrieval_ms']} ms`
- **平均重排耗時 (Rerank)**: `{summary['avg_rerank_ms']} ms`
- **⚡ 平均總檢索耗時**: `{summary['avg_latency_ms']} ms`

### 📝 明細列表
{df[['sample_id', 'query', 'hit', 'mrr', 'retrieval_ms', 'rerank_ms', 'latency_ms']].to_markdown(index=False)}
"""
            with open(report_md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
                
            logging.info(f"✅ 評測報告已成功導出至: {output_dir}")

        return summary, df

if __name__ == "__main__":
    dataset_file = os.path.join(CURRENT_DIR, "eval_dataset.json")
    output_directory = os.path.join(CURRENT_DIR, "reports")

    evaluator = RetrieverEvaluator()
    summary_metrics, _ = evaluator.run_evaluation(
        dataset_path=dataset_file,
        top_k_ret=10,
        top_k_rerank=3,
        output_dir=output_directory
    )
    print("\n" + "="*40)
    print("📊 評測結果摘要 (Summary):")
    print(json.dumps(summary_metrics, ensure_ascii=False, indent=2))
    print("="*40)