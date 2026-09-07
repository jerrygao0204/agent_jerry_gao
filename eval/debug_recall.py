# eval/debug_recall.py
import os
import sys
import json

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from search.retriever import FineBIRetriever
from search.reranker import FineBIReranker

def debug_eval_dataset(dataset_path: str):
    retriever = FineBIRetriever(cuda_device="0")
    reranker = FineBIReranker(cuda_device="0", min_prob=0.25)

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    print("\n" + "="*30 + " 初篩與精排診斷報告 " + "="*30)
    for sample in dataset:
        q_id = sample.get("query_id")
        query = sample.get("query")
        expected = set(sample.get("expected_chunk_ids", []))

        # 1. 執行混合檢索初篩 (Hybrid Search)
        raw_chunks = retriever.hybrid_search(query=query, top_k=10)
        raw_ids = [c.get("chunk_id") for c in raw_chunks]
        raw_hit = any(cid in expected for cid in raw_ids)

        # 2. 執行重排 (Rerank)
        ranked_chunks = reranker.rerank(query=query, documents=raw_chunks, top_n=3, disable_filtering=True)
        ranked_ids = [c.get("chunk_id") for c in ranked_chunks]
        final_hit = any(cid in expected for cid in ranked_ids)

        # 打印排查診斷
        if not final_hit:
            print(f"\n❌ [未命中 Sample]: {q_id} | Query: '{query}'")
            print(f"   - 期望 Chunk ID: {list(expected)}")
            print(f"   - 初篩 Top-10 召回: {raw_hit} -> {raw_ids}")
            print(f"   - 精排 Top-3  命中: {final_hit} -> {ranked_ids}")
            if not raw_hit:
                print("   👉 診斷結論：【初篩未召回】目標 Chunk 未進入 Hybrid Search Top-10。")
            else:
                print("   👉 診斷結論：【精排能力不足】目標 Chunk 在 Top-10 內，但未被 Reranker 排入 Top-3。")

if __name__ == "__main__":
    dataset_file = os.path.join(CURRENT_DIR, "eval_dataset.json")
    debug_eval_dataset(dataset_file)