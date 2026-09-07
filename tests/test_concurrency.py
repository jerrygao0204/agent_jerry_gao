# tests/test_concurrency.py
import os
import sys
import time
import asyncio
import pytest

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from generator.qa_chain import QAChain

@pytest.mark.asyncio
async def test_qa_chain_concurrency():
    print("\n🚀 初始化 QAChain 併發測試...")
    
    qa_chain = QAChain(
        cuda_device="0",
        prompt_hub_path=os.path.join(PROJECT_ROOT, "config", "prompt_hub.yaml")
    )

    query_text = "FineBI 如何配置數據庫連接？"

    # 1. 預熱：解析 Dict 結構，避免 TypeError；使用有效 Query 避免 Reranker 阻斷
    print("🔥 執行完整推理預熱 (Warm-up)...")
    warmup_gen = qa_chain.stream_answer(query=query_text)
    
    warmup_tokens = []
    for chunk in warmup_gen:
        if isinstance(chunk, dict):
            # 提取字典中的文本字段，若無則序列化為字串
            text = chunk.get("answer_chunk", chunk.get("text", str(chunk)))
            warmup_tokens.append(text)
        elif isinstance(chunk, str):
            warmup_tokens.append(chunk)

    print(f"✅ 預熱完成，輸出片段數: {len(warmup_tokens)}")

    concurrent_requests = 10  # 設定併發請求數量

    def run_single_request(req_id: int):
        t0 = time.perf_counter()
        try:
            stream_gen = qa_chain.stream_answer(query=query_text)
            tokens = []
            for chunk in stream_gen:
                if isinstance(chunk, dict):
                    tokens.append(chunk.get("answer_chunk", chunk.get("text", str(chunk))))
                elif isinstance(chunk, str):
                    tokens.append(chunk)
            
            full_answer = "".join(tokens)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            has_answer = len(full_answer.strip()) > 0
            return req_id, has_answer, elapsed_ms, None
        except Exception as e:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return req_id, False, elapsed_ms, str(e)

    print(f"⚡ 開始執行 QAChain 併發測試 (並行數: {concurrent_requests})...")
    
    tasks = [
        asyncio.to_thread(run_single_request, i) 
        for i in range(concurrent_requests)
    ]
    
    results = await asyncio.gather(*tasks)

    for r in results:
        if not r[1]:
            print(f"❌ 請求 {r[0]} 失敗, 耗時: {r[2]:.2f}ms, 錯誤資訊: {r[3]}")

    successes = [r for r in results if r[1]]
    latencies = [r[2] for r in results if r[1]]

    assert len(successes) == concurrent_requests, f"併發測試未完全成功: {len(successes)}/{concurrent_requests}"

    avg_latency = sum(latencies) / len(latencies)
    print(f"\n📊 QAChain 併發測試結果:")
    print(f" - 成功率: {len(successes)}/{concurrent_requests}")
    print(f" - 平均端到端延遲: {avg_latency:.2f} ms")