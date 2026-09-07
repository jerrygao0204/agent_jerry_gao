# # tests/test_concurrency.py
# import os
# import sys
# import time
# import asyncio
# import pytest

# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)

# from generator.qa_chain import QAChain

# @pytest.mark.asyncio
# async def test_qa_chain_concurrency():
#     print("\n🚀 初始化 QAChain 併發測試...")
    
#     qa_chain = QAChain(
#         cuda_device="0",
#         prompt_hub_path=os.path.join(PROJECT_ROOT, "config", "prompt_hub.yaml")
#     )

#     query_text = "FineBI 如何配置數據庫連接？"

#     # 1. 預熱：解析 Dict 結構，避免 TypeError；使用有效 Query 避免 Reranker 阻斷
#     print("🔥 執行完整推理預熱 (Warm-up)...")
#     warmup_gen = qa_chain.stream_answer(query=query_text)
    
#     warmup_tokens = []
#     for chunk in warmup_gen:
#         if isinstance(chunk, dict):
#             # 提取字典中的文本字段，若無則序列化為字串
#             text = chunk.get("answer_chunk", chunk.get("text", str(chunk)))
#             warmup_tokens.append(text)
#         elif isinstance(chunk, str):
#             warmup_tokens.append(chunk)

#     print(f"✅ 預熱完成，輸出片段數: {len(warmup_tokens)}")

#     concurrent_requests = 10  # 設定併發請求數量

#     def run_single_request(req_id: int):
#         t0 = time.perf_counter()
#         try:
#             stream_gen = qa_chain.stream_answer(query=query_text)
#             tokens = []
#             for chunk in stream_gen:
#                 if isinstance(chunk, dict):
#                     tokens.append(chunk.get("answer_chunk", chunk.get("text", str(chunk))))
#                 elif isinstance(chunk, str):
#                     tokens.append(chunk)
            
#             full_answer = "".join(tokens)
#             elapsed_ms = (time.perf_counter() - t0) * 1000
#             has_answer = len(full_answer.strip()) > 0
#             return req_id, has_answer, elapsed_ms, None
#         except Exception as e:
#             elapsed_ms = (time.perf_counter() - t0) * 1000
#             return req_id, False, elapsed_ms, str(e)

#     print(f"⚡ 開始執行 QAChain 併發測試 (並行數: {concurrent_requests})...")
    
#     tasks = [
#         asyncio.to_thread(run_single_request, i) 
#         for i in range(concurrent_requests)
#     ]
    
#     results = await asyncio.gather(*tasks)

#     for r in results:
#         if not r[1]:
#             print(f"❌ 請求 {r[0]} 失敗, 耗時: {r[2]:.2f}ms, 錯誤資訊: {r[3]}")

#     successes = [r for r in results if r[1]]
#     latencies = [r[2] for r in results if r[1]]

#     assert len(successes) == concurrent_requests, f"併發測試未完全成功: {len(successes)}/{concurrent_requests}"

#     avg_latency = sum(latencies) / len(latencies)
#     print(f"\n📊 QAChain 併發測試結果:")
#     print(f" - 成功率: {len(successes)}/{concurrent_requests}")
#     print(f" - 平均端到端延遲: {avg_latency:.2f} ms")


import os
import sys
import time
import asyncio
import statistics
import platform
from datetime import datetime

import pytest
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from generator.qa_chain import QAChain


def percentile(values, p: float) -> float:
    """
    计算百分位（线性插值），p 取值 [0, 100]
    """
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])

    vals = sorted(values)
    k = (len(vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    if f == c:
        return float(vals[f])
    d0 = vals[f] * (c - k)
    d1 = vals[c] * (k - f)
    return float(d0 + d1)


def get_env_info() -> dict:
    gpu_name = "cpu-only"
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "cuda_available": cuda_ok,
        "gpu": gpu_name,
        "torch_version": torch.__version__,
        "concurrency_model": "asyncio.to_thread",
    }


@pytest.mark.asyncio
async def test_qa_chain_concurrency():
    print("\n🚀 初始化 QAChain 併發測試...")
    env = get_env_info()
    print("🧾 測試環境資訊:")
    for k, v in env.items():
        print(f" - {k}: {v}")

    qa_chain = QAChain(
        cuda_device="0",
        prompt_hub_path=os.path.join(PROJECT_ROOT, "config", "prompt_hub.yaml"),
    )

    query_text = "FineBI 如何配置數據庫連接？"
    concurrent_requests = 10
    rounds = 3  # 多輪測試

    # 预热
    print("🔥 執行完整推理預熱 (Warm-up)...")
    warmup_gen = qa_chain.stream_answer(query=query_text)
    warmup_tokens = []
    for chunk in warmup_gen:
        if isinstance(chunk, dict):
            warmup_tokens.append(chunk.get("answer_chunk", chunk.get("text", str(chunk))))
        elif isinstance(chunk, str):
            warmup_tokens.append(chunk)
    print(f"✅ 預熱完成，輸出片段數: {len(warmup_tokens)}")

    # CUDA memory peak reset
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    all_latencies_ms = []
    total_success = 0
    total_requests = concurrent_requests * rounds
    t_global_start = time.perf_counter()

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

    for rd in range(1, rounds + 1):
        print(f"\n⚡ Round {rd}/{rounds}: 開始併發測試 (並行數: {concurrent_requests})...")
        tasks = [asyncio.to_thread(run_single_request, i) for i in range(concurrent_requests)]
        results = await asyncio.gather(*tasks)

        round_success = 0
        round_lat = []
        for r in results:
            if r[1]:
                round_success += 1
                round_lat.append(r[2])
                all_latencies_ms.append(r[2])
            else:
                print(f"❌ 請求 {r[0]} 失敗, 耗時: {r[2]:.2f}ms, 錯誤資訊: {r[3]}")

        total_success += round_success
        round_rate = round_success / concurrent_requests
        print(f"✅ Round {rd} 成功率: {round_success}/{concurrent_requests} ({round_rate:.1%})")
        if round_lat:
            print(
                f"   延遲(ms): avg={statistics.mean(round_lat):.2f}, "
                f"p50={percentile(round_lat, 50):.2f}, "
                f"p95={percentile(round_lat, 95):.2f}, "
                f"p99={percentile(round_lat, 99):.2f}"
            )

    total_elapsed_s = time.perf_counter() - t_global_start
    success_rate = total_success / total_requests
    qps = total_success / total_elapsed_s if total_elapsed_s > 0 else 0.0

    # 显存统计
    peak_alloc_mb = 0.0
    peak_reserved_mb = 0.0
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_alloc_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        peak_reserved_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)

    # 总断言：成功率与样本完整性
    assert total_success == total_requests, f"併發測試未完全成功: {total_success}/{total_requests}"
    assert len(all_latencies_ms) == total_success, "延遲樣本數與成功請求數不一致"

    print("\n📊 QAChain 併發基準結果（發布前指標）:")
    print(f" - 總請求數: {total_requests}")
    print(f" - 成功率: {total_success}/{total_requests} ({success_rate:.1%})")
    print(f" - 總耗時: {total_elapsed_s:.2f} s")
    print(f" - 吞吐量(QPS): {qps:.2f}")
    print(f" - 平均延遲: {statistics.mean(all_latencies_ms):.2f} ms")
    print(f" - P50 延遲: {percentile(all_latencies_ms, 50):.2f} ms")
    print(f" - P95 延遲: {percentile(all_latencies_ms, 95):.2f} ms")
    print(f" - P99 延遲: {percentile(all_latencies_ms, 99):.2f} ms")
    print(f" - GPU Peak Allocated: {peak_alloc_mb:.2f} MB")
    print(f" - GPU Peak Reserved : {peak_reserved_mb:.2f} MB")