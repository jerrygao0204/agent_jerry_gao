# eval/eval_generator.py
import os
import sys
import json
import logging
import time
import re
import pandas as pd
from typing import List, Dict, Any, Tuple

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from generator.qa_chain import QAChain
from generator.llm_client import FineBILLMClient
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

JUDGE_PROMPT_TEMPLATE = """你是一名嚴格的 RAG 系統生成質量評審專家。請根據【用戶問題】、【參考上下文】與【模型回答】，評估以下兩個指標並輸出 JSON 格式結果：

1. **Faithfulness (忠實度)** (0.0 ~ 1.0): 模型回答是否完全基於【參考上下文】？是否有不符合上下文的幻覺或編造內容？(1.0 表示完全忠實於上下文，無幻覺)
2. **Answer Relevance (相關性)** (0.0 ~ 1.0): 模型回答是否直接切中【用戶問題】的要點？是否提供了有用且精準的答覆？(1.0 表示完美回答問題)

【用戶問題】:
{query}

【參考上下文】:
{context}

【模型回答】:
{response}

請嚴格僅輸出以下格式的 JSON，不要包含任何額外 Markdown 標記或解釋：
{{
  "faithfulness": 0.0,
  "answer_relevance": 0.0,
  "reason": "簡短打分說明"
}}
"""

class GeneratorEvaluator:
    """RAG 回答生成質量自動化評估器 (LLM-as-a-Judge)"""

    def __init__(self, llm_model_name: str = "Qwen/Qwen3-4B", cuda_device: str = "0"):
        logging.info("🚀 初始化 GeneratorEvaluator: 載入 QAChain 與 Judge LLM...")
        self.qa_chain = QAChain(
            cuda_device=cuda_device,
            prompt_hub_path=os.path.join(PROJECT_ROOT, "config", "prompt_hub.yaml"),
            llm_short_name=llm_model_name
        )
        self.judge_client = FineBILLMClient(cuda_device=cuda_device)

    def load_dataset(self, dataset_path: str) -> List[Dict[str, Any]]:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"❌ 評測集檔案不存在: {dataset_path}")
        with open(dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def evaluate_generation(self, query: str, context: str, response: str) -> Dict[str, Any]:
        """使用 LLM 裁判計算 Faithfulness 與 Relevance (增強 JSON 解析健壯性)"""
        prompt = JUDGE_PROMPT_TEMPLATE.format(query=query, context=context, response=response)
        try:
            # 1. 調用 stream_generate 獲取串流生成器
            stream_res = self.judge_client.stream_generate(
                query=prompt,
                system_prompt="你是一個嚴格的 JSON 數據評估引擎。切勿輸出任何 <think> 思考過程或 Markdown 說明，只輸出合法的 JSON 對象。"
            )
            
            # 2. 拼接串流文字
            if hasattr(stream_res, "__iter__") and not isinstance(stream_res, (str, dict)):
                raw_text = "".join([str(chunk) for chunk in stream_res]).strip()
            else:
                raw_text = str(stream_res).strip()

            # 3. 過濾 <think>...</think> 標籤 (若模型開啟了思考鏈)
            clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()

            # 4. 正則匹配最外層的 { ... } 內容
            json_match = re.search(r'\{.*\}', clean_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                logging.error(f"⚠️ 無法提取 JSON，LLM 原始輸出: {raw_text[:200]}...")
                return {"faithfulness": 0.0, "answer_relevance": 0.0, "reason": "LLM 輸出非 JSON 格式"}

            # 5. 安全解析 JSON
            score_data = json.loads(json_str)
            return {
                "faithfulness": float(score_data.get("faithfulness", 0.0)),
                "answer_relevance": float(score_data.get("answer_relevance", 0.0)),
                "reason": score_data.get("reason", "無說明")
            }
        except Exception as e:
            logging.error(f"❌ LLM-as-a-Judge 評估失敗: {e}")
            return {"faithfulness": 0.0, "answer_relevance": 0.0, "reason": f"評估異常: {str(e)}"}
        
    def run_evaluation(self, dataset_path: str, output_dir: str = None) -> Tuple[Dict[str, Any], pd.DataFrame]:
        dataset = self.load_dataset(dataset_path)
        results = []

        # 1. Warm-up 預熱（傳入空列表，僅預熱 LLM 生成，跳過 Milvus 与 Reranker）
        logging.info("🔥 正在執行端到端 LLM 推理預熱 (Warm-up)...")
        try:
            for _ in self.qa_chain.stream_answer(query="预热查询", pre_retrieved_chunks=[]):
                pass
            logging.info("✅ 預熱完成，正式開始生成質量評測。")
        except Exception as e:
            logging.warning(f"⚠️ 預熱階段跳過: {e}")

        logging.info(f"⚡ 開始執行 RAG 生成質量評測，共 {len(dataset)} 筆 Sample...")
        for sample in dataset:
            query = sample.get("query", "")
            sample_id = sample.get("id", "N/A")

            t_start = time.perf_counter()
            t_first_token = None
            response_chunks = []

            # 🎯 步驟 A：在評測腳本層顯式進行 1 次混合檢索与重排
            raw_chunks = self.qa_chain.retriever.hybrid_search(
                query=query, 
                top_k=self.qa_chain.top_k_retrieval
            )
            retrieved_chunks = self.qa_chain.reranker.rerank(
                query=query, 
                documents=raw_chunks, 
                top_n=self.qa_chain.top_k_rerank
            )

            # 🎯 步驟 B：將重排結果傳給 pre_retrieved_chunks，強行堵死 QAChain 內部的重複檢索
            rag_stream = self.qa_chain.stream_answer(
                query=query,
                pre_retrieved_chunks=retrieved_chunks
            )
            
            for event in rag_stream:
                if isinstance(event, dict):
                    event_type = event.get("type")
                    data = event.get("data")
                    
                    if event_type == "text":
                        if t_first_token is None and data:
                            t_first_token = time.perf_counter()
                        response_chunks.append(str(data))
                    elif event_type == "security_block":
                        response_chunks = [str(data)]
                else:
                    if t_first_token is None:
                        t_first_token = time.perf_counter()
                    response_chunks.append(str(event))

            response_text = "".join(response_chunks)
            t_end = time.perf_counter()
            
            gen_latency_ms = (t_end - t_start) * 1000
            ttft_ms = ((t_first_token - t_start) * 1000) if t_first_token else gen_latency_ms

            # 2. 拼接 Context 供 Judge 評分
            context_text = "\n\n".join([c.get("content", "") for c in retrieved_chunks if isinstance(c, dict)])

            # 3. 調用 LLM-as-a-Judge 進行評分
            scores = self.evaluate_generation(query=query, context=context_text, response=response_text)

            results.append({
                "sample_id": sample_id,
                "query": query,
                "response": response_text,
                "faithfulness": scores["faithfulness"],
                "answer_relevance": scores["answer_relevance"],
                "reason": scores["reason"],
                "ttft_ms": round(ttft_ms, 2),
                "latency_ms": round(gen_latency_ms, 2)
            })
            logging.info(f"⏱️ Sample [{sample_id}] | Faithfulness: {scores['faithfulness']} | Relevance: {scores['answer_relevance']} | TTFT: {ttft_ms:.2f} ms | Total Latency: {gen_latency_ms:.2f} ms")

        df = pd.DataFrame(results)
        
        # 4. 統計指標 (包含 P95 與 TTFT)
        summary = {
            "total_samples": len(df),
            "avg_faithfulness": round(df["faithfulness"].mean(), 4),
            "avg_answer_relevance": round(df["answer_relevance"].mean(), 4),
            "avg_ttft_ms": round(df["ttft_ms"].mean(), 2),
            "avg_generation_latency_ms": round(df["latency_ms"].mean(), 2),
            "p95_generation_latency_ms": round(df["latency_ms"].quantile(0.95), 2)
        }

        # 3. 匯出評測報告
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            report_json_path = os.path.join(output_dir, "generator_eval_report.json")
            report_md_path = os.path.join(output_dir, "generator_eval_report.md")

            with open(report_json_path, "w", encoding="utf-8") as f:
                json.dump({"summary": summary, "details": results}, f, ensure_ascii=False, indent=2)

            md_content = f"""# 📊 RAG 生成質量評測報告 (Generator Evaluation Report)

                            ### 📈 核心生成指標
                            - **總測試 Sample 數**: `{summary['total_samples']}`
                            - **🛡️ 平均忠實度 (Faithfulness)**: `{summary['avg_faithfulness'] * 100:.2f}%`
                            - **🎯 平均回答相關性 (Answer Relevance)**: `{summary['avg_answer_relevance'] * 100:.2f}%`
                            - **⚡ 平均生成總延遲**: `{summary['avg_generation_latency_ms']} ms`

                            ### 📝 評估明細
                            {df[['sample_id', 'query', 'faithfulness', 'answer_relevance', 'reason']].to_markdown(index=False)}
                            """
            with open(report_md_path, "w", encoding="utf-8") as f:
                f.write(md_content)
                
            logging.info(f"✅ 生成評測報告已成功導出至: {output_dir}")

        return summary, df
if __name__ == "__main__":
    dataset_file = os.path.join(CURRENT_DIR, "eval_dataset.json")
    output_directory = os.path.join(CURRENT_DIR, "reports")

    evaluator = GeneratorEvaluator()
    summary_metrics, _ = evaluator.run_evaluation(
        dataset_path=dataset_file,
        output_dir=output_directory
    )
    print("\n" + "="*40)
    print("📊 生成評測結果摘要 (Summary):")
    print(json.dumps(summary_metrics, ensure_ascii=False, indent=2))
    print("="*40)