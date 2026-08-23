# ingest/validator.py 写入前标准校验 (JSON 格式/长度/字段等)
import os
import sys
import re
import json
import logging
from typing import List, Dict, Any, Tuple, Optional, Union
from pydantic import BaseModel, Field, ValidationError

# 📂 动态计算项目根目录并强行注入系统路径，确保全局工程内 factory 模块可见
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 统一从中央工厂引入枢纽
from factory.model_factory import ModelFactory

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers", "torch", "pydantic"])
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")


# =====================================================================
# 1. Pydantic 数据模型定义 (完美适配 JSON 结构)
# =====================================================================

class BusinessKeyword(BaseModel):
    word: str
    score: float


class CoreEntity(BaseModel):
    name: str
    type: str


class OperationConstraint(BaseModel):
    priority_level: Optional[int] = None
    operator: Optional[str] = None
    description: Optional[str] = None
    association_direction: Optional[str] = None
    constraint_type: Optional[str] = None


class BusinessProfileSchema(BaseModel):
    summary: str
    business_scene: str
    user_level: str
    content_keywords: List[BusinessKeyword] = Field(default_factory=list)
    core_entities: List[CoreEntity] = Field(default_factory=list)
    operation_constraints: List[
        Union[OperationConstraint, Dict[str, Any], str]
    ] = Field(default_factory=list)


class DocMetadataSchema(BaseModel):
    file_name: str = Field(..., description="原始 JSON/PDF 路径")
    file_url: Optional[str] = None
    path_hierarchy: List[str] = Field(default_factory=list)
    md5: str = Field(..., min_length=10)
    processed_at: str
    total_segments: int
    entity_summary: str
    all_entity_uuids: List[str] = Field(default_factory=list)
    business_profile: BusinessProfileSchema


class MilvusPayloadSchema(BaseModel):
    content: str = Field(..., min_length=1, description="物理分块核心文本")
    entity_uuids: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(..., description="分块自带的局部元数据，包含 chunk_id 等")


class FullRAGPayloadSchema(BaseModel):
    doc_metadata: DocMetadataSchema = Field(..., description="全局文档元数据")
    milvus_payloads: List[MilvusPayloadSchema] = Field(..., description="局部分块数据源")
    raw_markdown: Optional[str] = None


# =====================================================================
# 2. 托管大模型加载与预热驱动引擎
# =====================================================================

class Processor:
    """标准规范化的 LLM 评估模型加载器"""
    def __init__(self, prompt_hub_path: str = "prompt_hub.yaml", cuda_device: str = "0"):
        # 委派给大模型中央工厂，自动处理软链接构建
        self.factory = ModelFactory(prompt_hub_path=prompt_hub_path)
        self.device = self.factory.setup_cuda_device(cuda_device)
        
        self._llm_model = None
        self._llm_tokenizer = None

    def _get_llm(self,llm_model) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
        """单例模式加载与预热 LLM 智评底座"""
        if self._llm_model is None or self._llm_tokenizer is None:
            # 由工厂完成离线模型物理哈希寻址与修改时间排序
            model_dir = self.factory.resolve_model_path(llm_model)
            logging.info(f"🚀 [Offline Load] 正在冷启动加载语义智评大模型: {model_dir}")
            
            self._llm_tokenizer = AutoTokenizer.from_pretrained(
                model_dir, 
                local_files_only=True, 
                trust_remote_code=True
            )
            self._llm_model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2",
                use_cache=True,
                local_files_only=True,
                trust_remote_code=True
            )
            
            # 显存预热，分配静态缓冲池
            warmup_prompt = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant"
            inputs = self._llm_tokenizer(warmup_prompt, return_tensors="pt").to(self._llm_model.device)
            with torch.no_grad():
                _ = self._llm_model.generate(**inputs, max_new_tokens=5)
            logging.info("✅ 智评 LLM 物理引擎预热完毕，状态平稳。")
            
        return self._llm_model, self._llm_tokenizer


# =====================================================================
# 3. 校验与全面评分主类
# =====================================================================

class RAGDataValidator:
    def __init__(
        self, 
        processor: Optional[Processor] = None, 
        min_chunk_len: int = 4,           
        max_chunk_len: int = 300000,         
        score_threshold: float = 80.0,     
        bad_cases_dir: str = "./bad_cases",
        llm_sample_size: int = 9          
    ):
        self.processor = processor
        self.min_chunk_len = min_chunk_len
        self.max_chunk_len = max_chunk_len
        self.score_threshold = score_threshold
        self.bad_cases_dir = bad_cases_dir
        self.llm_sample_size = llm_sample_size
        self.logger = logging.getLogger("RAGDataValidator")

    def _evaluate_text_via_llm(self, context_summary: str, chunk_texts: List[str], llm_model: str) -> float:
        """调用本地大模型对抽样文本的语义连贯性与知识密度进行综合评分 (0-20分)"""
        if not self.processor:
            self.logger.warning("⚠️ 未注入 Processor，跳过 LLM 评估维度（默认给予基础分 15 分）")
            return 15.0

        try:
            model, tokenizer = self.processor._get_llm(llm_model)
            sampled_chunks_str = "\n".join([f"[分块 {i+1}]: {text}" for i, text in enumerate(chunk_texts)])
            
            prompt = (
                "<|im_start|>system\n"
                "你是一个冷酷、严厉的 RAG 数据质量评估机器。你不需要具备人类的社交礼仪，不需要解释，只能输出数字。\n"
                "<|im_end|>\n"
                f"<|im_start|>user\n"
                f"请根据提供的全局背景，评估待测局部分块文本的质量。\n\n"
                f"【文档全局背景总结】：\n{context_summary}\n\n"
                f"【待评估的局部分块文本】：\n{sampled_chunks_str}\n\n"
                f"【评估维度】：\n"
                f"1. 语义信息量：是否有实质内容，是否包含过多无意义的废话或格式乱码。\n"
                f"2. 上下文衔接度：分块边界是否导致严重的信息断层，实体指代是否模糊不清。\n"
                f"3. 噪声比例：是否存在大量的系统无关文本（如纯导航栏、页眉页脚片段）。\n\n"
                f"【⚠️核心指令】：\n"
                f"计算出最终的综合得分（范围 0 到 20 之间，允许保留一位小数，如 16.5）。\n"
                f"禁止输出任何分析、任何前缀（如“得分：”）、任何后缀（如“分”）、任何换行或标点符号。\n"
                f"如果你输出了除数字外的任何字符，系统将会崩溃。请直接给出最终的浮点数：<|im_end|>\n"
                f"<|im_start|>assistant\n"
                f"综合质量得分："
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=8, 
                    temperature=0.1,  
                    do_sample=False
                )
            
            response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
            match = re.search(r"\d+\.\d+|\d+", response)
            score = round(float(match[0]), 1)
            score = max(0.0, min(20.0, score))
            self.logger.info(f"🤖 LLM 语义维度的打分结果为: {score} / 20.0")
            return round(score, 1)
            
        except Exception as e:
            self.logger.error(f"❌ LLM 质量评估推理失败: {e}，启用安全降级策略分配 12.0 分")
            return 12.0

    def score_and_evaluate(self, raw_data: Dict[str, Any], llm_model:str) -> Tuple[float, Dict[str, float]]:
        """根据 5 维度质量模型对 JSON 数据执行全面打分 (满分 100分)"""
        scores = {
            "global_integrity": 20.0,  
            "local_structure": 20.0,   
            "text_length_suit": 20.0,  
            "text_cleanliness": 20.0,  
            "llm_semantic_quality": 0.0 
        }

        # --- 维度 1: 全局元数据与商业画像完整度评估 (20分) ---
        metadata = raw_data.get("doc_metadata", {})
        if not isinstance(metadata, dict):
            scores["global_integrity"] = 0.0
        else:
            profile = metadata.get("business_profile", {})
            if not profile:
                scores["global_integrity"] -= 8.0
            else:
                if len(profile.get("summary", "")) < 30:
                    scores["global_integrity"] -= 4.0
                if not profile.get("core_entities"):
                    scores["global_integrity"] -= 4.0
                if not profile.get("operation_constraints"):
                    scores["global_integrity"] -= 2.0
            if not metadata.get("all_entity_uuids"):
                scores["global_integrity"] -= 2.0
        scores["global_integrity"] = max(0.0, min(20.0, scores["global_integrity"]))

        # --- 维度 2: 局部物理分块格式评估 (20分) ---
        payloads = raw_data.get("milvus_payloads", [])
        if not payloads:
            scores["local_structure"] = 0.0
        else:
            orphan_count = 0
            for item in payloads:
                chunk_meta = item.get("metadata", {})
                if not chunk_meta.get("hierarchy"):
                    orphan_count += 1
            
            orphan_ratio = orphan_count / len(payloads)
            if orphan_ratio > 0.3:
                scores["local_structure"] -= 12.0
            elif orphan_ratio > 0:
                scores["local_structure"] -= 4.0
        scores["local_structure"] = max(0.0, scores["local_structure"])

        # --- 维度 3: 文本长度和段落合理度评估 (20分) ---
        total_len = 0
        extreme_len_count = 0
        for item in payloads:
            text_len = len(str(item.get("content", "")))
            total_len += text_len
            if text_len < 25 or text_len > 800:
                extreme_len_count += 1
                
        if payloads:
            avg_len = total_len / len(payloads)
            if not (100 <= avg_len <= 600):
                scores["text_length_suit"] -= 4.0
                
            bad_ratio = extreme_len_count / len(payloads)
            if bad_ratio > 0.2:
                scores["text_length_suit"] -= 12.0
            elif bad_ratio > 0.05:
                scores["text_length_suit"] -= 4.0
        scores["text_length_suit"] = max(0.0, scores["text_length_suit"])

        # --- 维度 4: 文本干净度与低噪性评估 (20分) ---
        total_clean_score = 0.0
        for item in payloads:
            text = str(item.get("content", ""))
            if not text:
                continue
            
            text_for_density = re.sub(r'<[^>]+>', '', text)
            text_for_density = re.sub(r'\s+', '', text_for_density)
            
            if len(text_for_density) > 0:
                alphanum_count = len([c for c in text_for_density if c.isalnum() or '\u4e00' <= c <= '\u9fff'])
                density = alphanum_count / len(text_for_density)
            else:
                density = 0.8  
            
            seg_clean_score = 20.0
            if density < 0.5:
                seg_clean_score -= 12.0
            elif density < 0.7:
                seg_clean_score -= 4.0
                
            text_stripped = text.strip()
            valid_endings = ("。", "！", "？", "”", "；", "...", "\n", "</table>", "</td>", "</tr>", "</div>", "|", "]")
            is_coherent = any(text_stripped.endswith(ending) for ending in valid_endings) if text_stripped else False
            if not is_coherent:
                seg_clean_score -= 2.0
                
            total_clean_score += max(0.0, seg_clean_score)
            
        if payloads:
            scores["text_cleanliness"] = round(total_clean_score / len(payloads), 1)
        else:
            scores["text_cleanliness"] = 0.0

        # --- 维度 5: LLM 语义合理性与知识密度评估 (20分) ---
        if payloads:
            context_summary = metadata.get("entity_summary", profile.get("summary", "无文档全局摘要"))
            step = max(1, len(payloads) // self.llm_sample_size)
            sampled_payloads = payloads[::step][:self.llm_sample_size]
            
            MIN_EVAL_LENGTH = 80  
            llm_eval_texts = []  
            short_chunk_scores = [] 
            
            for item in sampled_payloads:
                text = str(item.get("content", ""))
                text_clean = re.sub(r'<[^>]+>', '', text).strip()
                if len(text_clean) >= MIN_EVAL_LENGTH:
                    llm_eval_texts.append(text)
                else:
                    short_chunk_scores.append(17.0)

            if llm_eval_texts:
                llm_score = self._evaluate_text_via_llm(context_summary, llm_eval_texts,llm_model)
                if short_chunk_scores:
                    all_scores = [llm_score] * len(llm_eval_texts) + short_chunk_scores
                    scores["llm_semantic_quality"] = round(sum(all_scores) / len(all_scores), 1)
                else:
                    scores["llm_semantic_quality"] = round(llm_score, 1)
            else:
                if short_chunk_scores:
                    scores["llm_semantic_quality"] = round(sum(short_chunk_scores) / len(short_chunk_scores), 1)
                else:
                    scores["llm_semantic_quality"] = 0.0
        else:
            scores["llm_semantic_quality"] = 0.0

        total_score = sum(scores.values())
        return round(total_score, 1), scores

    def validate_json_file(self, file_path: str,llm_model: str) -> Tuple[bool, float, str]:
        """核心校验方法"""
        if not os.path.exists(file_path):
            err_msg = f"❌ 无法找到待校验的数据文件: {file_path}"
            self.logger.error(err_msg)
            return False, 0.0, err_msg

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        except json.JSONDecodeError as e:
            err_msg = f"❌ JSON 文件格式破损，解析失败。详情: {e}"
            self.logger.error(err_msg)
            return False, 0.0, err_msg

        # 1. Pydantic Schema 强规则硬校验
        try:
            validated_payload = FullRAGPayloadSchema(**raw_data)
            for idx, item in enumerate(validated_payload.milvus_payloads):
                text_len = len(item.content)
                if text_len < self.min_chunk_len or text_len > self.max_chunk_len:
                    raise ValueError(
                        f"分块校验失败！milvus_payloads[{idx}] 的 content 长度 ({text_len}) 违规。"
                        f"配置设定的限制区间为 [{self.min_chunk_len} - {self.max_chunk_len}] 字符。"
                    )
            is_schema_ok = True
            hard_check_msg = "PASSED"
        except (ValidationError, Exception) as e:
            is_schema_ok = False
            hard_check_msg = f"FAILED -> {str(e)}"

        # 2. 多维度软性打分（包含 LLM 打分）
        total_score, score_details = self.score_and_evaluate(raw_data,llm_model)
        is_approved = is_schema_ok and (total_score >= self.score_threshold)

        # 3. 输出体检报告
        file_name = os.path.basename(file_path)
        report = (
            f"\n┌────────────────────────────────────────────────────────┐\n"
            f"│            RAG 数据源质量体检报告 (LLM 智评融合版)     \n"
            f"├────────────────────────────────────────────────────────┤\n"
            f"│  📄 评估文件: {file_name}\n"
            f"│  🧩 物理分块总数: {len(raw_data.get('milvus_payloads', []))} 个\n"
            f"│  ⚙️  Schema 格式硬校验: {'🟢 完美合规' if is_schema_ok else '🔴 不合格'}\n"
            f"│  ⭐ 综合质量得分: {total_score} / 100.0 (拦截线: {self.score_threshold})\n"
            f"├────────────────────────────────────────────────────────┤\n"
            f"│  💡 各评估维度打分详情 (平均值):\n"
            f"│     - 全局文档元数据与商业画像 (20分): {score_details['global_integrity']} 分\n"
            f"│     - 局部分块层级格式与关联性 (20分): {score_details['local_structure']} 分\n"
            f"│     - 文本切分长度适宜度 (20分):       {score_details['text_length_suit']} 分\n"
            f"│     - 文本识别干净度与连贯度 (20分):   {score_details['text_cleanliness']} 分\n"
            f"│     - LLM 语义合理性与信息密度 (20分): {score_details['llm_semantic_quality']} 分\n"
            f"├────────────────────────────────────────────────────────┤\n"
            f"│  📢 终审放行结论: {'✅ 允许写入 Milvus 向量库' if is_approved else '❌ 拦截！数据未达标'}\n"
            f"│  ⚠️  异常/拦截原因: {'无 (PASS)' if is_approved else (hard_check_msg if not is_schema_ok else '格式正确，但综合质量得分未达到阈值')}\n"
            f"└────────────────────────────────────────────────────────┘"
        )
        print(report)

        # 4. 异常拦截归档
        if not is_approved:
            os.makedirs(self.bad_cases_dir, exist_ok=True)
            bad_case_file_path = os.path.join(
                self.bad_cases_dir, f"rejected_{total_score}_{file_name}"
            )
            raw_data["validation_error_log"] = hard_check_msg
            raw_data["validation_score_details"] = score_details
            with open(bad_case_file_path, 'w', encoding='utf-8') as f:
                json.dump(raw_data, f, ensure_ascii=False, indent=4)
            self.logger.warning(f"💾 已将不合规数据拦截并安全归档至: {bad_case_file_path}")

        return is_approved, total_score, report


# =====================================================================
# 🧪 本地链路质量体检实战验证
# =====================================================================
if __name__ == "__main__":
    # 1. 初始化引擎加载器并指派 CUDA 设备
    processor = Processor(cuda_device="0")
    
    # 2. 注入验证器
    validator = RAGDataValidator(
        processor=processor, 
        score_threshold=80.0, 
        llm_sample_size=25
    )
    
    # 3. 运行本地验证
    target_json = "/workspace/hf-conda/RAG/问答机器人/finebi_output/3_运算符和优先级.json"
    is_ok, score, report = validator.validate_json_file(target_json,llm_model="Qwen/Qwen3-32B")
