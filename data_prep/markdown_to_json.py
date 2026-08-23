# markdown_to_json.py
import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# 示例：安装 requests
install_package("langchain_text_splitters")
install_package("PyMuPDF")
install_package("pdfplumber")
install_package("openpyxl")
install_package("pymilvus")

import os
import re
import gc
import json
import uuid
import yaml
import logging
import hashlib
import subprocess
from datetime import datetime
from typing import Union, Dict, Any, List

import torch

from langchain_text_splitters import MarkdownHeaderTextSplitter

# 动态将当前脚本的上一级目录（即项目根目录 /workspace）加入 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from factory.model_factory import ModelFactory
from ingest.validator import Processor as ValidationProcessor, RAGDataValidator
from ingest.db_uploader import FineBIMilvusUploader
from data_prep.pdf_to_markdown import MarkdownProcessor

# 配置日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FineBIDocConfig:
    """FineBI 文档处理流水线配置管理类"""
    def __init__(
        self,
        cache_dir: str = "/workspace/hf-conda/hf_cache/hub",
        datalab_cache_dir: str = "/workspace/hf-conda/hf_cache/datalab",
        namespace_seed: str = "FineBI_RAG_2026",
        image_url_prefix: str = "",
        pdf_url_prefix: str = "",
        cuda_device: str = "0",
        yaml_rules_path: str = "heading_rules.yaml",    # 规则配置文件
        yaml_prompts_path: str = "prompt_hub.yaml"      # 提示词库文件
    ):
        self.cache_dir = cache_dir
        self.datalab_cache_dir = datalab_cache_dir
        self.namespace_seed = namespace_seed
        self.image_url_prefix = image_url_prefix
        self.pdf_url_prefix = pdf_url_prefix
        self.cuda_device = cuda_device
        self.yaml_rules_path = yaml_rules_path
        self.yaml_prompts_path = yaml_prompts_path
        
        # # 设置环境变量（在检测可用性之前设置）
        # os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
        # logging.info(f"CUDA 是否可用: {torch.cuda.is_available()}")
        
        # # 核心修复：加载并解析配置文件（规则和大模型提示词）
        # self.heading_rules: List[Dict[str, Any]] = []
        # self.prompts: Dict[str, str] = {}

        self._load_all_assets()

    def _load_all_assets(self):
        """同时加载规则配置和提示词库两个独立文件"""
        # ---- 1. 加载标题校准规则 (从 yaml_rules_path) ----
        if self.yaml_rules_path and os.path.exists(self.yaml_rules_path):
            with open(self.yaml_rules_path, 'r', encoding='utf-8') as f:
                rules_data = yaml.safe_load(f) or {}
                raw_rules = rules_data.get('heading_patterns', [])
                # 按正则长度降序排序，确保长模式优先匹配
                raw_rules.sort(key=lambda x: len(x['regex']), reverse=True)
                self.heading_rules = [{'reg': re.compile(i['regex']), 'level': i['level']} for i in raw_rules]
            logging.info(f"📂 成功加载并排序 {len(self.heading_rules)} 条标题切分规则。")
        else:
            logging.warning(f"⚠️ 找不到规则配置文件: {self.yaml_rules_path}")

        # ---- 2. 加载大模型提示词库 (从 yaml_prompts_path) ----
        if self.yaml_prompts_path and os.path.exists(self.yaml_prompts_path):
            with open(self.yaml_prompts_path, 'r', encoding='utf-8') as f:
                prompts_data = yaml.safe_load(f) or {}
                raw_prompts = prompts_data.get('prompts', [])
                self.prompts = {
                    item['name']: item['content'] 
                    for item in raw_prompts 
                    if 'name' in item and 'content' in item
                }
            logging.info(f"📂 成功加载 {len(self.prompts)} 个大模型提示词组件。")
        else:
            logging.warning(f"⚠️ 找不到提示词库文件: {self.yaml_prompts_path}")

        

class FineBIDocProcessor:
    """FineBI 文档语义解析与特征提取核心处理器"""
    
    def __init__(self, config: FineBIDocConfig):
        self.config = config
        
        # 延迟加载的单例模型变量
        self._llm_model = None
        self._llm_tokenizer = None
        
        # # 初始化系统软链接
        # self._initialize_environment()
        # # 加载 Markdown 校准规则
        # # 从 config 直接接管已经解析好的两套资产
        # self.heading_rules = self.config.heading_rules  # 对应第一个 YAML 的切分规则
        # self.prompts = self.config.prompts              # 对应第二个 YAML 的提示词库
        # 🟢 核心重构点：直接交由 ModelFactory 接管环境初始化（软链接、算力设备等）
        self.model_factory = ModelFactory(
            prompt_hub_path=config.yaml_prompts_path,
            cache_dir=config.cache_dir
        )

        # 统一设置 GPU 算力硬件
        self.device = self.model_factory.setup_cuda_device(config.cuda_device)
        
        # 懒加载变量预留（由工厂管理）
        self.model = None
        self.tokenizer = None

        # 加载 Markdown 校准规则与提示词
        self.heading_rules = self.config.heading_rules  # 对应第一个 YAML 的切分规则
        self.prompts = self.config.prompts              # 对应第二个 YAML 的提示词库

        logging.info("⚙️ FineBIDocProcessor 初始化完成，已成功绑定规则库与提示词库。")
        
    def _load_prompts(self) -> Dict[str, str]:
        """加载 Prompt Hub YAML 并转换为键值对"""
        if not os.path.exists(self.prompt_path):
            raise FileNotFoundError(f"❌ 提示词库配置文件未找到: {self.prompt_path}")
        
        with open(self.prompt_path, 'r', encoding='utf-8') as f:
            raw_data = yaml.safe_load(f)
            
        prompts_dict = {}
        # 如果 YAML 的最外层是一个列表
        if isinstance(raw_data, list):
            for item in raw_data:
                if isinstance(item, dict) and "name" in item and "content" in item:
                    prompts_dict[item["name"]] = item["content"]
        # 如果外层本身就是字典
        elif isinstance(raw_data, dict):
            # 兼容有些情况下以字典形式组织的名值对
            for k, v in raw_data.items():
                if isinstance(v, dict) and "content" in v:
                    prompts_dict[k] = v["content"]
                elif isinstance(v, str):
                    prompts_dict[k] = v
                    
        logging.info(f"🎯 已从 {self.prompt_path} 成功加载并解析了 {len(prompts_dict)} 个提示词模版")
        return prompts_dict

    def _get_llm(self):
        """🟢 核心重构：通过 ModelFactory 统一单例加载与预热 LLM"""
        if self._llm_model is None or self._llm_tokenizer is None:
            logging.info("🚀 正在通过 ModelFactory 加载纯文本 LLM 引擎...")
            self._llm_model, self._llm_tokenizer = self.model_factory.get_llm_model(
                llm_short_name="Qwen/Qwen3-32B"
            )
            
            # 预热分配显存
            warmup_prompt = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant"
            inputs = self._llm_tokenizer(warmup_prompt, return_tensors="pt").to(self._llm_model.device)
            _ = self._llm_model.generate(**inputs, max_new_tokens=5)
            logging.info("✅ LLM 预热完成，现在可以稳定工作了")
        return self._llm_model, self._llm_tokenizer

    def _init_llm_engine(self, model_name: str = "Qwen/Qwen3-32B"):
        """⚡ 通过工厂懒加载纯文本 LLM 引擎"""
        if self.model is None or self.tokenizer is None:
            logging.info("🤖 正在通过 ModelFactory 调起文本 LLM 引擎...")
            self.model, self.tokenizer = self.model_factory.get_llm_model(llm_short_name=model_name)

    def generate_entity_uuid(self, name: str, entity_type: str) -> str:
        """确定性 UUID v5 生成算法"""
        custom_namespace = uuid.uuid5(uuid.NAMESPACE_DNS, self.config.namespace_seed)
        norm_name = re.sub(r'\s+', ' ', str(name).strip()).upper()
        norm_type = re.sub(r'\s+', ' ', str(entity_type).strip()).upper()
        unique_key = f"{norm_type}:{norm_name}"
        return str(uuid.uuid5(custom_namespace, unique_key))

    @staticmethod
    def clean_generic_noise(entities: List[Dict[str, Any]], max_limit: int = 30) -> List[Dict[str, Any]]:
        """语义脱噪器：剔除检索价值低或字面量的无效实体"""
        cleaned = []
        seen = set()
        noise_literals = {"true", "false", "null", "none", "n/a", "undefined", "nan"}
        allowed_operators = {"+", "-", "*", "/", ">", "<", "=", "==", "!=", "&&", "||"}

        for ent in entities:
            if not isinstance(ent, dict): 
                continue
            name = str(ent.get("name", "")).strip().replace('"', "'")
            
            is_invalid = any([
                not name,
                name.isdigit(), 
                len(name) < 2 and name not in allowed_operators,
                name.lower() in noise_literals
            ])
            
            if not is_invalid and name.lower() not in seen:
                cleaned.append(ent)
                seen.add(name.lower())
                
            if len(cleaned) >= max_limit:
                break
        return cleaned

    @staticmethod
    def robust_json_stitcher(raw_text: str) -> dict:
        """强力 JSON 容错缝合器"""
        if not raw_text or not isinstance(raw_text, str):
            return {}

        clean_text = raw_text.strip()
        clean_text = re.sub(r'^```[a-zA-Z]*\s*', '', clean_text)
        clean_text = re.sub(r'\s*```$', '', clean_text).strip()

        try:
            return json.loads(clean_text)
        except json.JSONDecodeError:
            logging.warning("[Stitcher] 直接解析失败，尝试清洗和定位包裹...")

        start_idx = clean_text.find('{')
        end_idx = clean_text.rfind('}')
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            candidate = clean_text[start_idx:end_idx + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                try:
                    return json.loads(candidate, strict=False)
                except json.JSONDecodeError:
                    pass
            clean_text = candidate

        try:
            python_safe_text = (
                clean_text.replace('true', 'True')
                .replace('false', 'False')
                .replace('null', 'None')
            )
            import ast
            res = ast.literal_eval(python_safe_text)
            if isinstance(res, dict):
                return res
        except Exception as e:
            logging.error(f"[Stitcher] 二级解析依然失败: {e}。启动三级终极兜底...")

        # 三级正则提取兜底
        fallback_dict = {}
        patterns = {
            "chunk_type": r'"chunk_type"\s*:\s*"([^"]+)"',
            "chunk_summary": r'"chunk_summary"\s*:\s*"([^"]+)"',
            "operation_constraints": r'"operation_constraints"\s*:\s*\[(.*?)\]'
        }
        for key, pattern in patterns.items():
            match = re.search(pattern, raw_text, re.DOTALL)
            if match:
                if key == "operation_constraints":
                    try:
                        parsed_list = json.loads(f"[{match.group(1)}]")
                        fallback_dict[key] = [
                            item["constraint"] if isinstance(item, dict) and "constraint" in item else str(item)
                            for item in parsed_list
                        ]
                    except Exception:
                        items = re.findall(r'"([^"]+)"', match.group(1))
                        fallback_dict[key] = [x for x in items if x not in {"constraint", "priority"}]
                else:
                    fallback_dict[key] = match.group(1).strip()

        if "chunk_summary" not in fallback_dict:
            fallback_dict["chunk_summary"] = "由于格式损坏，未能自动提取特征，保留原分块数据。"
        return fallback_dict

    def _run_model_inference(self, prompt: str, step_label: str, base_tokens: int) -> dict:
        """内部通用大模型推理与异常捕获循环封装"""
        model, tokenizer = self._get_llm()
        max_attempts = 3
        
        for attempt in range(max_attempts):
            current_max = base_tokens + (attempt * 512)
            try:
                logging.info(f"🧠 [{step_label}] 尝试提取 (第 {attempt + 1} 次, max_tokens={current_max})...")
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                
                with torch.no_grad():
                    generate_kwargs = {
                        **inputs,
                        "max_new_tokens": current_max,
                        "temperature": 0.2 if attempt > 0 else 0.1,
                        "repetition_penalty": 1.05,
                        "pad_token_id": tokenizer.eos_token_id,
                        "stop_strings": ["<|im_end|>"],
                        "tokenizer": tokenizer
                    }
                    try:
                        outputs = model.generate(**generate_kwargs)
                    except TypeError as e:
                        if "stop_strings" in str(e):
                            generate_kwargs.pop("stop_strings")
                            generate_kwargs.pop("tokenizer")
                            outputs = model.generate(**generate_kwargs)
                        else:
                            raise e

                full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                answer = full_text.split("assistant")[-1].strip() if "assistant" in full_text else full_text.strip()
                
                # 特征预检（主要针对分块任务的复读断裂与脏数据保护）
                if step_label == "分块综合特征":
                    if '"operation_constraints"' not in answer or '}' not in answer[answer.rfind(']'):]:
                        raise ValueError("JSON 字段不完整，模型可能陷入了示例参数的复读陷阱")
                    names = re.findall(r'"name":\s*"(.*?)"', answer)
                    if len(names) > 15 and (sum(len(n) for n in names) / len(names)) < 2.5:
                        raise ValueError("检测到高频低价值输出（语义噪音），强制重试并提高惩罚")

                answer = re.sub(r'<think>.*?</think>|<tool_call>.*?</tool_call>', '', answer, flags=re.DOTALL).strip()
                answer = re.sub(r'```json\s*|\s*```', '', answer).strip()
                
                start_idx = answer.find('{')
                end_idx = answer.rfind('}')
                if start_idx == -1: 
                    raise ValueError("未找到 JSON 起始符 '{'")
                
                return self.robust_json_stitcher(answer[start_idx:end_idx+1])
                
            except Exception as e:
                logging.warning(f"🔄 [{step_label}] 第 {attempt + 1} 次解析失败: {e}")
                if attempt == max_attempts - 1:
                    return {}
        return {}

    def generate_business_features(self, text: str) -> dict:
        """提取全局宏观画像特征 (双步串行策略)"""
        content_sample = text if len(text) < 15000 else text[:8000] + "\n[...]\n" + text[-4000:]

        prompt_a_tmpl = self.prompts.get("finebi_doc_semantic_analysis")
        prompt_b_tmpl = self.prompts.get("doc_entity_constraint_extraction")
        prompt_a = prompt_a_tmpl.format(content_sample=content_sample)
        prompt_b = prompt_b_tmpl.format(content_sample=content_sample)

        res_a = self._run_model_inference(prompt_a, "语义画像", base_tokens=2048)
        self._clear_cuda_cache()
        
        res_b = self._run_model_inference(prompt_b, "技术实体", base_tokens=3072)
        refined_entities = self.clean_generic_noise(res_b.get("core_entities", []))
        self._clear_cuda_cache()

        return {
            "summary": res_a.get("summary", "语义解析失败"),
            "business_scene": res_a.get("business_scene", "通用"),
            "user_level": res_a.get("user_level", "未知"),
            "content_keywords": res_a.get("content_keywords", []),
            "core_entities": refined_entities,
            "operation_constraints": res_b.get("operation_constraints", [])
        }

    def generate_chunk_features(self, chunk_text: str) -> dict:
        """提取局部物理分块特征"""
        content_sample = chunk_text if len(chunk_text) < 12000 else chunk_text[:8000] + "\n[...截断...]\n" + chunk_text[-3000:]
        prompt_combined_tmpl = self.prompts.get("doc_chunk_five_dimension_parsing")
        prompt_combined = prompt_combined_tmpl.format(content_sample=content_sample)

        res = self._run_model_inference(prompt_combined, "分块综合特征", base_tokens=3072)
        refined_entities = self.clean_generic_noise(res.get("core_entities", []))
        self._clear_cuda_cache()

        return {
            "chunk_type": res.get("chunk_type", "Text"),
            "chunk_summary": res.get("chunk_summary", ""),
            "symbol_raw": res.get("symbol_raw", []),
            "core_entities": refined_entities,
            "operation_constraints": res.get("operation_constraints", [])
        }

    def refine_markdown_in_memory(self, original_md: str) -> str:
        """基于已加载的 YAML 规则，校准 Markdown 标题层级与去噪"""
        if not original_md or not self.heading_rules:
            return original_md

        lines = original_md.split('\n')
        refined_lines = []

        for line in lines:
            header_match = re.match(r'^(#+)\s+(.*)', line)
            if header_match:
                raw_title = header_match.group(2)
                clean_title = raw_title.replace("**", "").strip()
                
                target_level = None
                for rule in self.heading_rules:
                    if rule['reg'].match(clean_title):
                        target_level = rule['level']
                        break
                
                if target_level:
                    refined_lines.append(f"{'#' * target_level} {clean_title}")
                else:
                    old_hashes = header_match.group(1)
                    refined_lines.append(f"{old_hashes} {clean_title}")
            else:
                refined_lines.append(line)

        logging.info("✅ 已完成 Markdown 层级校准与标题去噪")
        return "\n".join(refined_lines)

    def prepare_milvus_payloads(self, clean_md: str, file_name: str, biz_features: dict) -> List[Dict[str, Any]]:
        """将清洗后的 Markdown 切分为多维度特征对齐的 Milvus 向量载荷"""
        headers_config = [("#", "H1"), ("##", "H2"), ("###", "H3")]
        h_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_config)
        chunks = h_splitter.split_text(clean_md)
        core_entities = biz_features.get("core_entities", [])
        all_payloads = []

        for i, chunk in enumerate(chunks):
            content = chunk.page_content.strip()
            if not content: 
                continue
            logging.info(f"   🧩 正在提取片段 [{i}] 的局部语义特征...")
            
            chunk_meta = self.generate_chunk_features(content)
            
            cleaned_metadata = {
                k: v.replace("**", "").strip() if isinstance(v, str) else v 
                for k, v in chunk.metadata.items()
            }

            raw_hierarchy = chunk_meta.get("hierarchy", cleaned_metadata.get("hierarchy", {}))
            if not isinstance(raw_hierarchy, dict):
                raw_hierarchy = {"H1": str(raw_hierarchy)}

            aligned_hierarchy = {
                "H1": raw_hierarchy.get("H1", None),
                "H2": raw_hierarchy.get("H2", None),
                "H3": raw_hierarchy.get("H3", None)
            }

            if not aligned_hierarchy["H1"]:
                aligned_hierarchy["H1"] = file_name

            if "metadata" in chunk_meta:
                chunk_meta["metadata"]["hierarchy"] = aligned_hierarchy
            else:
                chunk_meta["hierarchy"] = aligned_hierarchy

            img_matches = re.findall(r'!\[.*?\]\((.*?)\)', content)
            image_map = {img: f"{self.config.image_url_prefix.rstrip('/')}/{img}" for img in img_matches}
            
            # 确定性 UUID 注入
            chunk_uuids = [
                self.generate_entity_uuid(ent['name'], ent['type'])
                for ent in core_entities if ent['name'].upper() in content.upper()
            ]
            
            section_meta_str = str(sorted(chunk.metadata.items())) 
            fingerprint_base = f"{section_meta_str}_{content}"
            content_md5 = hashlib.md5(fingerprint_base.encode('utf-8')).hexdigest()

            header_levels = ["H3", "H2", "H1"]
            section_id = "Overview"
            for level in header_levels:
                if level in chunk.metadata:
                    section_id = chunk.metadata[level]
                    break
                    
            payload = {
                "content": content,
                "entity_uuids": list(set(chunk_uuids)),
                "metadata": {
                    # 1. 唯一标识与物理溯源 (ID & Source)
                    "chunk_id": f"{file_name}_h_{i}",
                    "source_file": file_name,
                    "md5": content_md5,
                    "local_index": i,
                    # 2. 链式上下文 (Context Chain) - 方便在 RAG 检索时拉取前后文
                    "prev_chunk_id": f"{file_name}_h_{i-1}" if i > 0 else None,
                    "next_chunk_id": f"{file_name}_h_{i+1}" if i < len(chunks) - 1 else None,
                    "hierarchy": cleaned_metadata,
                    # 3. 分块局部特征 (Local Features) - 检索匹配的核心
                    "chunk_type": chunk_meta.get("chunk_type", "Text"),
                    "chunk_summary": chunk_meta.get("chunk_summary", ""),
                    "symbol_raw": chunk_meta.get("symbol_raw", []),
                    "core_entities": chunk_meta.get("core_entities", []),
                    "operation_constraints": chunk_meta.get("operation_constraints", []),
                    # 4. 全局增强特征 (Global Augmented Features) - 用于 Milvus 标量过滤 (Scalar Filtering)
                    "scene": biz_features.get("business_scene", ""),
                    # 5. 多媒体与附件 (Multimedia)
                    "image_map": image_map,
                    "image_urls": list(image_map.values()),
                    "section_id": section_id,
                }
            }
            all_payloads.append(payload)
            # logging.info(f"====== 🔧 [RAG Debug] 最终分析 ======")
            # logging.info(f"LangChain 原始切分块数: {len(chunks)}")
            # logging.info(f"成功存入数组的有效块数: {len(all_payloads)}")
            # logging.info(f"各块的实际索引分布: {[p['metadata']['local_index'] for p in all_payloads]}")
            # logging.info(f"====================================")

            

        return all_payloads

    def process_pdf(self, pdf_path: str, md_input: Union[str, Any], output_dir: str) -> Dict[str, Any]:
        """执行端到端的 PDF-Markdown 转换与全维度数据归档存储流程"""
        os.makedirs(output_dir, exist_ok=True)
        file_name = os.path.basename(pdf_path)
        logging.info(f"📄 正在处理: {file_name} -> 目标目录: {output_dir}")
        
        # 兼容处理原始字符串输入或带 images 的 Marker 输出类对象
        if isinstance(md_input, str):
            class MockRendered:
                def __init__(self, text):
                    self.markdown = text
                    self.images = {}
                def model_dump(self):
                    return {"markdown": self.markdown}
            rendered = MockRendered(md_input)
        else:
            rendered = md_input

        # 1. 结构化修正与去噪
        output_dict = rendered.model_dump()
        mid_md = self.refine_markdown_in_memory(output_dict.get("markdown", ""))
        
        # 2. 调用单例大模型提取全局业务特征画像
        biz_features = self.generate_business_features(mid_md)
        
        # 3. 构建片段级 Milvus 载荷
        milvus_payloads = self.prepare_milvus_payloads(mid_md, file_name, biz_features)
        
        all_doc_uuids = set()
        for p in milvus_payloads:
            all_doc_uuids.update(p["entity_uuids"])
            
        summary_dict = {}
        for ent in biz_features.get("core_entities", []):
            summary_dict.setdefault(ent['type'], []).append(ent['name'])
        entity_summary = " | ".join([f"{k}({', '.join(set(v))})" for k, v in summary_dict.items()])

        # 4. 组装终极输出 JSON
        json_filename = file_name.replace(".pdf", ".json")
        json_path = os.path.join(output_dir, json_filename)
        
        with open(pdf_path, "rb") as f:
            doc_metadata_md5 = hashlib.md5(f.read()).hexdigest()
            
        path_hierarchy = [
            *os.path.relpath(json_path, output_dir).split(os.sep)[:-1],
            os.path.splitext(os.path.basename(json_path))[0]
        ]

        final_output = {
            "doc_metadata": {
                "file_name": json_path,
                "file_url": f"{self.config.pdf_url_prefix}{file_name}",
                "path_hierarchy": path_hierarchy,
                "md5": doc_metadata_md5,
                "processed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_segments": len(milvus_payloads),
                "entity_summary": entity_summary,
                "all_entity_uuids": list(all_doc_uuids),
                "business_profile": biz_features
            },
            "milvus_payloads": milvus_payloads,
            "raw_markdown": mid_md
        }
        
        # 落地持久化文件
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(final_output, f, ensure_ascii=False, indent=4)
        
        for img_name, img_obj in rendered.images.items():
            img_obj.save(os.path.join(output_dir, img_name))

        logging.info(f"✅ 处理完成: {json_filename} | 产生片段数: {len(milvus_payloads)}")
        return final_output

    @staticmethod
    def _clear_cuda_cache():
        """执行 CUDA 显存与垃圾回收强制清理"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    input_folder = "/workspace/hf-conda/RAG/问答机器人/other/finebi/函数专题/1_函数新手入门"
    # input_folder = "/workspace/hf-conda/RAG/问答机器人/finebi"
    output_root = "/workspace/hf-conda/RAG/问答机器人/finebi_output"
    img_prefix = "https://your-oss-bucket.com/finebi/docs/images"
    pdf_prefix = "https://your-oss-bucket.com/finebi/pdfs/"
    namespace_seed = "FineBI_RAG_2026"
    prompts_hub_path = "/workspace/hf-conda/RAG/问答机器人/config/prompt_hub.yaml"
    patterns_path = "/workspace/hf-conda/RAG/问答机器人/config/patterns.yaml"
    pdf_path = '/workspace/hf-conda/RAG/问答机器人/other/finebi/函数专题/1_函数新手入门/3_运算符和优先级.pdf'
    score_threshold = '80'
    
    # 初始化输出变量
    markdown_output = None
    is_quality_passed = False

    # ========================================================
    # 阶段一：运行 VLM 提取 Markdown，跑完立刻销毁并释放显存
    # ========================================================
    try:
        # 1. 创建处理器
        processor = MarkdownProcessor(prompt_hub_path=prompts_hub_path)
        
        # 2. 执行转换获取 Markdown 纯文本
        logging.info("🚀 开始运行 VLM 提取 PDF 结构及内容...")
        markdown_output = processor.main(
            pdf_path=pdf_path, 
            prompt_path=prompts_hub_path, 
            vlm="Qwen--Qwen3-VL-32B-Instruct"
        )
        logging.info("✅ Markdown 原始数据提取成功！")
    finally:
        # 3. 无论提取成功或失败，强制释放 VLM 显存，不让大模型多占用一秒钟
        logging.info("🧹 正在物理销毁大模型并释放显存资源...")
        if 'processor' in locals():
            # 获取内部持有的 VLM 模型（自适应属性名 self.model 或 self.vlm）
            target_model = None
            if hasattr(processor, "model") and processor.model is not None:
                target_model = processor.model
                processor.model = None
            elif hasattr(processor, "vlm") and processor.vlm is not None:
                target_model = processor.vlm
                processor.vlm = None

            # 调用写在类下面的静态销毁函数
            if target_model is not None:
                MarkdownProcessor._destroy_model(target_model)
                del target_model

            # 彻底干掉处理器实例
            del processor

        # 全局深度清理
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        logging.info("💾 显存深度清理完毕！当前 GPU 显存已被系统完全回收。")

    # ========================================================
    # 阶段二：进行后处理与 JSON 转换（利用重构后的类）
    # ========================================================
    if markdown_output:
        try:
            logging.info("⚙️ 正在初始化 FineBI 文档处理器...")
            # 1. 初始化配置类
            doc_config = FineBIDocConfig(
                namespace_seed=namespace_seed,
                image_url_prefix=img_prefix,
                pdf_url_prefix=pdf_prefix,
                cuda_device="0",  
                yaml_rules_path=patterns_path,      # 传入切分规则配置路径
                yaml_prompts_path=prompts_hub_path    # 传入提示词库路径
            )
            
            
            # 2. 实例化处理器
            doc_processor = FineBIDocProcessor(config=doc_config)
            logging.info("⚙️ 正在执行后处理、LLM 深度语义提取与完整 JSON 转换...")

            # 3. 调用核心处理方法（此方法内部会按需拉起并预热 LLM，跑完自动清理）
            doc_processor.process_pdf(
                pdf_path=pdf_path,
                md_input=markdown_output,
                output_dir=output_root
            )
            
            logging.info('🏁 所有 PDF 转换任务及后处理已圆满完成！')

        except Exception as e:
            logging.error(f"❌ 阶段二转换发生异常: {e}", exc_info=True)
        finally:
            # ⭐️ 直接触发 ModelFactory 的主动销毁熔断
            logging.info("🧹 [阶段二] 触发 ModelFactory 物理销毁阶段二 LLM 模型...")
            factory = ModelFactory(prompt_hub_path=prompts_hub_path)
            factory.destroy_llm_model()  # 触发 factory 熔断销毁
            
            if doc_processor is not None:
                del doc_processor

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            logging.info("💾 [阶段二] 显存已完全回收。")
    else:
        logging.error("❌ 因前置 VLM 提取失败，未执行后续阶段。")

    # ========================================================
    # 阶段三：使用独立的量化/轻量模型进行质量检验
    # ========================================================
    if markdown_output:
        logging.info("🔍 [阶段三] 启动数据质量检验引擎...")
        val_processor = None
        try:
            # 1. 动态生成目标 JSON 文件路径
            json_filename = os.path.basename(pdf_path).replace('.pdf', '.json')
            target_json = os.path.join(output_root, json_filename)

            if not os.path.exists(target_json):
                logging.error(f"❌ 找不到校验目标文件: {target_json}")
            else:
                # 2. 实例化验证引擎 (此时 GPU 处于完全空闲状态)
                val_processor = ValidationProcessor(
                    prompt_hub_path=prompts_hub_path, 
                    cuda_device="0"
                )
                
                validator = RAGDataValidator(
                    processor=val_processor, 
                    score_threshold=80.0, 
                    llm_sample_size=25
                )
                
                # 3. 指定用于评估的量化/轻量模型名称（如量化版 Qwen）
                quantized_eval_model = "Qwen/Qwen3-32B"
                
                logging.info(f"🤖 正在使用评估模型 [{quantized_eval_model}] 进行评分...")
                is_ok, score, report = validator.validate_json_file(
                    file_path=target_json,
                    llm_model=quantized_eval_model
                )
                
                logging.info(f"📊 质量检验完成 | 放行状态: {is_ok} | 最终得分: {score}")
                if score >= float(score_threshold):
                    is_quality_passed = True


        except Exception as e:
            logging.error(f"❌ 阶段三质量检验出现异常: {e}", exc_info=True)
        finally:
            # 4. 评估结束，再次调用 ModelFactory 销毁评估模型
            logging.info("🧹 [阶段三] 正在销毁评估模型并释放显存...")
            if val_processor is not None and hasattr(val_processor, "factory"):
                val_processor.factory.destroy_llm_model()
                del val_processor
            else:
                factory = ModelFactory(prompt_hub_path=prompts_hub_path)
                factory.destroy_llm_model()

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            logging.info("💾 全流程结束，显存归还系统。")

    # ========================================================
    # 阶段四：向量数据库 (Milvus) 写入与一致性审计
    # ========================================================
    if is_quality_passed:
        logging.info("🚀 [阶段四] 质量校验通过，准备将数据写入 Milvus 向量库...")
        uploader = None
        try:
            # 1. 初始化向量库上传器
            uploader = FineBIMilvusUploader(
                milvus_host="172.17.0.1",
                collection_name="finebi_knowledge_chunks",
                cuda_device="0"
            )
            
            # 2. 读取并写入向量库
            logging.info(f"📦 正在加载目标 JSON 数据源: {target_json}")
            with open(target_json, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            logging.info("⬆️ 正在执行 Milvus 数据分块写入与 Embedding 向量化...")
            uploader.upload_json_file(target_json)
            
            # 3. 运行一致性数据核对审计
            logging.info("🔎 正在执行写入后数据一致性审计...")
            uploader.audit_milvus_with_json(target_json)
            
            logging.info("🎉 [阶段四] Milvus 入库与一致性审计全部完成！")

        except Exception as e:
            logging.error(f"❌ [阶段四] Milvus 写入或审计过程发生错误: {e}", exc_info=True)
        finally:
            logging.info("🧹 [阶段四] 正在清理 Embedding 模型显存与连接资源...")
            if uploader is not None:
                # 如果 uploader 内部持有 embedding_model，尝试显式删除
                if hasattr(uploader, "embedding_model"):
                    del uploader.embedding_model
                del uploader

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            logging.info("💾 [阶段四] 全流程结束，GPU 显存与系统资源已被彻底完全释放！")
    else:
        logging.warning("⚠️ [阶段四] 因数据未通过阶段三质量检验 (或前置阶段失败)，已被拦截，拒绝写入 Milvus！")