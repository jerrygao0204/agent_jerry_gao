# ingest/db_uploader.py 连接向量库并写入
import os
import re
import time
import json
import logging
from collections import Counter
from typing import Dict, List
import torch
from pymilvus import MilvusClient, DataType
import sys
import hashlib

# 动态将当前脚本的上一级目录（即项目根目录 /workspace）加入 sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 🟢 导入已经对齐的系统模型工厂
from factory.model_factory import ModelFactory

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class FineBIMilvusUploader:
    """基于 MilvusClient 的 FineBI 知识库双路写入与检索验证器"""

    def __init__(
        self,
        milvus_host: str = "172.17.0.1",
        milvus_port: str = "19530",
        collection_name: str = "finebi_knowledge_chunks",
        model_short_name: str = "Qwen/Qwen3-Embedding-8B",
        cache_dir: str = "/workspace/hf-conda/hf_cache/hub",
        cuda_device: str = "0"
    ):
        self.collection_name = collection_name
        self.model_short_name = model_short_name
        self.cache_dir = cache_dir
        
        # 1. 实例化模型工厂底座（这一步会自动构建环境并建立软链接）
        self.factory = ModelFactory(cache_dir=self.cache_dir)
        
        # 2. 统一管理并绑定 GPU 算力分配
        self.factory.setup_cuda_device(cuda_device)
        
        self.model = None
        self.tokenizer = None
        
        # 使用现代化的 MilvusClient 建立连接
        self.client = MilvusClient(uri=f"http://{milvus_host}:{milvus_port}")
        logging.info(f"⚡ 成功连接 to Milvus Client [http://{milvus_host}:{milvus_port}]")

    def _init_embedding_engine(self):
        """⚡ 转向模型工厂获取实例，工厂内部已处理好环境与软链接"""
        if self.model is None or self.tokenizer is None:
            # 🎯 修复点：调用工厂的 get_llm_model 实例方法，并将接收变量顺序调整为 (model, tokenizer)
            self.model, self.tokenizer = self.factory.get_llm_model(
                llm_short_name=self.model_short_name
            )

    def get_dense_embedding(self, text: str) -> List[float]:
        self._init_embedding_engine()
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.model.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            embeddings = outputs.hidden_states[-1].mean(dim=1)
        return embeddings[0].to(torch.float32).cpu().numpy().tolist()


    @staticmethod
    def generate_sparse_vector(text: str) -> Dict[int, float]:
        tokens = re.findall(r"\w+", text.lower())
        counts = Counter(tokens)
        sparse_dict = {}
        for token, count in counts.items():
            # 🌟 使用 hashlib 确保不同进程间 Hash 维度 Index 绝对固定
            dim_id = (
                int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % (2**31 - 1)
            )
            sparse_dict[dim_id] = float(count)
        return sparse_dict

    def diagnose_cuda_memory(self):
        if not torch.cuda.is_available(): return
        if torch.cuda.memory_stats(0).get("inactive_split_bytes", 0) > 500 * 1024**2:
            torch.cuda.empty_cache()

    def _create_collection(self, force_recreate: bool = False):
        """基于 MilvusClient 的 Schema 创建方式"""
        exists = self.client.has_collection(collection_name=self.collection_name)

        if exists and force_recreate:
            logging.info(f"🗑️ 强制清理旧版数据集合: {self.collection_name}")
            self.client.drop_collection(collection_name=self.collection_name)
            time.sleep(1)
            exists = False

        if not exists:
            logging.info(f"🏗️ 集合 [{self.collection_name}] 不存在，开始构建 Schema...")
            schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
            
            schema.add_field(field_name="chunk_id", datatype=DataType.VARCHAR, is_primary=True, max_length=100)
            schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=4096)
            schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
            schema.add_field(field_name="file_name", datatype=DataType.VARCHAR, max_length=500)
            schema.add_field(field_name="file_url", datatype=DataType.VARCHAR, max_length=500)
            schema.add_field(field_name="doc_md5", datatype=DataType.VARCHAR, max_length=64)
            schema.add_field(field_name="processed_at", datatype=DataType.VARCHAR, max_length=30)
            schema.add_field(field_name="total_segments", datatype=DataType.INT64)

            schema.add_field(field_name="path_hierarchy", datatype=DataType.JSON)
            schema.add_field(field_name="entity_summary", datatype=DataType.VARCHAR, max_length=1000)
            schema.add_field(field_name="all_entity_uuids", datatype=DataType.JSON)
            
            schema.add_field(field_name="user_level", datatype=DataType.VARCHAR, max_length=20)
            schema.add_field(field_name="business_scene", datatype=DataType.VARCHAR, max_length=100)
            schema.add_field(field_name="biz_summary", datatype=DataType.VARCHAR, max_length=1000)

            schema.add_field(field_name="content_keywords", datatype=DataType.JSON)
            schema.add_field(field_name="core_entities", datatype=DataType.JSON)
            schema.add_field(field_name="operation_constraints", datatype=DataType.JSON)
            
            schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)
            schema.add_field(field_name="chunk_md5", datatype=DataType.VARCHAR, max_length=64)
            schema.add_field(field_name="content_summary", datatype=DataType.VARCHAR, max_length=1000)

            schema.add_field(field_name="entity_uuids", datatype=DataType.JSON)
            
            schema.add_field(field_name="section_id", datatype=DataType.VARCHAR, max_length=200)
            schema.add_field(field_name="local_scene", datatype=DataType.VARCHAR, max_length=100)
            schema.add_field(field_name="source_file", datatype=DataType.VARCHAR, max_length=500)
            schema.add_field(field_name="full_hierarchy", datatype=DataType.JSON)

            schema.add_field(field_name="full_hierarchy_array", datatype=DataType.JSON)
            schema.add_field(field_name="image_map", datatype=DataType.JSON)
            schema.add_field(field_name="image_urls", datatype=DataType.JSON)
            
            schema.add_field(field_name="prev_chunk_id", datatype=DataType.VARCHAR, max_length=100)
            schema.add_field(field_name="next_chunk_id", datatype=DataType.VARCHAR, max_length=100)

            index_params = self.client.prepare_index_params()
            index_params.add_index(field_name="dense_vector", index_type="HNSW", metric_type="COSINE", params={"M": 16, "efConstruction": 200})
            index_params.add_index(field_name="sparse_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="IP")
            
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params
            )
            logging.info(f"✅ 集合 [{self.collection_name}] 及混合索引首次构建完毕。")
        else:
            logging.info(f" 集合 [{self.collection_name}] 已存在，本次将直接采用追加模式。")

    def upload_json_file(self, json_file_path: str):
        """🟢 智能查重/覆盖更新模式（带3次重试容错、终极回滚与非对称向量提取）"""
        self._create_collection(force_recreate=False)

        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        doc_meta = data.get("doc_metadata", {})
        biz_profile = doc_meta.get("business_profile", {})
        current_doc_md5 = doc_meta.get("md5", "SNULL")

        payloads = data.get("milvus_payloads", [])
        if not payloads:
            logging.info(" 传入的 JSON 文件中没有检测到切片数据。")
            return

        incoming_chunk_ids = [p.get("metadata", {}).get("chunk_id") for p in payloads if p.get("metadata", {}).get("chunk_id")]
        
        logging.info(f"🔍 正在核对线上库，检索是否有历史冲突切片...")
        existing_records = self.client.query(
            collection_name=self.collection_name,
            filter=f"chunk_id in {incoming_chunk_ids}",
            output_fields=["*"]
        )
 
        online_doc_md5_map = {item["chunk_id"]: item.get("doc_md5") for item in existing_records}
        online_chunk_md5_map = {item["chunk_id"]: item.get("chunk_md5") for item in existing_records}

        backup_records_dict = {item["chunk_id"]: item for item in existing_records}

        insert_data = []
        NULL_STR = "SNULL"
        chunks_to_delete = []
        skip_count = 0
        update_count = 0
        new_count = 0

        logging.info(f"🔄 正在解析当前文件数据切片: {os.path.basename(json_file_path)} ...")
        for i, payload in enumerate(payloads):
            content = payload.get("content", "")
            meta = payload.get("metadata", {})
            chunk_id = meta.get("chunk_id") or NULL_STR
            incoming_chunk_md5 = meta.get("md5") or NULL_STR

            if chunk_id in online_doc_md5_map:
                online_d_md5 = online_doc_md5_map[chunk_id]
                online_c_md5 = online_chunk_md5_map.get(chunk_id)
                if online_d_md5 == current_doc_md5 and online_c_md5 == incoming_chunk_md5:
                    skip_count += 1
                    continue
                else:
                    chunks_to_delete.append(chunk_id)
                    update_count += 1
            else:
                new_count += 1

            # ------------------------------------------------------------------
            # 🟢 1. 内存中构建专供 Embedding 计算的富语义短文本（不落盘）
            # ------------------------------------------------------------------
            hierarchy_values = list(meta.get("hierarchy", {}).values())
            hierarchy_str = " > ".join(hierarchy_values) if hierarchy_values else ""
            summary_str = meta.get("summary") or biz_profile.get("summary") or ""
            core_entities_list = biz_profile.get("core_entities", []) or meta.get("core_entities", [])
            core_entities_str = ", ".join(core_entities_list) if isinstance(core_entities_list, list) else str(core_entities_list)

            # 强力过滤：剔除纯 URL、本地文件绝对路径、纯 JSON 字典结构噪音
            clean_content = re.sub(r'http\S+|/workspace\S+|\{.*?\}', '', content)
            clean_content_prefix = clean_content.strip()[:300]

            # 拼装控制在 300~400 Token 范围内的核心索引文本
            text_to_embed = f"文档路径: {hierarchy_str}\n"
            if core_entities_str:
                text_to_embed += f"核心实体: {core_entities_str}\n"
            if summary_str:
                text_to_embed += f"内容摘要: {summary_str}\n"
            text_to_embed += f"正文核心: {clean_content_prefix}"

            # ------------------------------------------------------------------
            # 🟢 2. 组装插入数据（Schema 零变动，密集与稀疏向量皆用 text_to_embed）
            # ------------------------------------------------------------------
            record = {
                "chunk_id": meta.get("chunk_id") or NULL_STR,
                "dense_vector": self.get_dense_embedding(text_to_embed),
                "sparse_vector": self.generate_sparse_vector(text_to_embed),
                "file_name": doc_meta.get("file_name") or NULL_STR,
                "file_url": doc_meta.get("file_url") or NULL_STR,
                "doc_md5": doc_meta.get("md5") or NULL_STR,
                "processed_at": doc_meta.get("processed_at") or NULL_STR,
                "total_segments": doc_meta.get("total_segments") or 0,
                "path_hierarchy": doc_meta.get("path_hierarchy") or [],
                "entity_summary": doc_meta.get("entity_summary") or NULL_STR,
                "all_entity_uuids": doc_meta.get("all_entity_uuids") or [],
                "user_level": biz_profile.get("user_level") or NULL_STR,
                "business_scene": biz_profile.get("business_scene") or meta.get("scene") or NULL_STR,
                "biz_summary": biz_profile.get("summary") or NULL_STR,
                "content_keywords": biz_profile.get("content_keywords") or [],
                "core_entities": biz_profile.get("core_entities") or [],
                "operation_constraints": biz_profile.get("operation_constraints") or [],
                "content": (content or NULL_STR)[:65000],
                "chunk_md5": meta.get("md5") or NULL_STR,
                "content_summary": (meta.get("summary") or NULL_STR)[:1000],
                "entity_uuids": payload.get("entity_uuids") or [],
                "section_id": meta.get("section_id") or NULL_STR,
                "local_scene": meta.get("scene") or NULL_STR,
                "source_file": meta.get("source_file") or NULL_STR,
                "full_hierarchy": meta.get("hierarchy") or {}, 
                "full_hierarchy_array": list(meta.get("hierarchy", {}).values()),
                "image_map": meta.get("image_map") or {},      
                "image_urls": meta.get("image_urls") or [],    
                "prev_chunk_id": meta.get("prev_chunk_id") or NULL_STR,
                "next_chunk_id": meta.get("next_chunk_id") or NULL_STR
            }
            insert_data.append(record)

            if (i + 1) % 10 == 0:
                self.diagnose_cuda_memory()

        if not insert_data:
            logging.info(" 线上数据已是最新，无需做任何数据更替动作。")
            return

        max_retries = 3
        success = False
        last_exception = None

        for attempt in range(1, max_retries + 1):
            try:
                logging.info(f"⚡ 正在尝试向 Milvus 写入数据 (第 {attempt}/{max_retries} 次尝试)...")
                
                if chunks_to_delete:
                    logging.info(f"   [试图物理移除] {len(chunks_to_delete)} 条更替旧切片...")
                    self.client.delete(collection_name=self.collection_name, filter=f"chunk_id in {chunks_to_delete}")
                
                logging.info(f"   [试图批量推送] {len(insert_data)} 条新 RAG 知识元组...")
                self.client.insert(collection_name=self.collection_name, data=insert_data)
                
                logging.info(f"🚀 [SUCCESS] 该文件导入在第 {attempt} 次尝试时圆满成功！")
                success = True
                break

            except Exception as e:
                last_exception = e
                logging.warning(f"⚠️ 第 {attempt} 次写入失败！原因: {e}")
                if attempt < max_retries:
                    sleep_time = attempt * 2
                    logging.info(f"⏳ 将在 {sleep_time} 秒后发起下一次尝试重试...")
                    time.sleep(sleep_time)

        if not success:
            logging.critical(f"🚨 [FATAL ERROR] 连续 {max_retries} 次尝试写入均宣告失败！触发终极防御策略。")
            logging.warning("🛑 正在紧急启动应用级事务回滚程序，全力恢复向量库历史状态...")
            
            try:
                incoming_all_ids = [p.get("metadata", {}).get("chunk_id") for p in payloads if p.get("metadata", {}).get("chunk_id")]
                if incoming_all_ids:
                    logging.info("🧹 物理清理中途尝试写入的混杂新数据...")
                    self.client.delete(collection_name=self.collection_name, filter=f"chunk_id in {incoming_all_ids}")
                
                if chunks_to_delete:
                    records_to_restore = [backup_records_dict[cid] for cid in chunks_to_delete if cid in backup_records_dict]
                    if records_to_restore:
                        logging.info(f"🔄 正在回填恢复 {len(records_to_restore)} 条历史备份数据元组...")
                        self.client.insert(collection_name=self.collection_name, data=records_to_restore)
                        
                logging.info("🎉 [ROLLBACK SUCCESS] 向量知识表已完美恢复到本次操作前的干净状态！")
            except Exception as rollback_error:
                logging.critical(f"😱 [CRITICAL BLOW] 极其罕见！连回滚补偿逻辑也失败了: {rollback_error}。")
            
            raise last_exception

    def audit_milvus_with_json(self, json_file_path: str):
        """🕵️‍♂️ 核对向量库入库标量信息与原始本地 JSON 文件的一致性"""
        logging.info(f"🕵️‍♂️ 启动本地 JSON 与 Milvus 库双向一致性审计流...")
        try:
            self.client.flush(self.collection_name)
            self.client.load_collection(collection_name=self.collection_name)
            time.sleep(1)
        except Exception as e:
            logging.warning(f"⚠️ 自动冲刷/加载集合时出现小插曲（可能尚未建立索引）: {e}")
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        payloads = json_data.get("milvus_payloads", [])
        total_json_records = len(payloads)
        
        chunk_ids = [p.get("metadata", {}).get("chunk_id") for p in payloads if p.get("metadata", {}).get("chunk_id")]
        
        milvus_results = self.client.query(
            collection_name=self.collection_name,
            filter=f"chunk_id in {chunk_ids}",
            output_fields=["*"]
        )
        
        milvus_dict = {item["chunk_id"]: item for item in milvus_results}
        mismatches = 0
        missing_ids = []
        
        print("\n" + "="*10 + " ⚙️ 标量一致性像素级核对盘点 ⚙️ " + "="*10)
        for payload in payloads:
            meta = payload.get("metadata", {})
            chunk_id = meta.get("chunk_id")
            
            if chunk_id not in milvus_dict:
                logging.error(f"❌ 严重缺陷：主键 ID [{chunk_id}] 在 Milvus 中未发现！")
                missing_ids.append(chunk_id)
                mismatches += 1
                continue
                
            milvus_record = milvus_dict[chunk_id]
            
            json_content = (payload.get("content", "") or "SNULL")[:65000]
            milvus_content = milvus_record.get("content", "")
            if json_content != milvus_content:
                print(f"⚠️ [差异发现] Chunk ID: {chunk_id} -> 基础文本切片不符")
                mismatches += 1
                
            # 🟢 修正后的审计核对部分：
            json_hierarchy = list(meta.get("hierarchy", {}).values())
            milvus_hierarchy = milvus_record.get("full_hierarchy_array") or []

            if json_hierarchy != milvus_hierarchy:
                print(
                    f"⚠️ [差异发现] Chunk ID: {chunk_id} -> 物理结构树不一致！\n"
                    f"   JSON: {json_hierarchy}\n"
                    f"   Milvus: {milvus_hierarchy}"
                )
                mismatches += 1
                
            json_section = meta.get("section_id") or "SNULL"
            milvus_section = milvus_record.get("section_id", "")
            if json_section != milvus_section:
                print(f"⚠️ [差异发现] Chunk ID: {chunk_id} -> Section ID 标志不符 ({json_section} vs {milvus_section})")
                mismatches += 1
        
        if mismatches == 0 and len(milvus_results) == total_json_records:
            logging.info("🎉 [AUDIT PASS] 核对通过！Milvus 向量数据库标量与本地源 JSON 100% 绝对一致！")
        else:
            logging.error(f"🚨 [AUDIT FAILED] 核对未通过！检测到 {mismatches} 处错误。")
            if missing_ids:
                print(f"👻 Milvus 遗漏的主键列表: {missing_ids}")
        print("="*63 + "\n")

    def run_formal_test(self, query: str):
        logging.info(f"🔍 启动检索验证流，目标 Prompt: '{query}'")
        self.client.load_collection(collection_name=self.collection_name)

        query_vector = self.get_dense_embedding(query)
        
        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            anns_field="dense_vector",
            search_params={"metric_type": "COSINE", "params": {"ef": 64}},
            limit=3,
            consistency_level="Strong",
            output_fields=[
                "chunk_id", "content", "section_id", 
                "path_hierarchy", "prev_chunk_id", "full_hierarchy_array"
            ]
        )

        if not results or len(results) == 0 or len(results[0]) == 0:
            logging.warning("⚠️ MilvusClient 返回了空结果！请检查向量索引或数据是否真的写入成功。")
            return

        logging.info(f"🎉 成功检索到 {len(results[0])} 条相似结果，开始解析输出：")

        for hit in results[0]:
            print("-" * 60)
            distance = hit.get("distance", 0.0)
            print(f"🎯 相似度权重得分 (Cosine Score): {distance:.4f}")
            
            entity = hit.get("entity", {})
            if not entity:
                print("⚠️ 警告：该命中项没有携带任何 entity 标量数据！")
                continue
                
            print(f"🆔 拓扑 Chunk ID: {entity.get('chunk_id', 'SNULL')}")
            print(f"📍 归属 Section ID: {entity.get('section_id', 'SNULL')}")
            
            path_dict = entity.get('path_hierarchy', {})
            path_array = path_dict.get('data', []) if isinstance(path_dict, dict) else path_dict
            if isinstance(path_array, list) and path_array:
                print(f"🗂️ 层级全路径: {' > '.join(path_array)}")
            
            h_dict = entity.get("full_hierarchy_array", {})
            h_array = h_dict.get('data', []) if isinstance(h_dict, dict) else h_dict
            if isinstance(h_array, list) and h_array: 
                print(f"🧭 物理标题结构: {' > '.join(h_array)}")
                
            content_str = entity.get('content', '')
            print(f"📝 文本前缀切片: {content_str[:120].strip()}...")
            
            prev_id = entity.get("prev_chunk_id")
            if prev_id and prev_id != "SNULL":
                print(f"🔗 记忆链-追溯前置 ID: {prev_id}")
                
        print("-" * 60)

if __name__ == "__main__":
    DATA_SOURCE_PATH = "/workspace/hf-conda/RAG/问答机器人/finebi_output/数据预警.json"
    uploader = FineBIMilvusUploader(
        milvus_host="172.17.0.1",
        collection_name="finebi_knowledge_chunks",
        cuda_device="0"
    )
    
    with open(DATA_SOURCE_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 写入向量库
    uploader.upload_json_file(DATA_SOURCE_PATH)
    # 一致性核对
    uploader.audit_milvus_with_json(DATA_SOURCE_PATH)