# # app_admin.py
# import subprocess
# import sys

# def install_package(package):
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package])
# install_package("gradio")

# import os
# import sys
# import gc
# import json
# import logging
# import torch
# import time
# import gradio as gr
# from pathlib import Path

# # 设置日志格式
# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# from factory.model_factory import ModelFactory
# from factory import ModelFactory, tool_factory, init_tools
# from config.config_loader import DEFAULT_CONFIG, config_loader # 配置文件加载器
# from data_prep.pdf_to_markdown import MarkdownProcessor
# from data_prep.markdown_to_json import FineBIDocConfig, FineBIDocProcessor
# from ingest.validator import Processor as ValidationProcessor, RAGDataValidator
# from ingest.db_uploader import FineBIMilvusUploader

# # 📌 获取当前脚本 (app_admin.py) 所在的绝对路径目录
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# # 配置文件持久化路径 (存放在当前 py 同级目录下)
# CONFIG_FILE_PATH = os.path.join(SCRIPT_DIR, "admin_config.json")

# # 初始化全局默认路径配置，将 output_root 默认指向当前 py 所在文件夹
# DEFAULT_CONFIG = {
#     "output_root": SCRIPT_DIR,
#     "prompts_hub_path": config_loader.prompt_hub_path,
#     "patterns_path": config_loader.patterns_path,
#     "img_prefix": "https://your-oss-bucket.com/finebi/docs/images",
#     "pdf_prefix": "https://your-oss-bucket.com/finebi/pdfs/",
#     "namespace_seed": "FineBI_RAG_2026",
#     "milvus_host": "172.17.0.1",
#     "collection_name": "finebi_knowledge_chunks",
#     "vlm_model_name": "Qwen--Qwen3-VL-32B-Instruct",
#     "llm_model_name": "Qwen/Qwen3-32B",
#     "score_threshold": 80.0
# }

# def load_config():
#     """读取本地 admin_config.json 并在绝对路径失效时自动修正"""
#     global DEFAULT_CONFIG
#     if os.path.exists(CONFIG_FILE_PATH):
#         try:
#             with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
#                 saved_config = json.load(f)
                
#                 # 校验路径是否存在，若不存在则回退至相对路径解析，防范硬编码迁移失效
#                 prompts_path = saved_config.get("prompts_hub_path", "")
#                 if not os.path.exists(prompts_path):
#                     saved_config["prompts_hub_path"] = config_loader.prompt_hub_path
#                     logging.warning(f"⚠️ 校验到原配置路径不存在: {prompts_path}，已自动更正为: {config_loader.prompt_hub_path}")

#                 patterns_path = saved_config.get("patterns_path", "")
#                 if not os.path.exists(patterns_path):
#                     saved_config["patterns_path"] = config_loader.patterns_path
#                     logging.warning(f"⚠️ 校验到原配置路径不存在: {patterns_path}，已自动更正为: {config_loader.patterns_path}")

#                 DEFAULT_CONFIG.update(saved_config)
#                 logging.info(f"⚙️ 成功加载持久化配置文件: {CONFIG_FILE_PATH}")
#         except Exception as e:
#             logging.error(f"❌ 读取配置文件异常: {e}")
            
#     os.makedirs(DEFAULT_CONFIG["output_root"], exist_ok=True)

# load_config()

# # ==========================================
# # 🛠️ 工厂与基础设施全局初始化
# # ==========================================
# # 1. 启动 ModelFactory 并加载全局模型的生命周期管理 (单例模式)
# model_factory = ModelFactory.get_instance()

# # 2. 初始化全局工具工厂 (后续如果创建了 retriever，在此处传入即可)
# # retriever = FineBIRetriever(...)
# init_tools(retriever=None)

# # ==========================================
# # 辅助函数：GPU 显存状态监控
# # ==========================================
# def get_gpu_memory_status():
#     """获取当前 GPU 显存使用情况"""
#     if not torch.cuda.is_available():
#         return "GPU 不可用 (CPU Mode)"
    
#     allocated = torch.cuda.memory_allocated() / (1024 ** 3)
#     reserved = torch.cuda.memory_reserved() / (1024 ** 3)
#     total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
#     return f"GPU 显存状态: 使用率 {allocated/total:.1%} | 实际分配 {allocated:.1f} GB | 已申领预留 {reserved:.1f} GB | 总共 {total:.1f} GB"

# def force_gc_cleanup():
#     """彻底回收 CUDA 显存与 Python 垃圾"""
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.synchronize()
#         torch.cuda.empty_cache()
#     return get_gpu_memory_status()

# def emergency_force_cleanup():
#     """🔥 应急熔断：一键彻底物理清空所有工厂模型并回收 CUDA 显存"""
#     logging.warning("🚨 [Emergency Cleanup] 用户触发了应急全量显存回收！")
#     if hasattr(ModelFactory, "destroy_all_models_cls"):
#         ModelFactory.destroy_all_models_cls()
#     else:
#         ModelFactory.destroy_vlm_model()
#         ModelFactory.destroy_llm_model()
#         if hasattr(ModelFactory, "destroy_embedding_model"):
#             ModelFactory.destroy_embedding_model()
#         force_gc_cleanup()
#     return get_gpu_memory_status()

# # ==========================================
# # 阶段 1-3：运行流水线解析与质检
# # ==========================================
# def run_parsing_and_validation(pdf_file, score_threshold):
#     start_time = time.time()
    
#     # 保持“启动”按钮处于禁用状态，直到后续确认/取消选择
#     btn_running = gr.update(value="⏳ 拼命解析中，请勿刷新页面...", variant="secondary", interactive=False)
#     btn_keep_disabled = gr.update(value="⏳ 请等待确认或取消写入", variant="secondary", interactive=False)
#     btn_write_disabled = gr.update(visible=False, interactive=False)
#     btn_cancel_disabled = gr.update(visible=False, interactive=False)
    
#     def status(msg):
#         return f"⏳ {msg} | 已耗时: {time.time() - start_time:.1f}s"

#     if not pdf_file:
#         btn_reset = gr.update(value="🚀 启动 1-3 阶段解析质检", variant="primary", interactive=True)
#         yield "❌ 请先输入或上传 PDF 文件！", "", "", get_gpu_memory_status(), "🔴 错误：未提供文件", btn_reset, btn_write_disabled, btn_cancel_disabled, ""
#         return
    
#     pdf_path = pdf_file.name if hasattr(pdf_file, "name") else str(pdf_file)
#     output_root = DEFAULT_CONFIG["output_root"]
#     logs = []

#     def make_log(msg):
#         logging.info(msg)
#         logs.append(msg)
#         return "\n".join(logs)

#     markdown_output = None
#     is_quality_passed = False
#     target_json = ""

#     # ----------------------------------------------------
#     # 阶段一：VLM 提取 Markdown
#     # ----------------------------------------------------
#     current_log = make_log("🚀 [阶段一] 开始运行 VLM 提取 PDF 结构及内容...")
#     yield current_log, "", "", get_gpu_memory_status(), status("阶段一：VLM 提取中..."), btn_running, btn_write_disabled, btn_cancel_disabled, ""

#     try:
#         # 获取绝对路径并确认文件存在
#         prompt_hub_path = os.path.abspath(DEFAULT_CONFIG["prompts_hub_path"])
#         if not os.path.exists(prompt_hub_path):
#             raise FileNotFoundError(f"配置文件路径不存在，请检查配置: {prompt_hub_path}")

#         logging.info(f"🔑 阶段一正在使用的 Prompt Path: {prompt_hub_path}")
#         input()
#         processor = MarkdownProcessor(prompt_hub_path=DEFAULT_CONFIG["prompts_hub_path"])
#         print(processor)
#         markdown_output = processor.main(
#             pdf_path=pdf_path, 
#             prompt_path=DEFAULT_CONFIG["prompts_hub_path"], 
#             vlm=DEFAULT_CONFIG["vlm_model_name"]
#         )
#         current_log = make_log("✅ [阶段一] Markdown 原始数据提取成功！")
#         yield current_log, markdown_output or "", "", get_gpu_memory_status(), status("阶段一：提取完毕，释放资源..."), btn_running, btn_write_disabled, btn_cancel_disabled, ""
#     except Exception as e:
#         current_log = make_log(f"❌ [阶段一] 发生异常: {e}")
#         yield current_log, "", "", get_gpu_memory_status(), status("阶段一：发生异常！"), btn_running, btn_write_disabled, btn_cancel_disabled, ""
#     finally:
#         current_log = make_log("🧹 [阶段一] 正在呼叫工厂物理销毁 VLM 模型并释放显存...")
#         if 'processor' in locals():
#             del processor
#         ModelFactory.destroy_vlm_model()
#         gpu_stat = get_gpu_memory_status()
#         current_log = make_log(f"💾 [阶段一] 显存已完全回收。{gpu_stat}")
#         yield current_log, markdown_output or "", "", gpu_stat, status("阶段一：完成"), btn_running, btn_write_disabled, btn_cancel_disabled, ""

#     # ----------------------------------------------------
#     # 阶段二：后处理与 JSON 转换
#     # ----------------------------------------------------
#     current_log = make_log("⚙️ [阶段二] 正在执行深度语义提取与 JSON 转换...")
#     yield current_log, markdown_output, "", get_gpu_memory_status(), status("阶段二：转换 JSON 中..."), btn_running, btn_write_disabled, btn_cancel_disabled, ""

#     doc_processor = None
#     try:
#         doc_config = FineBIDocConfig(
#             namespace_seed=DEFAULT_CONFIG["namespace_seed"],
#             image_url_prefix=DEFAULT_CONFIG["img_prefix"],
#             pdf_url_prefix=DEFAULT_CONFIG["pdf_prefix"],
#             cuda_device="0",  
#             yaml_rules_path=DEFAULT_CONFIG["patterns_path"],
#             yaml_prompts_path=DEFAULT_CONFIG["prompts_hub_path"]
#         )
#         doc_processor = FineBIDocProcessor(config=doc_config)
#         doc_processor.process_pdf(
#             pdf_path=pdf_path,
#             md_input=markdown_output,
#             output_dir=output_root
#         )
#         json_filename = os.path.basename(pdf_path).replace('.pdf', '.json')
#         target_json = os.path.join(output_root, json_filename)
#         current_log = make_log(f"🏁 [阶段二] JSON 转换完成，目标文件保存路径: {target_json}")
#     except Exception as e:
#         current_log = make_log(f"❌ [阶段二] 发生异常: {e}")
#     finally:
#         current_log = make_log("🧹 [阶段二] 物理销毁阶段二 LLM 模型...")
#         if "doc_processor" in locals() and doc_processor is not None:
#             del doc_processor
#         ModelFactory.destroy_llm_model()
#         gpu_stat = get_gpu_memory_status()
#         current_log = make_log(f"💾 [阶段二] 显存已完全回收。{gpu_stat}")
#         yield current_log, markdown_output or "", "", gpu_stat, status("阶段二：完成"), btn_running, btn_write_disabled, btn_cancel_disabled, target_json

#     # ----------------------------------------------------
#     # 阶段三：数据质量检验
#     # ----------------------------------------------------
#     current_log = make_log("🔍 [阶段三] 启动数据质量检验引擎...")
#     yield current_log, markdown_output, "", get_gpu_memory_status(), status("阶段三：模型质量打分中..."), btn_running, btn_write_disabled, btn_cancel_disabled, target_json

#     val_processor = None
#     try:
#         if not os.path.exists(target_json):
#             current_log = make_log(f"❌ [阶段三] 找不到校验目标文件: {target_json}")
#         else:
#             val_processor = ValidationProcessor(
#                 prompt_hub_path=DEFAULT_CONFIG["prompts_hub_path"], 
#                 cuda_device="0"
#             )
#             validator = RAGDataValidator(
#                 processor=val_processor, 
#                 score_threshold=float(score_threshold), 
#                 llm_sample_size=25
#             )
#             current_log = make_log(f"🤖 正在使用评估模型 [{DEFAULT_CONFIG['llm_model_name']}] 进行评分...")
#             yield current_log, markdown_output, "", get_gpu_memory_status(), status("阶段三：检验计算中..."), btn_running, btn_write_disabled, btn_cancel_disabled, target_json

#             is_ok, score, report = validator.validate_json_file(
#                 file_path=target_json,
#                 llm_model=DEFAULT_CONFIG["llm_model_name"]
#             )
#             current_log = make_log(f"📊 [阶段三] 质量检验完成 | 放行状态: {is_ok} | 得分: {score}")
#             if score >= float(score_threshold):
#                 is_quality_passed = True
#     except Exception as e:
#         current_log = make_log(f"❌ [阶段三] 发生异常: {e}")
#     finally:
#         current_log = make_log("🧹 [阶段三] 物理销毁阶段三评估模型并释放显存...")
#         if "val_processor" in locals(): 
#             del val_processor
#         ModelFactory.destroy_llm_model()
#         gpu_stat = get_gpu_memory_status()
#         current_log = make_log(f"💾 [阶段三] 显存已完全回收。{gpu_stat}")

#     # 格式化 JSON 预览
#     json_preview = ""
#     if os.path.exists(target_json):
#         with open(target_json, 'r', encoding='utf-8') as f:
#             content = json.load(f)
#             if isinstance(content, list):
#                 preview_data = content[:2]
#             elif isinstance(content, dict):
#                 if "chunks" in content and isinstance(content["chunks"], list):
#                     preview_data = {**{k: v for k, v in content.items() if k != "chunks"}, "chunks": content["chunks"][:2]}
#                 else:
#                     preview_data = {k: content[k] for k in list(content.keys())[:5]}
#             else:
#                 preview_data = content
#             json_preview = json.dumps(preview_data, ensure_ascii=False, indent=2)

#     if is_quality_passed:
#         current_log = make_log("🟢 [阶段三] 质量校验通过！请选择点击【确认写入】或【取消写入】。")
#         btn_write_enable = gr.update(value="📥 确认写入向量数据库 (Milvus)", variant="primary", visible=True, interactive=True)
#     else:
#         current_log = make_log("⚠️ [阶段三] 质量校验未达标！仍可手动强行点击写入，或选择取消。")
#         btn_write_enable = gr.update(value="⚠️ 强制写入向量数据库 (分值未达标)", variant="stop", visible=True, interactive=True)

#     btn_cancel_enable = gr.update(value="❌ 取消写入并清空显存", variant="secondary", visible=True, interactive=True)

#     # 🛑 保持“启动 1-3 阶段解析质检”灰色不可用 (btn_keep_disabled)
#     yield current_log, markdown_output or "", json_preview, get_gpu_memory_status(), f"⏸️ 1-3 阶段执行完毕，等待写入判定 (耗时: {time.time() - start_time:.1f}s)", btn_keep_disabled, btn_write_enable, btn_cancel_enable, target_json

# # ==========================================
# # 阶段 4：确认写入 Milvus
# # ==========================================
# def write_to_milvus_action(target_json, current_logs):
#     logs = [current_logs] if current_logs else []
#     def make_log(msg):
#         logging.info(msg)
#         logs.append(msg)
#         return "\n".join(logs)

#     btn_write_running = gr.update(value="⏳ 向 Milvus 写入数据中...", interactive=False)
#     btn_write_finished = gr.update(value="✅ 已成功入库", interactive=False, visible=True)
#     btn_cancel_hidden = gr.update(visible=False, interactive=False)
#     btn_start_unlocked = gr.update(value="🚀 启动 1-3 阶段解析质检", variant="primary", interactive=True)

#     if not target_json or not os.path.exists(target_json):
#         updated_log = make_log("❌ [阶段四] 找不到有效的 JSON 文件路径，无法写入！")
#         yield updated_log, get_gpu_memory_status(), "🔴 写入失败", btn_start_unlocked, btn_write_finished, btn_cancel_hidden
#         return

#     updated_log = make_log(f"🚀 [阶段四] 手动触发写入，正在解析: {target_json}")
#     yield updated_log, get_gpu_memory_status(), "阶段四：Milvus 入库中...", gr.update(interactive=False), btn_write_running, btn_cancel_hidden

#     uploader = None
#     try:
#         uploader = FineBIMilvusUploader(
#             milvus_host=DEFAULT_CONFIG["milvus_host"],
#             collection_name=DEFAULT_CONFIG["collection_name"],
#             cuda_device="0"
#         )
#         updated_log = make_log("⬆️ 正在向 Milvus 写入数据与向量...")
#         uploader.upload_json_file(target_json)
#         updated_log = make_log("🔎 正在执行写入后数据一致性审计...")
#         uploader.audit_milvus_with_json(target_json)
#         updated_log = make_log("🎉 [阶段四] Milvus 入库与审计全部顺利完成！")
#     except Exception as e:
#         updated_log = make_log(f"❌ [阶段四] 发生异常: {e}")
#     finally:
#         updated_log = make_log("🧹 [阶段四] 正在清理数据库连接与物理销毁 Embedding 模型...")
#         if 'uploader' in locals() and uploader is not None:
#             if hasattr(uploader, "close"):
#                 try:
#                     uploader.close()
#                 except Exception:
#                     pass
#             del uploader

#         if hasattr(ModelFactory, "destroy_embedding_model"):
#             ModelFactory.destroy_embedding_model()
#         ModelFactory.destroy_llm_model()

#         gpu_stat = get_gpu_memory_status()
#         updated_log = make_log(f"💾 [阶段四] 显存已彻底完全释放！{gpu_stat}")
#         # 🟢 完成写入后，解锁启动按钮
#         yield updated_log, gpu_stat, "✅ 全流程完整完成！", btn_start_unlocked, btn_write_finished, btn_cancel_hidden

# # ==========================================
# # 新增动作：取消写入向量库并彻底归零显存
# # ==========================================
# def cancel_write_action(current_logs):
#     logs = [current_logs] if current_logs else []
#     def make_log(msg):
#         logging.info(msg)
#         logs.append(msg)
#         return "\n".join(logs)

#     updated_log = make_log("🚫 用户点击【取消写入】，跳过阶段四直接物理清空所有显存模型...")
    
#     # ⚡ 需求 1：直接执行系统终极显存清零
#     logging.info("⚡ 正在执行系统退出前显存终极清零...")
#     ModelFactory.destroy_all_models_cls()
#     logging.info("✨ 显存已彻底清空，安全退出。")

#     gpu_stat = emergency_force_cleanup()
#     updated_log = make_log(f"✨ 显存归零操作执行完毕！{gpu_stat}")

#     # 解锁启动按钮，同时隐藏确认/取消按钮
#     btn_start_unlocked = gr.update(value="🚀 启动 1-3 阶段解析质检", variant="primary", interactive=True)
#     btn_write_hidden = gr.update(visible=False, interactive=False)
#     btn_cancel_hidden = gr.update(visible=False, interactive=False)

#     return updated_log, gpu_stat, "🔴 已取消写入 (显存已归零)", btn_start_unlocked, btn_write_hidden, btn_cancel_hidden

# # ==========================================
# # 系统配置保存与更新逻辑
# # ==========================================
# def save_system_config(
#     output_root, prompts_hub_path, patterns_path, img_prefix, 
#     pdf_prefix, namespace_seed, milvus_host, collection_name, 
#     vlm_model_name, llm_model_name, score_threshold
# ):
#     global DEFAULT_CONFIG
#     new_config = {
#         "output_root": output_root,
#         "prompts_hub_path": prompts_hub_path,
#         "patterns_path": patterns_path,
#         "img_prefix": img_prefix,
#         "pdf_prefix": pdf_prefix,
#         "namespace_seed": namespace_seed,
#         "milvus_host": milvus_host,
#         "collection_name": collection_name,
#         "vlm_model_name": vlm_model_name,
#         "llm_model_name": llm_model_name,
#         "score_threshold": float(score_threshold)
#     }
    
#     DEFAULT_CONFIG.update(new_config)
#     os.makedirs(DEFAULT_CONFIG["output_root"], exist_ok=True)
    
#     try:
#         with open(CONFIG_FILE_PATH, 'w', encoding='utf-8') as f:
#             json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=4)
#         return "✅ 系统基础配置已成功保存并同步生效！", DEFAULT_CONFIG
#     except Exception as e:
#         return f"❌ 保存配置文件失败: {e}", DEFAULT_CONFIG

# # ==========================================
# # Gradio 管理后台界面构建
# # ==========================================
# def build_admin_ui():
#     with gr.Blocks(title="FineBI 知识库后台管理系统", theme=gr.themes.Soft()) as demo:
#         target_json_state = gr.State("")

#         gr.Markdown("# 🛡️ FineBI 知识库离线清洗与向量入库系统 (Admin Portal)")
        
#         # 顶部显存监控栏与熔断按钮
#         with gr.Row():
#             gpu_status_box = gr.Textbox(
#                 label="🖥️ GPU 显存实时监控", 
#                 value=get_gpu_memory_status(), 
#                 interactive=False, 
#                 scale=3
#             )
#             refresh_gpu_btn = gr.Button("🔄 刷新显存", scale=1)
#             force_cleanup_btn = gr.Button("🔥 强制全量显存回收 (Emergency Cleanup)", variant="stop", scale=1)

#         refresh_gpu_btn.click(fn=get_gpu_memory_status, outputs=[gpu_status_box])
#         force_cleanup_btn.click(fn=emergency_force_cleanup, outputs=[gpu_status_box])
            
#         timer = gr.Timer(5)
#         timer.tick(fn=get_gpu_memory_status, inputs=None, outputs=gpu_status_box)

#         with gr.Tabs():
#             # ---------------------------------------------------------
#             # TAB 1: 解析质检与确认写入
#             # ---------------------------------------------------------
#             with gr.Tab("🚀 全流程解析与向量入库"):
#                 gr.Markdown("上传/输入 PDF 文件，运行 **VLM提取 -> JSON转换 -> 轻量LLM质检**。质检完成后，可手动选择**确认写入**或**取消写入**。")
                
#                 with gr.Row():
#                     pdf_input = gr.Textbox(label="输入 PDF 文件路径", placeholder="/workspace/.../xxx.pdf", scale=2)
#                     with gr.Column(scale=1):
#                         score_thresh_input = gr.Number(
#                             label="质量校验门槛分 (Score)", 
#                             value=DEFAULT_CONFIG["score_threshold"]
#                         )
#                         status_box = gr.Textbox(label="⏱️ 任务执行状态", value="🟢 待机中", interactive=False)
#                         run_btn = gr.Button("🚀 启动 1-3 阶段解析质检", variant="primary")
                        
#                         # 确认写入与取消写入按钮（默认隐藏）
#                         with gr.Row():
#                             write_milvus_btn = gr.Button(
#                                 "📥 确认写入向量数据库 (Milvus)", 
#                                 variant="primary", 
#                                 visible=False, 
#                                 interactive=False
#                             )
#                             cancel_write_btn = gr.Button(
#                                 "❌ 取消写入并清空显存", 
#                                 variant="stop", 
#                                 visible=False, 
#                                 interactive=False
#                             )
                
#                 pipeline_logs = gr.Textbox(label="全流程运行日志 (Realtime Logs)", lines=12, interactive=False)
                
#                 with gr.Accordion("中间产物预览 (Markdown & JSON)", open=False):
#                     with gr.Row():
#                         md_preview = gr.Markdown(label="阶段一 Markdown 产物预览")
#                         json_preview = gr.Code(label="阶段二 JSON 切片预览 (前2条)", language="json")

#                 # 1. 绑定 1-3 阶段解析质检
#                 run_btn.click(
#                     fn=run_parsing_and_validation,
#                     inputs=[pdf_input, score_thresh_input],
#                     outputs=[
#                         pipeline_logs, md_preview, json_preview, 
#                         gpu_status_box, status_box, run_btn, 
#                         write_milvus_btn, cancel_write_btn, target_json_state
#                     ],
#                     show_progress="minimal"
#                 )

#                 # 2. 绑定第 4 阶段确认写入逻辑（完成后重新解锁 run_btn）
#                 write_milvus_btn.click(
#                     fn=write_to_milvus_action,
#                     inputs=[target_json_state, pipeline_logs],
#                     outputs=[pipeline_logs, gpu_status_box, status_box, run_btn, write_milvus_btn, cancel_write_btn],
#                     show_progress="minimal"
#                 )

#                 # 3. 绑定取消写入逻辑（清空显存并重新解锁 run_btn）
#                 cancel_write_btn.click(
#                     fn=cancel_write_action,
#                     inputs=[pipeline_logs],
#                     outputs=[pipeline_logs, gpu_status_box, status_box, run_btn, write_milvus_btn, cancel_write_btn],
#                     show_progress="minimal"
#                 )

#             # ---------------------------------------------------------
#             # TAB 2: 系统参数与路径配置
#             # ---------------------------------------------------------
#             with gr.Tab("⚙️ 系统基础配置"):
#                 gr.Markdown("### 🛠️ 动态修改并保存系统运行环境与网络配置")
                
#                 config_status_msg = gr.Markdown("")
                
#                 with gr.Row():
#                     cfg_output_root = gr.Textbox(label="输出目录 (output_root)", value=DEFAULT_CONFIG["output_root"])
#                     cfg_prompts_hub_path = gr.Textbox(label="Prompt Hub 路径", value=DEFAULT_CONFIG["prompts_hub_path"])
                
#                 with gr.Row():
#                     cfg_patterns_path = gr.Textbox(label="Patterns Rules 路径", value=DEFAULT_CONFIG["patterns_path"])
#                     cfg_namespace_seed = gr.Textbox(label="命名空间种子 (namespace_seed)", value=DEFAULT_CONFIG["namespace_seed"])

#                 with gr.Row():
#                     cfg_img_prefix = gr.Textbox(label="图片 OSS 前缀", value=DEFAULT_CONFIG["img_prefix"])
#                     cfg_pdf_prefix = gr.Textbox(label="PDF OSS 前缀", value=DEFAULT_CONFIG["pdf_prefix"])

#                 with gr.Row():
#                     cfg_milvus_host = gr.Textbox(label="Milvus 主机地址", value=DEFAULT_CONFIG["milvus_host"])
#                     cfg_collection_name = gr.Textbox(label="Milvus 集合名称", value=DEFAULT_CONFIG["collection_name"])

#                 with gr.Row():
#                     cfg_vlm_model_name = gr.Textbox(label="VLM 模型名称", value=DEFAULT_CONFIG["vlm_model_name"])
#                     cfg_llm_model_name = gr.Textbox(label="LLM 模型名称", value=DEFAULT_CONFIG["llm_model_name"])
#                     cfg_score_threshold = gr.Number(label="默认质检门槛分", value=DEFAULT_CONFIG["score_threshold"])

#                 save_config_btn = gr.Button("💾 保存系统基础配置", variant="primary")
                
#                 config_json_preview = gr.JSON(label="当前全局生效配置视图", value=DEFAULT_CONFIG)

#                 save_config_btn.click(
#                     fn=save_system_config,
#                     inputs=[
#                         cfg_output_root, cfg_prompts_hub_path, cfg_patterns_path, 
#                         cfg_img_prefix, cfg_pdf_prefix, cfg_namespace_seed, 
#                         cfg_milvus_host, cfg_collection_name, cfg_vlm_model_name, 
#                         cfg_llm_model_name, cfg_score_threshold
#                     ],
#                     outputs=[config_status_msg, config_json_preview]
#                 )

#     return demo


# if __name__ == "__main__":
#     import atexit
#     try:
#         init_tools(retriever=None)
#         logging.info("🛠️ 后台管理系统成功初始化全局工具工厂！")
#     except Exception as e:
#         logging.warning(f"⚠️ 工具工厂初始化跳过/异常: {e}")

#     def on_app_shutdown():
#         """Gradio 进程关闭或 Ctrl+C 退出时的终极显存归零"""
#         logging.info("⚡ 正在执行系统退出前显存终极清零...")
#         ModelFactory.destroy_all_models_cls()
#         logging.info("✨ 显存已彻底清空，安全退出。")

#     atexit.register(on_app_shutdown)

#     ui = build_admin_ui()
#     ui.queue().launch(
#         server_name="0.0.0.0", 
#         server_port=7861,
#         root_path="/admin"
#     )

import subprocess
import sys

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

try:
    import gradio as gr
except ImportError:
    install_package("gradio")

import os
import sys
import gc
import json
import logging
import torch
import time
import atexit
import gradio as gr
from pathlib import Path

# 设置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

from factory.model_factory import ModelFactory
from factory import tool_factory, init_tools
from config.config_loader import DEFAULT_CONFIG, config_loader  # 配置文件加载器
from data_prep.pdf_to_markdown import MarkdownProcessor
from data_prep.markdown_to_json import FineBIDocConfig, FineBIDocProcessor
from ingest.validator import Processor as ValidationProcessor, RAGDataValidator
from ingest.db_uploader import FineBIMilvusUploader

# 📌 获取当前脚本 (app_admin.py) 所在的绝对路径目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 配置文件持久化路径 (存放在当前 py 同级目录下)
CONFIG_FILE_PATH = os.path.join(SCRIPT_DIR, "admin_config.json")

# 初始化全局默认路径配置
DEFAULT_CONFIG = {
    "output_root": SCRIPT_DIR,
    "prompts_hub_path": config_loader.prompt_hub_path,
    "patterns_path": config_loader.patterns_path,
    "img_prefix": "https://your-oss-bucket.com/finebi/docs/images",
    "pdf_prefix": "https://your-oss-bucket.com/finebi/pdfs/",
    "namespace_seed": "FineBI_RAG_2026",
    "milvus_host": "172.17.0.1",
    "collection_name": "finebi_knowledge_chunks",
    "vlm_model_name": "Qwen/Qwen3-VL-32B-Instruct",
    "llm_model_name": "Qwen/Qwen3-32B",
    "score_threshold": 80.0
}

def load_config():
    """读取本地 admin_config.json 并在绝对路径失效时自动修正"""
    global DEFAULT_CONFIG
    if os.path.exists(CONFIG_FILE_PATH):
        try:
            with open(CONFIG_FILE_PATH, 'r', encoding='utf-8') as f:
                saved_config = json.load(f)
                
                # 校验路径是否存在，若不存在则更正为当前配置路径
                prompts_path = saved_config.get("prompts_hub_path", "")
                if not os.path.exists(prompts_path):
                    saved_config["prompts_hub_path"] = config_loader.prompt_hub_path
                    logging.warning(f"⚠️ 校验到原配置路径不存在: {prompts_path}，已自动更正为: {config_loader.prompt_hub_path}")

                patterns_path = saved_config.get("patterns_path", "")
                if not os.path.exists(patterns_path):
                    saved_config["patterns_path"] = config_loader.patterns_path
                    logging.warning(f"⚠️ 校验到原配置路径不存在: {patterns_path}，已自动更正为: {config_loader.patterns_path}")

                DEFAULT_CONFIG.update(saved_config)
                logging.info(f"⚙️ 成功加载持久化配置文件: {CONFIG_FILE_PATH}")
        except Exception as e:
            logging.error(f"❌ 读取配置文件异常: {e}")
            
    os.makedirs(DEFAULT_CONFIG["output_root"], exist_ok=True)

load_config()

# ==========================================
# 🛠️ 工厂与基础设施全局初始化
# ==========================================
model_factory = ModelFactory()
init_tools(retriever=None)

# ==========================================
# 辅助函数：GPU 显存状态监控
# ==========================================
def get_gpu_memory_status():
    """获取当前 GPU 显存使用情况"""
    if not torch.cuda.is_available():
        return "GPU 不可用 (CPU Mode)"
    
    allocated = torch.cuda.memory_allocated() / (1024 ** 3)
    reserved = torch.cuda.memory_reserved() / (1024 ** 3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    return f"GPU 显存状态: 使用率 {allocated/total:.1%} | 实际分配 {allocated:.1f} GB | 已申领预留 {reserved:.1f} GB | 总共 {total:.1f} GB"

def force_gc_cleanup():
    """彻底回收 CUDA 显存与 Python 垃圾"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return get_gpu_memory_status()

def emergency_force_cleanup():
    """🔥 应急熔断：一键彻底物理清空所有工厂模型并回收 CUDA 显存"""
    logging.warning("🚨 [Emergency Cleanup] 用户触发了应急全量显存回收！")
    if hasattr(ModelFactory, "destroy_all_models_cls"):
        ModelFactory.destroy_all_models_cls()
    else:
        ModelFactory.destroy_vlm_model()
        ModelFactory.destroy_llm_model()
        if hasattr(ModelFactory, "destroy_embedding_model"):
            ModelFactory.destroy_embedding_model()
        force_gc_cleanup()
    return get_gpu_memory_status()

# ==========================================
# 阶段 1-3：运行流水线解析与质检
# ==========================================
def run_parsing_and_validation(pdf_file, score_threshold):
    start_time = time.time()
    
    btn_running = gr.update(value="⏳ 拼命解析中，请勿刷新页面...", variant="secondary", interactive=False)
    btn_keep_disabled = gr.update(value="⏳ 请等待确认或取消写入", variant="secondary", interactive=False)
    btn_write_disabled = gr.update(visible=False, interactive=False)
    btn_cancel_disabled = gr.update(visible=False, interactive=False)
    
    def status(msg):
        return f"⏳ {msg} | 已耗时: {time.time() - start_time:.1f}s"

    if not pdf_file:
        btn_reset = gr.update(value="🚀 启动 1-3 阶段解析质检", variant="primary", interactive=True)
        yield "❌ 请先输入或上传 PDF 文件！", "", "", get_gpu_memory_status(), "🔴 错误：未提供文件", btn_reset, btn_write_disabled, btn_cancel_disabled, ""
        return
    
    pdf_path = pdf_file.name if hasattr(pdf_file, "name") else str(pdf_file)
    output_root = DEFAULT_CONFIG["output_root"]
    logs = []

    def make_log(msg):
        logging.info(msg)
        logs.append(msg)
        return "\n".join(logs)

    markdown_output = None
    is_quality_passed = False
    target_json = ""
    
    # ----------------------------------------------------
    # 阶段一：VLM 提取 Markdown
    # ----------------------------------------------------
    current_log = make_log("🚀 [阶段一] 开始运行 VLM 提取 PDF 结构及内容...")
    yield current_log, "", "", get_gpu_memory_status(), status("阶段一：VLM 提取中..."), btn_running, btn_write_disabled, btn_cancel_disabled, ""

    try:
        prompt_hub_path = DEFAULT_CONFIG["prompts_hub_path"]
        vlm_model_name = DEFAULT_CONFIG["vlm_model_name"]

        if not os.path.exists(prompt_hub_path):
            raise FileNotFoundError(f"配置文件路径不存在，请检查配置: {prompt_hub_path}")

        logging.info(f"🔑 阶段一正在使用的 Prompt Path: {prompt_hub_path}")
        logging.info(f"🤖 阶段一正在使用的 VLM Model: {vlm_model_name}")

        # 1. 实例化处理器
        processor = MarkdownProcessor(prompt_hub_path=prompt_hub_path)
        
        # 2. 💡 显式将 pdf_path, prompt_path, vlm 三个参数全部传进去
        markdown_output = processor.main(
            pdf_path=pdf_path, 
            prompt_path=prompt_hub_path, 
            vlm=vlm_model_name
        )

        current_log = make_log("✅ [阶段一] Markdown 原始数据提取成功！")
        yield current_log, markdown_output or "", "", get_gpu_memory_status(), status("阶段一：提取完毕，释放资源..."), btn_running, btn_write_disabled, btn_cancel_disabled, ""

    except Exception as e:
        current_log = make_log(f"❌ [阶段一] 发生异常: {e}")
        yield current_log, "", "", get_gpu_memory_status(), status("阶段一：发生异常！"), btn_running, btn_write_disabled, btn_cancel_disabled, ""
    finally:
        current_log = make_log("🧹 [阶段一] 正在呼叫工厂物理销毁 VLM 模型并释放显存...")
        if 'processor' in locals():
            del processor
        ModelFactory.destroy_vlm_model()
        gpu_stat = get_gpu_memory_status()
        current_log = make_log(f"💾 [阶段一] 显存已完全回收。{gpu_stat}")
        yield current_log, markdown_output or "", "", gpu_stat, status("阶段一：完成"), btn_running, btn_write_disabled, btn_cancel_disabled, ""

    # ----------------------------------------------------
    # 阶段二：后处理与 JSON 转换
    # ----------------------------------------------------
    current_log = make_log("⚙️ [阶段二] 正在执行深度语义提取与 JSON 转换...")
    yield current_log, markdown_output, "", get_gpu_memory_status(), status("阶段二：转换 JSON 中..."), btn_running, btn_write_disabled, btn_cancel_disabled, ""

    doc_processor = None
    try:
        doc_config = FineBIDocConfig(
            namespace_seed=DEFAULT_CONFIG["namespace_seed"],
            image_url_prefix=DEFAULT_CONFIG["img_prefix"],
            pdf_url_prefix=DEFAULT_CONFIG["pdf_prefix"],
            cuda_device="0",  
            yaml_rules_path=DEFAULT_CONFIG["patterns_path"],
            yaml_prompts_path=DEFAULT_CONFIG["prompts_hub_path"]
        )
        doc_processor = FineBIDocProcessor(config=doc_config)
        doc_processor.process_pdf(
            pdf_path=pdf_path,
            md_input=markdown_output,
            output_dir=output_root
        )
        json_filename = os.path.basename(pdf_path).replace('.pdf', '.json')
        target_json = os.path.join(output_root, json_filename)
        current_log = make_log(f"🏁 [阶段二] JSON 转换完成，目标文件保存路径: {target_json}")
    except Exception as e:
        current_log = make_log(f"❌ [阶段二] 发生异常: {e}")
    finally:
        current_log = make_log("🧹 [阶段二] 物理销毁阶段二 LLM 模型...")
        if "doc_processor" in locals() and doc_processor is not None:
            del doc_processor
        ModelFactory.destroy_llm_model()
        gpu_stat = get_gpu_memory_status()
        current_log = make_log(f"💾 [阶段二] 显存已完全回收。{gpu_stat}")
        yield current_log, markdown_output or "", "", gpu_stat, status("阶段二：完成"), btn_running, btn_write_disabled, btn_cancel_disabled, target_json

    # ----------------------------------------------------
    # 阶段三：数据质量检验
    # ----------------------------------------------------
    current_log = make_log("🔍 [阶段三] 启动数据质量检验引擎...")
    yield current_log, markdown_output, "", get_gpu_memory_status(), status("阶段三：模型质量打分中..."), btn_running, btn_write_disabled, btn_cancel_disabled, target_json

    val_processor = None
    try:
        if not os.path.exists(target_json):
            current_log = make_log(f"❌ [阶段三] 找不到校验目标文件: {target_json}")
        else:
            val_processor = ValidationProcessor(
                prompt_hub_path=DEFAULT_CONFIG["prompts_hub_path"], 
                cuda_device="0"
            )
            validator = RAGDataValidator(
                processor=val_processor, 
                score_threshold=float(score_threshold), 
                llm_sample_size=25
            )
            current_log = make_log(f"🤖 正在使用评估模型 [{DEFAULT_CONFIG['llm_model_name']}] 进行评分...")
            yield current_log, markdown_output, "", get_gpu_memory_status(), status("阶段三：检验计算中..."), btn_running, btn_write_disabled, btn_cancel_disabled, target_json

            is_ok, score, report = validator.validate_json_file(
                file_path=target_json,
                llm_model=DEFAULT_CONFIG["llm_model_name"]
            )
            current_log = make_log(f"📊 [阶段三] 质量检验完成 | 放行状态: {is_ok} | 得分: {score}")
            if score >= float(score_threshold):
                is_quality_passed = True
    except Exception as e:
        current_log = make_log(f"❌ [阶段三] 发生异常: {e}")
    finally:
        current_log = make_log("🧹 [阶段三] 物理销毁阶段三评估模型并释放显存...")
        if "val_processor" in locals(): 
            del val_processor
        ModelFactory.destroy_llm_model()
        gpu_stat = get_gpu_memory_status()
        current_log = make_log(f"💾 [阶段三] 显存已完全回收。{gpu_stat}")

    # 格式化 JSON 预览
    json_preview = ""
    if os.path.exists(target_json):
        with open(target_json, 'r', encoding='utf-8') as f:
            content = json.load(f)
            if isinstance(content, list):
                preview_data = content[:2]
            elif isinstance(content, dict):
                if "chunks" in content and isinstance(content["chunks"], list):
                    preview_data = {**{k: v for k, v in content.items() if k != "chunks"}, "chunks": content["chunks"][:2]}
                else:
                    preview_data = {k: content[k] for k in list(content.keys())[:5]}
            else:
                preview_data = content
            json_preview = json.dumps(preview_data, ensure_ascii=False, indent=2)

    if is_quality_passed:
        current_log = make_log("🟢 [阶段三] 质量校验通过！请选择点击【确认写入】或【取消写入】。")
        btn_write_enable = gr.update(value="📥 确认写入向量数据库 (Milvus)", variant="primary", visible=True, interactive=True)
    else:
        current_log = make_log("⚠️ [阶段三] 质量校验未达标！仍可手动强行点击写入，或选择取消。")
        btn_write_enable = gr.update(value="⚠️ 强制写入向量数据库 (分值未达标)", variant="stop", visible=True, interactive=True)

    btn_cancel_enable = gr.update(value="❌ 取消写入并清空显存", variant="secondary", visible=True, interactive=True)

    yield current_log, markdown_output or "", json_preview, get_gpu_memory_status(), f"⏸️ 1-3 阶段执行完毕，等待写入判定 (耗时: {time.time() - start_time:.1f}s)", btn_keep_disabled, btn_write_enable, btn_cancel_enable, target_json

# ==========================================
# 阶段 4：确认写入 Milvus
# ==========================================
def write_to_milvus_action(target_json, current_logs):
    logs = [current_logs] if current_logs else []
    def make_log(msg):
        logging.info(msg)
        logs.append(msg)
        return "\n".join(logs)

    btn_write_running = gr.update(value="⏳ 向 Milvus 写入数据中...", interactive=False)
    btn_write_finished = gr.update(value="✅ 已成功入库", interactive=False, visible=True)
    btn_cancel_hidden = gr.update(visible=False, interactive=False)
    btn_start_unlocked = gr.update(value="🚀 启动 1-3 阶段解析质检", variant="primary", interactive=True)

    if not target_json or not os.path.exists(target_json):
        updated_log = make_log("❌ [阶段四] 找不到有效的 JSON 文件路径，无法写入！")
        yield updated_log, get_gpu_memory_status(), "🔴 写入失败", btn_start_unlocked, btn_write_finished, btn_cancel_hidden
        return

    updated_log = make_log(f"🚀 [阶段四] 手动触发写入，正在解析: {target_json}")
    yield updated_log, get_gpu_memory_status(), "阶段四：Milvus 入库中...", gr.update(interactive=False), btn_write_running, btn_cancel_hidden

    uploader = None
    try:
        uploader = FineBIMilvusUploader(
            milvus_host=DEFAULT_CONFIG["milvus_host"],
            collection_name=DEFAULT_CONFIG["collection_name"],
            cuda_device="0"
        )
        updated_log = make_log("⬆️ 正在向 Milvus 写入数据与向量...")
        uploader.upload_json_file(target_json)
        updated_log = make_log("🔎 正在执行写入后数据一致性审计...")
        uploader.audit_milvus_with_json(target_json)
        updated_log = make_log("🎉 [阶段四] Milvus 入库与审计全部顺利完成！")
    except Exception as e:
        updated_log = make_log(f"❌ [阶段四] 发生异常: {e}")
    finally:
        updated_log = make_log("🧹 [阶段四] 正在清理数据库连接与物理销毁 Embedding 模型...")
        if 'uploader' in locals() and uploader is not None:
            if hasattr(uploader, "close"):
                try:
                    uploader.close()
                except Exception:
                    pass
            del uploader

        if hasattr(ModelFactory, "destroy_embedding_model"):
            ModelFactory.destroy_embedding_model()
        ModelFactory.destroy_llm_model()

        gpu_stat = get_gpu_memory_status()
        updated_log = make_log(f"💾 [阶段四] 显存已彻底完全释放！{gpu_stat}")
        yield updated_log, gpu_stat, "✅ 全流程完整完成！", btn_start_unlocked, btn_write_finished, btn_cancel_hidden

# ==========================================
# 取消写入动作
# ==========================================
def cancel_write_action(current_logs):
    logs = [current_logs] if current_logs else []
    def make_log(msg):
        logging.info(msg)
        logs.append(msg)
        return "\n".join(logs)

    updated_log = make_log("🚫 用户点击【取消写入】，跳过阶段四直接物理清空所有显存模型...")
    
    logging.info("⚡ 正在执行系统退出前显存终极清零...")
    ModelFactory.destroy_all_models_cls()
    logging.info("✨ 显存已彻底清空，安全退出。")

    gpu_stat = emergency_force_cleanup()
    updated_log = make_log(f"✨ 显存归零操作执行完毕！{gpu_stat}")

    btn_start_unlocked = gr.update(value="🚀 启动 1-3 阶段解析质检", variant="primary", interactive=True)
    btn_write_hidden = gr.update(visible=False, interactive=False)
    btn_cancel_hidden = gr.update(visible=False, interactive=False)

    return updated_log, gpu_stat, "🔴 已取消写入 (显存已归零)", btn_start_unlocked, btn_write_hidden, btn_cancel_hidden

# ==========================================
# 系统配置保存与更新逻辑
# ==========================================
def save_system_config(
    output_root, prompts_hub_path, patterns_path, img_prefix, 
    pdf_prefix, namespace_seed, milvus_host, collection_name, 
    vlm_model_name, llm_model_name, score_threshold
):
    global DEFAULT_CONFIG
    new_config = {
        "output_root": output_root,
        "prompts_hub_path": prompts_hub_path,
        "patterns_path": patterns_path,
        "img_prefix": img_prefix,
        "pdf_prefix": pdf_prefix,
        "namespace_seed": namespace_seed,
        "milvus_host": milvus_host,
        "collection_name": collection_name,
        "vlm_model_name": vlm_model_name,
        "llm_model_name": llm_model_name,
        "score_threshold": float(score_threshold)
    }
    
    DEFAULT_CONFIG.update(new_config)
    os.makedirs(DEFAULT_CONFIG["output_root"], exist_ok=True)
    
    try:
        with open(CONFIG_FILE_PATH, 'w', encoding='utf-8') as f:
            json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=4)
        return "✅ 系统基础配置已成功保存并同步生效！", DEFAULT_CONFIG
    except Exception as e:
        return f"❌ 保存配置文件失败: {e}", DEFAULT_CONFIG

# ==========================================
# Gradio 管理后台界面构建
# ==========================================
def build_admin_ui():
    with gr.Blocks(title="FineBI 知识库后台管理系统", theme=gr.themes.Soft()) as demo:
        target_json_state = gr.State("")

        gr.Markdown("# 🛡️ FineBI 知识库离线清洗与向量入库系统 (Admin Portal)")
        
        with gr.Row():
            gpu_status_box = gr.Textbox(
                label="🖥️ GPU 显存实时监控", 
                value=get_gpu_memory_status(), 
                interactive=False, 
                scale=3
            )
            refresh_gpu_btn = gr.Button("🔄 刷新显存", scale=1)
            force_cleanup_btn = gr.Button("🔥 强制全量显存回收 (Emergency Cleanup)", variant="stop", scale=1)

        refresh_gpu_btn.click(fn=get_gpu_memory_status, outputs=[gpu_status_box])
        force_cleanup_btn.click(fn=emergency_force_cleanup, outputs=[gpu_status_box])
            
        timer = gr.Timer(5)
        timer.tick(fn=get_gpu_memory_status, inputs=None, outputs=gpu_status_box)

        with gr.Tabs():
            with gr.Tab("🚀 全流程解析与向量入库"):
                gr.Markdown("上传/输入 PDF 文件，运行 **VLM提取 -> JSON转换 -> 轻量LLM质检**。质检完成后，可手动选择**确认写入**或**取消写入**。")
                
                with gr.Row():
                    pdf_input = gr.Textbox(label="输入 PDF 文件路径", placeholder="/workspace/.../xxx.pdf", scale=2)
                    with gr.Column(scale=1):
                        score_thresh_input = gr.Number(
                            label="质量校验门槛分 (Score)", 
                            value=DEFAULT_CONFIG["score_threshold"]
                        )
                        status_box = gr.Textbox(label="⏱️ 任务执行状态", value="🟢 待机中", interactive=False)
                        run_btn = gr.Button("🚀 启动 1-3 阶段解析质检", variant="primary")
                        
                        with gr.Row():
                            write_milvus_btn = gr.Button(
                                "📥 确认写入向量数据库 (Milvus)", 
                                variant="primary", 
                                visible=False, 
                                interactive=False
                            )
                            cancel_write_btn = gr.Button(
                                "❌ 取消写入并清空显存", 
                                variant="stop", 
                                visible=False, 
                                interactive=False
                            )
                
                pipeline_logs = gr.Textbox(label="全流程运行日志 (Realtime Logs)", lines=12, interactive=False)
                
                with gr.Accordion("中间产物预览 (Markdown & JSON)", open=False):
                    with gr.Row():
                        md_preview = gr.Markdown(label="阶段一 Markdown 产物预览")
                        json_preview = gr.Code(label="阶段二 JSON 切片预览 (前2条)", language="json")

                run_btn.click(
                    fn=run_parsing_and_validation,
                    inputs=[pdf_input, score_thresh_input],
                    outputs=[
                        pipeline_logs, md_preview, json_preview, 
                        gpu_status_box, status_box, run_btn, 
                        write_milvus_btn, cancel_write_btn, target_json_state
                    ],
                    show_progress="minimal"
                )

                write_milvus_btn.click(
                    fn=write_to_milvus_action,
                    inputs=[target_json_state, pipeline_logs],
                    outputs=[pipeline_logs, gpu_status_box, status_box, run_btn, write_milvus_btn, cancel_write_btn],
                    show_progress="minimal"
                )

                cancel_write_btn.click(
                    fn=cancel_write_action,
                    inputs=[pipeline_logs],
                    outputs=[pipeline_logs, gpu_status_box, status_box, run_btn, write_milvus_btn, cancel_write_btn],
                    show_progress="minimal"
                )

            with gr.Tab("⚙️ 系统基础配置"):
                gr.Markdown("### 🛠️ 动态修改并保存系统运行环境与网络配置")
                
                config_status_msg = gr.Markdown("")
                
                with gr.Row():
                    cfg_output_root = gr.Textbox(label="输出目录 (output_root)", value=DEFAULT_CONFIG["output_root"])
                    cfg_prompts_hub_path = gr.Textbox(label="Prompt Hub 路径", value=DEFAULT_CONFIG["prompts_hub_path"])
                
                with gr.Row():
                    cfg_patterns_path = gr.Textbox(label="Patterns Rules 路径", value=DEFAULT_CONFIG["patterns_path"])
                    cfg_namespace_seed = gr.Textbox(label="命名空间种子 (namespace_seed)", value=DEFAULT_CONFIG["namespace_seed"])

                with gr.Row():
                    cfg_img_prefix = gr.Textbox(label="图片 OSS 前缀", value=DEFAULT_CONFIG["img_prefix"])
                    cfg_pdf_prefix = gr.Textbox(label="PDF OSS 前缀", value=DEFAULT_CONFIG["pdf_prefix"])

                with gr.Row():
                    cfg_milvus_host = gr.Textbox(label="Milvus 主机地址", value=DEFAULT_CONFIG["milvus_host"])
                    cfg_collection_name = gr.Textbox(label="Milvus 集合名称", value=DEFAULT_CONFIG["collection_name"])

                with gr.Row():
                    cfg_vlm_model_name = gr.Textbox(label="VLM 模型名称", value=DEFAULT_CONFIG["vlm_model_name"])
                    cfg_llm_model_name = gr.Textbox(label="LLM 模型名称", value=DEFAULT_CONFIG["llm_model_name"])
                    cfg_score_threshold = gr.Number(label="默认质检门槛分", value=DEFAULT_CONFIG["score_threshold"])

                save_config_btn = gr.Button("💾 保存系统基础配置", variant="primary")
                
                config_json_preview = gr.JSON(label="当前全局生效配置视图", value=DEFAULT_CONFIG)

                save_config_btn.click(
                    fn=save_system_config,
                    inputs=[
                        cfg_output_root, cfg_prompts_hub_path, cfg_patterns_path, 
                        cfg_img_prefix, cfg_pdf_prefix, cfg_namespace_seed, 
                        cfg_milvus_host, cfg_collection_name, cfg_vlm_model_name, 
                        cfg_llm_model_name, cfg_score_threshold
                    ],
                    outputs=[config_status_msg, config_json_preview]
                )

    return demo


if __name__ == "__main__":
    def on_app_shutdown():
        """Gradio 进程关闭或 Ctrl+C 退出时的终极显存归零"""
        logging.info("⚡ 正在执行系统退出前显存终极清零...")
        ModelFactory.destroy_all_models_cls()
        logging.info("✨ 显存已彻底清空，安全退出。")

    atexit.register(on_app_shutdown)

    ui = build_admin_ui()
    ui.queue().launch(
        server_name="0.0.0.0", 
        server_port=7861,
        root_path="/admin"
    )