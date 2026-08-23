# factory/model_factory.py
import os
import sys
import gc
import logging
import subprocess
import yaml
import torch
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s] - %(message)s")

try:
    from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
    from transformers import AutoProcessor, AutoTokenizer, AutoModelForCausalLM

# 条件导入多模态模型类
try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as Qwen3VLForConditionalGeneration
    except ImportError:
        from transformers import AutoModelForConditionalGeneration as Qwen3VLForConditionalGeneration


class ModelFactory:
    """
    模型与环境配置中心工厂（系统终极底座）
    全局接管：环境软链接、离线模型物理路径寻址、硬件算力分配、以及 VLM/LLM/Embedding 引擎的生命周期管理
    """
    
    # 🔒 静态类变量（全局唯一句柄）
    _LLM_MODEL = None
    _LLM_TOKENIZER = None
    
    _VL_MODEL = None
    _VL_PROCESSOR = None

    _EMB_MODEL = None
    _EMB_TOKENIZER = None

    _RERANKER_MODEL = None
    _RERANKER_TOKENIZER = None

    _instance = None

    def __new__(cls, *args, **kwargs):
        """真单例模式，阻断重复初始化"""
        if cls._instance is None:
            cls._instance = super(ModelFactory, cls).__new__(cls)
        return cls._instance

    def __init__(self, prompt_hub_path: str = "prompt_hub.yaml", cache_dir: str = "/workspace/hf-conda/hf_cache/hub"):

        if getattr(self, '_initialized', False):
            return
        
        if hasattr(self, '_initialized') and self._initialized:
            if prompt_hub_path and (not hasattr(self, 'prompts') or not self.prompts):
                self.prompt_hub_path = self._resolve_path(prompt_hub_path)
                self.prompts = self._load_prompts()
            return
        
        # 🌟 1. 将传入的路径统一转换为绝对路径
        self.prompt_hub_path = self._resolve_path(prompt_hub_path)
        self.cache_dir = cache_dir
        
        # 🌟 2. 使用绝对路径加载 Prompts
        self.prompts = self._load_prompts()

        # 建立全局软链接
        self._ensure_symlink("/workspace/hf-conda/hf_cache/datalab", "/root/.cache/datalab")
        self._ensure_symlink(self.cache_dir, "/root/.cache/huggingface")
        
        self._initialized = True

    def _resolve_path(self, path: str) -> str:
        """
        🌟 动态计算绝对路径：如果传入的是相对路径，则自动绑定到项目的 root/config/ 目录下
        """
        if os.path.isabs(path):
            return path
        # 获取 factory/ 目录的上一级目录（即项目 root 根目录）
        factory_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(factory_dir)
        
        # 提取文件名，直接绑定到 config 子目录下
        filename = os.path.basename(path)
        return os.path.join(project_root, "config", filename)

    def _load_prompts(self):
        """加载 Prompt Hub 文件"""
        # 🌟 3. 安全校验：防止绝对路径文件不存在
        if not os.path.exists(self.prompt_hub_path):
            logging.warning(f"⚠️ Prompt Hub 文件不存在，跳过加载: {self.prompt_hub_path}")
            return {}

        try:
            with open(self.prompt_hub_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            
            if not data or "prompts" not in data:
                logging.warning(f"⚠️ Prompt Hub 文件内容格式不正确: {self.prompt_hub_path}")
                return {}

            prompts_dict = {p["name"]: p["content"] for p in data.get("prompts", [])}
            logging.info(f"📂 Prompt Hub 资产加载成功，绝对路径: [{self.prompt_hub_path}]，可用 keys: {list(prompts_dict.keys())}")
            return prompts_dict
        except Exception as e:
            logging.error(f"❌ 加载 Prompt Hub 失败 ({self.prompt_hub_path}): {e}")
            return {}
        
    @classmethod
    def get_instance(cls, *args, **kwargs):
        """获取或创建 ModelFactory 全局单例句柄"""
        if not hasattr(cls, '_instance'):
            cls._instance = cls(*args, **kwargs)
        return cls._instance

    def resolve_model_path(self, short_name: str) -> str:
        """统一寻址算法 (Unified Path Resolver)"""
        if os.path.exists(short_name):
            return short_name
            
        safe_folder_name = f"models--{short_name.replace('/', '--')}"
        model_dir = os.path.join(self.cache_dir, safe_folder_name, "snapshots")
        
        if not os.path.isdir(model_dir):
            raise FileNotFoundError(f"❌ 工厂未在 {self.cache_dir} 寻寻找模型 [{short_name}] 的缓存文件夹。")
        
        snapshots = sorted(os.listdir(model_dir))
        if not snapshots:
            raise FileNotFoundError(f"❌ 模型 [{short_name}] 的 snapshots 目录为空。")
        
        real_path = os.path.join(model_dir, snapshots[-1])
        logging.info(f"🎯 工厂自动寻址成功 -> [{short_name}] 物理路径: {real_path}")
        return real_path

    def _ensure_symlink(self, source: str, target: str):
        if not os.path.islink(target):
            logging.info(f"🔧 构建全局软链接: {target} -> {source}")
            os.makedirs(os.path.dirname(target), exist_ok=True)
            if os.path.exists(target) and not os.path.islink(target):
                subprocess.run(f"rm -rf {target}", shell=True)
            os.symlink(source, target)

    @staticmethod
    def setup_cuda_device(cuda_device: str = "0") -> torch.device:
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f"🖥️ 硬件环境已绑定设备: {device}")
        return device

    # =====================================================================
    # 🛠️ 通用底层工具：极限原地解构模型 (Hard Cleansing Mechanism)
    # =====================================================================
    @classmethod
    def _hard_destroy_module(cls, model_obj):
        """原地拆解模型 Parameter/Buffer/Hooks，物理斩断 CUDA 显存强引用"""
        if model_obj is None:
            return
        
        try:
            # 1. 尝试解绑 Accelerate 挂载的 Hooks
            try:
                from accelerate.hooks import remove_hook_from_module
                remove_hook_from_module(model_obj, recurse=True)
            except Exception:
                pass

            # 2. 逐层将权重 Tensor 的内存置空 (0 字节)
            if hasattr(model_obj, "modules"):
                for module in model_obj.modules():
                    for param in list(module._parameters.keys()):
                        p = module._parameters[param]
                        if p is not None:
                            p.data = torch.empty(0, device=p.device)
                            module._parameters[param] = None
                    for buf in list(module._buffers.keys()):
                        b = module._buffers[buf]
                        if b is not None:
                            b.data = torch.empty(0, device=b.device)
                            module._buffers[buf] = None
                    for hook_dict in ('_backward_hooks', '_forward_hooks', '_forward_pre_hooks'):
                        if hasattr(module, hook_dict):
                            getattr(module, hook_dict).clear()
        except Exception as e:
            logging.warning(f"⚠️ 物理解构张量时发生非致命异常: {e}")

    @classmethod
    def _trigger_system_gc(cls):
        """系统级多层垃圾回收与 CUDA 缓存彻底清空"""
        gc.collect()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # =====================================================================
    # 🌌 核心引擎 1：纯文本 LLM 工厂驱动
    # =====================================================================
    def get_llm_model(self, llm_short_name: str = "Qwen/Qwen3-32B"):
        is_healthy = (
            ModelFactory._LLM_MODEL is not None 
            and hasattr(ModelFactory._LLM_MODEL, "generate")
        )

        if not is_healthy:
            model_dir = self.resolve_model_path(llm_short_name)
            logging.info(f"🚀 [Offline Load] 冷启动加载纯文本大模型: {model_dir}")
            
            ModelFactory._LLM_TOKENIZER = AutoTokenizer.from_pretrained(
                model_dir, 
                local_files_only=True, 
                trust_remote_code=True
            )
            ModelFactory._LLM_MODEL = AutoModelForCausalLM.from_pretrained(
                model_dir,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                attn_implementation="flash_attention_2",
                use_cache=True,
                local_files_only=True,
                trust_remote_code=True
            )
            
            warmup_prompt = "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant"
            inputs = ModelFactory._LLM_TOKENIZER(warmup_prompt, return_tensors="pt").to(ModelFactory._LLM_MODEL.device)
            
            logging.info("💡 正在执行纯文本大模型静态显存预热...")
            with torch.no_grad():
                _ = ModelFactory._LLM_MODEL.generate(**inputs, max_new_tokens=5)
            logging.info("✅ LLM 预热成功。")
            
            del inputs
            self._trigger_system_gc()
        else:
            logging.info("🟢 复用已存在的 LLM 实例。")
                
        return ModelFactory._LLM_MODEL, ModelFactory._LLM_TOKENIZER


    # =====================================================================
    # 🌌 核心引擎 2：多模态 VLM 工厂驱动
    # =====================================================================
    def get_vlm_model(self, vlm_short_name: str = 'Qwen/Qwen3-VL-32B-Instruct'):
        global Qwen3VLForConditionalGeneration
        if Qwen3VLForConditionalGeneration is None:
            raise ImportError("❌ 未找到对应的 Qwen VL 模型类。")

        is_old_model_alive = False
        if ModelFactory._VL_MODEL is not None:
            try:
                if next(ModelFactory._VL_MODEL.parameters()).numel() > 0:
                    is_old_model_alive = True
            except Exception:
                is_old_model_alive = False

        if is_old_model_alive:
            logging.info("♻️ 激活 VLM 绿色复用通道。")
            return ModelFactory._VL_MODEL, ModelFactory._VL_PROCESSOR

        if torch.cuda.is_available():
            device_idx = torch.cuda.current_device()
            free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
            used_bytes = total_bytes - free_bytes
            gpu_usage_ratio = used_bytes / total_bytes

            if gpu_usage_ratio > 0.70:
                self._trigger_system_gc()
                free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
                if ((total_bytes - free_bytes) / total_bytes) > 0.70:
                    raise RuntimeError(f"❌ [安全熔断] 显存不足以加载 {vlm_short_name}。")

        model_dir = self.resolve_model_path(vlm_short_name)
        logging.info(f"🚀 冷启动加载多模态模型: {model_dir}")
        
        ModelFactory._VL_PROCESSOR = AutoProcessor.from_pretrained(model_dir, local_files_only=True, trust_remote_code=True)
        ModelFactory._VL_MODEL = Qwen3VLForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        )

        warmup_img = Image.new("RGB", (1, 1), (255, 255, 255))
        messages = [{"role": "user", "content": [{"type": "image", "image": warmup_img}, {"type": "text", "text": "Hi"}]}]
        text = ModelFactory._VL_PROCESSOR.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = ModelFactory._VL_PROCESSOR(text=[text], images=[warmup_img], return_tensors="pt").to(ModelFactory._VL_MODEL.device)

        logging.info("💡 正在执行多模态模型预热...")
        with torch.no_grad():
            _ = ModelFactory._VL_MODEL.generate(**inputs, max_new_tokens=5)
        
        del inputs, text, messages, warmup_img
        self._trigger_system_gc()

        return ModelFactory._VL_MODEL, ModelFactory._VL_PROCESSOR

    @classmethod
    def destroy_llm_model(cls):
        """主动物理熔断销毁纯文本模型"""
        if cls._LLM_MODEL is not None:
            logging.info("🧹 正在主动物理清空纯文本大模型显存...")
            cls._hard_destroy_module(cls._LLM_MODEL)
            
            del cls._LLM_MODEL
            del cls._LLM_TOKENIZER
            cls._LLM_MODEL = None
            cls._LLM_TOKENIZER = None
            
            cls._trigger_system_gc()
            logging.info("✅ 纯文本大模型显存物理清理完毕。")

    @classmethod
    def destroy_vlm_model(cls):
        """物理销毁多模态模型"""
        if cls._VL_MODEL is not None:
            logging.info("🧹 正在物理拆解多模态大模型张量...")
            cls._hard_destroy_module(cls._VL_MODEL)

            del cls._VL_MODEL
            del cls._VL_PROCESSOR
            cls._VL_MODEL = None
            cls._VL_PROCESSOR = None

            cls._trigger_system_gc()
            logging.info("✅ 多模态模型显存清理完毕。")

    # =====================================================================
    # 🌌 核心引擎 3：新增 Embedding 模型管理
    # =====================================================================
    @classmethod
    def destroy_embedding_model(cls):
        """销毁 Embedding 向量模型"""
        if cls._EMB_MODEL is not None:
            logging.info("🧹 正在物理销毁 Embedding 模型...")
            cls._hard_destroy_module(cls._EMB_MODEL)
            
            del cls._EMB_MODEL
            del cls._EMB_TOKENIZER
            cls._EMB_MODEL = None
            cls._EMB_TOKENIZER = None
            
            cls._trigger_system_gc()
            logging.info("✅ Embedding 模型清理完毕。")

    # =====================================================================
    # 🌌 核心引擎 4：新增 reranker 模型管理
    # =====================================================================
    @classmethod
    def destroy_reranker_model(cls):
        """物理销毁 Reranker 重排模型 (Destroy Reranker)"""
        if cls._RERANKER_MODEL is not None:
            logging.info("🧹 正在物理销毁 Reranker 模型...")
            cls._hard_destroy_module(cls._RERANKER_MODEL)
            
            del cls._RERANKER_MODEL
            if hasattr(cls, "_RERANKER_TOKENIZER") and cls._RERANKER_TOKENIZER is not None:
                del cls._RERANKER_TOKENIZER
            cls._RERANKER_MODEL = None
            cls._RERANKER_TOKENIZER = None
            
            cls._trigger_system_gc()
            logging.info("✅ Reranker 模型显存物理清理完毕。")

    # =====================================================================
    # 🚨 终极核武器：一键物理清空工厂所有静态模型
    # =====================================================================
    @classmethod
    def destroy_all_models_cls(cls):
        """类级别无视状态强制清空一切挂载模型"""
        logging.info("🔥 触发工厂级终极全量显存回收...")
        cls.destroy_vlm_model()
        cls.destroy_llm_model()
        cls.destroy_embedding_model()
        cls.destroy_reranker_model()
        cls._trigger_system_gc()
        logging.info("✨ 工厂所有静态大模型已彻底卸载！")