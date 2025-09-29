import gc
import threading
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.configs.server_config import MODEL_DIR
from src.utils.file_helper import FileHelper
import torch


class ModelLoader:
    _instance = None
    _model = None
    _tokenizer = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ModelLoader, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_path, device_map="auto", torch_dtype=torch.bfloat16, attn_implementation="eager"):
        if not hasattr(self, '_initialized'):
            self.model_path = model_path
            self.torch_dtype = torch_dtype
            self.device_map = device_map
            self.attn_implementation = attn_implementation
            self._initialized = True
            
            # 自动检测并设置设备
            self._setup_device()

    def _setup_device(self):
        """自动设置设备配置"""
        if torch.cuda.is_available():
            # 有GPU可用
            self.device_map = "auto"
            # 确保使用GPU兼容的数据类型
            if self.torch_dtype == torch.float32:
                self.torch_dtype = torch.float16  # 在GPU上使用float16节省显存
            print(f"GPU detected, using device_map: {self.device_map}")
        else:
            # 只有CPU可用
            self.device_map = {"": "cpu"}
            print("Using CPU only")

    def load_model(self):
        if self._model is None:
            try:
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, 
                    device_map=self.device_map,
                    torch_dtype=self.torch_dtype,
                    attn_implementation=self.attn_implementation,
                    trust_remote_code=True  # 如果需要自定义模型代码
                )
                print(f"Model loaded on device: {self._model.device}")
            except Exception as e:
                print(f"Error loading model with GPU, falling back to CPU: {e}")
                # 如果GPU加载失败，回退到CPU
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_path, 
                    device_map={"": "cpu"},
                    torch_dtype=torch.float32,  # CPU上使用float32
                    attn_implementation=self.attn_implementation
                )
        return self._model

    def load_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True  # 如果需要自定义tokenizer
            )
            # tokenizer不需要device_map参数
        return self._tokenizer

    def unload_model(self):
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def unload_tokenizer(self):
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
            gc.collect()

    def switch_model(self, model_path):
        with self._lock:
            self.unload_model()
            self.unload_tokenizer()
            self.model_path = model_path
            self._setup_device()  # 重新设置设备
            self.load_model()
            self.load_tokenizer()

    def __enter__(self):
        self.load_model()
        self.load_tokenizer()
        return self._model, self._tokenizer

    def __exit__(self, exctype, excval, exctb):
        self.unload_model()
        self.unload_tokenizer()

# 初始化全局模型加载器
global_model_loader = ModelLoader(FileHelper.get_file_paths(MODEL_DIR)[0])
