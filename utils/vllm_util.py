import torch, os, peft
from vllm import LLM, SamplingParams

from copy import deepcopy

class vLLMGenerator:
    """vLLM生成器封装类"""
    def __init__(self, model_path, config, device):
        self.model_path = model_path
        self.config = config
        self.device = device
        self.llm = None
        self.sampling_params = None
        self._init_vllm()
    
    def _init_vllm(self):
        """初始化vLLM"""
        print(f"[vLLM] Initializing with model: {self.model_path}")
        
        # 验证模型路径和必要文件
        if not os.path.exists(self.model_path):
            raise ValueError(f"Model path does not exist: {self.model_path}")
        
        config_path = os.path.join(self.model_path, "config.json")
        if not os.path.exists(config_path):
            raise ValueError(f"config.json not found in {self.model_path}")
        
        print(f"[vLLM] Model validation:")
        print(f"  - config.json: ✓")
        
        # 释放之前的实例
        if self.llm is not None:
            del self.llm
            torch.cuda.empty_cache()
        
        # 初始化vLLM参数
        vllm_kwargs = {
            "model": self.model_path,
            "trust_remote_code": True,
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.5,
            "max_model_len": self.config.dataset.max_length + self.config.generation.max_new_tokens,
            "download_dir": self.config.cache_dir,
        }
        
        try:
            self.llm = LLM(**vllm_kwargs)
        except Exception as e:
            print(f"[vLLM] Initialization failed: {str(e)}")
            print(f"[vLLM] Attempting fallback: loading base model only")
            # 如果失败，尝试只加载base model
            vllm_kwargs["enable_lora"] = False
            self.llm = LLM(**vllm_kwargs)
        
        self.sampling_params = SamplingParams(
            temperature=self.config.generation.temperature,
            top_p=self.config.generation.top_p,
            top_k=self.config.generation.top_k,
            max_tokens=self.config.generation.max_new_tokens,
            skip_special_tokens=True,
        )
        
        print(f"[vLLM] Initialization complete")
    
    def reload_model(self, new_model_path):
        """重新加载模型"""
        print(f"\n[vLLM] Reloading model from: {new_model_path}")
        self.model_path = new_model_path
        self._init_vllm()
    
    def generate(self, prompts):
        """
        批量生成
        Args:
            prompts: List[str]
        Returns:
            List[str]: 生成的文本
        """
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]
    
    def cleanup(self):
        """清理资源"""
        if self.llm is not None:
            del self.llm
            self.llm = None
            torch.cuda.empty_cache()
            print("[vLLM] Cleaned up")


def merge_lora_for_vllm(model: peft.PeftMixedModel, tokenizer, save_path):
    """
    合并LoRA权重到base model并保存（vLLM需要完整模型）
    注意：这个函数不会修改原始model，只是保存一个合并后的副本
    
    Args:
        model: PEFT模型（训练中的模型，不会被修改）
        tokenizer: tokenizer
        save_path: 保存路径
    
    Returns:
        None (不返回任何东西，避免误用)
    """
    print(f"[Merge LoRA] Merging and saving to {save_path}")
    
    # 重要：merge_and_unload()返回一个新模型，不影响原始model
    # 我们只用它来保存，然后立即丢弃

    # # print the sum of all trainable parameters before merging
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"[Merge LoRA] Trainable parameters before merging: {trainable_params}", flush=True)
    model_ = deepcopy(model)
    merged_model = model_.merge_and_unload()
    # # print the sum of all trainable parameters after merging
    # trainable_params_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"[Merge LoRA] Trainable parameters after merging: {trainable_params_after}", flush=True)
    
    # 保存合并后的完整模型
    merged_model.save_pretrained(
        save_path,
        safe_serialization=True,  # 使用safetensors格式
    )
    tokenizer.save_pretrained(save_path)
    
    # 立即删除merged_model以释放显存
    del merged_model, model_
    torch.cuda.empty_cache()
    
    print(f"[Merge LoRA] ✓ Merged model saved and cleaned up")
    
    # 验证必要文件
    required_files = ['config.json', 'tokenizer_config.json']
    missing = [f for f in required_files if not os.path.exists(os.path.join(save_path, f))]
    if missing:
        print(f"[Merge LoRA] Warning: Missing files: {missing}")
    else:
        print(f"[Merge LoRA] ✓ All required files present")
    