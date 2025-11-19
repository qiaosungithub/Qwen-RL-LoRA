import torch
from vllm import LLM, SamplingParams

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
        
        # 释放之前的实例
        if self.llm is not None:
            del self.llm
            torch.cuda.empty_cache()
        
        self.llm = LLM(
            model=self.model_path,
            trust_remote_code=True,
            dtype="bfloat16",
            tensor_parallel_size=1,
            gpu_memory_utilization=0.5,  # 使用50%显存，另50%留给训练
            max_model_len=self.config.dataset.max_length + self.config.generation.max_new_tokens,
            download_dir=self.config.cache_dir,
        )
        
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


def merge_lora_for_vllm(model, tokenizer, save_path):
    """
    合并LoRA权重到base model并保存（vLLM需要完整模型）
    Args:
        model: PEFT模型
        tokenizer: tokenizer
        save_path: 保存路径
    """
    print(f"[Merge LoRA] Merging and saving to {save_path}")
    
    # 合并LoRA权重
    merged_model = model.merge_and_unload()
    
    # 保存合并后的模型
    merged_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    
    print(f"[Merge LoRA] Saved merged model")
    
    # 恢复LoRA
    # 注意：merge_and_unload()会返回base model，原model不受影响
    return merged_model