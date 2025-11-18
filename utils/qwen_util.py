import torch

def enable_kv_cache(model):
    """启用模型的KV cache（用于生成）"""
    # 对于标准的 Transformers 模型
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = True
    
    # 对于 PEFT 包装的模型，需要修改底层模型
    if hasattr(model, 'base_model'):
        if hasattr(model.base_model.config, 'use_cache'):
            model.base_model.config.use_cache = True
    
    # 对于某些模型可能需要显式设置
    for module in model.modules():
        if hasattr(module, 'use_cache'):
            module.use_cache = True


def disable_kv_cache(model):
    """禁用模型的KV cache（用于训练）"""
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = False
    
    if hasattr(model, 'base_model'):
        if hasattr(model.base_model.config, 'use_cache'):
            model.base_model.config.use_cache = False
    
    for module in model.modules():
        if hasattr(module, 'use_cache'):
            module.use_cache = False

@torch.no_grad()
def generate_multiple_times(model, generate_bs, input_ids, attention_mask, generate_kwargs):
    B = input_ids.shape[0]
    assert B % generate_bs == 0, "Batch size must be divisible by generate_bs"
    num_chunks = B // generate_bs
    generated_outputs = []
    for i in range(num_chunks):
        start_idx = i * generate_bs
        end_idx = (i + 1) * generate_bs
        input_ids_chunk = input_ids[start_idx:end_idx]
        attention_mask_chunk = attention_mask[start_idx:end_idx]
        outputs = model.generate(input_ids=input_ids_chunk, attention_mask=attention_mask_chunk, **generate_kwargs)
        generated_outputs.append(outputs)
    return torch.cat(generated_outputs, dim=0)

def model_forward_multiple_times(model, forward_bs, input_ids, attention_mask):
    B = input_ids.shape[0]
    if forward_bs is None: forward_bs = B
    assert B % forward_bs == 0, "Batch size must be divisible by forward_bs"
    num_chunks = B // forward_bs
    outputs = []
    for i in range(num_chunks):
        start_idx = i * forward_bs
        end_idx = (i + 1) * forward_bs
        input_ids_chunk = input_ids[start_idx:end_idx]
        attention_mask_chunk = attention_mask[start_idx:end_idx]
        output_chunk = model(input_ids=input_ids_chunk, attention_mask=attention_mask_chunk)
        outputs.append(output_chunk)
    return torch.cat(outputs, dim=0)
