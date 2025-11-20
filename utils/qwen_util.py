import torch

def enable_kv_cache(model):
    """启用模型的KV cache（用于生成）"""
    # 对于标准的 Transformers 模型
    if hasattr(model.config, 'use_cache'):
        model.config.use_cache = True
        print(f"Enabled KV cache for model: {model.__class__.__name__}")
        return
    
    # 对于 PEFT 包装的模型，需要修改底层模型
    if hasattr(model, 'base_model'):
        if hasattr(model.base_model.config, 'use_cache'):
            model.base_model.config.use_cache = True
            print(f"Enabled KV cache for base model: {model.base_model.__class__.__name__}")
            return
    
    # 对于某些模型可能需要显式设置
    for module in model.modules():
        if hasattr(module, 'use_cache'):
            module.use_cache = True
            print(f"Enabled KV cache for module: {module.__class__.__name__}")
            return
    assert False


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
    raise DeprecationWarning("This function is deprecated. Use vLLM for generation instead.")
    B = input_ids.shape[0]
    assert B % generate_bs == 0, "Batch size must be divisible by generate_bs"
    num_chunks = B // generate_bs
    generated_outputs = []
    
    for i in range(num_chunks):
        print(f"Generating chunk {i+1}/{num_chunks}...")
        start_idx = i * generate_bs
        end_idx = (i + 1) * generate_bs
        input_ids_chunk = input_ids[start_idx:end_idx]
        attention_mask_chunk = attention_mask[start_idx:end_idx]
        outputs = model.generate(
            input_ids=input_ids_chunk, 
            attention_mask=attention_mask_chunk, 
            **generate_kwargs
        )
        generated_outputs.append(outputs)
    
    # 找到最大长度
    max_length = max(output.shape[1] for output in generated_outputs)
    
    # 获取 pad_token_id
    pad_token_id = model.config.pad_token_id
    if pad_token_id is None:
        # 如果没有设置 pad_token_id，使用 eos_token_id
        pad_token_id = model.config.eos_token_id
    
    # 对所有输出进行左填充（left padding）
    padded_outputs = []
    for output in generated_outputs:
        if output.shape[1] < max_length:
            padding = torch.full(
                (output.shape[0], max_length - output.shape[1]),
                pad_token_id,
                dtype=output.dtype,
                device=output.device
            )
            # 左填充：padding 在前，output 在后
            padded_output = torch.cat([padding, output], dim=1)
        else:
            padded_output = output
        padded_outputs.append(padded_output)
    
    return torch.cat(padded_outputs, dim=0)

def model_forward_multiple_times(model, forward_bs, input_ids, attention_mask):
    B = input_ids.shape[0]
    if forward_bs is None: 
        forward_bs = B
    assert B % forward_bs == 0, "Batch size must be divisible by forward_bs"
    num_chunks = B // forward_bs
    logits_list = []  # 改名以更清晰
    for i in range(num_chunks):
        start_idx = i * forward_bs
        end_idx = (i + 1) * forward_bs
        input_ids_chunk = input_ids[start_idx:end_idx]
        attention_mask_chunk = attention_mask[start_idx:end_idx]
        output_chunk = model(input_ids=input_ids_chunk, attention_mask=attention_mask_chunk)
        # 关键修复：提取 logits
        logits_list.append(output_chunk.logits)
        print(f'finished forward chunk {i+1}/{num_chunks}')
    return torch.cat(logits_list, dim=0)
