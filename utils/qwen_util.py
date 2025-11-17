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