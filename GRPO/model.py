import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import MODEL_ID

def get_model_and_tokenizer():
    """
    Load Qwen3-8B in 4-bit QLoRA mode for training with GRPO.
    Designed for RTX 3090 (24GB).
    """
    # 4-bit NF4 quantization for QLoRA
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  # 3090: fp16
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # safer for generation

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,        # keep weights in fp16 on GPU
        device_map="auto",                # place on GPU automatically
        attn_implementation="sdpa",       # built-in SDPA (no flash-attn dependency)
    )

    # For training, disable cache
    model.config.use_cache = False

    return model, tokenizer
