from trl import GRPOConfig
from peft import LoraConfig

MODEL_ID = "Qwen/Qwen3-8B"

SYSTEM_PROMPT = """
Respond to the user's math problem. 
You must format your output as follows:
<think>
{reasoning}
</think>
<answer>
{final_answer}
</answer>
"""

def get_grpo_config() -> GRPOConfig:
    """
    GRPOConfig tuned for Qwen3-8B + QLoRA on a 24GB RTX 3090.
    """
    return GRPOConfig(
        output_dir="qwen3-8b-grpo-gsm8k",
        logging_steps=1,

        # Memory-aware settings for 24GB + 8B + GRPO
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,   # effective batch size 8
        num_generations=4,               # can try 6 or 8 later

        max_prompt_length=384,
        max_completion_length=256,

        learning_rate=5e-6,

        # Disable logging if you don't want W&B / others
        report_to="wandb",

        # 3090: use fp16, not bf16
        fp16=True,
        bf16=False,

        max_steps=200,

        # Saves VRAM at the cost of some compute
        gradient_checkpointing=True,
    )

# LoRA / QLoRA config
PEFT_CONFIG = LoraConfig(
    r=16,
    lora_alpha=32,
    # Qwen-style transformer modules
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    task_type="CAUSAL_LM",
)