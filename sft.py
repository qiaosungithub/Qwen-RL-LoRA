import re
import os
import yaml
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

BASE_CONFIG = {
    "model": {
        # "name": "Qwen/Qwen3-8B",
        "name": "Qwen/Qwen3-0.6B",
        "torch_dtype": "bfloat16",
        "attn_implementation": "flash_attention_2",
        "cache_dir": ".cache/models",
        "device": "cuda",
    },
    "lora": {
        "r": 32,
        "lora_alpha": 64,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "task_type": "CAUSAL_LM",
    },
    "training": {
        "max_steps": 500,
        "learning_rate": 1e-5,
        "gradient_accumulation_steps": 2,
        "logging_steps": 10,
        "bf16": True,
        "fp16": False,
        "max_prompt_length": 512,
        "max_completion_length": 1024,
        "do_sample": True,
        "temperature": 0.8,
        "top_p": 0.95,
    },
    "dataset": {
        "name": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "cache_dir": ".cache/datasets",
    },
    "reward": {
        "format_reward": 0.5,
        "correctness_reward": 1.5,
    },
    "wandb": {
        "project": "rl-qwen-gsm8k",
        "enabled": True,
    },
}

# ============================================================================
# System Prompt
# ============================================================================

SYSTEM_PROMPT = """Solve the math problem step by step. Put your reasoning inside <think>...</think> tags and your final numerical answer inside <answer>...</answer> tags."""


# ============================================================================
# Data Formatting
# ============================================================================

def format_data(example):
    """Format GSM8K example for training."""
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ],
        "answer": example["answer"],
    }


def extract_xml_answer(text: str) -> str:
    """Extract answer from XML tags."""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def check_format(text: str) -> bool:
    """Check if response follows the expected format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    return bool(re.search(pattern, text, re.DOTALL))


# ============================================================================
# Reward Functions
# ============================================================================

def create_reward_funcs(format_reward: float, correctness_reward: float):
    """Create reward functions with configurable reward values."""

    def format_reward_func(completions, **kwargs):
        pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
        responses = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, r, re.DOTALL) for r in responses]
        return [format_reward if match else 0.0 for match in matches]

    def correctness_reward_func(prompts, completions, answer, **kwargs):
        responses = [completion[0]["content"] for completion in completions]
        extracted_answers = [extract_xml_answer(r) for r in responses]

        rewards = []
        for extracted, correct in zip(extracted_answers, answer):
            correct_val = correct.split("#### ")[-1].strip()
            if extracted == correct_val:
                rewards.append(correctness_reward)
            else:
                rewards.append(0.0)
        return rewards

    return [format_reward_func, correctness_reward_func]


def compute_single_reward(response: str, correct_answer: str,
                          format_reward: float, correctness_reward: float) -> float:
    """Compute reward for a single response (used by PPO)."""
    reward = 0.0

    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    if re.search(pattern, response, re.DOTALL):
        reward += format_reward

    extracted = extract_xml_answer(response)
    correct_val = correct_answer.split("#### ")[-1].strip()
    if extracted == correct_val:
        reward += correctness_reward

    return reward


# ============================================================================
# Config Utilities
# ============================================================================

def deep_merge(base: dict, override: dict) -> dict:
    """
    Deep merge two dictionaries. Override values take precedence.
    Nested dicts are merged recursively, other values are replaced.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str, overrides: dict = None) -> dict:
    """
    Load YAML config with inheritance support.

    If the config contains a 'base' key, it will first load the base config
    and then merge the current config on top of it.

    Example:
        # configs/grpo.yaml
        base: base.yaml
        method: grpo
        training:
          output_dir: outputs/grpo
    """
    import os

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Handle inheritance from base config
    if "base" in config:
        base_path = config.pop("base")
        # Resolve relative path from the config file's directory
        config_dir = os.path.dirname(config_path)
        base_full_path = os.path.join(config_dir, base_path)

        # Load base config (recursively, to support multi-level inheritance)
        base_config = load_config(base_full_path)

        # Merge: base config + current config
        config = deep_merge(base_config, config)

    # Apply CLI overrides last (highest priority)
    if overrides:
        for key, value in overrides.items():
            keys = key.split(".")
            d = config
            for k in keys[:-1]:
                d = d.setdefault(k, {})
            try:
                d[keys[-1]] = yaml.safe_load(value)
            except:
                d[keys[-1]] = value

    return config


def get_torch_dtype(dtype_str: str):
    """Convert string to torch dtype."""
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return mapping.get(dtype_str, torch.bfloat16)

# ============================================================================
# End of Helpers
# ============================================================================


def train_sft(config: dict) -> str:
    """
    Train using SFT on GSM8K dataset with LoRA adaptation.
    
    Args:
        config: Configuration dictionary with model, lora, training, dataset, reward, wandb settings.

    Returns:
        Path to the saved model.
    """
    model_config = config["model"]
    lora_config = config["lora"]
    training_config = config["training"]
    dataset_config = config["dataset"]
    
    if config["wandb"]["enabled"]:
        os.environ["WANDB_PROJECT"] = config["wandb"]["project"]
        wandb.login()
        
    # Add timestamp to output_dir
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"{training_config['output_dir']}-{timestamp}"

    # Model
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name"],
        cache_dir=model_config["cache_dir"],
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_config["name"],
        dtype=get_torch_dtype(model_config["torch_dtype"]),
        attn_implementation=model_config["attn_implementation"],
        cache_dir=model_config["cache_dir"],
        device_map="cuda:0",
    )
    
    # Dataset
    dataset = load_dataset(
        dataset_config["name"],
        dataset_config["config"],
        split=dataset_config["split"],
        cache_dir=dataset_config["cache_dir"],
    )
    dataset = dataset.map(format_data)
    
    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        max_steps=training_config["max_steps"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        num_train_epochs=training_config["num_train_epochs"],
        learning_rate=training_config["learning_rate"],
        logging_steps=training_config["logging_steps"],
        bf16=training_config["bf16"],
        fp16=training_config["fp16"],
        # save_strategy="steps",
        # save_steps=training_config.get("save_steps", 100),
        # eval_strategy="steps",
        # eval_steps=training_config.get("eval_steps", 100),
        report_to="wandb" if config["wandb"]["enabled"] else "none",
        disable_tqdm=False,
        run_name=f"sft-{Path(output_dir).name}",
        temperature=training_config["temperature"],
        top_p=training_config["top_p"],
    )
  
    peft_config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        target_modules=lora_config["target_modules"],
        task_type=lora_config["task_type"],
    )
    
    trainer = Trainer(
        model=get_peft_model(model, peft_config),
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    
    # Train
    print(f"\n{'='*60}")
    print(f"Training with SFT on GSM8K")
    print(f"Output: {output_dir}")
    print(f"{'='*60}\n")
    
    trainer.train()
    
    # Save
    final_path = f"{output_dir}-final"
    trainer.save_model(final_path)
    print(f"\nModel saved to {final_path}")
    
    return final_path

def main(config: dict) -> str:
    final_path = train_sft(config)
    return final_path
    
if __name__ == "__main__":
    _SFT_OVERRIDES = {
        "training": {
            "output_dir": "outputs/qwen-sft-gsm8k",
        }
    }
    CONFIG = deep_merge(BASE_CONFIG, _SFT_OVERRIDES)
    final_path = main(CONFIG)
    print(f"Training completed. Model saved at: {final_path}")