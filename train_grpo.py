"""
GRPO (Group Relative Policy Optimization) Training Script

Usage:
    python train_grpo.py                           # standalone with defaults
    python train_rl.py --config configs/grpo.yaml  # via unified script
"""

import torch
import os
import wandb
from pathlib import Path

from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM

from rl_utils import (
    SYSTEM_PROMPT,
    format_data,
    create_reward_funcs,
    get_torch_dtype,
)


def train(config: dict) -> str:
    """
    Train using GRPO.

    Args:
        config: Configuration dictionary with model, lora, training, dataset, reward, wandb settings.

    Returns:
        Path to the saved model.
    """
    model_config = config["model"]
    lora_config = config["lora"]
    training_config = config["training"]
    dataset_config = config["dataset"]
    reward_config = config["reward"]

    # Setup wandb
    if config["wandb"]["enabled"]:
        os.environ["WANDB_PROJECT"] = config["wandb"]["project"]
        wandb.login()

    # Model
    tokenizer = AutoTokenizer.from_pretrained(model_config["name"])
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_config["name"],
        torch_dtype=get_torch_dtype(model_config["torch_dtype"]),
        attn_implementation=model_config.get("attn_implementation", "flash_attention_2"),
        device_map=None,
    ).to("cuda")

    # Dataset
    dataset = load_dataset(
        dataset_config["name"],
        dataset_config.get("config", "main"),
        split=dataset_config["split"]
    )
    dataset = dataset.map(format_data)

    # Training args
    training_args = GRPOConfig(
        output_dir=training_config["output_dir"],
        logging_steps=training_config.get("logging_steps", 1),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 8),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
        num_generations=training_config.get("num_generations", 8),
        max_prompt_length=training_config.get("max_prompt_length", 512),
        max_completion_length=training_config.get("max_completion_length", 512),
        learning_rate=training_config["learning_rate"],
        report_to="wandb" if config["wandb"]["enabled"] else "none",
        fp16=training_config.get("fp16", False),
        bf16=training_config.get("bf16", True),
        max_steps=training_config["max_steps"],
        run_name=f"grpo-{Path(training_config['output_dir']).name}",
    )

    peft_config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        target_modules=lora_config["target_modules"],
        task_type=lora_config["task_type"],
    )

    reward_funcs = create_reward_funcs(
        format_reward=reward_config.get("format_reward", 0.5),
        correctness_reward=reward_config.get("correctness_reward", 2.0),
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # Train
    print(f"\n{'='*60}")
    print(f"Training with GRPO")
    print(f"Output: {training_config['output_dir']}")
    print(f"{'='*60}\n")

    trainer.train()

    # Save
    final_path = f"{training_config['output_dir']}-final"
    trainer.save_model(final_path)
    print(f"\nModel saved to {final_path}")

    return final_path


# ============================================================================
# Standalone execution with default config
# ============================================================================

DEFAULT_CONFIG = {
    "method": "grpo",
    "model": {
        "name": "Qwen/Qwen2.5-1.5B-Instruct",
        "torch_dtype": "bfloat16",
        "attn_implementation": "flash_attention_2",
    },
    "lora": {
        "r": 16,
        "lora_alpha": 32,
        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "task_type": "CAUSAL_LM",
    },
    "training": {
        "output_dir": "qwen-grpo-gsm8k",
        "max_steps": 200,
        "learning_rate": 5e-6,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "num_generations": 8,
        "max_prompt_length": 512,
        "max_completion_length": 512,
        "logging_steps": 1,
        "bf16": True,
        "fp16": False,
    },
    "dataset": {
        "name": "openai/gsm8k",
        "config": "main",
        "split": "train",
    },
    "reward": {
        "format_reward": 0.5,
        "correctness_reward": 2.0,
    },
    "wandb": {
        "project": "grpo-qwen-gsm8k",
        "enabled": True,
    },
}


def main():
    train(DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
