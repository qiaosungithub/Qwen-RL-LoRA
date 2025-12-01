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
    format_data,
    create_reward_funcs,
    get_torch_dtype,
    deep_merge,
    BASE_CONFIG,
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
        attn_implementation=model_config["attn_implementation"],
        device_map=None,
    ).to("cuda")

    # Dataset
    dataset = load_dataset(
        dataset_config["name"],
        dataset_config["config"],
        split=dataset_config["split"]
    )
    dataset = dataset.map(format_data)

    # Training args
    training_args = GRPOConfig(
        output_dir=training_config["output_dir"],
        logging_steps=training_config["logging_steps"],
        per_device_train_batch_size=training_config["per_device_train_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        num_generations=training_config["num_generations"],
        max_prompt_length=training_config["max_prompt_length"],
        max_completion_length=training_config["max_completion_length"],
        learning_rate=training_config["learning_rate"],
        report_to="wandb" if config["wandb"]["enabled"] else "none",
        fp16=training_config["fp16"],
        bf16=training_config["bf16"],
        max_steps=training_config["max_steps"],
        run_name=f"grpo-{Path(training_config['output_dir']).name}",
        temperature=training_config["temperature"],
        top_p=training_config["top_p"],
    )

    peft_config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        target_modules=lora_config["target_modules"],
        task_type=lora_config["task_type"],
    )

    reward_funcs = create_reward_funcs(
        format_reward=reward_config["format_reward"],
        correctness_reward=reward_config["correctness_reward"],
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

# GRPO-specific overrides
_GRPO_OVERRIDES = {
    "method": "grpo",
    "training": {
        "output_dir": "outputs/qwen-grpo-gsm8k",
        "per_device_train_batch_size": 4,
        "num_generations": 16,
    },
}

DEFAULT_CONFIG = deep_merge(BASE_CONFIG, _GRPO_OVERRIDES)


def main():
    train(DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
