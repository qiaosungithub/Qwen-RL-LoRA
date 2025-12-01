"""
PPO (Proximal Policy Optimization) Training Script

Usage:
    python train_ppo.py                           # standalone with defaults
    python train_rl.py --config configs/ppo.yaml  # via unified script
"""

import torch
import os
import wandb

from datasets import load_dataset
from peft import LoraConfig
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from transformers import AutoTokenizer

from rl_utils import (
    format_data,
    compute_single_reward,
    get_torch_dtype,
)


def train(config: dict) -> str:
    """
    Train using PPO.

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

    # Model with value head
    tokenizer = AutoTokenizer.from_pretrained(model_config["name"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    peft_config = LoraConfig(
        r=lora_config["r"],
        lora_alpha=lora_config["lora_alpha"],
        target_modules=lora_config["target_modules"],
        task_type=lora_config["task_type"],
    )

    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_config["name"],
        torch_dtype=get_torch_dtype(model_config["torch_dtype"]),
        attn_implementation=model_config.get("attn_implementation", "flash_attention_2"),
        device_map=None,
        peft_config=peft_config,
    ).to("cuda")

    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
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

    # PPO config
    ppo_config = PPOConfig(
        output_dir=training_config["output_dir"],
        learning_rate=training_config["learning_rate"],
        batch_size=training_config.get("batch_size", 8),
        mini_batch_size=training_config.get("mini_batch_size", 4),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 1),
        ppo_epochs=training_config.get("ppo_epochs", 4),
        max_grad_norm=training_config.get("max_grad_norm", 0.5),
        target_kl=training_config.get("target_kl", 0.1),
        kl_penalty=training_config.get("kl_penalty", "kl"),
        log_with="wandb" if config["wandb"]["enabled"] else None,
    )

    trainer = PPOTrainer(
        config=ppo_config,
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
    )

    # Training params
    max_steps = training_config["max_steps"]
    max_new_tokens = training_config.get("max_new_tokens", 512)
    batch_size = training_config.get("batch_size", 8)
    do_sample = training_config.get("do_sample", True)
    top_p = training_config.get("top_p", 0.9)
    temperature = training_config.get("temperature", 0.7)
    logging_steps = training_config.get("logging_steps", 10)

    format_reward = reward_config.get("format_reward", 0.5)
    correctness_reward = reward_config.get("correctness_reward", 2.0)

    print(f"\n{'='*60}")
    print(f"Training with PPO")
    print(f"Output: {training_config['output_dir']}")
    print(f"{'='*60}\n")

    step = 0
    for epoch in range(10):
        for batch in dataset.iter(batch_size=batch_size):
            if step >= max_steps:
                break

            queries = []
            answers = []
            for i in range(len(batch["prompt"])):
                prompt = tokenizer.apply_chat_template(
                    batch["prompt"][i],
                    tokenize=False,
                    add_generation_prompt=True
                )
                queries.append(prompt)
                answers.append(batch["answer"][i])

            # Tokenize all queries as a batch
            query_encodings = tokenizer(
                queries,
                padding=True,
                return_tensors="pt",
            ).to("cuda")

            query_tensors = [
                query_encodings.input_ids[i][query_encodings.attention_mask[i].bool()]
                for i in range(len(queries))
            ]

            # Batched generation
            with torch.no_grad():
                outputs = trainer.generate(
                    query_encodings.input_ids,
                    attention_mask=query_encodings.attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    top_p=top_p,
                    temperature=temperature,
                    pad_token_id=tokenizer.pad_token_id,
                )

            # Extract response tensors (remove prompt tokens and trailing padding)
            response_tensors = []
            for i in range(len(queries)):
                prompt_len = query_encodings.attention_mask[i].sum().item()
                response = outputs[i, prompt_len:]
                # Remove trailing padding tokens
                non_pad_mask = response != tokenizer.pad_token_id
                if non_pad_mask.any():
                    last_non_pad = non_pad_mask.nonzero()[-1].item() + 1
                    response = response[:last_non_pad]
                response_tensors.append(response)

            responses = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]

            rewards = [
                torch.tensor(
                    compute_single_reward(resp, ans, format_reward, correctness_reward),
                    dtype=torch.float32
                ).to("cuda")
                for resp, ans in zip(responses, answers)
            ]

            stats = trainer.step(query_tensors, response_tensors, rewards)

            if step % logging_steps == 0:
                mean_reward = sum(r.item() for r in rewards) / len(rewards)
                print(f"Step {step}: mean_reward={mean_reward:.3f}")

            step += 1

        if step >= max_steps:
            break

    # Save
    final_path = f"{training_config['output_dir']}-final"
    trainer.save_pretrained(final_path)
    print(f"\nModel saved to {final_path}")

    return final_path


# ============================================================================
# Standalone execution with default config
# ============================================================================

DEFAULT_CONFIG = {
    "method": "ppo",
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
        "output_dir": "qwen-ppo-gsm8k",
        "max_steps": 200,
        "learning_rate": 5e-6,
        "batch_size": 8,
        "mini_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "ppo_epochs": 4,
        "max_grad_norm": 0.5,
        "target_kl": 0.1,
        "kl_penalty": "kl",
        "max_new_tokens": 512,
        "do_sample": True,
        "top_p": 0.9,
        "temperature": 0.7,
        "logging_steps": 10,
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
        "project": "ppo-qwen-gsm8k",
        "enabled": True,
    },
}


def main():
    train(DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
