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
    deep_merge,
    BASE_CONFIG,
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
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name"],
        cache_dir=model_config["cache_dir"],
    )
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
        attn_implementation=model_config["attn_implementation"],
        cache_dir=model_config["cache_dir"],
        device_map=None,
        peft_config=peft_config,
    ).to("cuda")

    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        model_config["name"],
        torch_dtype=get_torch_dtype(model_config["torch_dtype"]),
        attn_implementation=model_config["attn_implementation"],
        cache_dir=model_config["cache_dir"],
        device_map=None,
    ).to("cuda")

    # Dataset
    dataset = load_dataset(
        dataset_config["name"],
        dataset_config["config"],
        split=dataset_config["split"],
        cache_dir=dataset_config["cache_dir"],
    )
    dataset = dataset.map(format_data)

    # PPO config
    ppo_config = PPOConfig(
        output_dir=training_config["output_dir"],
        learning_rate=training_config["learning_rate"],
        batch_size=training_config["batch_size"],
        mini_batch_size=training_config["mini_batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        ppo_epochs=training_config["ppo_epochs"],
        max_grad_norm=training_config["max_grad_norm"],
        target_kl=training_config["target_kl"],
        kl_penalty=training_config["kl_penalty"],
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
    max_new_tokens = training_config["max_completion_length"]
    batch_size = training_config["batch_size"]
    do_sample = training_config["do_sample"]
    top_p = training_config["top_p"]
    temperature = training_config["temperature"]
    logging_steps = training_config["logging_steps"]

    format_reward = reward_config["format_reward"]
    correctness_reward = reward_config["correctness_reward"]

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

# PPO-specific overrides
_PPO_OVERRIDES = {
    "method": "ppo",
    "training": {
        "output_dir": "outputs/qwen-ppo-gsm8k",
        "batch_size": 4,
        "mini_batch_size": 2,
        "ppo_epochs": 4,
        "max_grad_norm": 1.0,
        "target_kl": 0.05,
        "kl_penalty": "kl",
    },
}

DEFAULT_CONFIG = deep_merge(BASE_CONFIG, _PPO_OVERRIDES)


def main():
    train(DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
