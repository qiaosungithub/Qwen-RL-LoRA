"""
Shared utilities for RL training scripts.
"""

import torch
import re
import yaml


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
# Shared Default Config (matches base.yaml)
# ============================================================================

BASE_CONFIG = {
    "model": {
        "name": "Qwen/Qwen3-8B",
        "torch_dtype": "bfloat16",
        "attn_implementation": "flash_attention_2",
        "cache_dir": ".cache/models",
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
