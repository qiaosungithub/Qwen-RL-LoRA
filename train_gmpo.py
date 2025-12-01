"""
GMPO (Geometric Mean Policy Optimization) Training Script

GMPO is a variant of GRPO that uses geometric mean instead of arithmetic mean
when aggregating per-token importance-weighted advantages.

Usage:
    python train_gmpo.py                           # standalone with defaults
    python train_rl.py --config configs/gmpo.yaml  # via unified script
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


# ============================================================================
# GMPO Trainer
# ============================================================================

class GMPOTrainer(GRPOTrainer):
    """
    GMPO (Geometric Mean Policy Optimization) Trainer.

    This is a patched version of GRPOTrainer that uses geometric mean instead of
    arithmetic mean when computing the loss. The key difference is in how we
    aggregate per-token importance-weighted advantages:

    GRPO: (1/|o|) * sum_t(rho_t * A)  (arithmetic mean)
    GMPO: [prod_t(|rho_t * A|)]^(1/|o|) * sgn(A)  (geometric mean)

    This change stabilizes policy updates by being more robust to outlier
    importance sampling ratios.
    """

    def __init__(self, *args, clip_epsilon: float = 0.4, debug: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.gmpo_clip_epsilon = clip_epsilon
        self.gmpo_debug = debug

    def _compute_loss(self, model, inputs):
        """
        Override the GRPO loss computation to use geometric mean (GMPO).

        According to the GMPO paper, instead of computing:
            GRPO: (1/|o|) * sum_t(min[rho_t * A, clip(rho_t) * A])

        We compute:
            GMPO: [prod_t(|min[rho_t * A, clip(rho_t) * A]|)]^(1/|o|) * sgn(A)

        In log space for numerical stability:
            geometric_mean = exp(sum(log|x|) / count) * sgn(A)
        """
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps, entropies = self._get_per_token_logps_and_entropies(
            model,
            input_ids,
            attention_mask,
            logits_to_keep,
            compute_entropy=True,
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            num_images=inputs.get("num_images"),
            pixel_attention_mask=inputs.get("pixel_attention_mask"),
            image_sizes=inputs.get("image_sizes"),
            token_type_ids=inputs.get("token_type_ids"),
        )

        advantages = inputs["advantages"]
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps

        # GMPO: geometric mean instead of arithmetic mean
        ratio = torch.exp(log_ratio)
        epsilon = self.gmpo_clip_epsilon
        ratio_clipped = torch.clamp(
            ratio,
            torch.exp(torch.tensor(-epsilon)).to(ratio.device),
            torch.exp(torch.tensor(epsilon)).to(ratio.device)
        )

        weighted_advantages_1 = ratio * advantages.unsqueeze(1)
        weighted_advantages_2 = ratio_clipped * advantages.unsqueeze(1)
        weighted_advantages_min = torch.min(weighted_advantages_1, weighted_advantages_2)

        # Geometric mean in log space
        abs_weighted_advantages = torch.abs(weighted_advantages_min)
        eps = 1e-8
        log_abs_weighted = torch.log(abs_weighted_advantages + eps)

        geometric_mean_log = (log_abs_weighted * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)
        geometric_mean = torch.exp(geometric_mean_log)

        sgn_A = torch.sign(advantages)
        gmpo_objective = geometric_mean * sgn_A

        loss = -gmpo_objective.mean()
        loss = loss / self.current_gradient_accumulation_steps

        # Logging
        mode = "train" if model.training else "eval"
        mean_entropy = ((entropies * completion_mask).sum() / completion_mask.sum().clamp(min=1.0))
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        is_low_clipped = (ratio < torch.exp(torch.tensor(-epsilon))) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (ratio > torch.exp(torch.tensor(epsilon))) & (advantages.unsqueeze(1) > 0)
        clip_ratio = ((is_low_clipped | is_high_clipped).float() * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        self._metrics[mode]["clip_ratio/region_mean"].append(self.accelerator.gather(clip_ratio).nanmean().item())

        # Debug output
        if self.gmpo_debug:
            print(f"\n=== GMPO Debug ===")
            print(f"Advantages: min={advantages.min().item():.4f}, max={advantages.max().item():.4f}, mean={advantages.mean().item():.4f}, std={advantages.std().item():.4f}")
            print(f"Geometric mean: min={geometric_mean.min().item():.4f}, max={geometric_mean.max().item():.4f}, mean={geometric_mean.mean().item():.4f}")
            print(f"GMPO objective: mean={gmpo_objective.mean().item():.6f}")
            print(f"Final loss: {loss.item():.6e}")
            print(f"==================\n")

        return loss


# ============================================================================
# Training Function
# ============================================================================

def train(config: dict) -> str:
    """
    Train using GMPO.

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
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name"],
        cache_dir=model_config["cache_dir"],
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
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
        run_name=f"gmpo-{Path(training_config['output_dir']).name}",
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

    clip_epsilon = training_config["clip_epsilon"]
    debug = training_config["debug"]

    trainer = GMPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        clip_epsilon=clip_epsilon,
        debug=debug,
    )

    # Train
    print(f"\n{'='*60}")
    print(f"Training with GMPO (clip_epsilon={clip_epsilon})")
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

# GMPO-specific overrides
_GMPO_OVERRIDES = {
    "method": "gmpo",
    "training": {
        "output_dir": "outputs/qwen-gmpo-gsm8k",
        "per_device_train_batch_size": 4,
        "num_generations": 16,
        "clip_epsilon": 0.4,
        "debug": False,
    },
}

DEFAULT_CONFIG = deep_merge(BASE_CONFIG, _GMPO_OVERRIDES)


def main():
    train(DEFAULT_CONFIG)


if __name__ == "__main__":
    main()
