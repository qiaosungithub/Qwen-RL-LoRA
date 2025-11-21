import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb
import re
import os

# Set environment variables
os.environ["WANDB_PROJECT"] = "gmpo-qwen-gsm8k"


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
        # Reuse most of the parent's computation
        # We'll compute per-token log probs and everything up to the loss aggregation step

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        # Compute the per_token_logps
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

        # Compute the loss
        advantages = inputs["advantages"]
        old_per_token_logps = inputs.get("old_per_token_logps")
        old_per_token_logps = per_token_logps.detach() if old_per_token_logps is None else old_per_token_logps

        log_ratio = per_token_logps - old_per_token_logps

        # ==================== KEY CHANGE: GMPO LOSS COMPUTATION ====================
        # GMPO: geometric mean of |min[rho_t * A, clip(rho_t) * A]| * sgn(A)
        # GRPO: arithmetic mean of min[rho_t * A, clip(rho_t) * A]

        # Compute importance sampling ratio
        ratio = torch.exp(log_ratio)  # (B, T)

        # Clip the ratio
        epsilon = 0.4  # GMPO uses wider clipping range (e^-0.4, e^0.4)
        ratio_clipped = torch.clamp(ratio, torch.exp(torch.tensor(-epsilon)), torch.exp(torch.tensor(epsilon)))

        # Compute importance-weighted advantages with clipping
        # Shape: advantages.unsqueeze(1) -> (B, 1), broadcasts to (B, T)
        weighted_advantages_1 = ratio * advantages.unsqueeze(1)
        weighted_advantages_2 = ratio_clipped * advantages.unsqueeze(1)

        # Take minimum (PPO-style clipping)
        weighted_advantages_min = torch.min(weighted_advantages_1, weighted_advantages_2)  # (B, T)

        # For GMPO: compute geometric mean of |weighted_advantages| over tokens
        # geometric_mean = (prod |x_t|)^(1/T) = exp(mean(log|x_t|))
        abs_weighted_advantages = torch.abs(weighted_advantages_min)  # (B, T)

        # Add small epsilon to avoid log(0)
        eps = 1e-8
        log_abs_weighted = torch.log(abs_weighted_advantages + eps)  # (B, T)

        # Compute mean of log values over valid tokens
        geometric_mean_log = (log_abs_weighted * completion_mask).sum(-1) / completion_mask.sum(-1).clamp(min=1.0)  # (B,)
        geometric_mean = torch.exp(geometric_mean_log)  # (B,)

        # Multiply by sign of advantage
        sgn_A = torch.sign(advantages)  # (B,)
        gmpo_objective = geometric_mean * sgn_A  # (B,)

        # Loss is negative objective (we want to maximize the objective)
        loss = -gmpo_objective.mean()
        loss = loss / self.current_gradient_accumulation_steps
        # ===========================================================================

        # Logging metrics (same as GRPO)
        mode = "train" if model.training else "eval"

        mean_entropy = ((entropies * completion_mask).sum() / completion_mask.sum().clamp(min=1.0))
        self._metrics[mode]["entropy"].append(self.accelerator.gather(mean_entropy).nanmean().item())

        # Compute clipping metrics
        ratio = torch.exp(log_ratio)
        is_low_clipped = (ratio < torch.exp(torch.tensor(-epsilon))) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (ratio > torch.exp(torch.tensor(epsilon))) & (advantages.unsqueeze(1) > 0)

        clip_ratio = ((is_low_clipped | is_high_clipped).float() * completion_mask).sum() / completion_mask.sum().clamp(min=1.0)
        self._metrics[mode]["clip_ratio/region_mean"].append(self.accelerator.gather(clip_ratio).nanmean().item())

        return loss


# System prompt and formatting
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


def format_data(example):
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ]
    }


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


# Reward 1: Format (Did they use the tags?)
def format_reward_func(completions, **kwargs):
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


# Reward 2: Correctness (Does the number match?)
def correctness_reward_func(prompts, completions, answer, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    extracted_answers = [extract_xml_answer(r) for r in responses]

    rewards = []
    for extracted, correct in zip(extracted_answers, answer):
        # Extract the number from the GSM8K solution text (usually last number)
        correct_val = correct.split("#### ")[-1].strip()
        if extracted == correct_val:
            rewards.append(2.0)  # High reward for correct answer
        else:
            rewards.append(0.0)
    return rewards


def main():
    # Login to wandb
    wandb.login()

    # Model setup
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map=None,
    ).to("cuda")

    # Dataset
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    dataset = dataset.map(format_data)

    # Training configuration
    training_args = GRPOConfig(
        output_dir="qwen-gmpo-gsm8k",
        logging_steps=1,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,
        num_generations=8,
        max_prompt_length=512,
        max_completion_length=512,
        learning_rate=5e-6,
        report_to="wandb",
        fp16=False,
        bf16=True,
        max_steps=200
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    # Use GMPOTrainer instead of GRPOTrainer
    trainer = GMPOTrainer(
        model=model,
        reward_funcs=[format_reward_func, correctness_reward_func],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer
    )

    # Train
    trainer.train()

    # Save the final model
    trainer.save_model("qwen-gmpo-gsm8k-final")


if __name__ == "__main__":
    main()
