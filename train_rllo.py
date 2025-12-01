import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import RLOOConfig, RLOOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM
import wandb
import re
import os

# Set environment variables
os.environ["WANDB_PROJECT"] = "rloo-qwen-gsm8k"

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
    # Robust extraction of the content between <answer> tags
    if "<answer>" not in text or "</answer>" not in text:
        return ""
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

# Reward 1: Format (Did they use the tags?)
def format_reward_func(completions, **kwargs):
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"

    # The Trainer handles the unpacking of the generated text.
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

# Reward 2: Correctness (Does the number match?)
# Note: 'answer' comes from the dataset column "answer"
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
    tokenizer.padding_side = "left" # Important for generation padding

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
        device_map="auto",
    )

    # Dataset
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    dataset = dataset.map(format_data)

    # Training configuration
    training_args = RLOOConfig(
        output_dir="qwen-rloo-gsm8k",
        run_name="rloo-gsm8k-run",
        logging_steps=1,
        per_device_train_batch_size=2,   # RLOO often requires smaller batch sizes per device due to multiple generations
        gradient_accumulation_steps=4,

        # RLOO Specifics
        num_generations=4,               # Number of samples (k) per prompt
        beta=0.05,                       # KL penalty coefficient (crucial for RLOO stability)

        max_prompt_length=512,
        max_completion_length=512,
        learning_rate=5e-6,
        report_to="wandb",
        fp16=False,
        bf16=True,
        max_steps=200,

        # Optimization
        gradient_checkpointing=True,     # Saves VRAM
    )

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )

    # Use RLOOTrainer
    trainer = RLOOTrainer(
        model=model,
        reward_funcs=[format_reward_func, correctness_reward_func],
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer, # Formerly 'tokenizer'
    )

    # Train
    trainer.train()

    # Save the final model
    trainer.save_model("qwen-rloo-gsm8k-final")

if __name__ == "__main__":
    main()