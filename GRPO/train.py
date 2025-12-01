# train.py
import torch
from trl import GRPOTrainer

from config import get_grpo_config, PEFT_CONFIG
from model import get_model_and_tokenizer
from data import get_gsm8k_dataset
from rewards import format_reward_func, correctness_reward_func

def main():
    # Basic sanity: must have a CUDA device
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please run on a GPU machine.")

    print("Loading model and tokenizer...")
    model, tokenizer = get_model_and_tokenizer()

    print("Loading GSM8K dataset...")
    dataset = get_gsm8k_dataset()

    print("Building GRPO config...")
    training_args = get_grpo_config()

    print("Initializing GRPOTrainer...")
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[format_reward_func, correctness_reward_func],
        args=training_args,
        train_dataset=dataset,
        peft_config=PEFT_CONFIG,
        processing_class=tokenizer,
    )

    print("Starting training...")
    trainer.train()

    print("Saving final model...")
    trainer.save_model(training_args.output_dir)
    print("Done. Model saved to:", training_args.output_dir)

if __name__ == "__main__":
    main()
