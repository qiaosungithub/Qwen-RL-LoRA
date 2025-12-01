# data.py
from datasets import load_dataset
from config import SYSTEM_PROMPT

def format_example(example):
    """
    Convert a GSM8K example into chat-style "prompt" for GRPO.
    Keeps "answer" as-is for reward function.
    """
    return {
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example["question"]},
        ],
        "answer": example["answer"],  # will be used by correctness reward
    }

def get_gsm8k_dataset():
    """
    Load and format the GSM8K train split.
    """
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    dataset = dataset.map(format_example)
    return dataset
