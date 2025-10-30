import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

class GSM8KDataset(Dataset):
    """
    GSM8K Dataset for Supervised or RL fine-tuning.
    Each item returns: {"input_ids", "attention_mask", "labels"} or raw text if no tokenizer.
    """
    def __init__(self, split="train", tokenizer=None, max_length=512):
        self.dataset = load_dataset("gsm8k", "main", split=split)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        question = item["question"]
        answer = item["answer"]

        if self.tokenizer is not None:
            prompt = f"Question: {question}\nAnswer:"
            input_ids = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            labels = self.tokenizer(
                answer,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": input_ids["input_ids"].squeeze(0),
                "attention_mask": input_ids["attention_mask"].squeeze(0),
                "labels": labels["input_ids"].squeeze(0),
                'original': item,
            }
        else:
            return {"question": question, "answer": answer}


def create_gsm8k_dataloader(
    tokenizer,
    split="train",
    batch_size=4,
    max_length=512,
    shuffle=True,
):
    """
    Creates a PyTorch DataLoader for GSM8K.
    - If model_name=None, returns raw text dataset.
    - You can plug this into any SFT or RL training loop.
    """
    dataset = GSM8KDataset(split=split, tokenizer=tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader


if __name__ == "__main__":
    loader, tokenizer = create_gsm8k_dataloader(
        model_name="Qwen/Qwen2-7B",
        split="train",
        batch_size=2,
    )

    for batch in loader:
        print({k: v.shape for k, v in batch.items()})
        print(tokenizer.decode(batch["input_ids"][0][:100]))
        print(tokenizer.decode(batch["labels"][0][:100]))
        break
