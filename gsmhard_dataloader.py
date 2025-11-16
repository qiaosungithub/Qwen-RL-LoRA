import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer

class GSMHardDataset(Dataset):
    '''
    GSM-Hard Dataset for supervised or RL fine-tuning.
    Fields: input (question), code, target (answer).
    '''
    def __init__(self, 
                 split='train', 
                 tokenizer=None, 
                 max_length=512,
                 repo_id="reasoning-machines/gsm-hard"):
        self.dataset = load_dataset(
            repo_id,
            split=split,
            cache_dir='/data/scratch-oc40/sqa/data'
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        print(f'Original item keys: {item.keys()}')
        print(f'Original item: {item}')
        # Note: GSM-Hard uses 'input' for questions and 'target' for answers
        question = item['input']
        answer = item['target']

        if self.tokenizer is not None:
            prompt = (
                "Please solve the following math problem. "
                "Return your final answer marking with ####, for example: #### 42.\n"
                f"{question}\nAnswer:"
            )

            input_enc = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt',
            )

            # label_enc = self.tokenizer(
            #     answer,
            #     truncation=True,
            #     max_length=self.max_length,
            #     padding='max_length',
            #     return_tensors='pt',
            # )

            return {
                'input_ids': input_enc['input_ids'].squeeze(0),
                'attention_mask': input_enc['attention_mask'].squeeze(0),
                # 'labels': label_enc['input_ids'].squeeze(0),
                'original': item,
                'target': answer,
            }

        else:
            return {"question": question, "answer": answer, "code": item.get('code', '')}


def create_gsmhard_dataloader(
    tokenizer,
    split='train',
    batch_size=4,
    max_length=512,
    shuffle=True,
    repo_id="reasoning-machines/gsm-hard",
):
    dataset = GSMHardDataset(
        split=split,
        tokenizer=tokenizer,
        max_length=max_length,
        repo_id=repo_id,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, len(loader)


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-7B')

    loader, steps = create_gsmhard_dataloader(
        tokenizer=tokenizer,
        split='train',
        batch_size=2,
    )

    for batch in loader:
        print({k: v.shape if isinstance(v, torch.Tensor) else type(v) for k, v in batch.items()})
        print("Input:", tokenizer.decode(batch['input_ids'][0][:120]))
        print("Label:", tokenizer.decode(batch['labels'][0][:120]))
        break