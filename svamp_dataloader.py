import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer


class SVAMPDataset(Dataset):
    '''
    SVAMP Dataset for supervised / RL fine-tuning.
    Each item returns: {'input_ids', 'attention_mask', 'labels'}.
    SVAMP fields: question, answer (numeric string).
    '''
    def __init__(self, split='train', tokenizer=None, max_length=512):
        # SVAMP is not split originally; we usually treat entire set as "train"
        # You can also manually split if needed.
        self.dataset = load_dataset(
            'ChilleD/SVAMP',
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
        question = item['question_concat']
        answer = str(item['answer'])  # ensure string

        if self.tokenizer is not None:
            prompt = (
                "Please solve the following math problem. "
                "Return your final answer marking with ####, for example: #### 42.\n"
                f"{question}\nAnswer:"
            )

            input_ids = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt',
            )

            labels = self.tokenizer(
                answer,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt',
            )

            return {
                'input_ids': input_ids['input_ids'].squeeze(0),
                'attention_mask': input_ids['attention_mask'].squeeze(0),
                'labels': labels['input_ids'].squeeze(0),
                'original': item,
            }

        else:
            return {'question': question, 'answer': answer}


def create_svamp_dataloader(
    tokenizer,
    split='train',
    batch_size=4,
    max_length=512,
    shuffle=True,
):
    '''
    Creates a PyTorch DataLoader for SVAMP.
    Returns: dataloader, num_batches
    '''
    dataset = SVAMPDataset(split=split, tokenizer=tokenizer, max_length=max_length)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, len(loader)


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-7B')

    loader, steps = create_svamp_dataloader(
        tokenizer=tokenizer,
        split='train',
        batch_size=2,
    )

    for batch in loader:
        print({k: v.shape for k, v in batch.items()})
        print(tokenizer.decode(batch['input_ids'][0][:120]))
        print(tokenizer.decode(batch['labels'][0][:120]))
        break
