import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import re

class GSMHardDataset(Dataset):
    '''
    GSM-Hard Dataset for Supervised or RL fine-tuning.
    Each item returns: {'input_ids', 'attention_mask', 'answer'} or raw text if no tokenizer.
    '''
    def __init__(self, split='train', tokenizer=None, max_length=512, cache_dir=None):
        self.dataset = load_dataset(
            'reasoning-machines/gsm-hard',
            split=split,
            cache_dir=cache_dir,
        )
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        question = item['input']  # GSM-Hard uses 'input' for questions
        answer = item['target']   # GSM-Hard uses 'target' for answers

        if self.tokenizer is not None:
            prompt = f'Please solve the following problem. Please reason step by step, and put your final answer within \\boxed{{}}.\n{question}\nAnswer:'
            messages = [{'role': 'user', 'content': prompt}]
            input_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
            input_ids = self.tokenizer(
                input_prompt,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt',
            )

            # Extract final answer - GSM-Hard target is already the final numeric answer
            # But let's handle it consistently with GSM8K
            final_answer = str(answer).strip()
            
            # Turn final_answer into int. Handle comma-separated numbers
            x = final_answer.replace(',', '')
            try:
                final_answer = int(x)
            except:
                # Some answers might be floats, try that
                try:
                    final_answer = float(x)
                    final_answer = int(final_answer)  # Convert to int if it's a whole number
                except:
                    raise ValueError(f'Cannot convert answer to int: {answer} with {x}')

            return {
                'input_ids': input_ids['input_ids'].squeeze(0),
                'attention_mask': input_ids['attention_mask'].squeeze(0),  # for ignoring padding tokens
                'answer': final_answer,
                'vis': {
                    'question': question,
                    'input_prompt': input_prompt,
                },
            }
        else:
            return {'question': question, 'answer': answer}


def create_gsmhard_dataloader(
    tokenizer,
    split='train',
    batch_size=4,
    max_length=512,
    shuffle=True,
    cache_dir=None,
):
    '''
    Creates a PyTorch DataLoader for GSM-Hard.
    - If tokenizer=None, returns raw text dataset.
    - You can plug this into any SFT or RL training loop.
    '''
    dataset = GSMHardDataset(split=split, tokenizer=tokenizer, max_length=max_length, cache_dir=cache_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, len(loader)


if __name__ == '__main__':
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2-7B')
    
    loader, steps = create_gsmhard_dataloader(
        tokenizer=tokenizer,
        split='train',
        batch_size=2,
    )

    for batch in loader:
        print({k: v.shape if isinstance(v, torch.Tensor) else type(v) for k, v in batch.items()})
        print("Question:", batch['vis']['question'][0])
        print("Input prompt preview:", batch['vis']['input_prompt'][0][:200])
        print("Answer:", batch['answer'][0])
        break