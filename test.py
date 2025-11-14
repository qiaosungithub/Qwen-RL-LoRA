import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset   

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # adjust to Qwen’s naming
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

gsm8k = load_dataset("openai/gsm8k", "main")

def format_example(ex):
    prompt = (
        "You are a helpful math tutor. Solve the following problem step by step.\n"
        "Show your reasoning clearly, and put the final answer in the form \"#### <answer>\".\n\n"
        f"Question:\n{ex['question']}\n\nAnswer:\n"
    )
    # GSM8K answer already ends with '#### <ans>'
    target = ex["answer"]
    full_text = prompt + target
    return {"text": full_text}

gsm8k = gsm8k.map(format_example)
train_data = gsm8k["train"]
test_data = gsm8k["test"]

def tokenize_fn(ex):
    out = tokenizer(
        ex["text"],
        truncation=True,
        max_length=1024,
    )
    out["labels"] = out["input_ids"].copy()
    return out

tokenized_train = train_data.map(tokenize_fn, batched=True, remove_columns=train_data.column_names)
# tokenized_train = tokenized_train.select(range(5000))  # Optionally limit to first 5000 samples for quicker training
tokenized_test = test_data.map(tokenize_fn, batched=True, remove_columns=test_data.column_names)

training_args = TrainingArguments(
    output_dir="./qwen-gsm8k-sft",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=True,
    logging_steps=20,
    save_strategy="epoch",
    evaluation_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
)
trainer.train()