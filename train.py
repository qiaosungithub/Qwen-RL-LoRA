# 基本依赖
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
import torch, torch.nn.functional as F

# 1. 模型加载 + LoRA
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-7B",
    device_map="auto",
    trust_remote_code=True
)
lora_cfg = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj","v_proj"])
model = get_peft_model(model, lora_cfg)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-7B", trust_remote_code=True)

# 2. 冻结除了 LoRA 层
for n, p in model.named_parameters():
    if 'lora' not in n:
        p.requires_grad = False

optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

# 3. RL 循环 (简化版 PPO/GRPO)
for batch in dataset:
    input_ids = tokenizer(batch["prompt"], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(**input_ids, max_new_tokens=128, do_sample=True)
    responses = output[:, input_ids["input_ids"].size(1):]

    # 计算 reward（自定义）
    rewards = compute_reward_fn(batch["prompt"], responses)

    # 重新前向算 logprobs
    logits = model(torch.cat([input_ids["input_ids"], responses], dim=1)).logits
    logprobs = F.log_softmax(logits, dim=-1)
    selected = logprobs.gather(-1, responses.unsqueeze(-1)).squeeze(-1)

    # PPO-like loss
    ratio = torch.exp(selected - old_logprobs)
    clipped = torch.clamp(ratio, 1-0.1, 1+0.1)
    policy_loss = -torch.mean(torch.min(ratio * rewards, clipped * rewards))

    optimizer.zero_grad()
    policy_loss.backward()
    optimizer.step()
