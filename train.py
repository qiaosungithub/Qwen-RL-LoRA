# 基本依赖
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
import torch, torch.nn.functional as F

from gsm8k_dataloader import create_gsm8k_dataloader

def train_and_evaluate(config, workdir):
    # 设定训练设备

    device = config.device
    print(f'ATTENTION: ######### Using device: {device} #########')

    # 1. 模型加载 + LoRA
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        device_map={'': device},
        trust_remote_code=True,
        cache_dir='/data/scratch-oc40/sqa/cache'
    )
    lora_cfg = LoraConfig(r=8, lora_alpha=32, target_modules=['q_proj','v_proj'])
    model = get_peft_model(model, lora_cfg) # wrap with LoRA
    # model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, trust_remote_code=True, cache_dir='/data/scratch-oc40/sqa/cache')

    # 2. 冻结除了 LoRA 层
    for n, p in model.named_parameters():
        if 'lora' not in n:
            p.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    ############## create dataset ##############
    train_loader, train_steps = create_gsm8k_dataloader(
        tokenizer=tokenizer,
        split='train',
        batch_size=config.training.batch_size,
        max_length=config.dataset.max_length,
        shuffle=True,
    )
    print(f'Train steps per epoch: {train_steps}')

    # 3. RL 循环 (简化版 PPO/GRPO)
    for batch in train_loader:
        # # for vis
        print(batch['original'])
        # print(batch['input_ids'].shape)
        # print(batch['attention_mask'].shape)
        # print(batch['labels'].shape)

        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        with torch.no_grad():
            output = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=128, do_sample=True)
        responses = output[:, input_ids.size(1):]

        # print the generated response
        decoded_responses = tokenizer.batch_decode(responses, skip_special_tokens=True)
        for i, resp in enumerate(decoded_responses):
            print(f"Response {i}: {resp}")
        assert False

        # 计算 reward（自定义）
        rewards = compute_reward_fn(batch['prompt'], responses)

        # 重新前向算 logprobs
        logits = model(torch.cat([input_ids, responses], dim=1)).logits
        logprobs = F.log_softmax(logits, dim=-1)
        selected = logprobs.gather(-1, responses.unsqueeze(-1)).squeeze(-1)

        # PPO-like loss
        ratio = torch.exp(selected - old_logprobs)
        clipped = torch.clamp(ratio, 1-0.1, 1+0.1)
        policy_loss = -torch.mean(torch.min(ratio * rewards, clipped * rewards))

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
