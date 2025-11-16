# 基本依赖
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
import torch, torch.nn.functional as F
import re

from gsm8k_dataloader import create_gsm8k_dataloader
from svamp_dataloader import create_svamp_dataloader
from gsmhard_dataloader import create_gsmhard_dataloader

def parse_answer(response):
    # grep the answer from the response
    answer_match = re.search(r'\\boxed{([^}]*)}', response)
    if answer_match:
        final_answer = answer_match.group(1).strip()
    else:
        final_answer = None
    
    try:
        final_answer = int(final_answer)
    except (ValueError, TypeError):
        final_answer = None
    
    return final_answer

def train_and_evaluate(config, workdir):
    # 设定训练设备

    device = config.device
    print(f'\n\nATTENTION: ######### Using device: {device} #########\n\n')

    # set random seed
    torch.manual_seed(config.training.seed)

    # 1. 模型加载 + LoRA
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        device_map={'': device},
        dtype=torch.bfloat16,
        trust_remote_code=True,
        cache_dir='/data/scratch-oc40/sqa/cache'
    ).eval()

    # model = model.to_bettertransformer() # accelerate BetterTransformer. not working now?

    generate_kwargs = dict(
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        do_sample=True,
        max_new_tokens=2048,
    ) # Qwen default


    lora_cfg = LoraConfig(r=8, lora_alpha=32, target_modules=['q_proj','v_proj'])
    model = get_peft_model(model, lora_cfg) # wrap with LoRA
    # model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.model.name, trust_remote_code=True, cache_dir='/data/scratch-oc40/sqa/cache')
    tokenizer.padding_side = "left"

    # 2. 冻结除了 LoRA 层
    for n, p in model.named_parameters():
        if 'lora' not in n:
            p.requires_grad = False

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    ############## create dataset ##############
    # train_loader, train_steps = create_gsm8k_dataloader(
    train_loader, train_steps = create_gsmhard_dataloader(
        tokenizer=tokenizer,
        split='train',
        batch_size=config.training.batch_size,
        max_length=config.dataset.max_length,
        shuffle=True,
    )
    print(f'Train steps per epoch: {train_steps}')

    ############## training loop ##############
    total_count = 0
    correct_count = 0
    model.train()

    for batch in train_loader:
        # # for vis
        # print(f'Question: {batch["vis"]["question"]}\n')
        # print(f'Model input: {batch["vis"]["input_prompt"]}\n')
        # print(f'Answer: {batch["answer"]}')
        # print(batch['input_ids'].shape)
        # print(batch['attention_mask'].shape)
        # print(batch['labels'].shape)

        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)

        # generate answer
        with torch.no_grad():
            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generate_kwargs,
            )
        responses = output[:, input_ids.size(1):]

        # decode the generated response
        decoded_responses = tokenizer.batch_decode(responses, skip_special_tokens=True)
        for i, resp in enumerate(decoded_responses):
            print(f"Response {i}: {resp}")

            final_answer = parse_answer(resp)
            total_count += 1
            s = 'wrong'
            if final_answer == int(batch['answer'][i]):
                s = 'correct'
                correct_count += 1
            
            print(f'Problem {total_count}: get {final_answer}, ground truth {batch["answer"][i]}. Evaluated as {s}.\n')
            
        if total_count > 80: break
        continue # just eval
    # 计算准确率
    if total_count > 0:
        accuracy = correct_count / total_count
        print(f'Accuracy: {accuracy:.4f}')

        exit('evalover')

        # # 计算 reward（自定义）
        # rewards = compute_reward_fn(batch['prompt'], responses)

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
