# 基本依赖
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
import torch
import torch.nn.functional as F
import re
import os, wandb
from datetime import datetime

from gsm8k_dataloader import create_gsm8k_dataloader
from svamp_dataloader import create_svamp_dataloader
from gsmhard_dataloader import create_gsmhard_dataloader

from utils.qwen_util import enable_kv_cache, disable_kv_cache

def parse_answer(response):
    """从response中提取答案"""
    answer_match = re.search(r'\\boxed{([^}]*)}', response)
    if answer_match:
        final_answer = answer_match.group(1).strip()
    else:
        final_answer = None
    
    try:
        # 处理逗号分隔的数字
        if final_answer:
            final_answer = final_answer.replace(',', '')
        final_answer = int(final_answer)
    except (ValueError, TypeError):
        final_answer = None
    
    return final_answer


def compute_rewards(responses, ground_truth_answers, tokenizer, reward_correct=1.0, reward_wrong=-0.5):
    """
    计算每个response的reward
    Args:
        responses: [batch_size, response_length] 生成的token ids
        ground_truth_answers: list of int，正确答案
        tokenizer: tokenizer
        reward_correct: 正确答案的奖励
        reward_wrong: 错误答案的惩罚
    Returns:
        rewards: [batch_size, response_length] 每个token位置的reward
    """
    batch_size = len(responses)
    rewards = []
    
    for i in range(batch_size):
        response_text = tokenizer.decode(responses[i], skip_special_tokens=True)
        predicted_answer = parse_answer(response_text)
        ground_truth = int(ground_truth_answers[i])
        
        # 给所有token相同的reward（整个推理链条对最终答案负责）
        response_length = responses[i].shape[0]
        
        if predicted_answer is not None and predicted_answer == ground_truth:
            token_rewards = torch.full((response_length,), reward_correct, device=responses[i].device)
        else:
            token_rewards = torch.full((response_length,), reward_wrong, device=responses[i].device)
        
        rewards.append(token_rewards)
    
    # Pad to same length
    max_len = max(r.shape[0] for r in rewards)
    padded_rewards = []
    for r in rewards:
        if r.shape[0] < max_len:
            padding = torch.zeros(max_len - r.shape[0], device=r.device)
            r = torch.cat([r, padding])
        padded_rewards.append(r)
    
    return torch.stack(padded_rewards)  # [batch_size, max_response_length]


def get_log_probs(model, input_ids, attention_mask, response_start_indices):
    """
    计算responses的log probabilities
    Args:
        model: 语言模型
        input_ids: [batch_size, seq_len] 完整的input (prompt + response)
        attention_mask: [batch_size, seq_len]
        response_start_indices: [batch_size] 每个样本response开始的位置
    Returns:
        log_probs: [batch_size, max_response_len] response部分的log probs
    """
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    logits = outputs.logits  # [batch_size, seq_len, vocab_size]
    
    batch_size = input_ids.shape[0]
    log_probs_list = []
    
    for i in range(batch_size):
        start_idx = response_start_indices[i]
        # 找到该样本的实际结束位置（排除padding）
        end_idx = attention_mask[i].sum().item()
        
        if end_idx <= start_idx:
            # 空response，创建一个dummy
            log_probs_list.append(torch.zeros(1, device=input_ids.device))
            continue
        
        # 提取response部分的logits (注意要取前一个位置的logits来预测当前token)
        response_logits = logits[i, start_idx-1:end_idx-1, :]  # [response_len, vocab_size]
        response_tokens = input_ids[i, start_idx:end_idx]      # [response_len]
        
        # 计算log probs
        log_probs = F.log_softmax(response_logits, dim=-1)
        selected_log_probs = log_probs.gather(
            dim=-1,
            index=response_tokens.unsqueeze(-1)
        ).squeeze(-1)  # [response_len]
        
        log_probs_list.append(selected_log_probs)
    
    # Pad to same length
    max_len = max(lp.shape[0] for lp in log_probs_list)
    padded_log_probs = []
    for lp in log_probs_list:
        if lp.shape[0] < max_len:
            # 用0填充（对应padding tokens的log prob）
            padding = torch.zeros(max_len - lp.shape[0], device=lp.device)
            lp = torch.cat([lp, padding])
        padded_log_probs.append(lp)
    
    return torch.stack(padded_log_probs)  # [batch_size, max_response_len]


def train_and_evaluate(config, workdir):
    """
    完整的PPO训练流程
    """
    # 设定训练设备
    device = config.device
    print(f'\n{"="*60}')
    print(f'Using device: {device}')
    print(f'{"="*60}\n')

    # 设置随机种子
    torch.manual_seed(config.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.training.seed)
    
    # init wandb
    wandb.init(project="qwen-ppo-lora", dir=workdir, tags=['try'])
    wandb.config.update(config.to_dict())
    wandb.run.notes = config.wandb_notes

    # ============ 1. 加载 Policy Model ============
    print("Loading policy model...")
    model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        device_map={'': device},
        dtype=torch.bfloat16,
        trust_remote_code=True,
        cache_dir=config.cache_dir,
        attn_implementation="flash_attention_2",  # Flash Attention 2 加速
        use_cache=True,  # KV cache
    )

    # ============ 2. 加载 Reference Model（冻结） ============
    print("Loading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.model.name,
        device_map={'': device},
        dtype=torch.bfloat16,
        trust_remote_code=True,
        cache_dir=config.cache_dir,
        attn_implementation="flash_attention_2",
        use_cache=False,
    )
    
    # 冻结 reference model
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()

    # ============ 3. 加载 Tokenizer ============
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.name,
        trust_remote_code=True,
        cache_dir=config.cache_dir
    )
    
    # 统一使用 left padding（适用于生成和训练）
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ============ 4. 添加 LoRA ============
    print("Adding LoRA adapters...")
    lora_cfg = LoraConfig(
        r=config.lora.rank,
        lora_alpha=config.lora.alpha,
        target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj'],
        lora_dropout=config.lora.dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_cfg)
    
    # 打印可训练参数
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.4f}%)")

    # ============ 5. 优化器（LoRA参数不使用weight decay） ============
    # 分离参数：LoRA参数 vs 其他参数
    lora_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'lora' in name.lower():
                lora_params.append(param)
            else:
                param.requires_grad = False
    
    optimizer = torch.optim.AdamW([
        {'params': lora_params, 'weight_decay': 0.0},
    ], lr=config.training.learning_rate)
    
    print(f"Optimizer: AdamW with lr={config.training.learning_rate}")
    print(f"  - LoRA params: {len(lora_params)} tensors")

    # ============ 6. 创建 DataLoader ============
    print(f"\nLoading dataset: {config.dataset.name}")
    # 使用 rollout_batch_size 作为 dataloader 的 batch_size
    if config.dataset.name == 'gsm8k':
        train_loader, train_steps = create_gsm8k_dataloader(
            tokenizer=tokenizer,
            split='train',
            batch_size=config.training.batch_size,  # 256
            max_length=config.dataset.max_length,
            shuffle=True,
        )
    elif config.dataset.name == 'gsmhard':
        train_loader, train_steps = create_gsmhard_dataloader(
            tokenizer=tokenizer,
            split='train',
            batch_size=config.training.batch_size,  # 256
            max_length=config.dataset.max_length,
            shuffle=True,
        )
    elif config.dataset.name == 'svamp':
        train_loader, train_steps = create_svamp_dataloader(
            tokenizer=tokenizer,
            split='train',
            batch_size=config.training.batch_size,  # 256
            max_length=config.dataset.max_length,
            shuffle=True,
        )
    else:
        raise ValueError(f"Unknown dataset: {config.dataset.name}")
    
    print(f'Train steps per epoch: {train_steps}')

    # ============ 7. 生成参数 ============
    generate_kwargs = dict(
        temperature=config.generation.temperature,
        top_p=config.generation.top_p,
        top_k=config.generation.top_k,
        do_sample=config.generation.do_sample,
        max_new_tokens=config.generation.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    )

    # ============ 8. PPO 超参数 ============
    clip_epsilon = config.ppo.clip_epsilon
    kl_coef = config.ppo.kl_coef
    train_batch_size = config.training.train_batch_size  # 32
    rounds_per_batch = config.training.rounds_per_batch  # 3
    
    print(f"\nPPO Hyperparameters:")
    print(f"  - Rollout batch size: {config.training.batch_size}")
    print(f"  - Train batch size: {train_batch_size}")
    print(f"  - Rounds per batch: {rounds_per_batch}")
    print(f"  - Clip epsilon: {clip_epsilon}")
    print(f"  - KL coefficient: {kl_coef}")

    # ============ 9. 训练循环 ============
    print(f"\n{'='*60}")
    print(f"Starting PPO Training")
    print(f"{'='*60}\n")

    global_step = 0
    best_accuracy = 0.0

    for epoch in range(config.training.num_epochs):
        model.train()
        epoch_loss = 0.0
        epoch_policy_loss = 0.0
        epoch_kl = 0.0
        epoch_correct = 0
        epoch_total = 0
        epoch_updates = 0  # 记录更新次数

        for batch_idx, batch in enumerate(train_loader):
            # ============ Phase 1: Rollout (生成256道题的回答) ============
            print(f"\n[Rollout] Processing batch {batch_idx+1}")
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ground_truth = batch['answer']
            
            rollout_batch_size = input_ids.shape[0]
            prompt_lengths = attention_mask.sum(dim=1)  # 每个样本的prompt长度

            # 生成 responses
            model.eval()
            enable_kv_cache(model)
            with torch.no_grad():
                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **generate_kwargs,
                )
            
            # 提取生成的部分
            responses = output[:, input_ids.size(1):]  # [rollout_batch_size, response_length]
            
            # 评估生成质量 & 计算 rewards
            decoded_responses = tokenizer.batch_decode(responses, skip_special_tokens=True)
            batch_correct = 0
            
            for i, resp in enumerate(decoded_responses):
                final_answer = parse_answer(resp)
                epoch_total += 1
                if final_answer is not None and final_answer == int(ground_truth[i]):
                    batch_correct += 1
                    epoch_correct += 1
            
            rewards = compute_rewards(
                responses, 
                ground_truth, 
                tokenizer,
                reward_correct=config.ppo.reward_correct,
                reward_wrong=config.ppo.reward_wrong
            )  # [rollout_batch_size, response_length]
            
            # 构建完整序列（prompt + response）
            full_input_ids = torch.cat([input_ids, responses], dim=1)
            full_attention_mask = torch.cat([
                attention_mask,
                torch.ones_like(responses)
            ], dim=1)
            
            # 记录每个样本response开始的位置
            response_start_indices = prompt_lengths.tolist()
            
            # 获取 old log probs（rollout时的log probs）
            model.eval()
            disable_kv_cache(model)
            with torch.no_grad():
                old_log_probs = get_log_probs(
                    model, 
                    full_input_ids, 
                    full_attention_mask, 
                    response_start_indices
                )  # [rollout_batch_size, max_response_len]
                
                # 同时获取 reference model 的 log probs（用于KL惩罚）
                ref_log_probs = get_log_probs(
                    ref_model,
                    full_input_ids,
                    full_attention_mask,
                    response_start_indices
                )  # [rollout_batch_size, max_response_len]
            
            # 创建response mask
            response_mask = torch.zeros_like(rewards, dtype=torch.bool)
            for i in range(rollout_batch_size):
                start_idx = response_start_indices[i]
                end_idx = full_attention_mask[i].sum().item()
                response_len = end_idx - start_idx
                response_mask[i, :response_len] = True
            
            # 统计rollout信息
            rollout_accuracy = batch_correct / rollout_batch_size
            # 现在每个有效token都有相同的reward，直接计算平均即可
            avg_reward = (rewards * response_mask).sum().item() / response_mask.sum().item()
            
            print(f"[Rollout] Generated {rollout_batch_size} responses | "
                  f"Accuracy: {rollout_accuracy:.2%} | "
                  f"Avg Reward: {avg_reward:.3f} | "
                  f"Correct: {batch_correct}, Wrong: {rollout_batch_size - batch_correct}")
            
            # ============ Phase 2: 多轮PPO更新 (对256题过3轮，每次32题) ============
            # 计算总共需要的更新步数：256 * 3 / 32 = 24步
            total_updates = (rollout_batch_size * rounds_per_batch) // train_batch_size
            
            for update_idx in range(total_updates):
                round_num = update_idx // (rollout_batch_size // train_batch_size) + 1
                step_in_round = update_idx % (rollout_batch_size // train_batch_size) + 1
                
                print(f"  [Update {update_idx+1}/{total_updates}] Round {round_num}/{rounds_per_batch}, Step {step_in_round}")
                
                # 随机采样train_batch_size个样本
                indices = torch.randperm(rollout_batch_size)[:train_batch_size]
                
                # 提取子batch
                train_input_ids = full_input_ids[indices]
                train_attention_mask = full_attention_mask[indices]
                train_old_log_probs = old_log_probs[indices]
                train_ref_log_probs = ref_log_probs[indices]
                train_rewards = rewards[indices]
                train_response_mask = response_mask[indices]
                train_response_start_indices = [response_start_indices[i] for i in indices]
                
                # 前向传播获取新的 log probs
                model.train()
                disable_kv_cache(model)
                new_log_probs = get_log_probs(
                    model,
                    train_input_ids,
                    train_attention_mask,
                    train_response_start_indices
                )  # [train_batch_size, max_response_len]
                
                # 计算 PPO loss
                # 1. Policy ratio
                ratio = torch.exp(new_log_probs - train_old_log_probs)
                
                # 2. Clipped ratio
                clipped_ratio = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
                
                # 3. PPO objective (只在有效token上计算)
                policy_objective = torch.min(
                    ratio * train_rewards,
                    clipped_ratio * train_rewards
                )
                policy_objective = (policy_objective * train_response_mask).sum() / train_response_mask.sum()
                policy_loss = -policy_objective
                
                # 4. KL divergence penalty
                kl_div = (new_log_probs - train_ref_log_probs)
                kl_div = (kl_div * train_response_mask).sum() / train_response_mask.sum()
                kl_penalty = kl_coef * kl_div
                
                # 5. Total loss
                loss = policy_loss + kl_penalty
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.training.max_grad_norm)
                optimizer.step()
                
                # 记录统计信息
                epoch_loss += loss.item()
                epoch_policy_loss += policy_loss.item()
                epoch_kl += kl_div.item()
                epoch_updates += 1
                global_step += 1
                
                print(f"    Loss: {loss.item():.4f} | "
                      f"Policy: {policy_loss.item():.4f} | "
                      f"KL: {kl_div.item():.4f}")
                
                # Log to wandb
                wandb.log({
                    'loss': loss.item(),
                    'policy_loss': policy_loss.item(),
                    'kl_div': kl_div.item(),
                    'rollout_accuracy': rollout_accuracy,
                    'avg_reward': avg_reward,
                    'epoch': epoch + 1,
                    'step': global_step,
                    'round': round_num,
                })
            
            # 限制每个epoch的最大步数
            if batch_idx >= config.training.max_steps_per_epoch - 1:
                break
        
        # ============ Epoch 总结 ============
        avg_loss = epoch_loss / epoch_updates if epoch_updates > 0 else 0
        avg_policy_loss = epoch_policy_loss / epoch_updates if epoch_updates > 0 else 0
        avg_kl = epoch_kl / epoch_updates if epoch_updates > 0 else 0
        epoch_accuracy = epoch_correct / epoch_total if epoch_total > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config.training.num_epochs} Summary:")
        print(f"  Total Updates: {epoch_updates}")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Policy Loss: {avg_policy_loss:.4f}")
        print(f"  KL Divergence: {avg_kl:.4f}")
        print(f"  Accuracy: {epoch_accuracy:.2%} ({epoch_correct}/{epoch_total})")
        print(f"{'='*60}\n")
        
        # ============ 保存 checkpoint ============
        if (epoch + 1) % config.training.save_every == 0:
            save_dir = os.path.join(workdir, f"checkpoint_epoch_{epoch+1}")
            os.makedirs(save_dir, exist_ok=True)
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)
            print(f"✓ Saved checkpoint to {save_dir}\n")
        
        # 保存最佳模型
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            best_dir = os.path.join(workdir, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            print(f"✓ Saved best model (accuracy: {best_accuracy:.2%}) to {best_dir}\n")

    print(f"\n{'='*60}")
    print("Training Completed!")
    print(f"Best Accuracy: {best_accuracy:.2%}")
    print(f"{'='*60}\n")

    wandb.finish()
    
    return model, tokenizer