# 基本依赖
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
import torch
import torch.nn.functional as F
import os, wandb

from gsm8k_dataloader import create_gsm8k_dataloader
from svamp_dataloader import create_svamp_dataloader
from gsmhard_dataloader import create_gsmhard_dataloader

from utils.qwen_util import enable_kv_cache, disable_kv_cache, generate_multiple_times, model_forward_multiple_times
from utils.reward_util import parse_answer, compute_rewards
from utils.vllm_util import vLLMGenerator, merge_lora_for_vllm

def save_response(responses, epoch, batch_idx, workdir):
    """将生成的响应保存到文件中"""
    response_dir = os.path.join(workdir, "responses")
    os.makedirs(response_dir, exist_ok=True)
    
    filename = f"epoch_{epoch+1}_batch_{batch_idx+1}.txt"
    filepath = os.path.join(response_dir, filename)
    
    with open(filepath, 'w') as f:
        for i, resp in enumerate(responses):
            f.write(f"Response {i+1}:\n{resp}\n\n")
    
    print(f"✓ Saved responses to {filepath}")

def get_log_probs_streaming(model, input_ids, attention_mask, response_start_indices, forward_bs=None):
    """
    流式计算 log probs，不存储整个 logits，可大幅减少显存（避免几十GB）
    """
    B, T = input_ids.shape
    if forward_bs is None:
        forward_bs = B
    assert B % forward_bs == 0
    num_chunks = B // forward_bs

    all_log_probs = []  # list of [response_len] tensors（每个样本独立）

    for ci in range(num_chunks):
        s = ci * forward_bs
        e = (ci + 1) * forward_bs

        ids_chunk = input_ids[s:e]
        mask_chunk = attention_mask[s:e]
        start_chunk = response_start_indices[s:e]

        # forward
        outputs = model(input_ids=ids_chunk, attention_mask=mask_chunk)
        logits_chunk = outputs.logits  # [chunk, T, V]
        V = logits_chunk.shape[-1]

        # 对 chunk 内的每一个样本提取 log prob
        for bi in range(forward_bs):
            start_idx = start_chunk[bi]
            end_idx = mask_chunk[bi].sum().item()

            if end_idx <= start_idx:
                all_log_probs.append(torch.zeros(1, device=input_ids.device))
                continue

            # logits 应该使用前一个 token 的位置预测当前 token
            resp_logits = logits_chunk[bi, start_idx-1:end_idx-1, :]  # [resp_len, V]
            resp_tokens = ids_chunk[bi, start_idx:end_idx]            # [resp_len]

            log_probs = F.log_softmax(resp_logits, dim=-1)
            selected = log_probs.gather(dim=-1, index=resp_tokens.unsqueeze(-1)).squeeze(-1)

            all_log_probs.append(selected)

        # ⚠️ 关键：丢弃大 tensor
        del logits_chunk, outputs
        torch.cuda.empty_cache()

        print(f"finished forward chunk {ci+1}/{num_chunks}")

    # === Padding ===
    max_len = max(lp.size(0) for lp in all_log_probs)
    padded = []
    for lp in all_log_probs:
        if lp.size(0) < max_len:
            pad = torch.zeros(max_len - lp.size(0), device=lp.device)
            lp = torch.cat([lp, pad])
        padded.append(lp)

    return torch.stack(padded)  # [B, max_len]


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
        attn_implementation="sdpa",  # Flash Attention 2 加速
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
        attn_implementation="sdpa",
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

    # ============ 7. PPO 超参数 ============
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

    # ============ 8. 初始化 vLLM ============
    # vLLM更新频率（每N个batch更新一次vLLM使用的模型）
    vllm_update_freq = getattr(config.training, 'vllm_update_freq', 5)
    print(f"\nvLLM Configuration:")
    print(f"  - Update frequency: every {vllm_update_freq} batches")
    
    # 创建vLLM模型保存目录
    vllm_model_dir = os.path.join(workdir, "vllm_model")
    os.makedirs(vllm_model_dir, exist_ok=True)
    
    # 初始保存base model（第一次使用）
    print("\n[Init] Saving initial model for vLLM...")
    merge_lora_for_vllm(model, tokenizer, vllm_model_dir)
    
    # 初始化vLLM生成器
    vllm_generator = vLLMGenerator(vllm_model_dir, config, device)

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
            # ============ Phase 0: 定期更新vLLM模型 ============
            if batch_idx % vllm_update_freq == 0 and batch_idx > 0:
                print(f"\n[Update] Merging LoRA at batch {batch_idx}...")
                merge_lora_for_vllm(model, tokenizer, vllm_model_dir)
                vllm_generator.reload_model(vllm_model_dir)
            
            # ============ Phase 1: Rollout ============
            print(f"\n[Rollout] Processing batch {batch_idx+1}")
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            ground_truth = batch['answer']
            
            rollout_batch_size = input_ids.shape[0]
            prompt_lengths = attention_mask.sum(dim=1)  # 每个样本的prompt长度

            # 将prompts转为文本
            prompts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            
            # 使用vLLM生成（超快！）
            responses_text = vllm_generator.generate(prompts)

            # save the responses in a file
            save_response(responses_text, epoch, batch_idx, workdir)

            batch_correct = 0
            
            for i, resp in enumerate(responses_text):
                final_answer = parse_answer(resp)
                epoch_total += 1
                if final_answer is not None and final_answer == int(ground_truth[i]):
                    batch_correct += 1
                    epoch_correct += 1
            
            # 将生成的文本转回token ids
            responses_encoded = tokenizer(
                responses_text,
                padding=True,
                truncation=True,
                max_length=config.generation.max_new_tokens,
                return_tensors="pt",
                add_special_tokens=False,  # 不要添加额外的special tokens
            )
            responses = responses_encoded['input_ids'].to(device)
            
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
            torch.cuda.empty_cache()
            with torch.no_grad():
                old_log_probs = get_log_probs_streaming(
                    model, 
                    full_input_ids, 
                    full_attention_mask, 
                    response_start_indices,
                    forward_bs=config.generate_bs
                )  # [rollout_batch_size, max_response_len]
                
                # 同时获取 reference model 的 log probs（用于KL惩罚）
                ref_log_probs = get_log_probs_streaming(
                    ref_model,
                    full_input_ids,
                    full_attention_mask,
                    response_start_indices,
                    forward_bs=config.generate_bs
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
            
            torch.cuda.empty_cache()
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
                new_log_probs = get_log_probs_streaming(
                    model,
                    train_input_ids,
                    train_attention_mask,
                    train_response_start_indices,
                    forward_bs=None,
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
                kl_div = (new_log_probs - train_ref_log_probs) ** 2
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

    vllm_generator.cleanup()

    wandb.finish()
    
    return model, tokenizer