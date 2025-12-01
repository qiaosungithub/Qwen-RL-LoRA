import torch
import re
from typing import List, Tuple

def identify_step_boundaries(response_text: str, tokenizer, response_ids: torch.Tensor, 
                             split_mode: str = 'newline') -> List[Tuple[int, int]]:
    """
    Identify token boundaries for each reasoning step.
    
    Args:
        response_text: Decoded response text
        tokenizer: Tokenizer
        response_ids: Token IDs [seq_len]
        split_mode: 'newline', 'sentence', or 'marker'
        
    Returns:
        List of (start_token_idx, end_token_idx) for each step
    """
    if split_mode == 'newline':
        # Split on newlines
        lines = response_text.split('\n')
        step_texts = [l.strip() for l in lines if l.strip()]
    elif split_mode == 'sentence':
        # Split on sentence boundaries
        step_texts = [s.strip() for s in re.split(r'[.!?]+', response_text) if s.strip()]
    elif split_mode == 'marker':
        # Look for explicit "Step N:" markers
        matches = list(re.finditer(r'Step\s+\d+:', response_text))
        if matches:
            step_texts = []
            for i, match in enumerate(matches):
                start = match.start()
                end = matches[i+1].start() if i+1 < len(matches) else len(response_text)
                step_texts.append(response_text[start:end].strip())
        else:
            # Fallback to newline if no markers found
            step_texts = [l.strip() for l in response_text.split('\n') if l.strip()]
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")
    
    # Map each step text to token indices
    boundaries = []
    current_pos = 0
    
    for step_text in step_texts:
        # Find step text in response
        step_start = response_text.find(step_text, current_pos)
        if step_start == -1:
            continue
        
        step_end = step_start + len(step_text)
        
        # Convert character positions to token positions
        # Tokenize prefix to find token start
        prefix = response_text[:step_start]
        prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
        start_token_idx = len(prefix_tokens)
        
        # Tokenize up to step end to find token end
        up_to_end = response_text[:step_end]
        end_tokens = tokenizer.encode(up_to_end, add_special_tokens=False)
        end_token_idx = len(end_tokens)
        
        boundaries.append((start_token_idx, end_token_idx))
        current_pos = step_end
    
    return boundaries


def compute_capo_rewards(responses, ground_truth, tokenizer, verifier_client, config):
    """
    Compute per-token rewards using CAPO credit assignment.
    
    Args:
        responses: [batch_size, response_length] token IDs
        ground_truth: List of correct answers
        tokenizer: Tokenizer
        verifier_client: VerifierClient instance
        config: Configuration object
        
    Returns:
        rewards: [batch_size, response_length] per-token rewards (same shape as PPO)
    """
    from utils.reward_util import parse_answer
    
    batch_size = len(responses)
    rewards = []
    
    # CAPO hyperparameters
    reward_correct = config.ppo.reward_correct
    reward_wrong = config.ppo.reward_wrong
    whole_weight = config.capo.whole_weight
    process_weight = config.capo.process_weight
    step_penalty = config.capo.step_penalty
    split_mode = config.capo.step_split_mode
    fallback_on_error = config.capo.fallback_on_error
    
    # Track metrics
    total_wrong_steps = 0
    total_samples = 0
    
    for i in range(batch_size):
        response_text = tokenizer.decode(responses[i], skip_special_tokens=True)
        predicted_answer = parse_answer(response_text)
        ground_truth_answer = int(ground_truth[i])
        response_length = responses[i].shape[0]
        
        # Determine base reward from final correctness
        if predicted_answer is not None and predicted_answer == ground_truth_answer:
            base_reward = reward_correct * whole_weight
        else:
            base_reward = reward_wrong * whole_weight
        
        # Initialize all tokens with base reward
        token_rewards = torch.full((response_length,), base_reward, 
                                   device=responses[i].device, dtype=torch.float32)
        
        # Call verifier to identify wrong steps
        verification = verifier_client.verify_reasoning(
            question="",  # Question not needed for dummy mode
            response=response_text,
            ground_truth=ground_truth_answer
        )
        
        if verification['success'] or not fallback_on_error:
            wrong_steps = verification['wrong_steps']
            total_steps = verification['total_steps']
            
            if wrong_steps and total_steps > 0:
                # Identify step boundaries
                try:
                    boundaries = identify_step_boundaries(
                        response_text, tokenizer, responses[i], split_mode
                    )
                    
                    # Apply penalty to tokens in wrong steps
                    for step_idx in wrong_steps:
                        if step_idx < len(boundaries):
                            start_tok, end_tok = boundaries[step_idx]
                            # Clamp to valid range
                            start_tok = max(0, min(start_tok, response_length))
                            end_tok = max(0, min(end_tok, response_length))
                            
                            # Add process penalty to these tokens
                            if start_tok < end_tok:
                                token_rewards[start_tok:end_tok] += step_penalty * process_weight
                    
                    total_wrong_steps += len(wrong_steps)
                    total_samples += 1
                    
                except Exception as e:
                    print(f"[CAPO] Step boundary error: {e}, using base reward only")
        
        rewards.append(token_rewards)
    
    # Pad to same length (matching compute_rewards behavior)
    max_len = max(r.shape[0] for r in rewards)
    padded_rewards = []
    for r in rewards:
        if r.shape[0] < max_len:
            padding = torch.zeros(max_len - r.shape[0], device=r.device, dtype=r.dtype)
            r = torch.cat([r, padding])
        padded_rewards.append(r)
    
    # Update verifier metrics with step info
    if hasattr(verifier_client, 'metrics'):
        verifier_client.metrics['avg_wrong_steps'] = total_wrong_steps / max(1, total_samples)
    
    return torch.stack(padded_rewards)  # [batch_size, max_response_length]
