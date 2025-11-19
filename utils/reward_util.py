import re, torch

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

