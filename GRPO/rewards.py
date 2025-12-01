# rewards.py
import re

def extract_xml_answer(text: str) -> str:
    """
    Extracts the content between <answer> ... </answer>.
    If not found, returns empty string.
    """
    if "<answer>" not in text or "</answer>" not in text:
        return ""
    answer = text.split("<answer>", 1)[-1]
    answer = answer.split("</answer>", 1)[0]
    return answer.strip()

# Reward 1: Format reward (did they use <think> and <answer> correctly?)
def format_reward_func(completions, **kwargs):
    """
    completions: list of list of dicts, e.g. [[{"role": "assistant", "content": "..."}], ...]
    Returns a list of float rewards.
    """
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.search(pattern, r, re.DOTALL) for r in responses]
    return [0.5 if match else 0.0 for match in matches]

# Reward 2: Correctness reward (does the extracted answer match GSM8K's numeric answer?)
def correctness_reward_func(prompts, completions, answer, **kwargs):
    """
    prompts: not used directly here, but part of signature.
    completions: same as above.
    answer: list of GSM8K solution strings (contains "#### <number>" at the end).
    """
    responses = [completion[0]["content"] for completion in completions]
    extracted_answers = [extract_xml_answer(r) for r in responses]

    rewards = []
    for extracted, correct in zip(extracted_answers, answer):
        # GSM8K format: "... #### 42"
        correct_val = correct.split("####")[-1].strip()
        if extracted == correct_val:
            rewards.append(2.0)  # high reward for correct numeric answer
        else:
            rewards.append(0.0)
    return rewards
