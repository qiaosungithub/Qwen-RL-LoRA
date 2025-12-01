import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import re
from tqdm import tqdm
import argparse


SYSTEM_PROMPT = """
Respond to the user's math problem.
You must format your output as follows:
<think>
{reasoning}
</think>
<answer>
{final_answer}
</answer>
"""


def extract_xml_answer(text: str) -> str:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()


def extract_ground_truth(answer_text: str) -> str:
    """Extract the final numerical answer from GSM8K solution."""
    return answer_text.split("#### ")[-1].strip()


def check_format(text: str) -> bool:
    """Check if response follows the expected format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    return bool(re.search(pattern, text, re.DOTALL))


def evaluate_model(model, tokenizer, dataset, max_new_tokens, desc="Evaluating"):
    """Evaluate a model on the dataset and return results."""
    correct = 0
    format_correct = 0
    total = 0
    results = []

    for i, example in enumerate(tqdm(dataset, desc=desc)):
        question = example["question"]
        ground_truth = extract_ground_truth(example["answer"])

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )

        extracted = extract_xml_answer(response)
        has_format = check_format(response)
        is_correct = extracted == ground_truth

        if has_format:
            format_correct += 1
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "response": response,
            "extracted": extracted,
            "is_correct": is_correct,
            "has_format": has_format,
        })

        if (i + 1) % 50 == 0:
            print(f"\n[{i+1}/{len(dataset)}] Running accuracy: {correct/total:.2%}, Format: {format_correct/total:.2%}")

    return {
        "accuracy": correct / total if total > 0 else 0,
        "format_accuracy": format_correct / total if total > 0 else 0,
        "total": total,
        "correct": correct,
        "format_correct": format_correct,
        "results": results,
    }


def print_results(name, metrics):
    """Print evaluation results for a model."""
    print(f"\n{'=' * 50}")
    print(f"{name} RESULTS")
    print(f"{'=' * 50}")
    print(f"Total samples: {metrics['total']}")
    print(f"Correct answers: {metrics['correct']} ({metrics['accuracy']:.2%})")
    print(f"Correct format: {metrics['format_correct']} ({metrics['format_accuracy']:.2%})")


def evaluate(
    model_path: str = "qwen-grpo-gsm8k-final",
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    max_samples: int = None,
    max_new_tokens: int = 512,
):
    print(f"Loading base model: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    print("Loading GSM8K test dataset...")
    dataset = load_dataset("openai/gsm8k", "main", split="test")

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Evaluating on {len(dataset)} samples...")

    # Evaluate base model first
    print("\n" + "=" * 50)
    print("EVALUATING BASE MODEL (no fine-tuning)")
    print("=" * 50)
    model.eval()
    base_metrics = evaluate_model(
        model, tokenizer, dataset, max_new_tokens, desc="Base Model"
    )
    print_results("BASE MODEL", base_metrics)

    # Load LoRA adapter and evaluate fine-tuned model
    print("\n" + "=" * 50)
    print(f"EVALUATING GRPO MODEL (from {model_path})")
    print("=" * 50)
    print(f"Loading LoRA adapter from: {model_path}")
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()

    grpo_metrics = evaluate_model(
        model, tokenizer, dataset, max_new_tokens, desc="GRPO Model"
    )
    print_results("GRPO MODEL", grpo_metrics)

    # Print comparison
    print("\n" + "=" * 50)
    print("COMPARISON: BASE vs GRPO")
    print("=" * 50)
    acc_diff = grpo_metrics["accuracy"] - base_metrics["accuracy"]
    fmt_diff = grpo_metrics["format_accuracy"] - base_metrics["format_accuracy"]

    print(f"{'Metric':<20} {'Base':>12} {'GRPO':>12} {'Δ':>12}")
    print("-" * 56)
    print(f"{'Accuracy':<20} {base_metrics['accuracy']:>11.2%} {grpo_metrics['accuracy']:>11.2%} {acc_diff:>+11.2%}")
    print(f"{'Format':<20} {base_metrics['format_accuracy']:>11.2%} {grpo_metrics['format_accuracy']:>11.2%} {fmt_diff:>+11.2%}")
    print(f"{'Correct':<20} {base_metrics['correct']:>12} {grpo_metrics['correct']:>12} {grpo_metrics['correct'] - base_metrics['correct']:>+12}")
    print("=" * 56)

    # Show examples where GRPO improved over base
    print("\n--- Examples where GRPO fixed Base errors ---")
    base_results = {r["question"]: r for r in base_metrics["results"]}
    grpo_results = {r["question"]: r for r in grpo_metrics["results"]}

    fixed_count = 0
    for q, grpo_r in grpo_results.items():
        base_r = base_results.get(q)
        if base_r and not base_r["is_correct"] and grpo_r["is_correct"]:
            if fixed_count < 3:
                print(f"\nFixed Example {fixed_count + 1}:")
                print(f"Q: {q[:100]}...")
                print(f"Ground truth: {grpo_r['ground_truth']}")
                print(f"Base extracted: {base_r['extracted']} (wrong)")
                print(f"GRPO extracted: {grpo_r['extracted']} (correct)")
            fixed_count += 1

    print(f"\nTotal fixed by GRPO: {fixed_count}")

    # Show examples where GRPO regressed
    regressed_count = 0
    for q, grpo_r in grpo_results.items():
        base_r = base_results.get(q)
        if base_r and base_r["is_correct"] and not grpo_r["is_correct"]:
            if regressed_count < 3:
                print(f"\nRegressed Example {regressed_count + 1}:")
                print(f"Q: {q[:100]}...")
                print(f"Ground truth: {grpo_r['ground_truth']}")
                print(f"Base extracted: {base_r['extracted']} (correct)")
                print(f"GRPO extracted: {grpo_r['extracted']} (wrong)")
            regressed_count += 1

    if regressed_count > 0:
        print(f"\nTotal regressed by GRPO: {regressed_count}")

    return {
        "base": base_metrics,
        "grpo": grpo_metrics,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GRPO model on GSM8K")
    parser.add_argument(
        "--model-path",
        type=str,
        default="qwen-grpo-gsm8k-final",
        help="Path to the LoRA adapter checkpoint",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Base model ID",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate",
    )
    args = parser.parse_args()

    evaluate(
        model_path=args.model_path,
        base_model=args.base_model,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
    )
