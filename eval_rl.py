"""
Unified RL Evaluation Script

Usage:
    python eval_rl.py --config configs/grpo.yaml
    python eval_rl.py --config configs/gmpo.yaml
    python eval_rl.py --config configs/ppo.yaml

    # Override config values:
    python eval_rl.py --config configs/grpo.yaml --max_samples 100

    # Specify model path explicitly:
    python eval_rl.py --config configs/grpo.yaml --model_path outputs/qwen-grpo-gsm8k-final
"""

import torch
import argparse
from tqdm import tqdm

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

from rl_utils import (
    SYSTEM_PROMPT,
    load_config,
    extract_xml_answer,
    check_format,
    get_torch_dtype,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Unified RL Evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--model_path", type=str, default=None, help="Override model path")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens to generate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for parallel evaluation")
    return parser.parse_args()


def extract_ground_truth(answer_text: str) -> str:
    """Extract the final numerical answer from GSM8K solution."""
    return answer_text.split("#### ")[-1].strip()


def evaluate_model(model, tokenizer, dataset, max_new_tokens, batch_size=8, desc="Evaluating"):
    """Evaluate a model on the dataset and return results."""
    correct = 0
    format_correct = 0
    total = 0
    results = []

    # Prepare all prompts and ground truths
    all_prompts = []
    all_questions = []
    all_ground_truths = []

    for example in dataset:
        question = example["question"]
        ground_truth = extract_ground_truth(example["answer"])

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        all_prompts.append(prompt)
        all_questions.append(question)
        all_ground_truths.append(ground_truth)

    # Process in batches
    num_batches = (len(all_prompts) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc=desc):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(all_prompts))

        batch_prompts = all_prompts[start_idx:end_idx]
        batch_questions = all_questions[start_idx:end_idx]
        batch_ground_truths = all_ground_truths[start_idx:end_idx]

        # Tokenize batch with padding
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        # Decode each response in the batch
        for i, (output, question, ground_truth) in enumerate(
            zip(outputs, batch_questions, batch_ground_truths)
        ):
            # Get only the generated part (after the prompt)
            prompt_len = inputs["input_ids"][i].ne(tokenizer.pad_token_id).sum()
            response = tokenizer.decode(output[prompt_len:], skip_special_tokens=True)

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

        # Progress update
        if (batch_idx + 1) % 10 == 0 or batch_idx == num_batches - 1:
            print(f"\n[{total}/{len(dataset)}] Running accuracy: {correct/total:.2%}, Format: {format_correct/total:.2%}")

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


def evaluate(config: dict, model_path: str = None, max_samples: int = None, max_new_tokens: int = 512, batch_size: int = 8):
    """Run evaluation comparing base model vs fine-tuned model."""
    model_config = config["model"]
    training_config = config["training"]
    dataset_config = config["dataset"]
    method = config["method"].upper()

    # Determine model path
    if model_path is None:
        model_path = f"{training_config['output_dir']}-final"

    base_model_name = model_config["name"]

    print(f"Loading base model: {base_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=get_torch_dtype(model_config["torch_dtype"]),
        attn_implementation=model_config.get("attn_implementation", "flash_attention_2"),
        device_map="auto",
    )

    # Load dataset (test split)
    print(f"Loading {dataset_config['name']} test dataset...")
    dataset = load_dataset(
        dataset_config["name"],
        dataset_config.get("config", "main"),
        split="test"
    )

    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))

    print(f"Evaluating on {len(dataset)} samples (batch_size={batch_size})...")

    # Evaluate base model first
    print("\n" + "=" * 50)
    print("EVALUATING BASE MODEL (no fine-tuning)")
    print("=" * 50)
    model.eval()
    base_metrics = evaluate_model(
        model, tokenizer, dataset, max_new_tokens, batch_size=batch_size, desc="Base Model"
    )
    print_results("BASE MODEL", base_metrics)

    # Load LoRA adapter and evaluate fine-tuned model
    print("\n" + "=" * 50)
    print(f"EVALUATING {method} MODEL (from {model_path})")
    print("=" * 50)
    print(f"Loading LoRA adapter from: {model_path}")

    try:
        model = PeftModel.from_pretrained(model, model_path)
        model.eval()

        finetuned_metrics = evaluate_model(
            model, tokenizer, dataset, max_new_tokens, batch_size=batch_size, desc=f"{method} Model"
        )
        print_results(f"{method} MODEL", finetuned_metrics)

        # Print comparison
        print("\n" + "=" * 50)
        print(f"COMPARISON: BASE vs {method}")
        print("=" * 50)
        acc_diff = finetuned_metrics["accuracy"] - base_metrics["accuracy"]
        fmt_diff = finetuned_metrics["format_accuracy"] - base_metrics["format_accuracy"]

        print(f"{'Metric':<20} {'Base':>12} {method:>12} {'Δ':>12}")
        print("-" * 56)
        print(f"{'Accuracy':<20} {base_metrics['accuracy']:>11.2%} {finetuned_metrics['accuracy']:>11.2%} {acc_diff:>+11.2%}")
        print(f"{'Format':<20} {base_metrics['format_accuracy']:>11.2%} {finetuned_metrics['format_accuracy']:>11.2%} {fmt_diff:>+11.2%}")
        print(f"{'Correct':<20} {base_metrics['correct']:>12} {finetuned_metrics['correct']:>12} {finetuned_metrics['correct'] - base_metrics['correct']:>+12}")
        print("=" * 56)

        # Show examples where fine-tuned model improved over base
        print(f"\n--- Examples where {method} fixed Base errors ---")
        base_results = {r["question"]: r for r in base_metrics["results"]}
        ft_results = {r["question"]: r for r in finetuned_metrics["results"]}

        fixed_count = 0
        for q, ft_r in ft_results.items():
            base_r = base_results.get(q)
            if base_r and not base_r["is_correct"] and ft_r["is_correct"]:
                if fixed_count < 3:
                    print(f"\nFixed Example {fixed_count + 1}:")
                    print(f"Q: {q[:100]}...")
                    print(f"Ground truth: {ft_r['ground_truth']}")
                    print(f"Base extracted: {base_r['extracted']} (wrong)")
                    print(f"{method} extracted: {ft_r['extracted']} (correct)")
                fixed_count += 1

        print(f"\nTotal fixed by {method}: {fixed_count}")

        # Show examples where fine-tuned model regressed
        regressed_count = 0
        for q, ft_r in ft_results.items():
            base_r = base_results.get(q)
            if base_r and base_r["is_correct"] and not ft_r["is_correct"]:
                if regressed_count < 3:
                    print(f"\nRegressed Example {regressed_count + 1}:")
                    print(f"Q: {q[:100]}...")
                    print(f"Ground truth: {ft_r['ground_truth']}")
                    print(f"Base extracted: {base_r['extracted']} (correct)")
                    print(f"{method} extracted: {ft_r['extracted']} (wrong)")
                regressed_count += 1

        if regressed_count > 0:
            print(f"\nTotal regressed by {method}: {regressed_count}")

        return {
            "base": base_metrics,
            method.lower(): finetuned_metrics,
        }

    except Exception as e:
        print(f"\nError loading fine-tuned model: {e}")
        print("Only base model results are available.")
        return {"base": base_metrics}


def main():
    args = parse_args()
    config = load_config(args.config)

    print(f"\nLoaded config from: {args.config}")
    print(f"Method: {config['method']}")

    evaluate(
        config=config,
        model_path=args.model_path,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
