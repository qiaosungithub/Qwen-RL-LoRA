"""
Batch Evaluation Script

Runs eval_rl.py on multiple (config, model_path) tuples and saves output to separate txt files.

Usage:
    # Define EVAL_CONFIGS in this file, then run:
    python batch_eval.py

    # Or use a file with tuples (config,model_path per line):
    python batch_eval.py --tuples_file eval_tuples.txt

    # Override output directory:
    python batch_eval.py --output_dir my_results/
"""

import sys
import subprocess
from pathlib import Path
from datetime import datetime
import argparse


# =============================================================================
# CONFIGURE YOUR EVALUATION RUNS HERE
# Each tuple is (config_path, model_path)
# =============================================================================
EVAL_CONFIGS = [
    ("configs/grpo.yaml", "outputs/qwen-grpo-gsm8k-final"),
    ("configs/gmpo.yaml", "outputs/qwen-gmpo-gsm8k-20251201-152642-final"),
    ("configs/rloo.yaml", "outputs/qwen-rloo-gsm8k-20251201-203336-final"),
]
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(description="Batch RL Evaluation")
    parser.add_argument("--tuples_file", type=str, help="File containing (config,model_path) tuples, one per line")
    parser.add_argument("--output_dir", type=str, default="eval_results", help="Directory for output files")
    parser.add_argument("--max_samples", type=int, default=None, help="Max samples to evaluate")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Max new tokens to generate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    return parser.parse_args()


def get_eval_tuples(args):
    """Collect (config, model_path) tuples from various sources."""
    tuples = []

    # First, add tuples from EVAL_CONFIGS
    tuples.extend(EVAL_CONFIGS)

    # Then, add from file if provided
    if args.tuples_file:
        with open(args.tuples_file, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    parts = line.split(",")
                    if len(parts) == 2:
                        config, model_path = parts[0].strip(), parts[1].strip()
                        tuples.append((config, model_path))

    return tuples


def path_to_filename(model_path):
    """Convert model path to a safe filename."""
    name = model_path.replace("/", "_").replace("\\", "_")
    name = name.strip("_")
    if name.startswith("outputs_"):
        name = name[8:]
    return name


def run_evaluation(config, model_path, output_file, max_samples=None, max_new_tokens=512, batch_size=16):
    """Run eval_rl.py for a single model and save output."""
    cmd = [
        sys.executable,
        "eval_rl.py",
        "--config", config,
        "--model_path", model_path,
        "--max_new_tokens", str(max_new_tokens),
        "--batch_size", str(batch_size),
    ]

    if max_samples is not None:
        cmd.extend(["--max_samples", str(max_samples)])

    print(f"\n{'=' * 60}")
    print(f"Config: {config}")
    print(f"Model: {model_path}")
    print(f"Output: {output_file}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}\n")

    with open(output_file, "w") as f:
        f.write(f"Config: {config}\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write("=" * 60 + "\n\n")
        f.flush()

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            print(line, end="")
            f.write(line)
            f.flush()

        process.wait()

        f.write(f"\n\nExit code: {process.returncode}\n")

    return process.returncode


def main():
    args = parse_args()

    eval_tuples = get_eval_tuples(args)

    if not eval_tuples:
        print("Error: No evaluation tuples specified.")
        print("Either edit EVAL_CONFIGS in this file or use --tuples_file")
        sys.exit(1)

    print(f"Found {len(eval_tuples)} evaluations to run:")
    for config, model_path in eval_tuples:
        print(f"  - {config} -> {model_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for config, model_path in eval_tuples:
        filename = path_to_filename(model_path) + ".txt"
        output_file = output_dir / filename

        exit_code = run_evaluation(
            config=config,
            model_path=model_path,
            output_file=str(output_file),
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
        )

        results.append({
            "config": config,
            "model_path": model_path,
            "output_file": str(output_file),
            "exit_code": exit_code,
        })

    print("\n" + "=" * 60)
    print("BATCH EVALUATION SUMMARY")
    print("=" * 60)

    for r in results:
        status = "OK" if r["exit_code"] == 0 else "FAIL"
        print(f"[{status}] {r['config']} -> {r['model_path']}")
        print(f"       -> {r['output_file']}")

    success = sum(1 for r in results if r["exit_code"] == 0)
    print(f"\nCompleted: {success}/{len(results)} successful")
    print(f"Results saved to: {output_dir}/")


if __name__ == "__main__":
    main()
