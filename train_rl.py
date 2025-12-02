"""
Unified RL Training Script

Dispatches to the appropriate trainer based on the config file.

Usage:
    python train_rl.py --config configs/grpo.yaml
    python train_rl.py --config configs/gmpo.yaml
    python train_rl.py --config configs/ppo.yaml
    python train_rl.py --config configs/rloo.yaml

    # With evaluation after training:
    python train_rl.py --config configs/grpo.yaml --eval
    python train_rl.py --config configs/grpo.yaml --eval --eval_samples 100

    # Override config values from command line:
    python train_rl.py --config configs/grpo.yaml --training.max_steps 500 --training.learning_rate 1e-5
"""

import argparse
import sys
from datetime import datetime
from io import StringIO
from pathlib import Path

from rl_utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Unified RL Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--eval", action="store_true", help="Run evaluation after training")
    parser.add_argument("--eval_samples", type=int, default=None, help="Number of samples for evaluation")
    parser.add_argument("--eval_max_tokens", type=int, default=512, help="Max new tokens for evaluation")
    parser.add_argument("--eval_batch_size", type=int, default=128, help="Batch size for evaluation")

    # Parse known args first, then collect overrides
    args, unknown = parser.parse_known_args()

    # Parse override arguments (--key.subkey value format)
    overrides = {}
    i = 0
    while i < len(unknown):
        if unknown[i].startswith("--"):
            key = unknown[i][2:]
            if i + 1 < len(unknown) and not unknown[i + 1].startswith("--"):
                overrides[key] = unknown[i + 1]
                i += 2
            else:
                overrides[key] = "true"
                i += 1
        else:
            i += 1

    return args, overrides


def main():
    args, overrides = parse_args()
    config = load_config(args.config, overrides)

    print(f"\nLoaded config from: {args.config}")
    print(f"Method: {config['method']}")
    if overrides:
        print(f"Overrides: {overrides}")

    method = config["method"].lower()

    # Train
    if method == "grpo":
        from train_grpo import train
        model_path = train(config)
    elif method == "gmpo":
        from train_gmpo import train
        model_path = train(config)
    elif method == "ppo":
        from train_ppo import train
        model_path = train(config)
    elif method == "rloo":
        from train_rloo import train
        model_path = train(config)
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: grpo, gmpo, ppo, rloo")

    # Evaluate if requested
    if args.eval:
        print(f"\n{'=' * 60}")
        print("RUNNING POST-TRAINING EVALUATION")
        print(f"{'=' * 60}")

        # Create eval output file path (next to model)
        eval_output_path = Path(model_path) / "eval_results.txt"

        class TeeOutput:
            """Write to both stdout and file."""
            def __init__(self, file, stream):
                self.file = file
                self.stream = stream

            def write(self, data):
                self.stream.write(data)
                self.file.write(data)
                self.file.flush()

            def flush(self):
                self.stream.flush()
                self.file.flush()

        from eval_rl import evaluate

        with open(eval_output_path, "w") as f:
            # Write header
            f.write(f"Config: {args.config}\n")
            f.write(f"Model Path: {model_path}\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Eval Samples: {args.eval_samples}\n")
            f.write(f"Batch Size: {args.eval_batch_size}\n")
            f.write("=" * 60 + "\n\n")

            # Capture stdout to file while also printing
            old_stdout = sys.stdout
            sys.stdout = TeeOutput(f, old_stdout)

            try:
                evaluate(
                    config=config,
                    model_path=model_path,
                    max_samples=args.eval_samples,
                    max_new_tokens=args.eval_max_tokens,
                    batch_size=args.eval_batch_size,
                )
            finally:
                sys.stdout = old_stdout

        print(f"\nEval results saved to: {eval_output_path}")


if __name__ == "__main__":
    main()
