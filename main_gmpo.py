"""
Main entry point for GMPO (Geometric Mean Policy Optimization) training.

This script provides a command-line interface for training language models
using GMPO, which is a more stable variant of GRPO that uses geometric mean
instead of arithmetic mean for aggregating token-level importance-weighted
advantages.

Usage:
    python main_gmpo.py --workdir=./logs/gmpo_run --config=configs/load_config.py:gmpo

The script will:
1. Load the model and tokenizer
2. Load and prepare the GSM8K dataset
3. Initialize the GMPOTrainer with custom loss computation
4. Train the model and save checkpoints
5. Log metrics to WandB
"""

from absl import app, flags
from ml_collections import config_flags
import os
import sys

# Add parent directory to path to import train_gmpo
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

FLAGS = flags.FLAGS

flags.DEFINE_string("workdir", None, "Directory to store model data and logs.")

config_flags.DEFINE_config_file(
    "config",
    help_string="File path to the GMPO training hyperparameter configuration.",
    lock_config=True,
)


def main(argv):
    """Main training function."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    # Import here to avoid loading heavy dependencies before flag parsing
    from train_gmpo import main as gmpo_main

    print(f"\n{'='*60}")
    print("GMPO Training - Geometric Mean Policy Optimization")
    print(f"{'='*60}\n")
    print(f"Working directory: {FLAGS.workdir}")
    print(f"Config: {FLAGS.config}")
    print()

    # Create working directory if it doesn't exist
    os.makedirs(FLAGS.workdir, exist_ok=True)

    # Run GMPO training
    return gmpo_main()


if __name__ == "__main__":
    flags.mark_flags_as_required(["workdir", "config"])
    app.run(main)
