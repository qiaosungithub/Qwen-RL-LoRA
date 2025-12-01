"""
Unified RL Training Script

Dispatches to the appropriate trainer based on the config file.

Usage:
    python train_rl.py --config configs/grpo.yaml
    python train_rl.py --config configs/gmpo.yaml
    python train_rl.py --config configs/ppo.yaml
    python train_rl.py --config configs/rloo.yaml

    # Override config values from command line:
    python train_rl.py --config configs/grpo.yaml --training.max_steps 500 --training.learning_rate 1e-5
"""

import argparse
from rl_utils import load_config


def parse_args():
    parser = argparse.ArgumentParser(description="Unified RL Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")

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

    return args.config, overrides


def main():
    config_path, overrides = parse_args()
    config = load_config(config_path, overrides)

    print(f"\nLoaded config from: {config_path}")
    print(f"Method: {config['method']}")
    if overrides:
        print(f"Overrides: {overrides}")

    method = config["method"].lower()

    if method == "grpo":
        from train_grpo import train
        train(config)
    elif method == "gmpo":
        from train_gmpo import train
        train(config)
    elif method == "ppo":
        from train_ppo import train
        train(config)
    elif method == "rloo":
        from train_rloo import train
        train(config)
    else:
        raise ValueError(f"Unknown method: {method}. Choose from: grpo, gmpo, ppo, rloo")


if __name__ == "__main__":
    main()
