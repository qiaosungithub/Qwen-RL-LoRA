#!/bin/bash
# GMPO Training Runner Script
# This script sets up the environment and launches GMPO training

# Activate conda environment (uncomment and modify as needed)
# conda activate your_env_name

HERE=$(pwd)

# Generate timestamp for unique job name
now=`date '+%Y%m%d_%H%M%S'`
JOBNAME=gmpo_Qwen_LoRA_${now}
LOGDIR=$HERE/logs/$JOBNAME

# Set WandB API key (replace with your own key or use environment variable)
export WANDB_API_KEY=${WANDB_API_KEY:-"73f8ff40bb7f8589e9bd1f476196a896f662cdfa"}

# Create log directory
mkdir -p ${LOGDIR}
echo "================================================"
echo "GMPO Training - Geometric Mean Policy Optimization"
echo "================================================"
echo "Job name: ${JOBNAME}"
echo "Log dir: ${LOGDIR}"
echo "================================================"
echo ""

# Login to WandB
echo "Logging in to WandB..."
python -m wandb login $WANDB_API_KEY
sleep 1
python -m wandb login

echo ""
echo "Starting GMPO training..."
echo ""

# Run GMPO training
# Note: Using the simplified train_gmpo.py directly instead of main_gmpo.py
# If you want to use the config system with main_gmpo.py, use:
# python main_gmpo.py --workdir=${LOGDIR} --config=configs/load_config.py:gmpo

python train_gmpo.py 2>&1 | tee ${LOGDIR}/train.log

echo ""
echo "================================================"
echo "Training completed!"
echo "Logs saved to: ${LOGDIR}"
echo "================================================"
