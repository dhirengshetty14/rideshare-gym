#!/bin/bash
#SBATCH --job-name=rideshare-collect-rollouts
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/collect-%j.out
#SBATCH --error=logs/collect-%j.err

# Generate diverse trajectories at temperature 0.7 for SFT/DPO data construction.

set -euo pipefail

module purge
module load python/3.12 cuda/12.4

cd "${PROJECT_ROOT:-/work/users/d/h/${USER}/rideshare-gym}"
source .venv/bin/activate

mkdir -p logs runs training_data

# 50 episodes per task * 12 tasks = 600 trajectories per pass.
# Run multiple passes to reach ~5000 total if needed.
python eval/run.py \
    --tasks "rideshare/*" \
    --agent trained_local \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --checkpoint Qwen/Qwen2.5-7B-Instruct \
    --temperature 0.7 \
    --load-in-4bit \
    --n-episodes 50 \
    --out runs/baseline-rollouts/

# Build SFT + DPO + GRPO datasets from the collected trajectories.
python -m training.data.trajectory_to_sft \
    --traj-dir runs/baseline-rollouts/trajectories \
    --output training_data/sft_v1.jsonl \
    --filter "success==True"

python -m training.data.trajectory_to_dpo \
    --traj-dir runs/baseline-rollouts/trajectories \
    --output training_data/dpo_v1.jsonl

python -m training.data.trajectory_to_grpo \
    --output training_data/grpo_prompts.jsonl \
    --seeds-per-task 50

# Failure mining report.
python -m training.data.failure_miner \
    --traj-dir runs/baseline-rollouts/trajectories \
    --output analysis/baseline_failure_report.json \
    --top-k 30
