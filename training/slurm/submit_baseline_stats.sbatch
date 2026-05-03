#!/bin/bash
#SBATCH --job-name=rideshare-baseline-stats
#SBATCH --partition=a100-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/baseline-stats-%j.out
#SBATCH --error=logs/baseline-stats-%j.err

# === STATS RUN: BASELINE ===
# 12 tasks * 50 seeds * 1 episode = 600 paired episodes at temperature 0.0.
# This is the reference distribution we compare against after training.

set -euo pipefail

module purge
module load python/3.12 cuda/12.4

cd "${PROJECT_ROOT:-/work/users/d/h/${USER}/rideshare-gym}"
source .venv/bin/activate

mkdir -p logs runs analysis

# Deterministic, paired evaluation. Seeds 0..49 across all 12 tasks.
python eval/run.py \
    --tasks "rideshare/*" \
    --agent trained_local \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --checkpoint Qwen/Qwen2.5-7B-Instruct \
    --temperature 0.0 \
    --load-in-4bit \
    --n-episodes 50 \
    --seed 0 \
    --out runs/stats-baseline/

# Compute the rich stats defined in analysis/STATS_PLAN.md.
python -m analysis.stats \
    --run-dir runs/stats-baseline \
    --output analysis/baseline_stats.json \
    --summary

echo "Baseline stats committed to analysis/baseline_stats.json"
echo "Now submit submit_collect_rollouts.sbatch then the training pipeline."
