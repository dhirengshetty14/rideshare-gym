#!/bin/bash
#SBATCH --job-name=rideshare-baseline
#SBATCH --partition=a100-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=logs/baseline-%j.out
#SBATCH --error=logs/baseline-%j.err

# Step 0 — measure baseline Qwen2.5-7B-Instruct success rate on rideshare-gym.

set -euo pipefail

module purge
module load python/3.12 cuda/12.4

cd "${PROJECT_ROOT:-/work/users/d/h/${USER}/rideshare-gym}"
source .venv/bin/activate

mkdir -p logs runs

python eval/run.py \
    --tasks "rideshare/*" \
    --agent trained_local \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --checkpoint Qwen/Qwen2.5-7B-Instruct \
    --temperature 0.0 \
    --load-in-4bit \
    --n-episodes 5 \
    --out runs/baseline-qwen7b/

echo "Baseline scorecard:"
cat runs/baseline-qwen7b/scorecard.json | python -m json.tool
