#!/bin/bash
#SBATCH --job-name=rideshare-sft
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=logs/sft-%j.out
#SBATCH --error=logs/sft-%j.err

# Recipe 1: Rejection-sampling SFT on Qwen2.5-7B-Instruct.

set -euo pipefail

module purge
module load python/3.12 cuda/12.4

cd "${PROJECT_ROOT:-/work/users/d/h/${USER}/rideshare-gym}"
source .venv/bin/activate

mkdir -p logs

CKPT_DIR="/work/users/d/h/${USER}/checkpoints/sft_v1"

python -m training.recipes.recipe_01_sft \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --dataset training_data/sft_v1.jsonl \
    --output "${CKPT_DIR}" \
    --use-lora \
    --load-in-4bit \
    --epochs 3 \
    --batch-size 1 \
    --grad-accum 8 \
    --learning-rate 1e-4 \
    --max-seq-length 2048

# Quick eval the SFT checkpoint.
python eval/run.py \
    --tasks "rideshare/*" \
    --agent trained_local \
    --checkpoint "${CKPT_DIR}" \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --temperature 0.0 \
    --n-episodes 5 \
    --out runs/sft_v1_eval/

echo "SFT scorecard:"
cat runs/sft_v1_eval/scorecard.json | python -m json.tool
