#!/bin/bash
#SBATCH --job-name=rideshare-dpo
#SBATCH --partition=a100-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --time=08:00:00
#SBATCH --output=logs/dpo-%j.out
#SBATCH --error=logs/dpo-%j.err

# Recipe 2: DPO from preference pairs, building on the SFT checkpoint.

set -euo pipefail

module purge
module load python/3.12 cuda/12.4

cd "${PROJECT_ROOT:-/work/users/d/h/${USER}/rideshare-gym}"
source .venv/bin/activate

mkdir -p logs

SFT_CKPT="/work/users/d/h/${USER}/checkpoints/sft_v1"
CKPT_DIR="/work/users/d/h/${USER}/checkpoints/dpo_v1"

python -m training.recipes.recipe_02_dpo \
    --base-model "${SFT_CKPT}" \
    --dataset training_data/dpo_v1.jsonl \
    --output "${CKPT_DIR}" \
    --use-lora \
    --epochs 1 \
    --learning-rate 5e-6 \
    --beta 0.1

python eval/run.py \
    --tasks "rideshare/*" \
    --agent trained_local \
    --checkpoint "${CKPT_DIR}" \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --temperature 0.0 \
    --n-episodes 5 \
    --out runs/dpo_v1_eval/

echo "DPO scorecard:"
cat runs/dpo_v1_eval/scorecard.json | python -m json.tool
