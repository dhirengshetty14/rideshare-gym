#!/bin/bash
#SBATCH --job-name=rideshare-grpo
#SBATCH --partition=a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:nvidia_a100-pcie-40gb:1
#SBATCH --mem=120G
#SBATCH --time=48:00:00
#SBATCH --output=logs/grpo-%j.out
#SBATCH --error=logs/grpo-%j.err

# Recipe 3: GRPO with verifiable rewards (DeepSeek-R1 / Tülu 3 RLVR style).
# Most expensive recipe — needs ~24-48h on 2x A100 80GB.

set -euo pipefail

module purge
module load python/3.12 cuda/12.4

cd "${PROJECT_ROOT:-/work/users/d/h/${USER}/rideshare-gym}"
source .venv/bin/activate

mkdir -p logs

DPO_CKPT="/work/users/d/h/${USER}/checkpoints/dpo_v1"
CKPT_DIR="/work/users/d/h/${USER}/checkpoints/grpo_v1"

python -m training.recipes.recipe_03_grpo \
    --base-model "${DPO_CKPT}" \
    --prompts-dataset training_data/grpo_prompts.jsonl \
    --output "${CKPT_DIR}" \
    --use-lora \
    --num-generations 8 \
    --batch-size 1 \
    --grad-accum 8 \
    --learning-rate 3e-6 \
    --kl-coef 0.001 \
    --max-completion-length 512 \
    --temperature 1.0 \
    --epochs 1

python eval/run.py \
    --tasks "rideshare/*" \
    --agent trained_local \
    --checkpoint "${CKPT_DIR}" \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --temperature 0.0 \
    --n-episodes 5 \
    --out runs/grpo_v1_eval/

echo "GRPO scorecard:"
cat runs/grpo_v1_eval/scorecard.json | python -m json.tool

# Final cross-stage comparison + chart.
python eval/compare_checkpoints.py \
    --baseline runs/baseline-qwen7b/scorecard.json \
    --sft      runs/sft_v1_eval/scorecard.json \
    --dpo      runs/dpo_v1_eval/scorecard.json \
    --grpo     runs/grpo_v1_eval/scorecard.json \
    --output analysis/training_comparison.json

python analysis/before_after_curves.py \
    --comparison analysis/training_comparison.json \
    --out analysis/training_curves.png
