#!/bin/bash
#SBATCH --job-name=rideshare-trained-stats
#SBATCH --partition=l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/trained-stats-%j.out
#SBATCH --error=logs/trained-stats-%j.err

# === STATS RUN: AFTER TRAINING ===
# Same 600 (task, seed) pairs as the baseline run, evaluated against the
# final GRPO checkpoint. Produces the paired numbers we compare in
# analysis/compare_runs.py.

set -euo pipefail

module purge
module load python/3.12 cuda/12.4

cd "${PROJECT_ROOT:-/work/users/d/h/${USER}/rideshare-gym}"
source .venv/bin/activate

mkdir -p logs runs analysis

# Default to GRPO checkpoint; override via env var to evaluate intermediate stages.
TRAINED_CKPT="${TRAINED_CKPT:-/work/users/d/h/${USER}/checkpoints/grpo_v1}"
STAGE_NAME="${STAGE_NAME:-grpo}"

echo "Evaluating checkpoint: ${TRAINED_CKPT}"
echo "Stage label: ${STAGE_NAME}"

python eval/run.py \
    --tasks "rideshare/*" \
    --agent trained_local \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --checkpoint "${TRAINED_CKPT}" \
    --temperature 0.0 \
    --n-episodes 50 \
    --seed 0 \
    --out "runs/stats-${STAGE_NAME}/"

python -m analysis.stats \
    --run-dir "runs/stats-${STAGE_NAME}" \
    --output "analysis/${STAGE_NAME}_stats.json" \
    --summary

# Paired comparison vs baseline + markdown report.
python -m analysis.compare_runs \
    --before runs/stats-baseline \
    --after  "runs/stats-${STAGE_NAME}" \
    --output "analysis/comparison_baseline_vs_${STAGE_NAME}.json" \
    --summary

python -m analysis.format_report \
    --comparison "analysis/comparison_baseline_vs_${STAGE_NAME}.json" \
    --before-stats analysis/baseline_stats.json \
    --after-stats "analysis/${STAGE_NAME}_stats.json" \
    --output "analysis/report_baseline_vs_${STAGE_NAME}.md" \
    --title "Baseline vs ${STAGE_NAME^^} — Qwen2.5-7B-Instruct on rideshare-gym"

echo
echo "All artifacts written to analysis/:"
ls -la analysis/${STAGE_NAME}_stats.json analysis/comparison_baseline_vs_${STAGE_NAME}.* 2>/dev/null
