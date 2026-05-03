#!/bin/bash
#SBATCH --job-name=rideshare-full-pipeline
#SBATCH --partition=a100-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pipeline-%j.out
#SBATCH --error=logs/pipeline-%j.err

# === FULL PIPELINE WITH STATS BOOKENDS ===
# Runs everything in one driver job (uses chain dependencies between phases).
# If you prefer manual control, submit each phase separately — same effect.
#
# Phase plan:
#   0. Baseline stats (50 seeds * 12 tasks = 600 episodes)
#   1. Collect rollouts at T=0.7 + build SFT/DPO/GRPO datasets
#   2. SFT
#   3. SFT eval + stats
#   4. DPO
#   5. DPO eval + stats
#   6. GRPO
#   7. GRPO eval + stats + final report

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/work/users/d/h/${USER}/rideshare-gym}"
SLURM_DIR="${PROJECT_ROOT}/training/slurm"

cd "${PROJECT_ROOT}"

echo "=== Submitting chained pipeline ==="
J0=$(sbatch --parsable "${SLURM_DIR}/submit_baseline_stats.sbatch")
echo "  baseline_stats: ${J0}"

J1=$(sbatch --parsable --dependency=afterok:${J0} \
    "${SLURM_DIR}/submit_collect_rollouts.sbatch")
echo "  collect_rollouts: ${J1}"

J2=$(sbatch --parsable --dependency=afterok:${J1} \
    "${SLURM_DIR}/submit_sft.sbatch")
echo "  sft: ${J2}"

J3=$(sbatch --parsable --dependency=afterok:${J2} \
    --export=ALL,TRAINED_CKPT=/work/users/d/h/${USER}/checkpoints/sft_v1,STAGE_NAME=sft \
    "${SLURM_DIR}/submit_trained_stats.sbatch")
echo "  sft_stats: ${J3}"

J4=$(sbatch --parsable --dependency=afterok:${J3} \
    "${SLURM_DIR}/submit_dpo.sbatch")
echo "  dpo: ${J4}"

J5=$(sbatch --parsable --dependency=afterok:${J4} \
    --export=ALL,TRAINED_CKPT=/work/users/d/h/${USER}/checkpoints/dpo_v1,STAGE_NAME=dpo \
    "${SLURM_DIR}/submit_trained_stats.sbatch")
echo "  dpo_stats: ${J5}"

J6=$(sbatch --parsable --dependency=afterok:${J5} \
    "${SLURM_DIR}/submit_grpo.sbatch")
echo "  grpo: ${J6}"

J7=$(sbatch --parsable --dependency=afterok:${J6} \
    --export=ALL,TRAINED_CKPT=/work/users/d/h/${USER}/checkpoints/grpo_v1,STAGE_NAME=grpo \
    "${SLURM_DIR}/submit_trained_stats.sbatch")
echo "  grpo_stats: ${J7}"

echo
echo "Pipeline submitted. Final report will be at:"
echo "  analysis/report_baseline_vs_grpo.md"
echo
echo "Monitor with: squeue -u \$USER"
