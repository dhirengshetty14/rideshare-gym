# SLURM scripts for UNC Longleaf

The full pipeline runs as 5 sequential SLURM jobs. Submit them in order; each
depends on the previous one's checkpoint.

## Setup (one time)

```bash
# On Longleaf login node:
ssh longleaf.unc.edu
cd /work/users/d/h/$USER          # your scratch space
git clone https://github.com/dhirengshetty14/rideshare-gym.git
cd rideshare-gym

# Build venv (use Longleaf's CUDA-aware Python).
module load python/3.12 cuda/12.4
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,training]"

# Set wandb (optional, but you want this for training curves).
echo "export WANDB_API_KEY=..." >> ~/.bashrc
echo "export PROJECT_ROOT=/work/users/d/h/$USER/rideshare-gym" >> ~/.bashrc
source ~/.bashrc

mkdir -p logs runs training_data analysis
```

## The pipeline (with statistics)

There are two ways to run it:

**A. Hands-off chain** (recommended) — one submission, jobs chain via SLURM dependencies:

```bash
bash training/slurm/submit_full_pipeline_with_stats.sbatch
```

**B. Manual** — submit each phase yourself in order:

```bash
# 0. Baseline stats (50 seeds × 12 tasks = 600 episodes, paired with later runs).
sbatch training/slurm/submit_baseline_stats.sbatch

# 1. Collect ~600 rollouts at temp 0.7 + build SFT/DPO/GRPO datasets.
sbatch training/slurm/submit_collect_rollouts.sbatch

# 2. SFT.
sbatch training/slurm/submit_sft.sbatch

# 3. Stats on SFT checkpoint (paired against baseline).
TRAINED_CKPT=/work/users/d/h/$USER/checkpoints/sft_v1 STAGE_NAME=sft \
    sbatch training/slurm/submit_trained_stats.sbatch

# 4. DPO from SFT checkpoint.
sbatch training/slurm/submit_dpo.sbatch

# 5. Stats on DPO checkpoint.
TRAINED_CKPT=/work/users/d/h/$USER/checkpoints/dpo_v1 STAGE_NAME=dpo \
    sbatch training/slurm/submit_trained_stats.sbatch

# 6. GRPO from DPO checkpoint.
sbatch training/slurm/submit_grpo.sbatch

# 7. Stats on the final GRPO checkpoint + cross-stage comparison report.
TRAINED_CKPT=/work/users/d/h/$USER/checkpoints/grpo_v1 STAGE_NAME=grpo \
    sbatch training/slurm/submit_trained_stats.sbatch
```

The final stats job produces:
- `analysis/baseline_stats.json`, `analysis/sft_stats.json`, `analysis/dpo_stats.json`, `analysis/grpo_stats.json`
- `analysis/comparison_baseline_vs_grpo.json` — paired McNemar, Cohen's h, per-task deltas with bootstrap CIs
- `analysis/report_baseline_vs_grpo.md` — **the deliverable Markdown report for the mentor**

## The statistical comparison

What makes the before/after comparison rigorous (not just "70% vs 85%"):

- **Paired** McNemar's test on the same `(task, seed)` pairs (not pooled proportions)
- **Bootstrap 95% CIs** on every delta
- **Cohen's h** effect size on success-rate differences (negligible / small / medium / large)
- **Paired Wilcoxon signed-rank** test on per-episode reward deltas
- **Failure-mode shift** — chi-square on the goal_incomplete / wrong_tool / wrong_args / crashed distribution before vs after
- **Tool-call quality** — wrong_args_rate and unknown_tool_rate before vs after, isolating whether training fixed format errors specifically

See `analysis/STATS_PLAN.md` for the full statistical rationale.

## Useful flags

```bash
squeue -u $USER                # see your jobs
scontrol show job <jobid>      # job details
scancel <jobid>                # cancel
sacct -u $USER --starttime=now-2days   # past jobs
```

## Expected timeline

| Step | Time | GPUs |
|---|---|---|
| Baseline eval | ~1h | 1× A100 |
| Rollout collection | ~6-12h | 1× A100 |
| SFT | ~3-6h | 1× A100 |
| DPO | ~4-8h | 1× A100 |
| GRPO | ~24-48h | 2× A100 |
| **Total wall-clock** | **~3-4 days** (with queue waits longer) | |

If A100s are queued up, fall back to L40-GPU (`--partition=l40-gpu --gres=gpu:l40:1`)
which is more available and works for everything except GRPO.

## Troubleshooting

- **OOM on SFT**: drop batch_size to 1 and bump grad_accum to 8. If still OOM,
  add `--load-in-4bit` to the recipe call (QLoRA mode).
- **Tokenizer error on chat template**: pin `transformers>=4.45.0` for native
  Qwen tool-calling.
- **wandb auth fail**: `wandb login` once on the login node before sbatch.
