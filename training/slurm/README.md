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

## The pipeline

```bash
# 0. Baseline measurement — what's our starting point?
sbatch training/slurm/submit_baseline_eval.sbatch

# 1. Collect ~600 trajectories at temperature 0.7 (more passes = more data).
sbatch training/slurm/submit_collect_rollouts.sbatch

# 2. Recipe 1 — SFT.
sbatch training/slurm/submit_sft.sbatch

# 3. Recipe 2 — DPO from SFT checkpoint.
sbatch training/slurm/submit_dpo.sbatch

# 4. Recipe 3 — GRPO from DPO checkpoint (longest job).
sbatch training/slurm/submit_grpo.sbatch
```

The GRPO script also produces the final comparison chart at
`analysis/training_curves.png`.

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
