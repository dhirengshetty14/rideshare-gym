# Training infrastructure for rideshare-gym

This module turns rideshare-gym from an *eval harness* into a *training factory*.
The same gym, same verifiers, same tasks — but now we use them to **improve a
model**, not just measure one.

## The pipeline in one picture

```
                    ┌─────────────────────┐
                    │  Baseline model     │      Qwen2.5-7B-Instruct
                    │  (0% trained)       │
                    └──────────┬──────────┘
                               │
                  collect rollouts (temperature=0.7)
                               │
                    ┌──────────▼──────────┐
                    │ ~600 trajectories   │      mix of success + fail
                    └──────────┬──────────┘
                               │
            ┌──────────────────┼──────────────────┐
            │                  │                  │
            ▼                  ▼                  ▼
      SFT examples       DPO pairs         GRPO prompts
   (only successes)   (success vs fail)   (just (task, seed))
            │                  │                  │
            ▼                  ▼                  ▼
     ┌─────────────┐   ┌─────────────┐   ┌─────────────────────┐
     │  Recipe 1   │   │  Recipe 2   │   │      Recipe 3       │
     │     SFT     │──>│    DPO      │──>│       GRPO          │
     │ (3-4 hours) │   │ (4-6 hours) │   │   (24-48 hours)     │
     │             │   │             │   │  rewards live       │
     │             │   │             │   │  from gym verifier  │
     └──────┬──────┘   └──────┬──────┘   └──────────┬──────────┘
            │                  │                    │
            ▼                  ▼                    ▼
   Eval on rideshare-gym → Eval → Eval → Final scorecard
                                          + before/after chart
```

## What goes where

```
training/
├── data/                                # Trajectory → training-data converters
│   ├── trajectory_to_sft.py             # Successful trajectories → (msgs, completion)
│   ├── trajectory_to_dpo.py             # success+fail pairs → (chosen, rejected)
│   ├── trajectory_to_grpo.py            # task list → prompt-only dataset
│   └── failure_miner.py                 # Cluster failures by pattern
│
├── recipes/                             # TRL-based training scripts
│   ├── recipe_01_sft.py                 # SFTTrainer (rejection-sampling SFT)
│   ├── recipe_02_dpo.py                 # DPOTrainer
│   └── recipe_03_grpo.py                # GRPOTrainer with our verifier as reward
│
├── reward.py                            # Verifier-as-reward function for GRPO
├── eval_callback.py                     # Periodic mid-training eval → wandb
└── slurm/                               # SLURM submission scripts for Longleaf
    ├── submit_baseline_eval.sbatch
    ├── submit_collect_rollouts.sbatch
    ├── submit_sft.sbatch
    ├── submit_dpo.sbatch
    └── submit_grpo.sbatch
```

## Key insight — the gym verifier IS the reward function

This is the design principle behind the whole pipeline. Look at `reward.py`:

```python
def gym_step_reward(*, task_id, seed, completion):
    """Spin a fresh gym episode, parse the model's completion as a tool call,
    step the env once, return the verifier's scalar reward."""
    ...
```

That single function is what GRPO uses to score every rollout. The gym's
verifier — the same code that decides "did the agent solve M2 fraud_ring?"
during eval — is now the reward signal during training. No learned reward
model. No human labelling. Just verifiable rewards from the simulator.

This is the **RLVR (RL with Verifiable Rewards)** pattern from Tülu 3 and
DeepSeek-R1. The whole reason the gym is useful for training is that we can
mechanically score any (task, seed, completion) tuple.

## How to use it

End-to-end on Longleaf, in order:

```bash
sbatch training/slurm/submit_baseline_eval.sbatch        # 1h
sbatch training/slurm/submit_collect_rollouts.sbatch     # 6-12h
sbatch training/slurm/submit_sft.sbatch                  # 3-6h
sbatch training/slurm/submit_dpo.sbatch                  # 4-8h
sbatch training/slurm/submit_grpo.sbatch                 # 24-48h
```

Total wall-clock: ~3-4 days. Total GPU-hours: roughly 60-100 A100-hours.

The final SLURM job emits the deliverable: `analysis/training_curves.png`
showing per-task success rate at each stage.

## Inspecting failures (the mentor's ask)

After collecting rollouts:

```bash
streamlit run analysis/failure_dashboard.py -- --traj-dir runs/baseline-rollouts/trajectories
```

A live dashboard listing the top 20 failure clusters by `(task, last_tool,
first_failed_assertion)`. Click into a cluster to see exact failing
trajectories. Use this to:

1. Understand WHY the baseline fails on specific tasks
2. Verify the trained model fixed those failures (re-run after SFT/DPO/GRPO)
3. Spot model-specific issues (e.g. "always forgets to call send_to_driver
   after adjust_driver_payout")

## References — what each recipe is based on

- **Recipe 1 (SFT)**: SWE-Gym paper, Tülu 3 SFT stage, standard rejection-sampling fine-tuning
- **Recipe 2 (DPO)**: Rafailov et al. 2023, Tülu 3 preference-tuning stage
- **Recipe 3 (GRPO)**: DeepSeek-R1 paper, HuggingFace open-r1 reproduction, Tülu 3 RLVR stage

All three trainers come from HuggingFace TRL. We don't reinvent any of this — we
just plug our gym in as the env / reward source.

## What success looks like

| Stage | Expected success rate | Improvement over baseline |
|---|---|---|
| Baseline | 30-50% | — |
| + SFT | 50-65% | +15-25 pp |
| + DPO | 55-70% | +25-30 pp |
| + GRPO | 60-75% | +30-45 pp |

Specifically, the hardest tasks (H1 realtime_dispatch, H3 coordinated_fraud)
should go from baseline ~5% to trained ≥30%. This is the artifact for
demonstrating that the gym actually improves agents, not just measures them.
