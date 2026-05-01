"""Training infrastructure for rideshare-gym.

Three escalating recipes:
  1. recipe_01_sft.py   — Rejection-sampling SFT (cheapest, biggest first gain)
  2. recipe_02_dpo.py   — DPO with success/fail preference pairs
  3. recipe_03_grpo.py  — GRPO with verifiable rewards (state of the art)

All recipes consume trajectories produced by `eval/run.py` against the same gym.
The verifier IS the reward function — see `reward.py`.
"""
