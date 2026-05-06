"""Recipe 3 — GRPO with verifiable rewards (the DeepSeek-R1 / Tülu 3 RLVR recipe).

Takes a prompt-only dataset (from `training/data/trajectory_to_grpo.py`),
generates N rollouts per prompt, scores each via the gym verifier, computes
group-relative advantages, and updates the policy. Ends with the most
RL-tuned checkpoint of the pipeline.

Builds on the DPO checkpoint (--base-model). Most expensive recipe — needs
2-4× A100 80GB and 24-48h on Longleaf.

Usage:
    sbatch training/slurm/submit_grpo.sbatch

Or directly:
    python -m training.recipes.recipe_03_grpo \
        --base-model /work/users/d/h/$USER/checkpoints/dpo_v1 \
        --prompts-dataset training_data/grpo_prompts.jsonl \
        --output /work/users/d/h/$USER/checkpoints/grpo_v1 \
        --use-lora --num-generations 8
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True,
                     help="Path to DPO (or SFT) checkpoint to start GRPO from")
    ap.add_argument("--prompts-dataset", required=True, type=Path,
                     help="JSONL of {prompt: messages, task_id, seed}")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--learning-rate", type=float, default=3e-6,
                     help="DeepSeek-R1 default")
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--num-generations", type=int, default=8,
                     help="Rollouts per prompt; larger = better advantage est.")
    ap.add_argument("--max-prompt-length", type=int, default=2048)
    ap.add_argument("--max-completion-length", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--kl-coef", type=float, default=0.001,
                     help="KL penalty against reference policy.")
    ap.add_argument("--use-lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--logging-steps", type=int, default=5)
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wandb-project", default="rideshare-gym-grpo")
    ap.add_argument("--wandb-run-name", default=None)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    from training.reward import grpo_reward_fn

    if args.wandb_project and os.environ.get("WANDB_API_KEY"):
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    print(f"Loading {args.prompts_dataset} ...")
    raw = load_jsonl(args.prompts_dataset)
    print(f"  {len(raw)} prompts")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Render messages → prompt strings via the model's chat template.
    rendered: list[dict[str, Any]] = []
    for ex in raw:
        try:
            prompt = tokenizer.apply_chat_template(
                ex["prompt"], tools=ex.get("tools") or None,
                tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            prompt = tokenizer.apply_chat_template(
                ex["prompt"], tokenize=False, add_generation_prompt=True,
            )
        rendered.append({
            "prompt": prompt,
            "task_id": ex["task_id"],
            "seed": ex["seed"],
        })
    dataset = Dataset.from_list(rendered)

    # Build model loading kwargs FIRST. In newer TRL these go on
    # GRPOConfig (via model_init_kwargs), not on GRPOTrainer.
    model_kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    grpo_config = GRPOConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        beta=args.kl_coef,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        seed=args.seed,
        bf16=torch.cuda.is_available(),
        report_to=("wandb" if os.environ.get("WANDB_API_KEY") else "none"),
        run_name=args.wandb_run_name or args.output.name,
        gradient_checkpointing=True,
        warmup_ratio=0.05,
        model_init_kwargs=model_kwargs,
    )

    peft_config = None
    if args.use_lora:
        from peft import LoraConfig
        peft_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                             "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )

    trainer = GRPOTrainer(
        model=args.base_model,
        reward_funcs=grpo_reward_fn,
        args=grpo_config,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    print("Starting GRPO ...")
    trainer.train()
    print(f"Saving to {args.output} ...")
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print("Done.")


if __name__ == "__main__":
    main()
