"""Recipe 2 — DPO with success/fail preference pairs.

Takes preference pairs from `training/data/trajectory_to_dpo.py` and trains
the model to prefer the chosen action over the rejected one. Builds on the
SFT checkpoint (--base-model).

Usage:
    sbatch training/slurm/submit_dpo.sbatch

Or directly:
    python -m training.recipes.recipe_02_dpo \
        --base-model /work/users/d/h/$USER/checkpoints/sft_v1 \
        --dataset training_data/dpo_v1.jsonl \
        --output /work/users/d/h/$USER/checkpoints/dpo_v1 \
        --use-lora --epochs 1
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


def format_pair(example: dict[str, Any], tokenizer) -> dict[str, str]:
    """Render a (messages, chosen, rejected) tuple into TRL-DPO format —
    which expects strings for prompt / chosen / rejected."""
    messages = example["messages"]
    tools = example.get("tools") or None
    try:
        prompt = tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )

    def render_response(resp: dict[str, Any]) -> str:
        if isinstance(resp.get("content"), list):
            return json.dumps(resp["content"], default=str)
        if "tool_calls" in resp:
            return (resp.get("content") or "") + "\n" + json.dumps(
                resp["tool_calls"], default=str)
        return resp.get("content") or json.dumps(resp, default=str)

    return {
        "prompt": prompt,
        "chosen": render_response(example["chosen"]),
        "rejected": render_response(example["rejected"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True,
                     help="Path to SFT checkpoint (or HF id) to start DPO from")
    ap.add_argument("--dataset", required=True, type=Path,
                     help="JSONL produced by trajectory_to_dpo.py")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--learning-rate", type=float, default=5e-6)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=4096)
    ap.add_argument("--max-prompt-length", type=int, default=2048)
    ap.add_argument("--beta", type=float, default=0.1,
                     help="DPO temperature; 0.1 is the canonical default.")
    ap.add_argument("--use-lora", action="store_true")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--load-in-4bit", action="store_true")
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wandb-project", default="rideshare-gym-dpo")
    ap.add_argument("--wandb-run-name", default=None)
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    if args.wandb_project and os.environ.get("WANDB_API_KEY"):
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    print(f"Loading {args.dataset} ...")
    raw = load_jsonl(args.dataset)
    print(f"  {len(raw)} preference pairs")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    formatted = [format_pair(ex, tokenizer) for ex in raw]
    dataset = Dataset.from_list(formatted)

    # Build model loading kwargs FIRST. In newer TRL these go on
    # DPOConfig (via model_init_kwargs), not on DPOTrainer.
    model_kwargs: dict[str, Any] = {"torch_dtype": torch.bfloat16}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    dpo_config = DPOConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
        beta=args.beta,
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

    trainer = DPOTrainer(
        model=args.base_model,
        ref_model=None,                # TRL clones for ref policy automatically
        args=dpo_config,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
    )
    print("Starting DPO ...")
    trainer.train()
    print(f"Saving to {args.output} ...")
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print("Done.")


if __name__ == "__main__":
    main()
