"""Recipe 1 — Rejection-sampling SFT (RFT).

Take a JSONL of (messages, completion, tools) examples produced by
`training/data/trajectory_to_sft.py`, render each through the model's chat
template, and fine-tune via TRL's SFTTrainer.

This is the cheapest recipe and typically delivers 60-80% of the total
improvement vs the baseline. Run this FIRST.

Usage on Longleaf (via SLURM):
    sbatch training/slurm/submit_sft.sbatch

Or directly:
    python -m training.recipes.recipe_01_sft \
        --base-model Qwen/Qwen2.5-7B-Instruct \
        --dataset training_data/sft_v1.jsonl \
        --output /work/users/d/h/$USER/checkpoints/sft_v1 \
        --use-lora --epochs 3
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


def format_example(example: dict[str, Any], tokenizer) -> dict[str, str]:
    """Render one (messages, completion) pair as a single text string the
    SFTTrainer can train on. The completion is appended to the chat history;
    the trainer masks the prompt portion of loss."""
    messages = example["messages"] + [example["completion"]]
    tools = example.get("tools") or None
    try:
        text = tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
    return {"text": text}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dataset", required=True, type=Path,
                     help="JSONL produced by trajectory_to_sft.py")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--learning-rate", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--max-seq-length", type=int, default=4096)
    ap.add_argument("--use-lora", action="store_true",
                     help="Recommended; full FT also works on A100 80GB.")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--load-in-4bit", action="store_true",
                     help="QLoRA. Use on smaller GPUs.")
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--save-steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--wandb-project", default="rideshare-gym-sft")
    ap.add_argument("--wandb-run-name", default=None)
    args = ap.parse_args()

    # Lazy imports — these are heavy and only needed at training time.
    import torch
    from datasets import Dataset
    from transformers import AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    if args.wandb_project and os.environ.get("WANDB_API_KEY"):
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)

    print(f"Loading dataset from {args.dataset} ...")
    raw = load_jsonl(args.dataset)
    print(f"  {len(raw)} examples")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Formatting examples through chat template ...")
    formatted = [format_example(ex, tokenizer) for ex in raw]
    dataset = Dataset.from_list(formatted)

    # Build model loading kwargs FIRST. In newer TRL (>=0.12) these go on
    # SFTConfig (via model_init_kwargs), not on SFTTrainer.
    model_kwargs: dict[str, Any] = {
        "torch_dtype": torch.bfloat16,
    }
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    sft_config = SFTConfig(
        output_dir=str(args.output),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        max_length=args.max_seq_length,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
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

    trainer = SFTTrainer(
        model=args.base_model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config,
        tokenizer=tokenizer,
    )
    print("Starting SFT ...")
    trainer.train()
    print(f"Saving to {args.output} ...")
    trainer.save_model(str(args.output))
    tokenizer.save_pretrained(str(args.output))
    print("Done.")


if __name__ == "__main__":
    main()
