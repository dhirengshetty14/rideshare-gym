"""Read scorecards from multiple eval runs and emit a side-by-side comparison.

Use after running baseline + sft + dpo + grpo evals to see the per-task
training curve.

Usage:
    python eval/compare_checkpoints.py \
        --baseline runs/baseline-qwen7b/scorecard.json \
        --sft runs/sft_v1_eval/scorecard.json \
        --dpo runs/dpo_v1_eval/scorecard.json \
        --grpo runs/grpo_v1_eval/scorecard.json \
        --output analysis/compare.json

The output is both a stdout table and a JSON file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Allow `python eval/compare_checkpoints.py`.
_PROJ_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJ_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJ_ROOT))


def load_scorecard(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def per_task_table(scorecards: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Returns {task_id: {stage: success_rate}} for every task that appears
    in any scorecard."""
    all_tasks: set[str] = set()
    for sc in scorecards.values():
        all_tasks.update(sc.get("tasks", {}).keys())

    out: dict[str, dict[str, float]] = {}
    for tid in sorted(all_tasks):
        out[tid] = {}
        for stage_name, sc in scorecards.items():
            task_metrics = (sc.get("tasks") or {}).get(tid, {})
            out[tid][stage_name] = float(task_metrics.get("success_rate", 0.0))
    return out


def print_table(table: dict[str, dict[str, float]], stage_order: list[str]) -> None:
    width_task = max(len(t) for t in table) if table else 30
    header = " | ".join([f"{'task':<{width_task}}"] + [f"{s:>10}" for s in stage_order])
    print(header)
    print("-" * len(header))
    for tid, stages in table.items():
        row = [f"{tid:<{width_task}}"]
        for s in stage_order:
            v = stages.get(s)
            row.append(f"{v:>9.0%}" if v is not None else f"{'?':>10}")
        print(" | ".join(row))
    print("-" * len(header))


def overall_deltas(scorecards: dict[str, dict[str, Any]],
                    base_stage: str) -> dict[str, float]:
    base_rate = (scorecards[base_stage].get("overall") or {}).get(
        "success_rate", 0.0)
    out: dict[str, float] = {}
    for stage, sc in scorecards.items():
        rate = (sc.get("overall") or {}).get("success_rate", 0.0)
        out[stage] = rate - base_rate
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True, type=Path)
    ap.add_argument("--sft", type=Path, default=None)
    ap.add_argument("--dpo", type=Path, default=None)
    ap.add_argument("--grpo", type=Path, default=None)
    ap.add_argument("--extra", nargs="*", default=[],
                     help="More scorecards as name=path pairs (e.g. dpo_v2=runs/.../scorecard.json)")
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    scorecards: dict[str, dict[str, Any]] = {}
    for stage, path in [
        ("baseline", args.baseline),
        ("sft", args.sft),
        ("dpo", args.dpo),
        ("grpo", args.grpo),
    ]:
        if path is not None:
            scorecards[stage] = load_scorecard(path)
    for kv in args.extra:
        if "=" not in kv:
            continue
        name, path = kv.split("=", 1)
        scorecards[name] = load_scorecard(Path(path))

    if not scorecards:
        sys.exit("no scorecards provided")

    stage_order = list(scorecards.keys())
    table = per_task_table(scorecards)
    print(f"Per-task success rate by training stage")
    print()
    print_table(table, stage_order)
    print()
    deltas = overall_deltas(scorecards, base_stage="baseline")
    print("Overall delta vs baseline:")
    for s, d in deltas.items():
        sign = "+" if d >= 0 else ""
        print(f"  {s:>10}  {sign}{d:.1%}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({
            "stages": stage_order,
            "per_task_success_rates": table,
            "overall_deltas_vs_baseline": deltas,
            "scorecards": scorecards,
        }, indent=2, default=str), encoding="utf-8")
        print(f"\nWrote comparison to {args.output}")


if __name__ == "__main__":
    main()
