"""Plot training-stage progress as a single chart: baseline → SFT → DPO → GRPO.

This is the deliverable image for the mentor — one chart showing the model
improving across the three training recipes on each of the 12 tasks, with
the gym's KPI thresholds drawn as horizontal reference lines where relevant.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison", required=True, type=Path,
                     help="Output of eval/compare_checkpoints.py --output")
    ap.add_argument("--out", type=Path, default=Path("analysis/training_curves.png"))
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = load(args.comparison)
    stages: list[str] = data["stages"]
    per_task: dict[str, dict[str, float]] = data["per_task_success_rates"]

    tasks = sorted(per_task.keys())

    fig, ax = plt.subplots(figsize=(11, 7))
    x = list(range(len(stages)))
    cmap = plt.get_cmap("tab20")
    for i, tid in enumerate(tasks):
        y = [per_task[tid].get(s, 0.0) for s in stages]
        short = tid.replace("rideshare/", "")
        ax.plot(x, y, marker="o", label=short, color=cmap(i % 20))

    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in stages])
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Success rate")
    ax.set_xlabel("Training stage")
    ax.set_title("Per-task success rate across training stages")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8,
              frameon=False)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
