"""Generate a human-readable Markdown report from a comparison.json.

This is the artefact you share with your mentor — one file with the headline
number, per-task table, statistical tests, acceptance criteria, and a few
key plots.

Usage:
    python -m analysis.format_report \
        --comparison analysis/comparison_baseline_vs_grpo.json \
        --output analysis/report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def fmt_pct(x: float) -> str:
    return f"{x:.1%}" if isinstance(x, (int, float)) else "?"


def fmt_signed_pp(x: float) -> str:
    pp = x * 100
    sign = "+" if pp >= 0 else ""
    return f"{sign}{pp:.1f} pp"


def fmt_signed(x: float, fmt: str = ".3f") -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}{format(x, fmt)}"


def render_per_task_detailed(
    comparison: dict[str, Any],
    before_stats: dict[str, Any] | None,
    after_stats: dict[str, Any] | None,
) -> str:
    """One subsection per task with the full per-task breakdown — error
    categories, tool-call quality, mean reward + steps. This is the
    "stats for all the tasks" the mentor asked for, in expanded form."""
    out = ["## Detailed per-task breakdown\n"]
    for tid in sorted(comparison["per_task"]):
        c = comparison["per_task"][tid]
        b_stats = (before_stats or {}).get("per_task", {}).get(tid, {})
        a_stats = (after_stats or {}).get("per_task", {}).get(tid, {})

        short = tid.replace("rideshare/", "")
        out.append(f"### `{short}` ({c['difficulty']})")
        out.append("")
        out.append(f"- **Success rate:** "
                    f"{fmt_pct(c['success_rate_before'])} → "
                    f"{fmt_pct(c['success_rate_after'])} "
                    f"(**{fmt_signed_pp(c['delta'])}**, "
                    f"95% CI [{fmt_signed_pp(c['delta_ci_95'][0])}, "
                    f"{fmt_signed_pp(c['delta_ci_95'][1])}], "
                    f"Cohen's h = {c['cohens_h']:+.2f})")
        out.append(f"- **Episodes paired:** {c['n_pairs']} "
                    f"(improved: {c['improved_episodes']}, "
                    f"regressed: {c['regressed_episodes']}, "
                    f"unchanged-pass: {c['unchanged_pass']}, "
                    f"unchanged-fail: {c['unchanged_fail']})")

        if b_stats and a_stats:
            out.append(f"- **Mean reward:** "
                        f"{b_stats.get('mean_reward', 0):.3f} → "
                        f"{a_stats.get('mean_reward', 0):.3f} "
                        f"(Δ = "
                        f"{fmt_signed(a_stats.get('mean_reward', 0) - b_stats.get('mean_reward', 0))})")
            out.append(f"- **Mean steps:** "
                        f"{b_stats.get('mean_steps', 0):.1f} → "
                        f"{a_stats.get('mean_steps', 0):.1f} "
                        f"(Δ = "
                        f"{fmt_signed(a_stats.get('mean_steps', 0) - b_stats.get('mean_steps', 0), '.1f')})")

            be = b_stats.get("error_breakdown", {})
            ae = a_stats.get("error_breakdown", {})
            if be or ae:
                out.append("- **Error breakdown** (count of failures by category):")
                out.append("")
                out.append("    | Category | Before | After | Delta |")
                out.append("    |---|---|---|---|")
                for cat in ("goal_incomplete", "wrong_tool", "wrong_args", "crashed"):
                    bv = be.get(cat, 0)
                    av = ae.get(cat, 0)
                    out.append(f"    | `{cat}` | {bv} | {av} | {av-bv:+d} |")

            out.append("- **Tool-call rates on this task:**")
            for k, label in [
                ("tool_call_validity_rate", "completed without error"),
                ("wrong_args_rate", "JSON-Schema arg errors"),
                ("unknown_tool_rate", "calls to undefined tools"),
            ]:
                bv = b_stats.get(k, 0)
                av = a_stats.get(k, 0)
                out.append(f"    - `{k}` ({label}): "
                            f"{fmt_pct(bv)} → {fmt_pct(av)} "
                            f"({fmt_signed_pp(av-bv)})")
        out.append("")
    return "\n".join(out) + "\n"


def render_overall(comparison: dict[str, Any]) -> str:
    ov = comparison["overall"]
    h = ov["cohens_h"]
    h_label = comparison["overall"].get("cohens_h_label", "")
    p = ov["mcnemar"]["p_value"]
    n = ov["n_pairs"]
    sig = ("statistically significant" if p < 0.05
            else f"NOT statistically significant (p={p:.3f})")
    return f"""## Headline

After training, overall success on the rideshare-gym benchmark went from
**{fmt_pct(ov['success_rate_before'])}** to
**{fmt_pct(ov['success_rate_after'])}** — a delta of
**{fmt_signed_pp(ov['success_rate_delta'])}** ({sig}, McNemar p = {p:.4f}).
Cohen's h = {h:+.3f} ({h_label} effect size).

Paired on the same `(task, seed)` pairs, n = {n} episodes per stage.

| | Before | After | Delta |
|---|---|---|---|
| Success rate | {fmt_pct(ov['success_rate_before'])} | {fmt_pct(ov['success_rate_after'])} | {fmt_signed_pp(ov['success_rate_delta'])} |
| Mean reward | {ov['mean_reward_before']:.3f} | {ov['mean_reward_after']:.3f} | {(ov['mean_reward_after']-ov['mean_reward_before']):+.3f} |
| Mean reward 95% CI on delta | | | [{ov['mean_reward_delta_ci_95'][0]:+.3f}, {ov['mean_reward_delta_ci_95'][1]:+.3f}] |
| Wilcoxon p-value (reward) | | | {ov['wilcoxon_reward']['p_value']:.4f} |
"""


def render_per_task(comparison: dict[str, Any]) -> str:
    rows = ["| Task | Difficulty | Before | After | Delta | 95% CI | Improved | Regressed |",
             "|---|---|---|---|---|---|---|---|"]
    for tid, m in sorted(comparison["per_task"].items()):
        rows.append(
            f"| `{tid.replace('rideshare/', '')}` | {m['difficulty']} | "
            f"{fmt_pct(m['success_rate_before'])} | "
            f"{fmt_pct(m['success_rate_after'])} | "
            f"**{fmt_signed_pp(m['delta'])}** | "
            f"[{m['delta_ci_95'][0]:+.2f}, {m['delta_ci_95'][1]:+.2f}] | "
            f"{m['improved_episodes']} | {m['regressed_episodes']} |"
        )
    return "## Per-task results\n\n" + "\n".join(rows) + "\n"


def render_failure_modes(comparison: dict[str, Any]) -> str:
    fs = comparison["failure_mode_shift"]
    rows = ["| Failure category | Before count | After count | Delta |",
             "|---|---|---|---|"]
    for k in ("goal_incomplete", "wrong_tool", "wrong_args", "crashed"):
        b = fs["before"].get(k, 0)
        a = fs["after"].get(k, 0)
        rows.append(f"| `{k}` | {b} | {a} | {a-b:+d} |")
    return "## Failure-mode shift\n\n" + "\n".join(rows) + "\n"


def render_tool_quality(comparison: dict[str, Any]) -> str:
    tq = comparison["tool_call_quality"]
    rows = ["| Metric | Before | After | Delta |",
             "|---|---|---|---|"]
    for k in ("tool_call_validity_rate", "wrong_args_rate", "unknown_tool_rate"):
        b = tq[k]["before"]
        a = tq[k]["after"]
        rows.append(f"| `{k}` | {fmt_pct(b)} | {fmt_pct(a)} | {fmt_signed_pp(a-b)} |")
    return "## Tool-call quality\n\n" + "\n".join(rows) + "\n"


def render_acceptance(comparison: dict[str, Any]) -> str:
    ac = comparison["acceptance"]
    out = ["## Acceptance criteria\n"]
    for k, v in ac.items():
        if k == "details":
            continue
        mark = "✅" if v else "❌"
        out.append(f"- {mark} **{k}**")
    out.append("")
    out.append("Details:")
    out.append("```json")
    out.append(json.dumps(ac["details"], indent=2, default=str))
    out.append("```")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison", required=True, type=Path)
    ap.add_argument("--before-stats", type=Path, default=None,
                     help="Optional: detailed per-task stats from baseline run "
                          "(produced by analysis/stats.py). When provided, the "
                          "report includes a Detailed per-task breakdown section.")
    ap.add_argument("--after-stats", type=Path, default=None)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--title", default="Rideshare-gym training results")
    args = ap.parse_args()

    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    before_stats = (json.loads(args.before_stats.read_text(encoding="utf-8"))
                     if args.before_stats and args.before_stats.exists() else None)
    after_stats = (json.loads(args.after_stats.read_text(encoding="utf-8"))
                    if args.after_stats and args.after_stats.exists() else None)

    parts = [
        f"# {args.title}",
        f"_Comparison: `{comparison['before_run']}` → `{comparison['after_run']}`_",
        render_overall(comparison),
        render_per_task(comparison),
    ]
    if before_stats and after_stats:
        parts.append(render_per_task_detailed(comparison, before_stats, after_stats))
    parts.extend([
        render_failure_modes(comparison),
        render_tool_quality(comparison),
        render_acceptance(comparison),
    ])
    md = "\n\n".join(parts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
