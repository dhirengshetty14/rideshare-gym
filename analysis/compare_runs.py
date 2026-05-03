"""Paired statistical comparison between two run directories (e.g. baseline
and trained checkpoint). Outputs all the §5 numbers from STATS_PLAN.md.

Critical assumption: BOTH runs used the same (task, seed) pairs. We pair
trajectories on (task_id, seed) and compute paired tests on the deltas.

Usage:
    python -m analysis.compare_runs \
        --before runs/baseline-qwen7b \
        --after  runs/grpo_v1_eval \
        --output analysis/comparison_baseline_vs_grpo.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from analysis.stats import (
    _difficulty_of,
    bootstrap_ci,
    compute_run_stats,
    trajectory_features,
)


# --------------------------------------------------------------------------- #
# Pairing
# --------------------------------------------------------------------------- #

def pair_features(
    before: list[dict[str, Any]], after: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Match features on (task_id, seed). Drops unpaired."""
    by_key = {(f["task_id"], f["seed"]): f for f in before}
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for fa in after:
        key = (fa["task_id"], fa["seed"])
        if key in by_key:
            pairs.append((by_key[key], fa))
    return pairs


# --------------------------------------------------------------------------- #
# Paired statistical tests
# --------------------------------------------------------------------------- #

def mcnemar_test(b_pass: int, b_fail: int, c_pass: int, c_fail: int,
                  *, exact_threshold: int = 25) -> dict[str, float]:
    """McNemar's test on paired binary outcomes.

    The four counts come from the 2×2 contingency table:

                    AFTER pass    AFTER fail
    BEFORE pass         a            b
    BEFORE fail         c            d

    `b` = before-pass, after-fail (regression)
    `c` = before-fail, after-pass (improvement)

    H_0: b == c. Statistic = (|b - c| - 1)^2 / (b + c).
    """
    b, c = b_fail, c_pass    # naming follows stats convention
    if b + c == 0:
        return {"b": b, "c": c, "statistic": 0.0, "p_value": 1.0,
                "method": "mcnemar (no discordant pairs)"}

    if b + c < exact_threshold:
        # Exact binomial test on min(b, c) ~ Binomial(b+c, 0.5).
        from math import comb
        n = b + c
        k = min(b, c)
        # Two-sided p-value: 2 * P(K <= k) for k <= n/2.
        p_one = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
        p = min(1.0, 2 * p_one)
        return {"b": b, "c": c, "statistic": float("nan"), "p_value": p,
                "method": "mcnemar exact (small n)"}

    chi2 = ((abs(b - c) - 1) ** 2) / (b + c)
    # χ²₁ → p-value via survival of standard normal (Wilson approximation):
    # for chi2 with 1 df, p = erfc(sqrt(chi2/2)).
    p = math.erfc(math.sqrt(chi2 / 2))
    return {"b": b, "c": c, "statistic": chi2, "p_value": p,
            "method": "mcnemar with continuity correction"}


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h for proportion differences. p1 = before, p2 = after."""
    p1 = max(0.0, min(1.0, p1))
    p2 = max(0.0, min(1.0, p2))
    return 2 * (math.asin(math.sqrt(p2)) - math.asin(math.sqrt(p1)))


def cohens_h_label(h: float) -> str:
    a = abs(h)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


def wilcoxon_signed_rank(deltas: list[float]) -> dict[str, float]:
    """Two-sided Wilcoxon signed-rank test on paired deltas. Standard
    rank-sum approximation — good for n >= 20.

    Returns a dict with statistic and approximate p-value."""
    nonzero = [d for d in deltas if d != 0]
    n = len(nonzero)
    if n == 0:
        return {"n": 0, "statistic": 0.0, "p_value": 1.0,
                "method": "wilcoxon (all zero)"}
    abs_deltas = sorted(((abs(d), d) for d in nonzero), key=lambda x: x[0])
    # Rank with ties → average rank (simple version).
    ranks: dict[float, float] = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs_deltas[j + 1][0] == abs_deltas[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1   # 1-indexed
        for k in range(i, j + 1):
            ranks[id(abs_deltas[k])] = avg_rank
        i = j + 1

    w_plus = sum(ranks[id(item)] for item in abs_deltas if item[1] > 0)
    w_minus = sum(ranks[id(item)] for item in abs_deltas if item[1] < 0)
    w = min(w_plus, w_minus)

    # Normal approximation for n >= 20.
    mean = n * (n + 1) / 4
    var = n * (n + 1) * (2 * n + 1) / 24
    if var == 0:
        return {"n": n, "statistic": w, "p_value": 1.0,
                "method": "wilcoxon"}
    z = (w - mean) / math.sqrt(var)
    p = math.erfc(abs(z) / math.sqrt(2))
    return {"n": n, "statistic": w, "z": z, "p_value": p,
            "method": "wilcoxon (normal approx)"}


# --------------------------------------------------------------------------- #
# Comparison aggregations
# --------------------------------------------------------------------------- #

def overall_comparison(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    if not pairs:
        return {"n": 0}

    n = len(pairs)
    # 2×2 table on success.
    a = sum(1 for b, af in pairs if b["success"] and af["success"])
    bb = sum(1 for b, af in pairs if b["success"] and not af["success"])  # before-pass / after-fail
    c = sum(1 for b, af in pairs if not b["success"] and af["success"])    # before-fail / after-pass
    d = sum(1 for b, af in pairs if not b["success"] and not af["success"])

    p_before = (a + bb) / n
    p_after = (a + c) / n

    mcn = mcnemar_test(a, bb, c, d)
    h = cohens_h(p_before, p_after)

    # Reward delta paired test.
    deltas = [af["final_reward"] - b["final_reward"] for b, af in pairs]
    wcx = wilcoxon_signed_rank(deltas)
    delta_lo, delta_hi = bootstrap_ci(deltas, n_iter=1000)

    return {
        "n_pairs": n,
        "success_rate_before": p_before,
        "success_rate_after": p_after,
        "success_rate_delta": p_after - p_before,
        "contingency_table": {"both_pass": a, "before_pass_after_fail": bb,
                                 "before_fail_after_pass": c, "both_fail": d},
        "mcnemar": mcn,
        "cohens_h": h,
        "cohens_h_label": cohens_h_label(h),
        "mean_reward_before": sum(b["final_reward"] for b, _ in pairs) / n,
        "mean_reward_after": sum(af["final_reward"] for _, af in pairs) / n,
        "mean_reward_delta": sum(deltas) / n,
        "mean_reward_delta_ci_95": [delta_lo, delta_hi],
        "wilcoxon_reward": wcx,
    }


def per_task_comparison(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, dict[str, Any]]:
    by_task: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = {}
    for b, af in pairs:
        by_task.setdefault(b["task_id"], []).append((b, af))

    out: dict[str, dict[str, Any]] = {}
    for tid, ps in by_task.items():
        n = len(ps)
        before_succ = sum(1 for b, _ in ps if b["success"])
        after_succ = sum(1 for _, af in ps if af["success"])
        p_before = before_succ / n
        p_after = after_succ / n

        # Paired changes on this task.
        a = sum(1 for b, af in ps if b["success"] and af["success"])
        bb = sum(1 for b, af in ps if b["success"] and not af["success"])
        c = sum(1 for b, af in ps if not b["success"] and af["success"])
        d = sum(1 for b, af in ps if not b["success"] and not af["success"])

        deltas_succ = [int(af["success"]) - int(b["success"]) for b, af in ps]
        delta_lo, delta_hi = bootstrap_ci(deltas_succ, n_iter=1000)

        out[tid] = {
            "n_pairs": n,
            "difficulty": _difficulty_of(tid),
            "success_rate_before": p_before,
            "success_rate_after": p_after,
            "delta": p_after - p_before,
            "delta_ci_95": [delta_lo, delta_hi],
            "improved_episodes": c,
            "regressed_episodes": bb,
            "unchanged_pass": a,
            "unchanged_fail": d,
            "cohens_h": cohens_h(p_before, p_after),
        }
    return out


def failure_mode_shift(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    """Compare failure-mode distribution between before and after."""
    cats = ("goal_incomplete", "wrong_tool", "wrong_args", "crashed")
    before_counts = {k: 0 for k in cats}
    after_counts = {k: 0 for k in cats}
    for b, af in pairs:
        if b["error_category"] in before_counts:
            before_counts[b["error_category"]] += 1
        if af["error_category"] in after_counts:
            after_counts[af["error_category"]] += 1
    return {
        "before": before_counts,
        "after": after_counts,
        "delta": {k: after_counts[k] - before_counts[k] for k in cats},
    }


def tool_call_quality_shift(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any]:
    def rates(items, getter_ok, getter_total):
        ok = sum(getter_ok(x) for x in items)
        total = sum(getter_total(x) for x in items)
        return ok / total if total else 0.0

    before = [b for b, _ in pairs]
    after = [af for _, af in pairs]
    return {
        "tool_call_validity_rate": {
            "before": rates(before, lambda x: x["tool_calls_ok"],
                             lambda x: x["tool_calls_made"]),
            "after": rates(after, lambda x: x["tool_calls_ok"],
                            lambda x: x["tool_calls_made"]),
        },
        "wrong_args_rate": {
            "before": rates(before, lambda x: x["tool_calls_wrong_args"],
                             lambda x: x["tool_calls_made"]),
            "after": rates(after, lambda x: x["tool_calls_wrong_args"],
                            lambda x: x["tool_calls_made"]),
        },
        "unknown_tool_rate": {
            "before": rates(before, lambda x: x["tool_calls_unknown_tool"],
                             lambda x: x["tool_calls_made"]),
            "after": rates(after, lambda x: x["tool_calls_unknown_tool"],
                            lambda x: x["tool_calls_made"]),
        },
    }


def acceptance_criteria_check(comparison: dict[str, Any]) -> dict[str, Any]:
    """Apply STATS_PLAN.md §7 acceptance criteria."""
    ov = comparison["overall"]
    per_task = comparison["per_task"]

    delta_pp = (ov["success_rate_delta"]) * 100
    p_value = ov["mcnemar"]["p_value"]
    h = ov["cohens_h"]

    hard_tasks_breakthrough = any(
        per_task[t]["difficulty"] == "hard"
        and per_task[t]["success_rate_before"] < 0.10
        and per_task[t]["success_rate_after"] >= 0.30
        for t in per_task
    )

    tool_validity_delta = (
        comparison["tool_call_quality"]["tool_call_validity_rate"]["after"]
        - comparison["tool_call_quality"]["tool_call_validity_rate"]["before"]
    )

    no_catastrophic_regression = all(
        per_task[t]["delta"] >= -0.10 for t in per_task
    )

    return {
        "criterion_1_overall_improvement_pp_geq_15_p_lt_05_h_geq_03": (
            delta_pp >= 15 and p_value < 0.05 and h >= 0.3
        ),
        "criterion_2_hard_task_breakthrough": hard_tasks_breakthrough,
        "criterion_3_tool_validity_improvement_geq_10pp": tool_validity_delta >= 0.10,
        "criterion_4_no_task_regresses_more_than_10pp": no_catastrophic_regression,
        "details": {
            "overall_delta_pp": delta_pp,
            "mcnemar_p_value": p_value,
            "cohens_h": h,
            "tool_validity_delta": tool_validity_delta,
            "tasks_regressed_more_than_10pp": [
                t for t, m in per_task.items() if m["delta"] < -0.10
            ],
        },
    }


# --------------------------------------------------------------------------- #
# Top-level
# --------------------------------------------------------------------------- #

def compare(before_dir: Path, after_dir: Path) -> dict[str, Any]:
    before_stats = compute_run_stats(before_dir)
    after_stats = compute_run_stats(after_dir)
    pairs = pair_features(
        before_stats["_trajectory_features"],
        after_stats["_trajectory_features"],
    )

    overall = overall_comparison(pairs)
    per_task = per_task_comparison(pairs)
    fail_shift = failure_mode_shift(pairs)
    tool_shift = tool_call_quality_shift(pairs)
    comparison = {
        "before_run": str(before_dir),
        "after_run": str(after_dir),
        "n_paired": len(pairs),
        "n_before_only": (len(before_stats["_trajectory_features"]) - len(pairs)),
        "n_after_only": (len(after_stats["_trajectory_features"]) - len(pairs)),
        "overall": overall,
        "per_task": per_task,
        "failure_mode_shift": fail_shift,
        "tool_call_quality": tool_shift,
    }
    comparison["acceptance"] = acceptance_criteria_check(comparison)
    return comparison


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--before", required=True, type=Path)
    ap.add_argument("--after", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()

    out = compare(args.before, args.after)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {args.output}")

    if args.summary:
        ov = out["overall"]
        ac = out["acceptance"]
        print()
        print(f"  paired episodes: {out['n_paired']}")
        print(f"  before success:  {ov['success_rate_before']:.1%}")
        print(f"  after  success:  {ov['success_rate_after']:.1%}")
        print(f"  delta:           {ov['success_rate_delta']:+.1%}")
        print(f"  McNemar p-value: {ov['mcnemar']['p_value']:.4f}")
        print(f"  Cohen's h:       {ov['cohens_h']:+.3f} ({out['overall']['cohens_h_label']})")
        print()
        print("Acceptance criteria:")
        for k, v in ac.items():
            if k == "details":
                continue
            mark = "PASS" if v else "FAIL"
            print(f"  [{mark}] {k}")


if __name__ == "__main__":
    main()
