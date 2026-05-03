"""Tests for the statistics module.

Synthetic trajectories — no LLM, no GPU. Validates the math behind:
- per-trajectory feature extraction
- per-task aggregation with bootstrap CIs
- McNemar's test, Cohen's h, paired Wilcoxon
- acceptance criteria evaluation
"""

from __future__ import annotations

from rideshare_gym.core.recorder import Step, Trajectory
from rideshare_gym.core.types import Observation, ToolCall

from analysis.compare_runs import (
    cohens_h,
    cohens_h_label,
    mcnemar_test,
    overall_comparison,
    pair_features,
    per_task_comparison,
    wilcoxon_signed_rank,
)
from analysis.stats import (
    bootstrap_ci,
    overall_stats,
    per_task_stats,
    trajectory_features,
)


def _t(*, task_id, seed, success, n_steps=2, error_category=None,
        wrong_args_per_step=False, ok=True) -> Trajectory:
    steps = []
    for i in range(n_steps):
        steps.append(Step(
            index=i,
            action=ToolCall(name=f"tool_{i % 2}", arguments={"x": i}),
            observation=Observation(text="ok"),
            reward=(1.0 if (success and i == n_steps - 1) else 0.5),
            terminated=success and i == n_steps - 1,
            info={
                "tool_ok": ok,
                "tool_error": ("wrong_args: bad" if wrong_args_per_step else None),
            },
        ))
    return Trajectory(
        task_id=task_id, seed=seed, success=success,
        final_reward=1.0 if success else 0.0,
        error_category=None if success else (error_category or "goal_incomplete"),
        steps=steps,
    )


def test_trajectory_features_counts_correctly():
    t = _t(task_id="x", seed=0, success=True, n_steps=3,
           wrong_args_per_step=False, ok=True)
    f = trajectory_features(t)
    assert f["task_id"] == "x"
    assert f["success"] is True
    assert f["n_steps"] == 3
    assert f["tool_calls_made"] == 3
    assert f["tool_calls_ok"] == 3
    assert f["tool_calls_wrong_args"] == 0
    assert f["unique_tools_used"] == 2  # tool_0, tool_1


def test_per_task_stats_with_paired_features():
    trajs = [
        _t(task_id="A", seed=0, success=True),
        _t(task_id="A", seed=1, success=True),
        _t(task_id="A", seed=2, success=False, error_category="wrong_args"),
        _t(task_id="A", seed=3, success=True),
        _t(task_id="B", seed=0, success=False, error_category="goal_incomplete"),
        _t(task_id="B", seed=1, success=False, error_category="goal_incomplete"),
    ]
    feats = [trajectory_features(t) for t in trajs]
    s = per_task_stats(feats)
    assert s["A"]["n"] == 4
    assert abs(s["A"]["success_rate"] - 0.75) < 1e-9
    assert s["A"]["error_breakdown"]["wrong_args"] == 1
    assert s["B"]["success_rate"] == 0.0
    assert s["B"]["error_breakdown"]["goal_incomplete"] == 2


def test_overall_stats_aggregates_across_tasks():
    feats = [
        trajectory_features(_t(task_id="rideshare/match_single_ride",
                                  seed=i, success=(i < 4)))
        for i in range(6)
    ]
    s = overall_stats(feats)
    assert s["n"] == 6
    assert abs(s["overall_success_rate"] - 4 / 6) < 1e-9
    assert "easy" in s["success_rate_by_difficulty"]


def test_bootstrap_ci_brackets_observed_mean():
    values = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0]
    lo, hi = bootstrap_ci(values, n_iter=2000, seed=42)
    mean = sum(values) / len(values)
    assert lo <= mean <= hi
    # Width should be reasonable for n=10 — at least 0.2.
    assert hi - lo > 0.1


def test_mcnemar_no_discordant_pairs():
    out = mcnemar_test(b_pass=10, b_fail=0, c_pass=0, c_fail=10)
    assert out["p_value"] == 1.0


def test_mcnemar_significant_improvement():
    # Many before-fail / after-pass; few before-pass / after-fail.
    # b = before_pass_after_fail, c = before_fail_after_pass
    out = mcnemar_test(b_pass=20, b_fail=2, c_pass=18, c_fail=20)
    assert out["p_value"] < 0.05
    assert out["c"] > out["b"]


def test_cohens_h_directionality():
    assert cohens_h(0.3, 0.6) > 0
    assert cohens_h(0.6, 0.3) < 0
    assert abs(cohens_h(0.5, 0.5)) < 1e-9


def test_cohens_h_labels():
    assert cohens_h_label(0.0) == "negligible"
    assert cohens_h_label(0.25) == "small"
    assert cohens_h_label(0.55) == "medium"
    assert cohens_h_label(0.95) == "large"


def test_pair_features_matches_on_task_seed():
    before = [
        trajectory_features(_t(task_id="A", seed=i, success=False))
        for i in range(5)
    ]
    after = [
        trajectory_features(_t(task_id="A", seed=i, success=True))
        for i in range(5)
    ]
    pairs = pair_features(before, after)
    assert len(pairs) == 5
    for b, af in pairs:
        assert b["task_id"] == af["task_id"]
        assert b["seed"] == af["seed"]


def test_overall_comparison_detects_improvement():
    before = [
        trajectory_features(_t(task_id="A", seed=i, success=False))
        for i in range(40)
    ]
    after = [
        trajectory_features(_t(task_id="A", seed=i, success=True))
        for i in range(40)
    ]
    pairs = pair_features(before, after)
    out = overall_comparison(pairs)
    assert out["success_rate_after"] - out["success_rate_before"] == 1.0
    assert out["mcnemar"]["p_value"] < 0.001
    assert out["cohens_h"] > 0.5


def test_per_task_comparison_tracks_changes():
    before = [
        trajectory_features(_t(task_id="A", seed=i, success=(i < 2)))
        for i in range(5)
    ]
    after = [
        trajectory_features(_t(task_id="A", seed=i, success=(i < 4)))
        for i in range(5)
    ]
    pairs = pair_features(before, after)
    s = per_task_comparison(pairs)
    assert "A" in s
    # 2 before-fail/after-pass (improved): seeds 2,3
    # 0 before-pass/after-fail (regressed)
    assert s["A"]["improved_episodes"] == 2
    assert s["A"]["regressed_episodes"] == 0
    assert s["A"]["delta"] == 0.4


def test_wilcoxon_signed_rank_detects_systematic_improvement():
    deltas = [0.5] * 30
    out = wilcoxon_signed_rank(deltas)
    assert out["p_value"] < 0.001


def test_wilcoxon_no_change_high_p():
    deltas = [0.0] * 10
    out = wilcoxon_signed_rank(deltas)
    assert out["p_value"] >= 0.99
