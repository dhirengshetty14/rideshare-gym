"""Tests for the trajectory → training-data converters.

These work on synthetic Trajectory objects (no LLM / GPU needed) so they
run in the standard pytest sweep on any machine."""

from __future__ import annotations

import json
from pathlib import Path

from rideshare_gym.core.recorder import Step, Trajectory
from rideshare_gym.core.types import Observation, ToolCall

from training.data.failure_miner import (
    cluster_failures, per_task_summary,
)
from training.data.trajectory_to_dpo import build_pairs
from training.data.trajectory_to_sft import (
    filter_trajectory, trajectory_to_sft_examples,
)


def _synthetic_traj(
    *, task_id: str, seed: int, success: bool,
    n_steps: int = 2, error_category: str | None = None,
) -> Trajectory:
    """Build a synthetic LLM-style trajectory (with assistant_message + tool_result_message)."""
    steps = []
    for i in range(n_steps):
        steps.append(Step(
            index=i,
            action=ToolCall(name="some_tool", arguments={"x": i}),
            observation=Observation(text=f"obs_{i}", data={"i": i}),
            reward=(1.0 if (success and i == n_steps - 1) else 0.0),
            terminated=success and i == n_steps - 1,
            assistant_message={
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": "some_tool",
                        "arguments": json.dumps({"x": i}),
                    },
                }],
            },
            tool_result_message={
                "role": "tool",
                "tool_call_id": f"call_{i}",
                "content": json.dumps({"text": f"obs_{i}", "data": {"i": i}}),
            },
            info={"verifier": {
                "verifier": "assertion_list",
                "passed": ["a"] if success else [],
                "failed": [] if success else ["a"],
            }},
        ))
    return Trajectory(
        task_id=task_id, seed=seed,
        success=success,
        final_reward=1.0 if success else 0.0,
        error_category=None if success else (error_category or "goal_incomplete"),
        steps=steps,
        system_prompt="You are a helpful platform agent.",
        tools_serialized=[{"name": "some_tool", "description": "a tool"}],
        initial_user_message="Solve task X",
    )


def test_sft_extracts_one_example_per_step():
    t = _synthetic_traj(task_id="rideshare/match_single_ride",
                         seed=0, success=True, n_steps=3)
    examples = trajectory_to_sft_examples(t)
    assert len(examples) == 3
    assert all(ex["completion"]["role"] == "assistant" for ex in examples)
    # Each example's "messages" should grow as we go (history accumulates).
    assert len(examples[0]["messages"]) == 2     # system + user
    assert len(examples[1]["messages"]) == 4     # + asst + tool
    assert len(examples[2]["messages"]) == 6


def test_sft_skips_gold_oracle_trajectories():
    """Gold oracle trajectories have no assistant_message — converter should
    return empty list, not crash."""
    t = Trajectory(task_id="x", seed=0, success=True,
                    steps=[Step(index=0,
                                action=ToolCall(name="t", arguments={}),
                                observation=Observation(text=""),
                                reward=1.0, terminated=True)])
    # No system_prompt set → returns []
    assert trajectory_to_sft_examples(t) == []


def test_filter_trajectory_expression():
    t = _synthetic_traj(task_id="x", seed=0, success=True)
    assert filter_trajectory(t, "success==True") is True
    assert filter_trajectory(t, "success==False") is False
    assert filter_trajectory(t, "final_reward>=0.5") is True
    assert filter_trajectory(t, "final_reward>2.0") is False


def test_dpo_pair_at_first_divergence():
    """Two trajectories from same (task, seed) should pair at first divergent step."""
    success = _synthetic_traj(task_id="x", seed=0, success=True, n_steps=2)
    failure = _synthetic_traj(task_id="x", seed=0, success=False, n_steps=2)
    # Force a divergence at step 0 by changing the failure's first action arg.
    failure.steps[0].assistant_message["tool_calls"][0]["function"]["arguments"] = json.dumps({"x": 999})
    pairs = build_pairs(success, failure)
    assert len(pairs) == 1
    assert pairs[0]["chosen"]["tool_calls"][0]["function"]["arguments"] != \
           pairs[0]["rejected"]["tool_calls"][0]["function"]["arguments"]


def test_dpo_no_pair_when_identical():
    success = _synthetic_traj(task_id="x", seed=0, success=True, n_steps=2)
    # success-shaped failure (same actions, different outcome) → no divergence
    failure = _synthetic_traj(task_id="x", seed=0, success=False, n_steps=2)
    failure.success = False
    failure.steps[-1].terminated = False
    failure.steps[-1].reward = 0.0
    # Actions identical → no pair
    assert build_pairs(success, failure) == []


def test_failure_miner_clusters():
    trajs = [
        _synthetic_traj(task_id="rideshare/M1", seed=i,
                         success=False, error_category="goal_incomplete")
        for i in range(3)
    ] + [
        _synthetic_traj(task_id="rideshare/M2", seed=i,
                         success=False, error_category="wrong_args")
        for i in range(2)
    ] + [
        _synthetic_traj(task_id="rideshare/M1", seed=10, success=True),
    ]
    clusters = cluster_failures(trajs)
    # Two clusters: M1 goal_incomplete (3) and M2 wrong_args (2).
    assert len(clusters) == 2
    assert clusters[0].count == 3
    assert clusters[0].task_id == "rideshare/M1"
    assert clusters[1].count == 2
    assert clusters[1].task_id == "rideshare/M2"


def test_per_task_summary_counts_correctly():
    trajs = [
        _synthetic_traj(task_id="A", seed=i, success=True) for i in range(3)
    ] + [
        _synthetic_traj(task_id="A", seed=i, success=False,
                         error_category="wrong_args") for i in range(2)
    ]
    summary = per_task_summary(trajs)
    assert summary["A"]["n_episodes"] == 5
    assert abs(summary["A"]["success_rate"] - 0.6) < 1e-9
    assert summary["A"]["error_breakdown"] == {"wrong_args": 2}
