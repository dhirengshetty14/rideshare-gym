"""Unit tests for trajectory recorder + JSONL serialization."""

from __future__ import annotations

import json
from pathlib import Path

from rideshare_gym.core.recorder import Trajectory, TrajectoryRecorder, write_run_index
from rideshare_gym.core.types import Observation, ToolCall


def test_recorder_basic_flow(tmp_path: Path):
    rec = TrajectoryRecorder(task_id="shopify/test", seed=42, ground_truth={"x": 1})
    rec.set_initial("h0")
    rec.record(
        action=ToolCall(name="t", arguments={"a": 1}),
        observation=Observation(text="ok", data={"r": 1}),
        reward=0.5,
        terminated=False,
        latency_ms=12.0,
    )
    rec.record(
        action=ToolCall(name="t", arguments={"a": 2}),
        observation=Observation(text="done", data={}),
        reward=1.0,
        terminated=True,
        latency_ms=8.0,
        tokens_in=100,
        tokens_out=50,
    )
    traj = rec.finalize(final_state_hash="h1", success=True)

    assert traj.task_id == "shopify/test"
    assert traj.seed == 42
    assert traj.initial_state_hash == "h0"
    assert traj.final_state_hash == "h1"
    assert len(traj.steps) == 2
    assert traj.success is True
    assert traj.final_reward == 1.0
    assert traj.meta["n_steps"] == 2
    assert traj.meta["wall_time_seconds"] >= 0


def test_trajectory_jsonl_roundtrip(tmp_path: Path):
    rec = TrajectoryRecorder(task_id="shopify/test", seed=0)
    rec.record(
        action=ToolCall(name="t", arguments={}),
        observation=Observation(text="ok"),
        reward=1.0,
        terminated=True,
    )
    traj = rec.finalize(success=True)
    path = tmp_path / "ep.jsonl"
    traj.write(path)

    raw = path.read_text(encoding="utf-8").strip()
    parsed = Trajectory.model_validate_json(raw)
    assert parsed.task_id == traj.task_id
    assert parsed.success is True
    assert len(parsed.steps) == 1


def test_run_index_aggregation(tmp_path: Path):
    trajs = []
    for i in range(3):
        rec = TrajectoryRecorder(task_id=f"shopify/t{i}", seed=i)
        rec.record(
            action=ToolCall(name="t", arguments={}),
            observation=Observation(text="ok"),
            reward=1.0,
            terminated=True,
        )
        trajs.append(rec.finalize(success=True))
    idx_path = write_run_index(tmp_path, trajs)
    lines = idx_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    parsed = [json.loads(line) for line in lines]
    assert {p["task_id"] for p in parsed} == {"shopify/t0", "shopify/t1", "shopify/t2"}
    assert all(p["success"] for p in parsed)
