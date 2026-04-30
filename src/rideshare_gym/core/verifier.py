"""Task verifiers — given a sandbox at end-of-episode, return (reward, done, info).

Three concrete patterns:
  * StateEqualityVerifier   — deterministic post-state hash compare (τ-bench style)
  * AssertionListVerifier   — list of callables on final state (AppWorld style)
  * MetricThresholdVerifier — F1 / accuracy / latency threshold (ML-shaped tasks)

Plus CompositeVerifier(all_of=...) to combine them.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel


class VerifierResult(BaseModel):
    """The triple every verifier returns."""

    reward: float
    done: bool
    info: dict[str, Any] = {}


class Verifier(Protocol):
    """Validate the post-action state. Called every step by GymEnvironment."""

    def validate(self, sandbox: Any) -> VerifierResult: ...


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def canonicalize(obj: Any, ignore_paths: tuple[str, ...] = ()) -> Any:
    """Deterministically normalise a JSON-like object for hashing/equality.
    Strips fields whose dotted-path matches any pattern in `ignore_paths`."""

    def _walk(node: Any, path: str) -> Any:
        if any(_path_matches(path, ig) for ig in ignore_paths):
            return None
        if isinstance(node, dict):
            return {k: _walk(v, f"{path}.{k}" if path else k) for k, v in sorted(node.items())}
        if isinstance(node, list):
            return [_walk(v, f"{path}[]") for v in node]
        return node

    return _walk(obj, "")


def _path_matches(path: str, pattern: str) -> bool:
    """Glob-style match. `*` matches one segment; `**` matches any depth."""
    if pattern == path:
        return True
    if "**" in pattern:
        return path.startswith(pattern.split("**")[0])
    p_parts = pattern.split(".")
    a_parts = path.split(".")
    if len(p_parts) != len(a_parts):
        return False
    return all(p == "*" or p == a for p, a in zip(p_parts, a_parts))


def state_hash(state: dict[str, Any], ignore_paths: tuple[str, ...] = ()) -> str:
    """Stable SHA256 hash of a canonicalised state dict."""
    norm = canonicalize(state, ignore_paths)
    payload = json.dumps(norm, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------- #
# Concrete verifiers
# --------------------------------------------------------------------------- #

@dataclass
class StateEqualityVerifier:
    """Compares snapshot of the sandbox state to an expected dict, deep + stable.
    `ignore_paths` patterns are stripped before comparison (timestamps, ids)."""

    expected_state: dict[str, Any]
    snapshot_fn: Callable[[Any], dict[str, Any]]
    ignore_paths: tuple[str, ...] = ()
    success_reward: float = 1.0
    failure_reward: float = 0.0

    def validate(self, sandbox: Any) -> VerifierResult:
        observed = self.snapshot_fn(sandbox)
        h_obs = state_hash(observed, self.ignore_paths)
        h_exp = state_hash(self.expected_state, self.ignore_paths)
        if h_obs == h_exp:
            return VerifierResult(
                reward=self.success_reward, done=True,
                info={"verifier": "state_equality", "match": True})
        diff = _shallow_diff(self.expected_state, observed, self.ignore_paths)
        return VerifierResult(
            reward=self.failure_reward, done=False,
            info={"verifier": "state_equality", "match": False, "diff": diff[:20]})


@dataclass
class AssertionListVerifier:
    """A list of `(name, predicate)` pairs evaluated against the snapshot.
    Reward = fraction passed; done iff ALL pass."""

    assertions: list[tuple[str, Callable[[dict[str, Any]], bool]]]
    snapshot_fn: Callable[[Any], dict[str, Any]]
    require_all: bool = True
    weights: dict[str, float] | None = None

    def validate(self, sandbox: Any) -> VerifierResult:
        snap = self.snapshot_fn(sandbox)
        results: list[tuple[str, bool]] = []
        for name, pred in self.assertions:
            try:
                results.append((name, bool(pred(snap))))
            except Exception as e:
                results.append((name, False))
                snap.setdefault("_assertion_errors", {})[name] = str(e)
        if self.weights:
            total_w = sum(self.weights.get(n, 1.0) for n, _ in results)
            got_w = sum(self.weights.get(n, 1.0) for n, ok in results if ok)
            reward = got_w / total_w if total_w > 0 else 0.0
        else:
            reward = sum(1 for _, ok in results if ok) / max(1, len(results))
        all_passed = all(ok for _, ok in results)
        done = all_passed if self.require_all else reward > 0
        return VerifierResult(
            reward=reward, done=done,
            info={
                "verifier": "assertion_list",
                "passed": [n for n, ok in results if ok],
                "failed": [n for n, ok in results if not ok],
            },
        )


@dataclass
class MetricThresholdVerifier:
    """Computes a metric (F1, accuracy, latency, …) and compares to a threshold.
    Reward is the metric value (clipped to [0, 1]) or a step function."""

    metric_fn: Callable[[Any, dict[str, Any]], float]
    """Signature: `(sandbox, ground_truth) -> float in [0, 1]`."""
    ground_truth: dict[str, Any]
    threshold: float = 0.9
    metric_name: str = "metric"
    binary_reward: bool = False
    """If True: reward is 1.0 iff metric >= threshold else 0.0."""

    def validate(self, sandbox: Any) -> VerifierResult:
        try:
            value = float(self.metric_fn(sandbox, self.ground_truth))
        except Exception as e:
            return VerifierResult(
                reward=0.0, done=False,
                info={"verifier": "metric_threshold", "error": str(e)})
        passed = value >= self.threshold
        reward = (1.0 if passed else 0.0) if self.binary_reward else max(0.0, min(1.0, value))
        return VerifierResult(
            reward=reward, done=passed,
            info={
                "verifier": "metric_threshold",
                "metric": self.metric_name,
                "value": value,
                "threshold": self.threshold,
                "passed": passed,
            },
        )


@dataclass
class CompositeVerifier:
    """Combines verifiers with `all_of` (every one must succeed) or `any_of`."""

    children: list[Verifier]
    mode: str = "all_of"  # all_of | any_of
    weights: list[float] | None = None

    def validate(self, sandbox: Any) -> VerifierResult:
        results = [c.validate(sandbox) for c in self.children]
        weights = self.weights or [1.0] * len(results)
        total = sum(weights)
        reward = sum(r.reward * w for r, w in zip(results, weights)) / total
        if self.mode == "all_of":
            done = all(r.done for r in results)
        elif self.mode == "any_of":
            done = any(r.done for r in results)
        else:
            raise ValueError(f"unknown mode: {self.mode}")
        return VerifierResult(
            reward=reward, done=done,
            info={"verifier": "composite", "mode": self.mode,
                  "children": [r.info for r in results]},
        )


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #

def _shallow_diff(
    expected: dict[str, Any],
    actual: dict[str, Any],
    ignore_paths: tuple[str, ...] = (),
) -> list[dict[str, Any]]:
    """Best-effort diff for human inspection. Not used for reward."""
    diffs: list[dict[str, Any]] = []

    def walk(e: Any, a: Any, path: str) -> None:
        if any(_path_matches(path, ig) for ig in ignore_paths):
            return
        if isinstance(e, dict) and isinstance(a, dict):
            for k in set(e) | set(a):
                walk(e.get(k), a.get(k), f"{path}.{k}" if path else k)
        elif e != a:
            diffs.append({"path": path, "expected": e, "actual": a})

    walk(expected, actual, "")
    return diffs
