"""Unit tests for verifier patterns."""

from __future__ import annotations

from rideshare_gym.core.sandbox import StubSandbox
from rideshare_gym.core.verifier import (
    AssertionListVerifier,
    CompositeVerifier,
    MetricThresholdVerifier,
    StateEqualityVerifier,
    canonicalize,
    state_hash,
)


def test_canonicalize_sorts_and_ignores():
    a = {"b": 1, "a": [3, 2, 1], "_ts": "2026-01-01"}
    out = canonicalize(a, ignore_paths=("_ts",))
    assert list(out.keys()) == ["_ts", "a", "b"]  # all keys present, sorted
    assert out["_ts"] is None  # stripped


def test_state_hash_is_stable_under_key_order():
    h1 = state_hash({"a": 1, "b": 2})
    h2 = state_hash({"b": 2, "a": 1})
    assert h1 == h2


def test_state_equality_pass():
    sb = StubSandbox()
    sb.set("orders", [{"id": 1, "status": "paid"}])
    v = StateEqualityVerifier(
        expected_state={"orders": [{"id": 1, "status": "paid"}]},
        snapshot_fn=lambda s: s.snapshot(),
    )
    r = v.validate(sb)
    assert r.done is True
    assert r.reward == 1.0
    assert r.info["match"] is True


def test_state_equality_fail_returns_diff():
    sb = StubSandbox()
    sb.set("orders", [{"id": 1, "status": "open"}])
    v = StateEqualityVerifier(
        expected_state={"orders": [{"id": 1, "status": "paid"}]},
        snapshot_fn=lambda s: s.snapshot(),
    )
    r = v.validate(sb)
    assert r.done is False
    assert r.reward == 0.0
    assert "diff" in r.info


def test_assertion_list_partial_credit():
    sb = StubSandbox()
    sb.set("inventory", 0)
    sb.set("emails_sent", 1)
    v = AssertionListVerifier(
        assertions=[
            ("inventory_zero", lambda s: s["inventory"] == 0),
            ("emails_sent", lambda s: s["emails_sent"] >= 1),
            ("refunded", lambda s: s.get("refunded", False)),
        ],
        snapshot_fn=lambda s: s.snapshot(),
    )
    r = v.validate(sb)
    # 2/3 pass
    assert abs(r.reward - 2 / 3) < 1e-9
    assert r.done is False  # require_all=True
    assert "refunded" in r.info["failed"]


def test_assertion_list_all_pass_marks_done():
    sb = StubSandbox()
    sb.set("ok", True)
    v = AssertionListVerifier(
        assertions=[("ok", lambda s: s["ok"] is True)],
        snapshot_fn=lambda s: s.snapshot(),
    )
    r = v.validate(sb)
    assert r.done is True
    assert r.reward == 1.0


def test_metric_threshold_binary_and_continuous():
    sb = StubSandbox()
    sb.set("predictions", [1, 1, 0, 1])
    sb.set("labels", [1, 0, 0, 1])  # 3/4 correct = 0.75

    def metric(sandbox, gt):
        snap = sandbox.snapshot()
        preds, labels = snap["predictions"], snap["labels"]
        return sum(int(p == l) for p, l in zip(preds, labels)) / len(labels)

    v_cont = MetricThresholdVerifier(
        metric_fn=metric, ground_truth={}, threshold=0.9, metric_name="acc")
    v_bin = MetricThresholdVerifier(
        metric_fn=metric, ground_truth={}, threshold=0.9, metric_name="acc",
        binary_reward=True)
    r1 = v_cont.validate(sb)
    r2 = v_bin.validate(sb)
    assert r1.reward == 0.75
    assert r1.done is False  # below 0.9
    assert r2.reward == 0.0


def test_composite_all_of():
    sb = StubSandbox()
    sb.set("flag_a", True)
    sb.set("flag_b", True)
    a = AssertionListVerifier(
        assertions=[("a", lambda s: s["flag_a"] is True)],
        snapshot_fn=lambda s: s.snapshot())
    b = AssertionListVerifier(
        assertions=[("b", lambda s: s["flag_b"] is True)],
        snapshot_fn=lambda s: s.snapshot())
    c = CompositeVerifier(children=[a, b], mode="all_of")
    r = c.validate(sb)
    assert r.done is True
    assert r.reward == 1.0


def test_composite_any_of_partial():
    sb = StubSandbox()
    sb.set("flag_a", False)
    sb.set("flag_b", True)
    a = AssertionListVerifier(
        assertions=[("a", lambda s: s["flag_a"] is True)],
        snapshot_fn=lambda s: s.snapshot())
    b = AssertionListVerifier(
        assertions=[("b", lambda s: s["flag_b"] is True)],
        snapshot_fn=lambda s: s.snapshot())
    c = CompositeVerifier(children=[a, b], mode="any_of")
    r = c.validate(sb)
    assert r.done is True  # any_of: b passed
    assert r.reward == 0.5
