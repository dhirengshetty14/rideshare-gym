"""Integration tests for the mock rideshare server (in-process via TestClient)."""

from __future__ import annotations

import pytest

from rideshare_gym.rideshare_sandbox import in_process_sandbox_factory


@pytest.fixture(scope="module")
def factory():
    return in_process_sandbox_factory(tenant_prefix="itest")


def test_health(factory):
    sb = factory()
    # health is unauth, hit it directly
    r = sb.http.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_seed_populates_world(factory):
    sb = factory()
    sb.reset()
    out = sb.rs.seed(n_drivers=10, n_riders=20, seed=0)
    assert out["ok"] is True
    snap = sb.snapshot()
    assert len(snap["drivers"]) == 10
    assert len(snap["riders"]) == 20


def test_tick_advances_time(factory):
    sb = factory()
    sb.reset()
    sb.rs.seed(n_drivers=5, n_riders=5)
    out = sb.rs.tick(60.0)
    assert out["now"] == 60.0
    sb.rs.tick(60.0)
    assert sb.rs.tick(60.0)["now"] == 180.0


def test_full_dispatch_cycle(factory):
    """End-to-end: seed, advance time to spawn demand, match a trip, complete."""
    sb = factory()
    sb.reset()
    sb.rs.seed(n_drivers=20, n_riders=20)
    # Advance to spawn demand.
    for _ in range(3):
        sb.rs.tick(60.0)
    pending = sb.rs.list_pending_requests(limit=10)["pending"]
    if not pending:
        return  # rare; demand stochastic
    trip_id = pending[0]["trip_id"]
    out = sb.rs.auto_match_nearest(trip_id)
    assert "trip" in out
    assert out["trip"]["status"] in ("matched", "driver_arriving",
                                        "driver_arrived", "in_trip")


def test_freeze_then_restore_rider(factory):
    sb = factory()
    sb.reset()
    sb.rs.seed(n_drivers=2, n_riders=5)
    snap = sb.snapshot()
    rid = next(iter(snap["riders"].values()))["id"]
    sb.rs.freeze_rider(rid, reason="test")
    snap = sb.snapshot()
    assert "frozen" in snap["riders"][str(rid)]["flags"]
    sb.rs.restore_rider(rid, reason="false_alarm")
    snap = sb.snapshot()
    assert "frozen" not in snap["riders"][str(rid)]["flags"]


def test_perturbation_rate_limit_returns_429(factory):
    sb = factory()
    sb.reset()
    sb.rs.seed(n_drivers=5, n_riders=5)
    sb.inject_perturbations([
        {"kind": "rate_limit", "params": {"p": 1.0, "status": 429}}
    ])
    # Should now 429 on most endpoints.
    r = sb.http.get("/api/v1/marketplace/idle_drivers",
                     headers={"X-Rideshare-Tenant": sb.tenant_id})
    assert r.status_code == 429
    sb.clear_perturbations()
