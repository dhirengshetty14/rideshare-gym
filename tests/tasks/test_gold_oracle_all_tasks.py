"""Gold-oracle smoke tests across all 12 tasks.

These confirm:
- Every task is registered, runnable, and produces a valid trajectory.
- The 10 deterministic tasks complete with reward 1.0.
- The 2 real-time tasks (H1, M1) hit the verifier's KPI ceiling — they don't
  fully succeed but produce reward >= 0.5. This is by design — the verifier
  thresholds are tight on purpose so stronger agents have headroom.
"""

from __future__ import annotations

import pytest

from agents.gold_oracle import GoldOracleAgent
from rideshare_gym.core.env import GymEnvironment
from rideshare_gym.rideshare_sandbox import in_process_sandbox_factory
from rideshare_gym.tasks import REGISTRY

DETERMINISTIC_TASKS = [
    "rideshare/match_single_ride",
    "rideshare/refund_cancelled_trip",
    "rideshare/verify_driver_documents",
    "rideshare/fraud_ring_detection",
    "rideshare/lost_item_recovery",
    "rideshare/driver_pay_dispute",
    "rideshare/accident_incident_response",
    "rideshare/account_takeover_response",
    "rideshare/event_surge_planning",
    "rideshare/coordinated_fraud_response",
]

PARTIAL_TASKS = [
    "rideshare/realtime_dispatch_window",
    "rideshare/surge_demand_spike",
]


@pytest.fixture(scope="module")
def factory():
    return in_process_sandbox_factory(tenant_prefix="gold")


@pytest.fixture(scope="module")
def agent():
    return GoldOracleAgent()


@pytest.mark.parametrize("task_id", DETERMINISTIC_TASKS)
def test_oracle_solves_deterministic_task(task_id, factory, agent):
    env = GymEnvironment(task=REGISTRY[task_id](seed=0), sandbox_factory=factory)
    traj = agent.run(env)
    env.close()
    assert traj.success, (
        f"oracle failed {task_id}: reward={traj.final_reward}, "
        f"err={traj.error_category}, "
        f"verifier={traj.steps[-1].info.get('verifier') if traj.steps else None}"
    )
    assert traj.final_reward == 1.0


@pytest.mark.parametrize("task_id", PARTIAL_TASKS)
def test_oracle_partial_credit_on_realtime_task(task_id, factory, agent):
    """Real-time tasks have KPI thresholds the greedy oracle doesn't fully hit.
    We assert reward >= 0.5 — verifier credits the assertions oracle DID nail
    (e.g. surge raised) but flags missed KPIs (e.g. unmatched ratio)."""
    env = GymEnvironment(task=REGISTRY[task_id](seed=0), sandbox_factory=factory)
    traj = agent.run(env)
    env.close()
    assert traj.final_reward >= 0.5, (
        f"oracle on {task_id} should produce >= 0.5 reward (passes some assertions); "
        f"got {traj.final_reward}"
    )
