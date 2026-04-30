"""M1 — Set surge in response to a sudden demand spike.

Setup: 5 simulated minutes of 10x demand in downtown; supply unchanged.
Verifier: surge raised in downtown to >= 1.5x; cumulative unmatched ratio < 0.4.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from rideshare_gym.core.task import InitialState
from rideshare_gym.core.tools import ToolSpec
from rideshare_gym.core.verifier import (
    AssertionListVerifier, CompositeVerifier, MetricThresholdVerifier, Verifier,
)
from rideshare_gym.mock_server.seed import seed_drivers
from rideshare_gym.mock_server.store import get_world
from rideshare_gym.rideshare_sandbox import RideshareSandbox
from rideshare_gym.tasks._base import RideshareTask
from rideshare_gym.tools import select
from rideshare_gym.world.events import WorldEvent


class SurgeDemandSpikeTask(RideshareTask):
    task_id = "rideshare/surge_demand_spike"
    difficulty = "medium"
    max_steps = 25

    def tools(self) -> list[ToolSpec]:
        return select(
            "zone_supply_demand", "list_pending_requests",
            "list_idle_drivers", "set_surge",
            "auto_match_nearest", "tick",
        )

    def setup(self, sandbox: RideshareSandbox) -> InitialState:
        sandbox.rs.reset()
        w = get_world(sandbox.tenant_id)
        w.reseed(self.seed)

        # 25 drivers normally; demand spike + their movement should expose mismatch.
        seed_drivers(w, n_drivers=25, seed=self.seed, online_pct=1.0)

        # Inject a 5-minute concert_let_out-like event amplifying downtown demand 10x.
        spike = WorldEvent(
            kind="concert_let_out",
            started_at=w.clock.now,
            duration_seconds=300.0,
            affected_zones=["downtown"],
            severity=1.0,
            metadata={"reason": "demand_spike_test"},
        )
        w.events.append(spike)

        # Pre-spawn ~30 pending downtown requests so the agent doesn't have to wait
        # ticks to see demand.
        downtown = w.city.zone_by_id("downtown")
        for _ in range(30):
            w._spawn_request(downtown)

        ground_truth = {"affected_zone": "downtown", "spike_duration_s": 300.0}
        sandbox.rs.set_metadata({"ground_truth": ground_truth})
        return InitialState(
            summary=(
                "DEMAND ALERT: Downtown is experiencing a 10x demand spike for "
                "the next 5 simulated minutes. ~30 pending requests already; "
                "supply is limited. Use set_surge on the downtown zone to "
                "balance demand, then dispatch as many as you can. "
                "Goal: keep unmatched ratio under 40% cumulative."
            ),
            snapshot={"affected_zone": "downtown", "duration_s": 300.0,
                      "pre_spawned": 30},
            ground_truth=ground_truth,
        )

    def verifier(self) -> Verifier:
        def _snap(sandbox: RideshareSandbox) -> dict[str, Any]:
            w = get_world(sandbox.tenant_id)
            from rideshare_gym.world.trips import TripStatus
            n_pending = sum(1 for t in w.trips.values()
                            if t.status == TripStatus.REQUESTED)
            n_total = len(w.trips)
            n_unmatched_terminal = sum(
                1 for t in w.trips.values()
                if t.status in (TripStatus.REQUESTED, TripStatus.CANCELLED)
            )
            return {
                "downtown_surge": float(w.surge_zones.get("downtown", Decimal("1.0"))),
                "n_total_trips": n_total,
                "n_pending": n_pending,
                "unmatched_ratio": (n_unmatched_terminal / max(1, n_total)),
                "kpis": w.kpis(),
            }

        def _unmatched_metric(sandbox: RideshareSandbox, gt: dict) -> float:
            snap = _snap(sandbox)
            ratio = snap["unmatched_ratio"]
            return max(0.0, 1.0 - ratio / 0.4)

        return CompositeVerifier(
            children=[
                AssertionListVerifier(
                    assertions=[
                        ("surge_raised_downtown",
                            lambda s: s["downtown_surge"] >= 1.5),
                    ],
                    snapshot_fn=_snap,
                ),
                MetricThresholdVerifier(
                    metric_fn=_unmatched_metric,
                    ground_truth={},
                    threshold=0.5,
                    metric_name="unmatched_under_40pct",
                ),
            ],
            mode="all_of",
        )
