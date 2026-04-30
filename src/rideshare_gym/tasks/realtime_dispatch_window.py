"""H1 (FLAGSHIP) — Real-time dispatch over a 30-minute simulated window.

Setup: 30 sim-minutes of demand across 6 zones with a peak around minute 18.
~30 drivers, online_pct=0.85. Agent calls tick() itself between match calls.

Verifier (composite, all_of):
- mean_pickup_wait_minutes < 4.0
- completion_rate >= 0.5
- cancellation_rate < 0.3
- revenue >= some baseline

This is the marquee long-horizon task. Expects ~80-150 tool calls per episode.
"""

from __future__ import annotations

from typing import Any

from rideshare_gym.core.task import InitialState
from rideshare_gym.core.tools import ToolSpec
from rideshare_gym.core.verifier import (
    AssertionListVerifier, CompositeVerifier, Verifier,
)
from rideshare_gym.mock_server.seed import seed_drivers, seed_riders
from rideshare_gym.mock_server.store import get_world
from rideshare_gym.rideshare_sandbox import RideshareSandbox
from rideshare_gym.tasks._base import RideshareTask
from rideshare_gym.tools import select


class RealtimeDispatchWindowTask(RideshareTask):
    task_id = "rideshare/realtime_dispatch_window"
    difficulty = "hard"
    max_steps = 250
    step_penalty = 0.0

    def tools(self) -> list[ToolSpec]:
        return select(
            "list_pending_requests", "list_idle_drivers",
            "auto_match_nearest", "match_ride", "set_surge",
            "rebalance_driver", "zone_supply_demand", "tick",
        )

    def setup(self, sandbox: RideshareSandbox) -> InitialState:
        sandbox.rs.reset()
        w = get_world(sandbox.tenant_id)
        w.reseed(self.seed)
        seed_drivers(w, n_drivers=30, seed=self.seed, online_pct=0.85)
        seed_riders(w, n_riders=20, seed=self.seed)

        # Plant a peak event around the 18-minute mark.
        from rideshare_gym.world.events import WorldEvent
        w.events.append(WorldEvent(
            kind="rush_hour", started_at=18 * 60, duration_seconds=300,
            affected_zones=["downtown", "stadium"], severity=0.6,
        ))

        ground_truth = {
            "max_sim_seconds": 30 * 60,
            "kpi_targets": {
                "mean_pickup_wait_minutes_max": 4.0,
                "completion_rate_min": 0.5,
                "cancellation_rate_max": 0.3,
            },
        }
        sandbox.rs.set_metadata({"ground_truth": ground_truth})
        return InitialState(
            summary=(
                "REAL-TIME DISPATCH: Run 30 simulated minutes (60 ticks of 30s). "
                "Use `tick()` to advance simulator time between actions. Demand "
                "varies across 6 zones with a peak around minute 18 (rush_hour). "
                "Goals: mean pickup wait < 4 min, completion rate >= 0.5, "
                "cancellation rate < 0.3.\n\n"
                "Loop pattern: auto_match_nearest pending trips -> tick(60) -> "
                "repeat. Optionally use set_surge or rebalance_driver to manage "
                "supply imbalances. Stop when the simulation has run >= 30 min."
            ),
            snapshot={"target_sim_seconds": 30 * 60,
                      "kpi_targets": ground_truth["kpi_targets"]},
            ground_truth=ground_truth,
        )

    def verifier(self) -> Verifier:
        def _snap(sandbox: RideshareSandbox) -> dict[str, Any]:
            w = get_world(sandbox.tenant_id)
            kpis = w.kpis()
            return {
                "now": w.clock.now,
                "kpis": kpis,
                "ran_full_window": w.clock.now >= 30 * 60,
            }

        return CompositeVerifier(
            children=[
                AssertionListVerifier(
                    assertions=[
                        ("ran_full_window",
                            lambda s: s["ran_full_window"]),
                        ("mean_wait_under_4min",
                            lambda s: s["kpis"]["mean_pickup_wait_minutes"] < 4.0),
                        ("completion_rate_over_50pct",
                            lambda s: s["kpis"]["completion_rate"] >= 0.5),
                        ("cancellation_rate_under_30pct",
                            lambda s: s["kpis"]["cancellation_rate"] < 0.3),
                    ],
                    snapshot_fn=_snap,
                ),
            ],
            mode="all_of",
        )
