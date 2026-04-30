"""H2 — Event surge planning. Predictive action under explicit deadline.

Setup: A `concert_let_out` event is scheduled to start at t=90s in stadium zone.
Agent has 90 sim-seconds to pre-position drivers and raise surge.

Verifier (assertion_list):
- >= 6 drivers rebalanced to stadium before t=90s
- Surge in stadium >= 1.8 by t=90s
"""

from __future__ import annotations

from typing import Any

from rideshare_gym.core.task import InitialState
from rideshare_gym.core.tools import ToolSpec
from rideshare_gym.core.verifier import AssertionListVerifier, Verifier
from rideshare_gym.mock_server.seed import seed_drivers, seed_riders
from rideshare_gym.mock_server.store import get_world
from rideshare_gym.rideshare_sandbox import RideshareSandbox
from rideshare_gym.tasks._base import RideshareTask
from rideshare_gym.tools import select
from rideshare_gym.world.events import WorldEvent


class EventSurgePlanningTask(RideshareTask):
    task_id = "rideshare/event_surge_planning"
    difficulty = "hard"
    max_steps = 40

    def tools(self) -> list[ToolSpec]:
        return select(
            "list_idle_drivers", "list_surge_zones", "zone_supply_demand",
            "rebalance_driver", "set_surge", "tick", "list_pending_requests",
        )

    def setup(self, sandbox: RideshareSandbox) -> InitialState:
        sandbox.rs.reset()
        w = get_world(sandbox.tenant_id)
        w.reseed(self.seed)

        # Seed 25 drivers concentrated in suburbs (not at the stadium).
        seed_drivers(w, n_drivers=25, seed=self.seed, online_pct=1.0)
        # Force their home zones away from stadium.
        for d in w.drivers.values():
            if d.home_zone_id == "stadium":
                d.home_zone_id = "suburb_s"
                d.location = w.city.zone_by_id("suburb_s").centroid
        seed_riders(w, n_riders=20, seed=self.seed)

        # Concert lets out at t=90s for 5 minutes.
        w.events.append(WorldEvent(
            kind="concert_let_out", started_at=90.0, duration_seconds=300.0,
            affected_zones=["stadium"], severity=1.0,
            metadata={"reason": "Memorial Stadium event"},
        ))
        ground_truth = {
            "deadline_seconds": 90.0,
            "min_drivers_at_stadium": 6,
            "min_surge": 1.8,
        }
        sandbox.rs.set_metadata({"ground_truth": ground_truth})
        return InitialState(
            summary=(
                "EVENT INCOMING: A 5-minute concert_let_out event in the "
                "STADIUM zone fires at t=90s. Currently 25 idle drivers are "
                "spread across the suburbs. Pre-position >= 6 drivers to "
                "stadium AND set surge in stadium to >= 1.8x BEFORE t=90s. "
                "Use `tick()` sparingly — every tick burns your deadline."
            ),
            snapshot={
                "deadline_seconds": 90.0,
                "min_drivers": 6, "min_surge": 1.8,
                "current_now": 0.0,
            },
            ground_truth=ground_truth,
        )

    def verifier(self) -> Verifier:
        def _snap(sandbox: RideshareSandbox) -> dict[str, Any]:
            w = get_world(sandbox.tenant_id)
            from rideshare_gym.world.drivers import DriverStatus
            stadium_drivers = sum(
                1 for d in w.drivers.values()
                if d.home_zone_id == "stadium"
                   and d.status == DriverStatus.IDLE
            )
            stadium_surge = float(w.surge_zones.get("stadium",
                                                     w.surge_zones.get("stadium",
                                                                          1.0)))
            return {
                "now": w.clock.now,
                "stadium_drivers_idle": stadium_drivers,
                "stadium_surge": stadium_surge,
            }

        return AssertionListVerifier(
            assertions=[
                ("at_least_6_drivers_at_stadium",
                    lambda s: s["stadium_drivers_idle"] >= 6),
                ("stadium_surge_at_least_1_8x",
                    lambda s: s["stadium_surge"] >= 1.8),
            ],
            snapshot_fn=_snap,
        )
