"""E1 — Match a single pending ride to the best (nearest-ETA) idle driver.

Setup: 1 pending request, 5 idle drivers in different zones.
Verifier: trip MATCHED, driver picked is the closest by ETA, dispatch_log entry.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from rideshare_gym.core.task import InitialState
from rideshare_gym.core.tools import ToolSpec
from rideshare_gym.core.verifier import AssertionListVerifier, Verifier
from rideshare_gym.mock_server.store import get_world
from rideshare_gym.rideshare_sandbox import RideshareSandbox
from rideshare_gym.tasks._base import RideshareTask
from rideshare_gym.tools import select
from rideshare_gym.world.drivers import Driver, DriverStatus
from rideshare_gym.world.matching import eta_minutes
from rideshare_gym.world.riders import Rider
from rideshare_gym.world.trips import Trip, TripStatus


class MatchSingleRideTask(RideshareTask):
    task_id = "rideshare/match_single_ride"
    difficulty = "easy"
    max_steps = 8

    def tools(self) -> list[ToolSpec]:
        return select(
            "list_pending_requests", "list_idle_drivers",
            "match_ride", "auto_match_nearest", "get_trip",
        )

    def setup(self, sandbox: RideshareSandbox) -> InitialState:
        # Plant 5 drivers spread across zones, then 1 pending request.
        sandbox.rs.reset()
        world = get_world(sandbox.tenant_id)
        world.reseed(self.seed)

        zones_to_seed = ["downtown", "airport", "university",
                          "stadium", "suburb_n"]
        driver_locations = [
            world.city.zone_by_id(z).centroid for z in zones_to_seed
        ]
        for i, loc in enumerate(driver_locations):
            did = world.next_id()
            world.drivers[did] = Driver(
                id=did, name=f"driver_{i}", location=loc,
                status=DriverStatus.IDLE, vehicle_type="uberx",
                rating=4.85, docs_verified=True,
                home_zone_id=zones_to_seed[i],
                speed_kmh=30.0, device_fingerprint=f"dfp_{did}",
            )

        # Create the rider + pending request near downtown.
        downtown = world.city.zone_by_id("downtown")
        pickup = downtown.centroid
        dropoff = world.city.zone_by_id("airport").centroid

        rider_id = world.next_id()
        world.riders[rider_id] = Rider(
            id=rider_id, name="rider", email="rider@example.com",
            payment_method_id=f"pm_{rider_id}",
        )
        trip_id = world.next_id()
        world.trips[trip_id] = Trip(
            id=trip_id, rider_id=rider_id, pickup=pickup, dropoff=dropoff,
            pickup_zone_id="downtown", dropoff_zone_id="airport",
            vehicle_type="uberx", status=TripStatus.REQUESTED,
            requested_at=world.clock.now,
            surge_at_request=Decimal("1.0"),
        )

        # Compute the correct (closest-ETA) driver.
        best_driver = min(
            world.drivers.values(),
            key=lambda d: eta_minutes(world.city, d, pickup),
        )
        ground_truth = {"trip_id": trip_id, "best_driver_id": best_driver.id}
        sandbox.rs.set_metadata({"ground_truth": ground_truth})

        return InitialState(
            summary=(
                f"One pending ride: trip {trip_id} from downtown to airport. "
                f"Five idle drivers are available across zones. Match the trip "
                f"to the BEST (closest-ETA) driver."
            ),
            snapshot={"trip_id": trip_id, "n_idle_drivers": 5},
            ground_truth=ground_truth,
        )

    def verifier(self) -> Verifier:
        def _snap(sandbox: RideshareSandbox) -> dict[str, Any]:
            world = get_world(sandbox.tenant_id)
            gt = world.metadata.get("ground_truth", {})
            trip = world.trips.get(gt.get("trip_id", -1))
            log_entries = [
                e for e in world.dispatch_log if e.trip_id == (trip.id if trip else -1)
            ]
            return {
                "trip_status": trip.status.value if trip else None,
                "trip_driver_id": trip.driver_id if trip else None,
                "ground_truth_best": gt.get("best_driver_id"),
                "n_dispatch_log_entries": len(log_entries),
            }

        return AssertionListVerifier(
            assertions=[
                ("trip_matched", lambda s: s["trip_status"] == "matched"
                                 or s["trip_status"] in ("driver_arriving",
                                                          "driver_arrived",
                                                          "in_trip", "completed")),
                ("picked_best_driver",
                    lambda s: s["trip_driver_id"] == s["ground_truth_best"]),
                ("dispatch_logged", lambda s: s["n_dispatch_log_entries"] >= 1),
            ],
            snapshot_fn=_snap,
        )
