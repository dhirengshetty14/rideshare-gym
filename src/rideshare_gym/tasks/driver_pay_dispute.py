"""M4 — Driver pay dispute. Forensic reconstruction from logs.

Setup: A trip's surge was 1.8x but the driver was paid as if 1.0x.
Agent must inspect the trip + GPS log + surge state, recompute the correct
driver payout, and adjust.
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
from rideshare_gym.world.pricing import compute_fare
from rideshare_gym.world.riders import Rider
from rideshare_gym.world.trips import GpsPoint, Trip, TripStatus


class DriverPayDisputeTask(RideshareTask):
    task_id = "rideshare/driver_pay_dispute"
    difficulty = "medium"
    max_steps = 12

    def tools(self) -> list[ToolSpec]:
        return select(
            "get_trip", "get_trip_gps_log", "get_driver",
            "list_surge_zones", "adjust_driver_payout", "send_to_driver",
        )

    def setup(self, sandbox: RideshareSandbox) -> InitialState:
        sandbox.rs.reset()
        w = get_world(sandbox.tenant_id)
        w.reseed(self.seed)

        did = w.next_id()
        w.drivers[did] = Driver(
            id=did, name="aggrieved driver", location=(15.0, 15.0),
            status=DriverStatus.IDLE, vehicle_type="uberx",
            cumulative_earnings_today=Decimal("85.50"),
            home_zone_id="downtown",
        )
        rid = w.next_id()
        w.riders[rid] = Rider(
            id=rid, name="rider", email="rider@example.com",
            payment_method_id=f"pm_{rid}",
        )

        # The disputed trip.
        pickup = w.city.zone_by_id("downtown").centroid
        dropoff = w.city.zone_by_id("airport").centroid
        actual_surge = Decimal("1.8")
        # Compute the CORRECT fare at 1.8x surge.
        correct_fare = compute_fare(
            distance_km=w.city.distance_km(pickup, dropoff),
            duration_min=18.0,
            surge=actual_surge,
            vehicle_type="uberx",
        )
        # Compute the WRONG fare at 1.0x — what driver was paid.
        wrong_fare = compute_fare(
            distance_km=w.city.distance_km(pickup, dropoff),
            duration_min=18.0,
            surge=Decimal("1.0"),
            vehicle_type="uberx",
        )
        underpayment = correct_fare.driver_payout() - wrong_fare.driver_payout()

        trip_id = w.next_id()
        w.trips[trip_id] = Trip(
            id=trip_id, rider_id=rid, driver_id=did,
            pickup=pickup, dropoff=dropoff,
            pickup_zone_id="downtown", dropoff_zone_id="airport",
            vehicle_type="uberx", status=TripStatus.COMPLETED,
            requested_at=w.clock.now - 1500.0,
            matched_at=w.clock.now - 1450.0,
            picked_up_at=w.clock.now - 1300.0,
            completed_at=w.clock.now - 200.0,
            surge_at_request=actual_surge,    # this is the truth
            fare={
                "vehicle_type": "uberx",
                "base_fare": str(wrong_fare.base_fare),
                "distance_fare": str(wrong_fare.distance_fare),
                "time_fare": str(wrong_fare.time_fare),
                "surge_multiplier": "1.0",     # <- the BUG: paid out at 1.0
                "booking_fee": str(wrong_fare.booking_fee),
                "safety_fee": str(wrong_fare.safety_fee),
                "subtotal": str(wrong_fare.subtotal),
                "tax": str(wrong_fare.tax),
                "total": str(wrong_fare.total),
                "distance_km": wrong_fare.distance_km,
                "duration_min": wrong_fare.duration_min,
                "driver_payout": str(wrong_fare.driver_payout()),
            },
        )
        # GPS log so the agent can verify distance.
        for i in range(20):
            t = w.trips[trip_id].picked_up_at + i * 60
            w.trips[trip_id].gps_log.append(
                GpsPoint(t=t, location=(pickup[0] + i * 0.6,
                                          pickup[1] - i * 0.4),
                          speed_kmh=30.0))

        ground_truth = {
            "trip_id": trip_id,
            "driver_id": did,
            "underpayment_dollars": float(underpayment),
            "correct_surge": str(actual_surge),
        }
        sandbox.rs.set_metadata({"ground_truth": ground_truth})
        return InitialState(
            summary=(
                f"Driver {did} disputes pay on trip {trip_id}. They claim "
                f"surge was 1.8x but they were paid at 1.0x. Verify by "
                f"inspecting the trip's surge_at_request field, recompute "
                f"the correct driver payout, and apply the difference as a "
                f"payout adjustment. Notify the driver."
            ),
            snapshot={"trip_id": trip_id, "driver_id": did,
                      "current_payout_listed": str(wrong_fare.driver_payout())},
            ground_truth=ground_truth,
        )

    def verifier(self) -> Verifier:
        def _snap(sandbox: RideshareSandbox) -> dict[str, Any]:
            w = get_world(sandbox.tenant_id)
            gt = w.metadata.get("ground_truth", {})
            driver_id = gt.get("driver_id", -1)
            adj_for_driver = [a for a in w.payout_adjustments
                              if a.driver_id == driver_id]
            messages = [m for m in w.sent_messages
                        if m.to == f"driver:{driver_id}"]
            return {
                "n_adjustments": len(adj_for_driver),
                "total_adjusted": (float(sum((a.amount for a in adj_for_driver),
                                              start=Decimal("0")))
                                   if adj_for_driver else 0.0),
                "expected_underpayment": gt.get("underpayment_dollars", 0.0),
                "n_messages": len(messages),
            }

        return AssertionListVerifier(
            assertions=[
                ("at_least_one_adjustment", lambda s: s["n_adjustments"] >= 1),
                ("amount_within_tolerance",
                    lambda s: abs(s["total_adjusted"]
                                  - s["expected_underpayment"]) <= 0.50),
                ("driver_notified", lambda s: s["n_messages"] >= 1),
            ],
            snapshot_fn=_snap,
        )
