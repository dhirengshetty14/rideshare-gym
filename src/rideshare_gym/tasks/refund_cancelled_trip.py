"""E2 — Refund a trip the driver cancelled after the rider waited.

Setup: 1 trip cancelled by the driver 12 minutes after pickup ETA.
Verifier: refund issued for full amount, reason references driver cancellation,
rider notified.
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
from rideshare_gym.world.trips import Trip, TripStatus


class RefundCancelledTripTask(RideshareTask):
    task_id = "rideshare/refund_cancelled_trip"
    difficulty = "easy"
    max_steps = 10

    def tools(self) -> list[ToolSpec]:
        return select(
            "list_trips", "get_trip", "get_rider",
            "create_refund", "send_to_rider",
        )

    def setup(self, sandbox: RideshareSandbox) -> InitialState:
        sandbox.rs.reset()
        w = get_world(sandbox.tenant_id)
        w.reseed(self.seed)

        # Driver who will cancel.
        did = w.next_id()
        w.drivers[did] = Driver(
            id=did, name="cancelling driver", location=(15.0, 15.0),
            status=DriverStatus.IDLE, vehicle_type="uberx",
            home_zone_id="downtown",
        )
        # Rider with email.
        rid = w.next_id()
        w.riders[rid] = Rider(
            id=rid, name="waiting rider", email="rider@example.com",
            payment_method_id=f"pm_{rid}",
        )
        # The cancelled trip — fare computed for documentation; rider was charged.
        pickup = w.city.zone_by_id("downtown").centroid
        dropoff = w.city.zone_by_id("airport").centroid
        distance_km = w.city.distance_km(pickup, dropoff)
        fare = compute_fare(
            distance_km=distance_km, duration_min=20.0,
            surge=Decimal("1.0"), vehicle_type="uberx",
        )
        trip_id = w.next_id()
        w.trips[trip_id] = Trip(
            id=trip_id, rider_id=rid, driver_id=did,
            pickup=pickup, dropoff=dropoff,
            pickup_zone_id="downtown", dropoff_zone_id="airport",
            vehicle_type="uberx",
            status=TripStatus.CANCELLED,
            requested_at=w.clock.now - 720.0,         # 12 min ago
            matched_at=w.clock.now - 600.0,
            cancelled_at=w.clock.now - 60.0,
            cancelled_by="driver",
            cancel_reason="driver_no_show",
            surge_at_request=Decimal("1.0"),
            fare={
                "vehicle_type": "uberx",
                "base_fare": str(fare.base_fare),
                "distance_fare": str(fare.distance_fare),
                "time_fare": str(fare.time_fare),
                "surge_multiplier": str(fare.surge_multiplier),
                "booking_fee": str(fare.booking_fee),
                "safety_fee": str(fare.safety_fee),
                "subtotal": str(fare.subtotal),
                "tax": str(fare.tax),
                "total": str(fare.total),
                "distance_km": fare.distance_km,
                "duration_min": fare.duration_min,
            },
        )
        ground_truth = {
            "trip_id": trip_id, "rider_id": rid,
            "expected_refund_amount": float(fare.total),
        }
        sandbox.rs.set_metadata({"ground_truth": ground_truth})
        return InitialState(
            summary=(
                f"Trip {trip_id} was cancelled by the driver after the rider "
                f"waited 12 minutes. Issue a full refund and notify the rider. "
                f"Expected refund amount equals the trip total fare."
            ),
            snapshot={"trip_id": trip_id, "rider_id": rid,
                      "fare_total": str(fare.total)},
            ground_truth=ground_truth,
        )

    def verifier(self) -> Verifier:
        def _snap(sandbox: RideshareSandbox) -> dict[str, Any]:
            w = get_world(sandbox.tenant_id)
            gt = w.metadata.get("ground_truth", {})
            trip_id = gt.get("trip_id", -1)
            rider_id = gt.get("rider_id", -1)
            refunds_for_trip = [r for r in w.refunds.values()
                                 if r.trip_id == trip_id]
            messages_to_rider = [
                m for m in w.sent_messages if m.to == f"rider:{rider_id}"
            ]
            return {
                "n_refunds": len(refunds_for_trip),
                "refund_amount": (float(refunds_for_trip[0].amount)
                                    if refunds_for_trip else None),
                "refund_reason": (refunds_for_trip[0].reason
                                    if refunds_for_trip else None),
                "expected_amount": gt.get("expected_refund_amount"),
                "n_messages_to_rider": len(messages_to_rider),
            }

        return AssertionListVerifier(
            assertions=[
                ("exactly_one_refund", lambda s: s["n_refunds"] == 1),
                ("refund_amount_correct",
                    lambda s: s["refund_amount"] is not None
                              and abs(s["refund_amount"]
                                      - s["expected_amount"]) <= 0.01),
                ("rider_notified",
                    lambda s: s["n_messages_to_rider"] >= 1),
            ],
            snapshot_fn=_snap,
        )
