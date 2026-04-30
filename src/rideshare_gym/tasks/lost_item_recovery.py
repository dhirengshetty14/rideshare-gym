"""M3 — Run the full lost-item recovery flow.

Setup: A rider's wallet was left in a completed trip. Driver has done 3 more
trips since. Agent must create lost_item, assign to original driver, schedule
the return pickup, and notify both parties with the handoff code.
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
from rideshare_gym.world.riders import Rider
from rideshare_gym.world.trips import Trip, TripStatus


class LostItemRecoveryTask(RideshareTask):
    task_id = "rideshare/lost_item_recovery"
    difficulty = "medium"
    max_steps = 14

    def tools(self) -> list[ToolSpec]:
        return select(
            "list_trips", "get_trip", "get_driver",
            "create_lost_item", "assign_lost_item",
            "schedule_lost_item_pickup", "send_to_rider", "send_to_driver",
        )

    def setup(self, sandbox: RideshareSandbox) -> InitialState:
        sandbox.rs.reset()
        w = get_world(sandbox.tenant_id)
        w.reseed(self.seed)

        did = w.next_id()
        w.drivers[did] = Driver(
            id=did, name="original driver",
            location=(15.0, 15.0), status=DriverStatus.IDLE,
            vehicle_type="uberx", home_zone_id="downtown",
        )
        rid = w.next_id()
        w.riders[rid] = Rider(
            id=rid, name="forgetful rider",
            email="forget@example.com",
            payment_method_id=f"pm_{rid}",
        )

        # The completed trip where the wallet was left.
        trip_id = w.next_id()
        pickup = w.city.zone_by_id("downtown").centroid
        dropoff = w.city.zone_by_id("airport").centroid
        w.trips[trip_id] = Trip(
            id=trip_id, rider_id=rid, driver_id=did,
            pickup=pickup, dropoff=dropoff,
            pickup_zone_id="downtown", dropoff_zone_id="airport",
            vehicle_type="uberx", status=TripStatus.COMPLETED,
            requested_at=w.clock.now - 1800.0,
            matched_at=w.clock.now - 1700.0,
            picked_up_at=w.clock.now - 1500.0,
            completed_at=w.clock.now - 600.0,
            surge_at_request=Decimal("1.0"),
        )

        # The driver did 3 more trips since.
        for i in range(3):
            tid = w.next_id()
            w.trips[tid] = Trip(
                id=tid, rider_id=rid, driver_id=did,
                pickup=pickup, dropoff=dropoff,
                pickup_zone_id="downtown", dropoff_zone_id="airport",
                vehicle_type="uberx", status=TripStatus.COMPLETED,
                requested_at=w.clock.now - 540.0 + i * 60,
                completed_at=w.clock.now - 240.0 + i * 60,
            )

        ground_truth = {
            "trip_id": trip_id,
            "rider_id": rid,
            "driver_id": did,
            "expected_description": "wallet",
        }
        sandbox.rs.set_metadata({"ground_truth": ground_truth})
        return InitialState(
            summary=(
                f"Rider on trip {trip_id} reports they left their wallet in "
                f"the vehicle. The driver has completed 3 more trips since "
                f"and is currently idle. Run the lost-item recovery flow: "
                f"create the report, assign to the ORIGINAL driver, schedule "
                f"the return pickup at a downtown spot in 10 minutes, and "
                f"notify both parties with the handoff code."
            ),
            snapshot={"trip_id": trip_id, "rider_id": rid, "driver_id": did},
            ground_truth=ground_truth,
        )

    def verifier(self) -> Verifier:
        def _snap(sandbox: RideshareSandbox) -> dict[str, Any]:
            w = get_world(sandbox.tenant_id)
            gt = w.metadata.get("ground_truth", {})
            trip_id = gt.get("trip_id", -1)
            rider_id = gt.get("rider_id", -1)
            driver_id = gt.get("driver_id", -1)

            lis = [li for li in w.lost_items.values() if li.trip_id == trip_id]
            li = lis[0] if lis else None
            rider_msgs = [
                m for m in w.sent_messages
                if m.to == f"rider:{rider_id}"
                   and "lost_item" in m.template
            ]
            driver_msgs = [
                m for m in w.sent_messages
                if m.to == f"driver:{driver_id}"
                   and "lost_item" in m.template
            ]
            return {
                "lost_item_present": li is not None,
                "assigned_to_original_driver":
                    bool(li) and li.assigned_driver_id == driver_id,
                "scheduled": bool(li) and li.scheduled_pickup_at is not None,
                "has_handoff_code": bool(li) and bool(li.handoff_code),
                "rider_notified": len(rider_msgs) >= 1,
                "driver_notified": len(driver_msgs) >= 1,
            }

        return AssertionListVerifier(
            assertions=[
                ("lost_item_created", lambda s: s["lost_item_present"]),
                ("assigned_to_original_driver",
                    lambda s: s["assigned_to_original_driver"]),
                ("pickup_scheduled", lambda s: s["scheduled"]),
                ("handoff_code_generated", lambda s: s["has_handoff_code"]),
                ("rider_notified", lambda s: s["rider_notified"]),
                ("driver_notified", lambda s: s["driver_notified"]),
            ],
            snapshot_fn=_snap,
        )
