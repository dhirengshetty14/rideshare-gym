"""M5 — Accident incident response. Multi-action coordination under safety constraints.

Setup: A safety incident on trip Z (mid-trip vehicle damage) is open.
Verifier: incident escalated to T2 or T3, emergency contacted, trip refunded,
driver compensated, both parties notified.
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
from rideshare_gym.world.world import SafetyIncident


class AccidentIncidentResponseTask(RideshareTask):
    task_id = "rideshare/accident_incident_response"
    difficulty = "medium"
    max_steps = 14

    def tools(self) -> list[ToolSpec]:
        return select(
            "list_safety_incidents", "get_incident", "get_trip",
            "escalate_incident", "contact_emergency", "attach_evidence",
            "create_refund", "adjust_driver_payout",
            "send_to_rider", "send_to_driver",
        )

    def setup(self, sandbox: RideshareSandbox) -> InitialState:
        sandbox.rs.reset()
        w = get_world(sandbox.tenant_id)
        w.reseed(self.seed)

        did = w.next_id()
        w.drivers[did] = Driver(
            id=did, name="affected driver", location=(15.0, 15.0),
            status=DriverStatus.OFFLINE, vehicle_type="uberx",
            home_zone_id="downtown",
        )
        rid = w.next_id()
        w.riders[rid] = Rider(
            id=rid, name="affected rider",
            email="rider@example.com", payment_method_id=f"pm_{rid}",
        )

        pickup = w.city.zone_by_id("downtown").centroid
        dropoff = w.city.zone_by_id("airport").centroid
        trip_id = w.next_id()
        # The trip got cancelled mid-flight due to vehicle damage.
        w.trips[trip_id] = Trip(
            id=trip_id, rider_id=rid, driver_id=did,
            pickup=pickup, dropoff=dropoff,
            pickup_zone_id="downtown", dropoff_zone_id="airport",
            vehicle_type="uberx", status=TripStatus.CANCELLED,
            requested_at=w.clock.now - 600.0,
            matched_at=w.clock.now - 540.0,
            picked_up_at=w.clock.now - 480.0,
            cancelled_at=w.clock.now - 60.0,
            cancelled_by="system",
            cancel_reason="safety_concern",
            surge_at_request=Decimal("1.0"),
            fare={
                "vehicle_type": "uberx", "subtotal": "18.50", "tax": "1.48",
                "total": "19.98", "distance_km": 12.0, "duration_min": 24.0,
                "surge_multiplier": "1.0",
            },
        )
        incident_id = w.next_id()
        w.incidents[incident_id] = SafetyIncident(
            id=incident_id, trip_id=trip_id, kind="vehicle_damage",
            severity=2, reported_at=w.clock.now - 120.0,
            escalation_level=0, emergency_contacted=False,
        )
        ground_truth = {
            "incident_id": incident_id, "trip_id": trip_id,
            "rider_id": rid, "driver_id": did,
            "expected_refund_amount": 19.98,
        }
        sandbox.rs.set_metadata({"ground_truth": ground_truth})
        return InitialState(
            summary=(
                f"SAFETY INCIDENT: Mid-trip vehicle damage on trip "
                f"{trip_id}. Incident {incident_id}, severity T2. "
                f"Required actions: escalate to >= T2, contact emergency "
                f"services, refund the rider in full ($19.98), compensate "
                f"the driver for the lost trip, notify both parties."
            ),
            snapshot={"incident_id": incident_id, "trip_id": trip_id,
                      "severity": 2},
            ground_truth=ground_truth,
        )

    def verifier(self) -> Verifier:
        def _snap(sandbox: RideshareSandbox) -> dict[str, Any]:
            w = get_world(sandbox.tenant_id)
            gt = w.metadata.get("ground_truth", {})
            incident = w.incidents.get(gt.get("incident_id", -1))
            trip_id = gt.get("trip_id", -1)
            rider_id = gt.get("rider_id", -1)
            driver_id = gt.get("driver_id", -1)
            refunds = [r for r in w.refunds.values() if r.trip_id == trip_id]
            payouts = [a for a in w.payout_adjustments
                        if a.driver_id == driver_id]
            r_msgs = [m for m in w.sent_messages
                      if m.to == f"rider:{rider_id}"]
            d_msgs = [m for m in w.sent_messages
                      if m.to == f"driver:{driver_id}"]
            return {
                "escalation_level":
                    incident.escalation_level if incident else 0,
                "emergency_contacted":
                    incident.emergency_contacted if incident else False,
                "n_refunds": len(refunds),
                "refund_amount":
                    float(refunds[0].amount) if refunds else 0.0,
                "expected_refund": gt.get("expected_refund_amount", 0.0),
                "n_payouts": len(payouts),
                "rider_notified": len(r_msgs) >= 1,
                "driver_notified": len(d_msgs) >= 1,
            }

        return AssertionListVerifier(
            assertions=[
                ("escalated_to_T2_or_higher",
                    lambda s: s["escalation_level"] >= 2),
                ("emergency_contacted",
                    lambda s: s["emergency_contacted"] is True),
                ("rider_refunded_full",
                    lambda s: s["n_refunds"] >= 1
                              and abs(s["refund_amount"]
                                       - s["expected_refund"]) <= 0.05),
                ("driver_compensated",
                    lambda s: s["n_payouts"] >= 1),
                ("rider_notified", lambda s: s["rider_notified"]),
                ("driver_notified", lambda s: s["driver_notified"]),
            ],
            snapshot_fn=_snap,
        )
