"""The World — composes city + drivers + riders + trips + ops records into one
mutable simulator state. Owns the tick loop. One World per tenant.
"""

from __future__ import annotations

import random
import threading
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

from rideshare_gym.world.city import City, default_city
from rideshare_gym.world.clock import SimClock
from rideshare_gym.world.demand import DemandGenerator
from rideshare_gym.world.drivers import Driver, DriverStatus, step_driver
from rideshare_gym.world.events import WorldEvent
from rideshare_gym.world.pricing import compute_fare
from rideshare_gym.world.riders import Rider
from rideshare_gym.world.surge import (
    SURGE_MIN, apply_hysteresis, compute_surge,
)
from rideshare_gym.world.trips import GpsPoint, Trip, TripStatus


@dataclass
class Refund:
    id: int
    trip_id: int
    amount: Decimal
    reason: str
    issued_at: float
    notify_rider: bool = True


@dataclass
class Dispute:
    id: int
    trip_id: int
    reason: str
    status: str = "needs_response"
    response: dict[str, Any] | None = None
    created_at: float = 0.0
    deadline: float = 0.0


@dataclass
class SafetyIncident:
    id: int
    trip_id: int
    kind: str           # "accident" | "verbal_aggression" | "vehicle_damage" | "weapon" | "other"
    severity: int = 1   # 1..3
    reported_at: float = 0.0
    escalation_level: int = 0
    emergency_contacted: bool = False
    evidence: list[str] = field(default_factory=list)


@dataclass
class LostItem:
    id: int
    trip_id: int
    description: str
    reported_at: float = 0.0
    assigned_driver_id: int | None = None
    return_method: str | None = None
    scheduled_pickup_at: float | None = None
    return_pickup_location: tuple[float, float] | None = None
    handoff_code: str = ""
    confirmed_at: float | None = None


@dataclass
class SentMessage:
    id: int
    to: str        # "rider:{id}" | "driver:{id}"
    template: str
    variables: dict[str, Any] = field(default_factory=dict)
    sent_at: float = 0.0


@dataclass
class DispatchLogEntry:
    trip_id: int
    driver_id: int
    decided_at: float
    eta_minutes_at_dispatch: float
    alternatives_considered: int = 0


@dataclass
class PayoutAdjustment:
    id: int
    driver_id: int
    amount: Decimal
    reason: str
    related_trip_id: int | None = None
    issued_at: float = 0.0


@dataclass
class AccountAction:
    """Freeze / restore / ban events captured for verifier use."""
    id: int
    target_kind: str   # "rider" | "driver" | "device"
    target_id: str
    action: str        # "freeze" | "restore" | "ban"
    reason: str = ""
    at: float = 0.0


@dataclass
class World:
    """The whole simulator state for ONE tenant."""

    tenant_id: str
    clock: SimClock = field(default_factory=SimClock)
    city: City = field(default_factory=default_city)

    drivers: dict[int, Driver] = field(default_factory=dict)
    riders: dict[int, Rider] = field(default_factory=dict)
    trips: dict[int, Trip] = field(default_factory=dict)

    surge_zones: dict[str, Decimal] = field(default_factory=dict)
    """zone_id -> current surge multiplier."""

    events: list[WorldEvent] = field(default_factory=list)

    # Operations records.
    refunds: dict[int, Refund] = field(default_factory=dict)
    disputes: dict[int, Dispute] = field(default_factory=dict)
    incidents: dict[int, SafetyIncident] = field(default_factory=dict)
    lost_items: dict[int, LostItem] = field(default_factory=dict)

    # Side effects (queryable by verifiers).
    sent_messages: list[SentMessage] = field(default_factory=list)
    dispatch_log: list[DispatchLogEntry] = field(default_factory=list)
    payout_adjustments: list[PayoutAdjustment] = field(default_factory=list)
    account_actions: list[AccountAction] = field(default_factory=list)
    banned_devices: dict[str, str] = field(default_factory=dict)
    """fingerprint -> reason."""

    # Adversarial perturbations (installed by FixtureMutator).
    perturbations: list[dict[str, Any]] = field(default_factory=list)

    # Freeform task scratch (ground truth, hints, etc.).
    metadata: dict[str, Any] = field(default_factory=dict)

    # Internal.
    rng: random.Random = field(default_factory=lambda: random.Random(0))
    demand_gen: DemandGenerator = field(default_factory=DemandGenerator)
    _next_id: int = 1000
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def next_id(self) -> int:
        with self._lock:
            self._next_id += 1
            return self._next_id

    def reseed(self, seed: int) -> None:
        self.rng = random.Random(seed)
        self.demand_gen = DemandGenerator(rng=random.Random(seed + 7))

    # ------------------------------------------------------------------ #
    # Tick loop
    # ------------------------------------------------------------------ #

    def tick(self, dt_seconds: float | None = None) -> None:
        dt = self.clock.step_seconds if dt_seconds is None else float(dt_seconds)
        self.clock.tick(dt)

        # 1. Move drivers physically.
        for d in self.drivers.values():
            step_driver(d, self, dt)

        # 2. Advance trip state machine.
        for t in list(self.trips.values()):
            self._advance_trip(t, dt)

        # 3. Sample new demand.
        self._spawn_demand(dt)

        # 4. Recompute surge per zone.
        self._update_surge()

    # ------------------------------------------------------------------ #
    # Trip state machine
    # ------------------------------------------------------------------ #

    def _advance_trip(self, trip: Trip, dt: float) -> None:
        """Move a trip's state forward based on driver location."""
        if trip.status == TripStatus.MATCHED:
            driver = self.drivers.get(trip.driver_id) if trip.driver_id else None
            if driver is None:
                return
            # Sample a GPS point.
            trip.gps_log.append(GpsPoint(t=self.clock.now, location=driver.location,
                                          speed_kmh=driver.speed_kmh))
            # Did driver arrive at pickup?
            if _at(driver.location, trip.pickup, eps=0.05):
                trip.status = TripStatus.DRIVER_ARRIVED
                trip.pickup_arrived_at = self.clock.now
                driver.status = DriverStatus.PICKING_UP
                # Auto-pickup after 30s of dwell.
                trip.metadata["pickup_dwell_started_at"] = self.clock.now
        elif trip.status == TripStatus.DRIVER_ARRIVED:
            driver = self.drivers.get(trip.driver_id) if trip.driver_id else None
            if driver is None:
                return
            dwell_start = trip.metadata.get("pickup_dwell_started_at", self.clock.now)
            if self.clock.now - dwell_start >= 30.0:
                # Pickup happens; transition to in-trip.
                trip.status = TripStatus.IN_TRIP
                trip.picked_up_at = self.clock.now
                driver.status = DriverStatus.IN_TRIP
                driver.target_location = trip.dropoff
        elif trip.status == TripStatus.IN_TRIP:
            driver = self.drivers.get(trip.driver_id) if trip.driver_id else None
            if driver is None:
                return
            trip.gps_log.append(GpsPoint(t=self.clock.now, location=driver.location,
                                          speed_kmh=driver.speed_kmh))
            if _at(driver.location, trip.dropoff, eps=0.05):
                self._complete_trip(trip, driver)

    def _complete_trip(self, trip: Trip, driver: Driver) -> None:
        trip.status = TripStatus.COMPLETED
        trip.completed_at = self.clock.now
        # Compute fare.
        distance_km = self.city.distance_km(trip.pickup, trip.dropoff)
        if trip.picked_up_at is not None:
            duration_min = max(1.0, (trip.completed_at - trip.picked_up_at) / 60.0)
        else:
            duration_min = max(1.0, distance_km * 60.0 / max(self.city.avg_speed_kmh, 1.0))
        fare = compute_fare(
            distance_km=distance_km,
            duration_min=duration_min,
            surge=trip.surge_at_request,
            vehicle_type=trip.vehicle_type,
        )
        trip.fare = {
            "vehicle_type": fare.vehicle_type,
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
            "driver_payout": str(fare.driver_payout()),
        }
        # Driver post-trip.
        driver.status = DriverStatus.IDLE
        driver.target_location = None
        driver.current_trip_id = None
        driver.completed_trips_today += 1
        driver.cumulative_earnings_today += fare.driver_payout()

    # ------------------------------------------------------------------ #
    # Demand + surge
    # ------------------------------------------------------------------ #

    def _spawn_demand(self, dt: float) -> None:
        zone_event_mult: dict[str, float] = {}
        for ev in self.events:
            for z in ev.affected_zones:
                zone_event_mult[z] = zone_event_mult.get(z, 1.0) * \
                    ev.demand_multiplier_for(z, self.clock.now)
        arrivals = self.demand_gen.sample_arrivals(
            city=self.city, dt_seconds=dt, now_seconds=self.clock.now,
            zone_event_multipliers=zone_event_mult,
        )
        for zone_id, n in arrivals.items():
            zone = self.city.zone_by_id(zone_id)
            if zone is None:
                continue
            for _ in range(n):
                self._spawn_request(zone)

    def _spawn_request(self, origin_zone) -> None:
        """Create a Trip in REQUESTED status with a synthetic rider if needed."""
        # Find or create a casual rider (we don't need detailed bio for synthetic demand).
        rider_id = self.next_id()
        rider = Rider(
            id=rider_id,
            name=f"rider_{rider_id}",
            email=f"rider_{rider_id}@example.com",
            payment_method_id=f"pm_{rider_id}",
            payment_bin="424242",
            device_fingerprint=f"fp_{rider_id}",
            home_zone_id=origin_zone.id,
        )
        self.riders[rider_id] = rider

        dest_zone_id = self.demand_gen.sample_destination(self.city, origin_zone.id)
        dest_zone = self.city.zone_by_id(dest_zone_id) or origin_zone
        pickup = origin_zone.random_point(self.rng)
        dropoff = dest_zone.random_point(self.rng)

        trip_id = self.next_id()
        trip = Trip(
            id=trip_id,
            rider_id=rider_id,
            pickup=pickup,
            dropoff=dropoff,
            pickup_zone_id=origin_zone.id,
            dropoff_zone_id=dest_zone.id,
            vehicle_type="uberx",
            status=TripStatus.REQUESTED,
            requested_at=self.clock.now,
            surge_at_request=self.surge_zones.get(origin_zone.id, SURGE_MIN),
        )
        self.trips[trip_id] = trip

    def _update_surge(self) -> None:
        for zone in self.city.zones:
            demand = sum(
                1 for t in self.trips.values()
                if t.status == TripStatus.REQUESTED and t.pickup_zone_id == zone.id
            )
            supply = sum(
                1 for d in self.drivers.values()
                if d.status == DriverStatus.IDLE and d.home_zone_id == zone.id
            )
            target = compute_surge(demand=demand, supply=supply)
            current = self.surge_zones.get(zone.id, SURGE_MIN)
            self.surge_zones[zone.id] = apply_hysteresis(current, target)

    # ------------------------------------------------------------------ #
    # KPIs (used by realtime tasks' verifiers)
    # ------------------------------------------------------------------ #

    def kpis(self) -> dict[str, Any]:
        completed = [t for t in self.trips.values() if t.status == TripStatus.COMPLETED]
        cancelled = [t for t in self.trips.values() if t.status == TripStatus.CANCELLED]
        all_terminal = completed + cancelled
        all_seen = list(self.trips.values())
        wait_times = [
            (t.matched_at - t.requested_at) / 60.0
            for t in all_seen if t.matched_at is not None
        ]
        revenue = sum(
            float(t.fare["subtotal"]) for t in completed if t.fare and "subtotal" in t.fare
        )
        return {
            "n_trips_seen": len(all_seen),
            "n_completed": len(completed),
            "n_cancelled": len(cancelled),
            "n_unmatched_pending": sum(
                1 for t in all_seen if t.status == TripStatus.REQUESTED),
            "completion_rate": len(completed) / max(1, len(all_terminal)),
            "cancellation_rate": len(cancelled) / max(1, len(all_terminal)),
            "mean_pickup_wait_minutes": (
                sum(wait_times) / len(wait_times) if wait_times else 0.0),
            "revenue": round(revenue, 2),
        }

    # ------------------------------------------------------------------ #
    # Snapshot for verifier diff / state-hash
    # ------------------------------------------------------------------ #

    def snapshot(self) -> dict[str, Any]:
        return {
            "now": self.clock.now,
            "drivers": {
                str(d.id): {
                    "id": d.id, "name": d.name, "location": list(d.location),
                    "status": d.status.value, "rating": d.rating,
                    "vehicle_type": d.vehicle_type, "current_trip_id": d.current_trip_id,
                    "docs_verified": d.docs_verified, "docs_expiry": d.docs_expiry,
                    "flags": list(d.flags),
                } for d in self.drivers.values()
            },
            "riders": {
                str(r.id): {
                    "id": r.id, "name": r.name, "email": r.email,
                    "rating": r.rating, "payment_bin": r.payment_bin,
                    "device_fingerprint": r.device_fingerprint,
                    "flags": list(r.flags),
                } for r in self.riders.values()
            },
            "trips": {
                str(t.id): {
                    "id": t.id, "rider_id": t.rider_id, "driver_id": t.driver_id,
                    "status": t.status.value, "pickup_zone_id": t.pickup_zone_id,
                    "dropoff_zone_id": t.dropoff_zone_id,
                    "vehicle_type": t.vehicle_type,
                    "requested_at": t.requested_at,
                    "matched_at": t.matched_at,
                    "completed_at": t.completed_at,
                    "cancelled_at": t.cancelled_at,
                    "cancelled_by": t.cancelled_by,
                    "cancel_reason": t.cancel_reason,
                    "surge_at_request": str(t.surge_at_request),
                    "fare": t.fare,
                    "flags": list(t.flags),
                } for t in self.trips.values()
            },
            "surge_zones": {k: str(v) for k, v in self.surge_zones.items()},
            "refunds": {str(r.id): {"id": r.id, "trip_id": r.trip_id,
                                     "amount": str(r.amount), "reason": r.reason,
                                     "issued_at": r.issued_at}
                        for r in self.refunds.values()},
            "disputes": {str(d.id): {"id": d.id, "trip_id": d.trip_id,
                                      "status": d.status, "response": d.response}
                         for d in self.disputes.values()},
            "incidents": {str(i.id): {"id": i.id, "trip_id": i.trip_id,
                                        "kind": i.kind, "severity": i.severity,
                                        "escalation_level": i.escalation_level,
                                        "emergency_contacted": i.emergency_contacted}
                          for i in self.incidents.values()},
            "lost_items": {str(li.id): {"id": li.id, "trip_id": li.trip_id,
                                          "description": li.description,
                                          "assigned_driver_id": li.assigned_driver_id,
                                          "scheduled_pickup_at": li.scheduled_pickup_at,
                                          "handoff_code": li.handoff_code,
                                          "confirmed_at": li.confirmed_at}
                           for li in self.lost_items.values()},
            "sent_messages": [
                {"to": m.to, "template": m.template, "variables": m.variables,
                 "sent_at": m.sent_at}
                for m in self.sent_messages
            ],
            "dispatch_log": [
                {"trip_id": e.trip_id, "driver_id": e.driver_id,
                 "decided_at": e.decided_at,
                 "eta_minutes_at_dispatch": e.eta_minutes_at_dispatch,
                 "alternatives_considered": e.alternatives_considered}
                for e in self.dispatch_log
            ],
            "payout_adjustments": [
                {"driver_id": p.driver_id, "amount": str(p.amount),
                 "reason": p.reason, "related_trip_id": p.related_trip_id,
                 "issued_at": p.issued_at}
                for p in self.payout_adjustments
            ],
            "account_actions": [
                {"target_kind": a.target_kind, "target_id": a.target_id,
                 "action": a.action, "reason": a.reason, "at": a.at}
                for a in self.account_actions
            ],
            "banned_devices": dict(self.banned_devices),
            "events": [
                {"kind": e.kind, "started_at": e.started_at,
                 "duration_seconds": e.duration_seconds,
                 "affected_zones": list(e.affected_zones),
                 "severity": e.severity}
                for e in self.events
            ],
            "metadata": dict(self.metadata),
            "kpis": self.kpis(),
        }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _at(a: tuple[float, float], b: tuple[float, float], eps: float = 0.1) -> bool:
    return abs(a[0] - b[0]) <= eps and abs(a[1] - b[1]) <= eps
