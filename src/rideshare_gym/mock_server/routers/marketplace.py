"""Marketplace dispatching — match rides, set surge, rebalance fleet, offer incentives."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query, Request

from rideshare_gym.mock_server.perturbations import get_tenant_id
from rideshare_gym.mock_server.store import get_world
from rideshare_gym.world.drivers import DriverStatus
from rideshare_gym.world.matching import eta_minutes, nearest_driver_for
from rideshare_gym.world.surge import SURGE_MAX, SURGE_MIN
from rideshare_gym.world.trips import TripStatus
from rideshare_gym.world.world import DispatchLogEntry

router = APIRouter(prefix="/api/v1/marketplace", tags=["marketplace"])


@router.get("/pending_requests")
def list_pending_requests(
    request: Request,
    zone_id: str | None = Query(None),
    limit: int = Query(50, le=500),
) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    out = [t for t in w.trips.values() if t.status == TripStatus.REQUESTED]
    if zone_id:
        out = [t for t in out if t.pickup_zone_id == zone_id]
    out.sort(key=lambda t: t.requested_at)
    return {
        "pending": [
            {
                "trip_id": t.id,
                "rider_id": t.rider_id,
                "pickup": list(t.pickup),
                "dropoff": list(t.dropoff),
                "pickup_zone_id": t.pickup_zone_id,
                "dropoff_zone_id": t.dropoff_zone_id,
                "vehicle_type": t.vehicle_type,
                "requested_at": t.requested_at,
                "wait_seconds": w.clock.now - t.requested_at,
                "surge_at_request": str(t.surge_at_request),
            }
            for t in out[:limit]
        ],
        "now": w.clock.now,
    }


@router.get("/idle_drivers")
def list_idle_drivers(
    request: Request,
    zone_id: str | None = Query(None),
    vehicle_type: str | None = Query(None),
    limit: int = Query(100, le=500),
) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    out = [
        d for d in w.drivers.values()
        if d.status == DriverStatus.IDLE and d.docs_verified and "frozen" not in d.flags
    ]
    if zone_id:
        out = [d for d in out if d.home_zone_id == zone_id]
    if vehicle_type:
        out = [d for d in out if d.vehicle_type == vehicle_type]
    return {
        "idle_drivers": [
            {
                "driver_id": d.id,
                "name": d.name,
                "location": list(d.location),
                "rating": d.rating,
                "vehicle_type": d.vehicle_type,
                "home_zone_id": d.home_zone_id,
            }
            for d in out[:limit]
        ],
    }


@router.post("/match")
def match_ride(
    request: Request,
    trip_id: int = Body(...),
    driver_id: int = Body(...),
) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    trip = w.trips.get(trip_id)
    driver = w.drivers.get(driver_id)
    if trip is None:
        raise HTTPException(status_code=404, detail="trip_not_found")
    if driver is None:
        raise HTTPException(status_code=404, detail="driver_not_found")
    if trip.status != TripStatus.REQUESTED:
        raise HTTPException(status_code=422, detail=f"trip_not_requested:{trip.status.value}")
    if driver.status != DriverStatus.IDLE:
        raise HTTPException(status_code=422, detail=f"driver_not_idle:{driver.status.value}")
    if not driver.docs_verified:
        raise HTTPException(status_code=422, detail="driver_docs_not_verified")
    if "frozen" in driver.flags:
        raise HTTPException(status_code=422, detail="driver_frozen")

    # Compute alternatives count for the dispatch_log.
    n_alt = sum(
        1 for d in w.drivers.values()
        if d.status == DriverStatus.IDLE and d.id != driver_id
        and d.docs_verified and "frozen" not in d.flags
    )
    eta = eta_minutes(w.city, driver, trip.pickup)

    trip.status = TripStatus.MATCHED
    trip.driver_id = driver_id
    trip.matched_at = w.clock.now
    driver.status = DriverStatus.DISPATCHED
    driver.current_trip_id = trip_id
    driver.target_location = trip.pickup

    w.dispatch_log.append(DispatchLogEntry(
        trip_id=trip_id,
        driver_id=driver_id,
        decided_at=w.clock.now,
        eta_minutes_at_dispatch=eta,
        alternatives_considered=n_alt,
    ))
    return {"trip": _trip_dict(trip), "eta_minutes": eta}


@router.post("/cancel_trip")
def cancel_trip(
    request: Request,
    trip_id: int = Body(...),
    reason: str = Body(...),
    cancelled_by: str = Body(default="system"),
) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    trip = w.trips.get(trip_id)
    if trip is None:
        raise HTTPException(status_code=404, detail="trip_not_found")
    if trip.status in (TripStatus.COMPLETED, TripStatus.CANCELLED):
        raise HTTPException(status_code=422, detail=f"trip_terminal:{trip.status.value}")
    trip.status = TripStatus.CANCELLED
    trip.cancelled_at = w.clock.now
    trip.cancelled_by = cancelled_by
    trip.cancel_reason = reason
    if trip.driver_id is not None:
        d = w.drivers.get(trip.driver_id)
        if d is not None:
            d.status = DriverStatus.IDLE
            d.target_location = None
            d.current_trip_id = None
            if cancelled_by == "driver":
                d.cancellation_count_today += 1
    return {"trip": _trip_dict(trip)}


@router.post("/auto_match_nearest")
def auto_match_nearest(
    request: Request,
    trip_id: int = Body(...),
) -> dict[str, Any]:
    """Convenience: pick the nearest idle driver and match. Useful for the
    realtime task agent if it doesn't want to enumerate manually."""
    w = get_world(get_tenant_id(request))
    trip = w.trips.get(trip_id)
    if trip is None:
        raise HTTPException(status_code=404, detail="trip_not_found")
    driver = nearest_driver_for(
        w.city, w.drivers, trip.pickup, vehicle_type=trip.vehicle_type)
    if driver is None:
        raise HTTPException(status_code=409, detail="no_idle_drivers")
    return match_ride(request, trip_id=trip_id, driver_id=driver.id)


@router.post("/set_surge")
def set_surge(
    request: Request,
    zone_id: str = Body(...),
    multiplier: float = Body(...),
    ttl_minutes: float = Body(default=5.0),
) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    if w.city.zone_by_id(zone_id) is None:
        raise HTTPException(status_code=404, detail="zone_not_found")
    m = max(float(SURGE_MIN), min(float(SURGE_MAX), float(multiplier)))
    w.surge_zones[zone_id] = Decimal(str(round(m, 2)))
    w.metadata.setdefault("surge_overrides", {})[zone_id] = {
        "multiplier": str(m), "set_at": w.clock.now, "ttl_minutes": ttl_minutes,
    }
    return {"ok": True, "zone_id": zone_id, "multiplier": str(w.surge_zones[zone_id])}


@router.post("/rebalance_driver")
def rebalance_driver(
    request: Request,
    driver_id: int = Body(...),
    target_zone_id: str = Body(...),
) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    driver = w.drivers.get(driver_id)
    if driver is None:
        raise HTTPException(status_code=404, detail="driver_not_found")
    target_zone = w.city.zone_by_id(target_zone_id)
    if target_zone is None:
        raise HTTPException(status_code=404, detail="zone_not_found")
    if driver.status != DriverStatus.IDLE:
        raise HTTPException(status_code=422, detail=f"driver_not_idle:{driver.status.value}")
    # Set the centroid of the target zone as the driver's home; physically
    # teleport (real apps would offer a guidance/incentive — we abstract).
    driver.home_zone_id = target_zone_id
    driver.location = target_zone.centroid
    return {"ok": True, "driver_id": driver_id, "new_zone": target_zone_id}


@router.post("/offer_incentive")
def offer_incentive(
    request: Request,
    driver_id: int = Body(...),
    type: str = Body(default="bonus"),
    amount: float = Body(default=10.0),
) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    driver = w.drivers.get(driver_id)
    if driver is None:
        raise HTTPException(status_code=404, detail="driver_not_found")
    w.metadata.setdefault("incentives", []).append({
        "driver_id": driver_id, "type": type, "amount": amount, "at": w.clock.now,
    })
    return {"ok": True}


@router.get("/zone_supply_demand")
def zone_supply_demand(request: Request) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    out: list[dict[str, Any]] = []
    for z in w.city.zones:
        demand = sum(
            1 for t in w.trips.values()
            if t.status == TripStatus.REQUESTED and t.pickup_zone_id == z.id
        )
        supply = sum(
            1 for d in w.drivers.values()
            if d.status == DriverStatus.IDLE and d.home_zone_id == z.id
        )
        out.append({
            "zone_id": z.id,
            "name": z.name,
            "demand": demand,
            "supply": supply,
            "current_surge": str(w.surge_zones.get(z.id, SURGE_MIN)),
        })
    return {"zones": out, "now": w.clock.now}


@router.get("/dispatch_log")
def get_dispatch_log(request: Request, limit: int = Query(100, le=2000)) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    return {
        "dispatch_log": [
            {"trip_id": e.trip_id, "driver_id": e.driver_id,
             "decided_at": e.decided_at,
             "eta_minutes_at_dispatch": e.eta_minutes_at_dispatch,
             "alternatives_considered": e.alternatives_considered}
            for e in w.dispatch_log[-limit:]
        ],
    }


def _trip_dict(t) -> dict[str, Any]:
    return {
        "id": t.id, "rider_id": t.rider_id, "driver_id": t.driver_id,
        "status": t.status.value, "pickup": list(t.pickup),
        "dropoff": list(t.dropoff),
        "pickup_zone_id": t.pickup_zone_id, "dropoff_zone_id": t.dropoff_zone_id,
        "vehicle_type": t.vehicle_type,
        "requested_at": t.requested_at, "matched_at": t.matched_at,
        "completed_at": t.completed_at, "cancelled_at": t.cancelled_at,
        "cancelled_by": t.cancelled_by, "cancel_reason": t.cancel_reason,
        "surge_at_request": str(t.surge_at_request), "fare": t.fare,
        "flags": list(t.flags),
    }
