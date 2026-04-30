"""Trip CRUD + history + GPS log."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from rideshare_gym.mock_server.perturbations import get_tenant_id
from rideshare_gym.mock_server.store import get_world
from rideshare_gym.world.trips import TripStatus

router = APIRouter(prefix="/api/v1", tags=["trips"])


@router.get("/trips")
def list_trips(
    request: Request,
    rider_id: int | None = Query(None),
    driver_id: int | None = Query(None),
    status: str | None = Query(None),
    limit: int = Query(100, le=500),
) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    out = list(w.trips.values())
    if rider_id is not None:
        out = [t for t in out if t.rider_id == rider_id]
    if driver_id is not None:
        out = [t for t in out if t.driver_id == driver_id]
    if status is not None:
        out = [t for t in out if t.status.value == status]
    out.sort(key=lambda t: t.requested_at, reverse=True)
    return {
        "trips": [
            {
                "id": t.id, "rider_id": t.rider_id, "driver_id": t.driver_id,
                "status": t.status.value,
                "pickup_zone_id": t.pickup_zone_id, "dropoff_zone_id": t.dropoff_zone_id,
                "vehicle_type": t.vehicle_type,
                "requested_at": t.requested_at, "completed_at": t.completed_at,
                "cancelled_at": t.cancelled_at, "cancelled_by": t.cancelled_by,
                "cancel_reason": t.cancel_reason,
                "surge_at_request": str(t.surge_at_request),
                "fare_total": t.fare["total"] if t.fare else None,
                "flags": list(t.flags),
            }
            for t in out[:limit]
        ],
    }


@router.get("/trips/{trip_id}")
def get_trip(request: Request, trip_id: int) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    t = w.trips.get(trip_id)
    if t is None:
        raise HTTPException(status_code=404, detail="trip_not_found")
    rider = w.riders.get(t.rider_id)
    return {
        "trip": {
            "id": t.id, "rider_id": t.rider_id, "driver_id": t.driver_id,
            "status": t.status.value,
            "pickup": list(t.pickup), "dropoff": list(t.dropoff),
            "pickup_zone_id": t.pickup_zone_id, "dropoff_zone_id": t.dropoff_zone_id,
            "vehicle_type": t.vehicle_type,
            "requested_at": t.requested_at, "matched_at": t.matched_at,
            "pickup_arrived_at": t.pickup_arrived_at,
            "picked_up_at": t.picked_up_at, "completed_at": t.completed_at,
            "cancelled_at": t.cancelled_at, "cancelled_by": t.cancelled_by,
            "cancel_reason": t.cancel_reason,
            "surge_at_request": str(t.surge_at_request),
            "fare": t.fare,
            "rider_rating_of_driver": t.rider_rating_of_driver,
            "driver_rating_of_rider": t.driver_rating_of_rider,
            "tip_amount": str(t.tip_amount),
            "refund_id": t.refund_id, "dispute_id": t.dispute_id,
            "lost_item_id": t.lost_item_id, "incident_id": t.incident_id,
            "flags": list(t.flags),
            "metadata": t.metadata,
            "rider_email": rider.email if rider else "",
            "rider_payment_bin": rider.payment_bin if rider else "",
            "rider_device_fingerprint": rider.device_fingerprint if rider else "",
        }
    }


@router.get("/trips/{trip_id}/gps_log")
def get_trip_gps(request: Request, trip_id: int) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    t = w.trips.get(trip_id)
    if t is None:
        raise HTTPException(status_code=404, detail="trip_not_found")
    return {
        "trip_id": trip_id,
        "gps_log": [
            {"t": p.t, "location": list(p.location), "speed_kmh": p.speed_kmh}
            for p in t.gps_log
        ],
    }
