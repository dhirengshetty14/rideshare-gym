"""Driver-side endpoints — docs, payouts, performance."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request

from rideshare_gym.mock_server.perturbations import get_tenant_id
from rideshare_gym.mock_server.store import get_world
from rideshare_gym.world.world import PayoutAdjustment

router = APIRouter(prefix="/api/v1/drivers", tags=["drivers"])


@router.get("/{driver_id}")
def get_driver(request: Request, driver_id: int) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    d = w.drivers.get(driver_id)
    if d is None:
        raise HTTPException(status_code=404, detail="driver_not_found")
    return {
        "driver": {
            "id": d.id, "name": d.name, "location": list(d.location),
            "status": d.status.value, "rating": d.rating,
            "vehicle_type": d.vehicle_type, "current_trip_id": d.current_trip_id,
            "docs_verified": d.docs_verified, "docs_expiry": d.docs_expiry,
            "home_zone_id": d.home_zone_id,
            "completed_trips_today": d.completed_trips_today,
            "cumulative_earnings_today": str(d.cumulative_earnings_today),
            "cancellation_count_today": d.cancellation_count_today,
            "device_fingerprint": d.device_fingerprint,
            "flags": list(d.flags),
        }
    }


@router.get("/{driver_id}/documents")
def get_driver_documents(request: Request, driver_id: int) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    d = w.drivers.get(driver_id)
    if d is None:
        raise HTTPException(status_code=404, detail="driver_not_found")
    return {
        "driver_id": driver_id,
        "docs_verified": d.docs_verified,
        "docs_expiry": d.docs_expiry,
        "documents": [
            {"kind": "drivers_license", "expiry": d.docs_expiry, "verified": d.docs_verified},
            {"kind": "vehicle_registration", "expiry": "2027-01-01", "verified": True},
            {"kind": "background_check", "expiry": "2027-12-31", "verified": True},
        ],
    }


@router.post("/{driver_id}/verify_documents")
def verify_documents(request: Request, driver_id: int) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    d = w.drivers.get(driver_id)
    if d is None:
        raise HTTPException(status_code=404, detail="driver_not_found")
    d.docs_verified = True
    return {"ok": True, "driver_id": driver_id, "docs_verified": True}


@router.post("/{driver_id}/payout_adjust")
def payout_adjust(
    request: Request,
    driver_id: int,
    amount: float = Body(...),
    reason: str = Body(...),
    related_trip_id: int | None = Body(default=None),
) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    d = w.drivers.get(driver_id)
    if d is None:
        raise HTTPException(status_code=404, detail="driver_not_found")
    pid = w.next_id()
    p = PayoutAdjustment(
        id=pid, driver_id=driver_id,
        amount=Decimal(str(round(amount, 2))),
        reason=reason, related_trip_id=related_trip_id,
        issued_at=w.clock.now,
    )
    w.payout_adjustments.append(p)
    d.cumulative_earnings_today += p.amount
    return {"adjustment": {"id": pid, "driver_id": driver_id,
                            "amount": str(p.amount), "reason": reason,
                            "related_trip_id": related_trip_id}}


@router.get("/{driver_id}/performance")
def get_performance(request: Request, driver_id: int) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    d = w.drivers.get(driver_id)
    if d is None:
        raise HTTPException(status_code=404, detail="driver_not_found")
    completed_today = sum(
        1 for t in w.trips.values()
        if t.driver_id == driver_id and t.status.value == "completed"
    )
    return {
        "driver_id": driver_id,
        "rating": d.rating,
        "completed_trips_today": completed_today,
        "cancellation_count_today": d.cancellation_count_today,
        "cumulative_earnings_today": str(d.cumulative_earnings_today),
    }
