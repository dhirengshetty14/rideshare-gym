"""Rider-side endpoints — accounts, freeze/restore, login history."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException, Request

from rideshare_gym.mock_server.perturbations import get_tenant_id
from rideshare_gym.mock_server.store import get_world
from rideshare_gym.world.world import AccountAction

router = APIRouter(prefix="/api/v1/riders", tags=["riders"])


@router.get("/{rider_id}")
def get_rider(request: Request, rider_id: int) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    r = w.riders.get(rider_id)
    if r is None:
        raise HTTPException(status_code=404, detail="rider_not_found")
    return {
        "rider": {
            "id": r.id, "name": r.name, "email": r.email, "phone": r.phone,
            "rating": r.rating, "payment_method_id": r.payment_method_id,
            "payment_bin": r.payment_bin, "device_fingerprint": r.device_fingerprint,
            "home_zone_id": r.home_zone_id, "typical_login_zone_id": r.typical_login_zone_id,
            "completed_trips": r.completed_trips, "chargeback_count": r.chargeback_count,
            "lifetime_spent": str(r.lifetime_spent), "created_at": r.created_at,
            "flags": list(r.flags),
        }
    }


@router.post("/{rider_id}/freeze")
def freeze_rider(
    request: Request,
    rider_id: int,
    reason: str = Body(...),
) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    r = w.riders.get(rider_id)
    if r is None:
        raise HTTPException(status_code=404, detail="rider_not_found")
    if "frozen" not in r.flags:
        r.flags.append("frozen")
    w.account_actions.append(AccountAction(
        id=w.next_id(), target_kind="rider", target_id=str(rider_id),
        action="freeze", reason=reason, at=w.clock.now,
    ))
    return {"ok": True, "flags": list(r.flags)}


@router.post("/{rider_id}/restore")
def restore_rider(
    request: Request,
    rider_id: int,
    reason: str = Body(default=""),
) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    r = w.riders.get(rider_id)
    if r is None:
        raise HTTPException(status_code=404, detail="rider_not_found")
    r.flags = [f for f in r.flags if f != "frozen"]
    w.account_actions.append(AccountAction(
        id=w.next_id(), target_kind="rider", target_id=str(rider_id),
        action="restore", reason=reason, at=w.clock.now,
    ))
    return {"ok": True, "flags": list(r.flags)}


@router.get("/{rider_id}/login_history")
def login_history(request: Request, rider_id: int) -> dict[str, Any]:
    """Stub login history — real apps log every auth attempt with device, location,
    timestamp. We synthesise from typical_login_* fields + recent anomalies on
    the rider record."""
    w = get_world(get_tenant_id(request))
    r = w.riders.get(rider_id)
    if r is None:
        raise HTTPException(status_code=404, detail="rider_not_found")
    typical = {
        "device_fingerprint": r.device_fingerprint,
        "zone_id": r.typical_login_zone_id,
        "hour_window": list(r.typical_login_hour_window),
    }
    recent_anomalies = w.metadata.get("login_anomalies", {}).get(str(rider_id), [])
    return {
        "rider_id": rider_id,
        "typical": typical,
        "recent_anomalies": recent_anomalies,
    }
