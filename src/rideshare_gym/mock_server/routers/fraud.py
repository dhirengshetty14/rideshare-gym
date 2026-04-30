"""Fraud detection — flag, freeze, ban devices."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query, Request

from rideshare_gym.mock_server.perturbations import get_tenant_id
from rideshare_gym.mock_server.store import get_world
from rideshare_gym.world.world import AccountAction

router = APIRouter(prefix="/api/v1/fraud", tags=["fraud"])


@router.get("/flagged_trips")
def list_flagged_trips(request: Request) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    flagged = [t for t in w.trips.values() if "flagged_fraud" in t.flags or t.dispute_id is not None]
    return {
        "flagged_trips": [
            {"id": t.id, "rider_id": t.rider_id, "driver_id": t.driver_id,
             "flags": list(t.flags), "dispute_id": t.dispute_id}
            for t in flagged
        ],
    }


@router.get("/flagged_accounts")
def list_flagged_accounts(request: Request) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    riders = [r for r in w.riders.values() if r.flags]
    drivers = [d for d in w.drivers.values() if d.flags]
    return {
        "flagged_riders": [
            {"id": r.id, "name": r.name, "email": r.email,
             "payment_bin": r.payment_bin, "device_fingerprint": r.device_fingerprint,
             "flags": list(r.flags)}
            for r in riders
        ],
        "flagged_drivers": [
            {"id": d.id, "name": d.name, "vehicle_type": d.vehicle_type,
             "device_fingerprint": d.device_fingerprint,
             "flags": list(d.flags)}
            for d in drivers
        ],
    }


@router.post("/freeze_account")
def freeze_account(
    request: Request,
    target_kind: str = Body(...),
    target_id: int = Body(...),
    reason: str = Body(...),
) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    if target_kind == "rider":
        r = w.riders.get(target_id)
        if r is None:
            raise HTTPException(status_code=404, detail="rider_not_found")
        if "frozen" not in r.flags:
            r.flags.append("frozen")
    elif target_kind == "driver":
        d = w.drivers.get(target_id)
        if d is None:
            raise HTTPException(status_code=404, detail="driver_not_found")
        if "frozen" not in d.flags:
            d.flags.append("frozen")
    else:
        raise HTTPException(status_code=400, detail="bad_target_kind")
    w.account_actions.append(AccountAction(
        id=w.next_id(), target_kind=target_kind, target_id=str(target_id),
        action="freeze", reason=reason, at=w.clock.now,
    ))
    return {"ok": True}


@router.post("/ban_device")
def ban_device(
    request: Request,
    fingerprint: str = Body(...),
    reason: str = Body(...),
) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    w.banned_devices[fingerprint] = reason
    w.account_actions.append(AccountAction(
        id=w.next_id(), target_kind="device", target_id=fingerprint,
        action="ban", reason=reason, at=w.clock.now,
    ))
    return {"ok": True, "fingerprint": fingerprint}


@router.get("/cluster_by_fingerprint")
def cluster_by_fingerprint(
    request: Request,
    fingerprint: str = Query(...),
) -> dict[str, Any]:
    """Return all riders + drivers + trips that share the given fingerprint."""
    w = get_world(get_tenant_id(request))
    riders = [r for r in w.riders.values() if r.device_fingerprint == fingerprint]
    drivers = [d for d in w.drivers.values() if d.device_fingerprint == fingerprint]
    rider_ids = {r.id for r in riders}
    trips = [t for t in w.trips.values() if t.rider_id in rider_ids]
    return {
        "fingerprint": fingerprint,
        "rider_ids": [r.id for r in riders],
        "driver_ids": [d.id for d in drivers],
        "trip_ids": [t.id for t in trips],
        "n_riders": len(riders), "n_drivers": len(drivers), "n_trips": len(trips),
    }
