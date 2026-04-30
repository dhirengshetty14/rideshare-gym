"""Refunds + payment credits."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query, Request

from rideshare_gym.mock_server.perturbations import get_tenant_id
from rideshare_gym.mock_server.store import get_world
from rideshare_gym.world.world import Refund, SentMessage

router = APIRouter(prefix="/api/v1", tags=["refunds"])


@router.post("/refunds")
def create_refund(
    request: Request,
    trip_id: int = Body(...),
    amount: float = Body(...),
    reason: str = Body(...),
    notify_rider: bool = Body(default=True),
) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    trip = w.trips.get(trip_id)
    if trip is None:
        raise HTTPException(status_code=404, detail="trip_not_found")
    rid = w.next_id()
    r = Refund(
        id=rid, trip_id=trip_id, amount=Decimal(str(round(amount, 2))),
        reason=reason, issued_at=w.clock.now, notify_rider=notify_rider,
    )
    w.refunds[rid] = r
    trip.refund_id = rid
    if notify_rider:
        w.sent_messages.append(SentMessage(
            id=w.next_id(), to=f"rider:{trip.rider_id}", template="trip.refunded",
            variables={"trip_id": trip_id, "amount": str(r.amount), "reason": reason},
            sent_at=w.clock.now,
        ))
    return {"refund": {"id": rid, "trip_id": trip_id, "amount": str(r.amount),
                        "reason": reason, "issued_at": w.clock.now}}


@router.get("/refunds")
def list_refunds(
    request: Request,
    trip_id: int | None = Query(None),
) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    out = list(w.refunds.values())
    if trip_id is not None:
        out = [r for r in out if r.trip_id == trip_id]
    return {
        "refunds": [
            {"id": r.id, "trip_id": r.trip_id, "amount": str(r.amount),
             "reason": r.reason, "issued_at": r.issued_at}
            for r in out
        ],
    }
