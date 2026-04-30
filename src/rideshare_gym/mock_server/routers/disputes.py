"""Chargeback disputes (rider files chargeback with their bank)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query, Request

from rideshare_gym.mock_server.perturbations import get_tenant_id
from rideshare_gym.mock_server.store import get_world

router = APIRouter(prefix="/api/v1", tags=["disputes"])


@router.get("/disputes")
def list_disputes(
    request: Request,
    status: str | None = Query(None),
) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    out = list(w.disputes.values())
    if status is not None:
        out = [d for d in out if d.status == status]
    return {"disputes": [_dispute_dict(d) for d in out]}


@router.get("/disputes/{dispute_id}")
def get_dispute(request: Request, dispute_id: int) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    d = w.disputes.get(dispute_id)
    if d is None:
        raise HTTPException(status_code=404, detail="dispute_not_found")
    return {"dispute": _dispute_dict(d)}


@router.post("/disputes/{dispute_id}/submit_response")
def submit_dispute_response(
    request: Request,
    dispute_id: int,
    response: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    d = w.disputes.get(dispute_id)
    if d is None:
        raise HTTPException(status_code=404, detail="dispute_not_found")
    if d.status != "needs_response":
        raise HTTPException(status_code=422, detail=f"dispute_not_actionable:{d.status}")
    if d.deadline and w.clock.now > d.deadline:
        raise HTTPException(status_code=422, detail="dispute_deadline_passed")
    d.response = dict(response)
    d.status = "under_review"
    return {"dispute": _dispute_dict(d)}


def _dispute_dict(d) -> dict[str, Any]:
    return {
        "id": d.id, "trip_id": d.trip_id, "reason": d.reason,
        "status": d.status, "response": d.response,
        "created_at": d.created_at, "deadline": d.deadline,
    }
