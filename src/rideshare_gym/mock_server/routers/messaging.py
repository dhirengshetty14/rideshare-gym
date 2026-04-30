"""Messaging — push/SMS to riders and drivers (captured for verifier use)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query, Request

from rideshare_gym.mock_server.perturbations import get_tenant_id
from rideshare_gym.mock_server.store import get_world
from rideshare_gym.world.world import SentMessage

router = APIRouter(prefix="/api/v1/messaging", tags=["messaging"])


@router.post("/send_to_rider")
def send_to_rider(
    request: Request,
    rider_id: int = Body(...),
    template: str = Body(...),
    variables: dict[str, Any] = Body(default_factory=dict),
) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    if rider_id not in w.riders:
        raise HTTPException(status_code=404, detail="rider_not_found")
    msg = SentMessage(
        id=w.next_id(), to=f"rider:{rider_id}", template=template,
        variables=dict(variables), sent_at=w.clock.now,
    )
    w.sent_messages.append(msg)
    return {"ok": True, "message_id": msg.id}


@router.post("/send_to_driver")
def send_to_driver(
    request: Request,
    driver_id: int = Body(...),
    template: str = Body(...),
    variables: dict[str, Any] = Body(default_factory=dict),
) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    if driver_id not in w.drivers:
        raise HTTPException(status_code=404, detail="driver_not_found")
    msg = SentMessage(
        id=w.next_id(), to=f"driver:{driver_id}", template=template,
        variables=dict(variables), sent_at=w.clock.now,
    )
    w.sent_messages.append(msg)
    return {"ok": True, "message_id": msg.id}


@router.get("/messages")
def list_messages(
    request: Request,
    to: str | None = Query(None),
    template: str | None = Query(None),
) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    out = w.sent_messages
    if to is not None:
        out = [m for m in out if m.to == to]
    if template is not None:
        out = [m for m in out if m.template == template]
    return {
        "messages": [
            {"id": m.id, "to": m.to, "template": m.template,
             "variables": m.variables, "sent_at": m.sent_at}
            for m in out
        ],
    }
