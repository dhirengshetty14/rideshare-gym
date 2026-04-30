"""Lost-item recovery flow — modelled on Uber's actual workflow."""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query, Request

from rideshare_gym.mock_server.perturbations import get_tenant_id
from rideshare_gym.mock_server.store import get_world
from rideshare_gym.world.world import LostItem, SentMessage

router = APIRouter(prefix="/api/v1/lost_items", tags=["lost_items"])


@router.get("")
def list_lost_items(
    request: Request,
    trip_id: int | None = Query(None),
    confirmed: bool | None = Query(None),
) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    out = list(w.lost_items.values())
    if trip_id is not None:
        out = [li for li in out if li.trip_id == trip_id]
    if confirmed is not None:
        if confirmed:
            out = [li for li in out if li.confirmed_at is not None]
        else:
            out = [li for li in out if li.confirmed_at is None]
    return {"lost_items": [_lost_dict(li) for li in out]}


@router.post("")
def create_lost_item(
    request: Request,
    trip_id: int = Body(...),
    description: str = Body(...),
) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    trip = w.trips.get(trip_id)
    if trip is None:
        raise HTTPException(status_code=404, detail="trip_not_found")
    li_id = w.next_id()
    li = LostItem(
        id=li_id, trip_id=trip_id, description=description,
        reported_at=w.clock.now,
    )
    w.lost_items[li_id] = li
    trip.lost_item_id = li_id
    return {"lost_item": _lost_dict(li)}


@router.post("/{lost_item_id}/assign")
def assign_lost_item(
    request: Request,
    lost_item_id: int,
    driver_id: int = Body(...),
    return_method: str = Body(default="next_idle_window"),
) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    li = w.lost_items.get(lost_item_id)
    if li is None:
        raise HTTPException(status_code=404, detail="lost_item_not_found")
    if driver_id not in w.drivers:
        raise HTTPException(status_code=404, detail="driver_not_found")
    li.assigned_driver_id = driver_id
    li.return_method = return_method
    li.handoff_code = uuid.uuid4().hex[:6].upper()
    return {"lost_item": _lost_dict(li)}


@router.post("/{lost_item_id}/schedule_pickup")
def schedule_pickup(
    request: Request,
    lost_item_id: int,
    pickup_at: float = Body(...),    # absolute sim seconds
    pickup_location: list[float] = Body(...),
    notify_rider: bool = Body(default=True),
    notify_driver: bool = Body(default=True),
) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    li = w.lost_items.get(lost_item_id)
    if li is None:
        raise HTTPException(status_code=404, detail="lost_item_not_found")
    if li.assigned_driver_id is None:
        raise HTTPException(status_code=422, detail="lost_item_not_assigned")
    li.scheduled_pickup_at = pickup_at
    li.return_pickup_location = (float(pickup_location[0]), float(pickup_location[1]))
    trip = w.trips.get(li.trip_id)
    if notify_rider and trip is not None:
        w.sent_messages.append(SentMessage(
            id=w.next_id(), to=f"rider:{trip.rider_id}",
            template="lost_item.pickup_scheduled",
            variables={"lost_item_id": li.id, "pickup_at": pickup_at,
                       "code": li.handoff_code},
            sent_at=w.clock.now,
        ))
    if notify_driver:
        w.sent_messages.append(SentMessage(
            id=w.next_id(), to=f"driver:{li.assigned_driver_id}",
            template="lost_item.return_pickup",
            variables={"lost_item_id": li.id, "pickup_at": pickup_at,
                       "code": li.handoff_code},
            sent_at=w.clock.now,
        ))
    return {"lost_item": _lost_dict(li)}


@router.post("/{lost_item_id}/confirm_handoff")
def confirm_handoff(
    request: Request,
    lost_item_id: int,
    code: str = Body(...),
) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    li = w.lost_items.get(lost_item_id)
    if li is None:
        raise HTTPException(status_code=404, detail="lost_item_not_found")
    if code != li.handoff_code:
        raise HTTPException(status_code=422, detail="invalid_handoff_code")
    li.confirmed_at = w.clock.now
    return {"lost_item": _lost_dict(li)}


def _lost_dict(li) -> dict[str, Any]:
    return {
        "id": li.id, "trip_id": li.trip_id, "description": li.description,
        "reported_at": li.reported_at,
        "assigned_driver_id": li.assigned_driver_id,
        "return_method": li.return_method,
        "scheduled_pickup_at": li.scheduled_pickup_at,
        "return_pickup_location": (list(li.return_pickup_location)
                                     if li.return_pickup_location else None),
        "handoff_code": li.handoff_code,
        "confirmed_at": li.confirmed_at,
    }
