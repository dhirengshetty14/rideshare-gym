"""Safety incidents — T1/T2/T3 escalation tiers."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query, Request

from rideshare_gym.mock_server.perturbations import get_tenant_id
from rideshare_gym.mock_server.store import get_world
from rideshare_gym.world.world import SafetyIncident, SentMessage

router = APIRouter(prefix="/api/v1/safety", tags=["safety"])


@router.get("/incidents")
def list_incidents(
    request: Request,
    severity_min: int | None = Query(None),
) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    out = list(w.incidents.values())
    if severity_min is not None:
        out = [i for i in out if i.severity >= severity_min]
    return {
        "incidents": [
            {
                "id": i.id, "trip_id": i.trip_id, "kind": i.kind,
                "severity": i.severity, "reported_at": i.reported_at,
                "escalation_level": i.escalation_level,
                "emergency_contacted": i.emergency_contacted,
                "evidence": list(i.evidence),
            } for i in out
        ],
    }


@router.get("/incidents/{incident_id}")
def get_incident(request: Request, incident_id: int) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    i = w.incidents.get(incident_id)
    if i is None:
        raise HTTPException(status_code=404, detail="incident_not_found")
    return {
        "incident": {
            "id": i.id, "trip_id": i.trip_id, "kind": i.kind,
            "severity": i.severity, "reported_at": i.reported_at,
            "escalation_level": i.escalation_level,
            "emergency_contacted": i.emergency_contacted,
            "evidence": list(i.evidence),
        }
    }


@router.post("/escalate")
def escalate_incident(
    request: Request,
    incident_id: int = Body(...),
    level: int = Body(...),
    notify_parties: bool = Body(default=True),
) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    i = w.incidents.get(incident_id)
    if i is None:
        raise HTTPException(status_code=404, detail="incident_not_found")
    if level < i.escalation_level:
        raise HTTPException(status_code=422, detail="cannot_de_escalate")
    i.escalation_level = level
    if notify_parties:
        trip = w.trips.get(i.trip_id)
        if trip is not None:
            w.sent_messages.append(SentMessage(
                id=w.next_id(), to=f"rider:{trip.rider_id}",
                template="safety.incident_update",
                variables={"incident_id": i.id, "level": level},
                sent_at=w.clock.now,
            ))
            if trip.driver_id is not None:
                w.sent_messages.append(SentMessage(
                    id=w.next_id(), to=f"driver:{trip.driver_id}",
                    template="safety.incident_update",
                    variables={"incident_id": i.id, "level": level},
                    sent_at=w.clock.now,
                ))
    return {"ok": True, "incident_id": i.id, "level": i.escalation_level}


@router.post("/contact_emergency")
def contact_emergency(
    request: Request,
    incident_id: int = Body(...),
    kind: str = Body(default="911"),
) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    i = w.incidents.get(incident_id)
    if i is None:
        raise HTTPException(status_code=404, detail="incident_not_found")
    i.emergency_contacted = True
    return {"ok": True, "kind": kind, "incident_id": i.id}


@router.post("/attach_evidence")
def attach_evidence(
    request: Request,
    incident_id: int = Body(...),
    evidence_kind: str = Body(...),
    note: str = Body(default=""),
) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    i = w.incidents.get(incident_id)
    if i is None:
        raise HTTPException(status_code=404, detail="incident_not_found")
    i.evidence.append(f"{evidence_kind}: {note}")
    return {"ok": True, "n_evidence": len(i.evidence)}
