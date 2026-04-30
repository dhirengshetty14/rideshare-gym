"""Test/gym-only admin endpoints (not part of any real ride-sharing API)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Query

from rideshare_gym.mock_server.seed import seed_drivers, seed_riders
from rideshare_gym.mock_server.store import all_tenants, get_world, reset_world

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/reset")
def reset(tenant: str = Query(...)) -> dict[str, Any]:
    reset_world(tenant)
    return {"ok": True, "tenant": tenant}


@router.post("/snapshot")
def snapshot(tenant: str = Query(...)) -> dict[str, Any]:
    return get_world(tenant).snapshot()


@router.post("/perturb")
def perturb(
    tenant: str = Query(...),
    perturbations: list[dict[str, Any]] = Body(...),
) -> dict[str, Any]:
    w = get_world(tenant)
    w.perturbations = list(perturbations)
    return {"ok": True, "n_active": len(w.perturbations)}


@router.post("/clear_perturbations")
def clear_perturbations(tenant: str = Query(...)) -> dict[str, Any]:
    w = get_world(tenant)
    w.perturbations.clear()
    return {"ok": True}


@router.post("/tick")
def tick(
    tenant: str = Query(...),
    dt_seconds: float = Body(default=30.0, embed=True),
) -> dict[str, Any]:
    w = get_world(tenant)
    w.tick(dt_seconds)
    return {"ok": True, "now": w.clock.now, "kpis": w.kpis()}


@router.post("/seed")
def seed_endpoint(
    tenant: str = Query(...),
    n_drivers: int = Body(default=50),
    n_riders: int = Body(default=100),
    seed: int = Body(default=0),
    online_pct: float = Body(default=0.85),
) -> dict[str, Any]:
    w = get_world(tenant)
    w.reseed(seed)
    drivers = seed_drivers(w, n_drivers=n_drivers, seed=seed, online_pct=online_pct)
    riders = seed_riders(w, n_riders=n_riders, seed=seed)
    return {"ok": True, "drivers": len(drivers), "riders": len(riders)}


@router.post("/set_metadata")
def set_metadata(
    tenant: str = Query(...),
    metadata: dict[str, Any] = Body(...),
) -> dict[str, Any]:
    get_world(tenant).metadata.update(metadata)
    return {"ok": True}


@router.get("/metadata")
def get_metadata(tenant: str = Query(...)) -> dict[str, Any]:
    return get_world(tenant).metadata


@router.get("/tenants")
def list_tenants() -> dict[str, Any]:
    return {"tenants": all_tenants()}


@router.post("/inject_event")
def inject_event(
    tenant: str = Query(...),
    kind: str = Body(...),
    started_at: float = Body(default=0.0),
    duration_seconds: float = Body(default=300.0),
    affected_zones: list[str] = Body(default_factory=list),
    severity: float = Body(default=0.5),
    metadata: dict[str, Any] = Body(default_factory=dict),
) -> dict[str, Any]:
    from rideshare_gym.world.events import WorldEvent

    w = get_world(tenant)
    e = WorldEvent(
        kind=kind,  # type: ignore[arg-type]
        started_at=started_at if started_at > 0 else w.clock.now,
        duration_seconds=duration_seconds,
        affected_zones=list(affected_zones),
        severity=severity,
        metadata=dict(metadata),
    )
    w.events.append(e)
    return {"ok": True, "event_count": len(w.events)}
