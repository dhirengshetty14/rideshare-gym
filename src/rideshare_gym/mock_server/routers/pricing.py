"""Pricing — fare quotes + surge zone listing."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request

from rideshare_gym.mock_server.perturbations import get_tenant_id
from rideshare_gym.mock_server.store import get_world
from rideshare_gym.world.pricing import compute_fare
from rideshare_gym.world.surge import SURGE_MIN

router = APIRouter(prefix="/api/v1/pricing", tags=["pricing"])


@router.get("/quote")
def quote(
    request: Request,
    pickup_x: float = Query(...),
    pickup_y: float = Query(...),
    dropoff_x: float = Query(...),
    dropoff_y: float = Query(...),
    vehicle_type: str = Query("uberx"),
) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    pickup = (pickup_x, pickup_y)
    dropoff = (dropoff_x, dropoff_y)
    distance_km = w.city.distance_km(pickup, dropoff)
    duration_min = w.city.travel_time_minutes(pickup, dropoff)
    pickup_zone = w.city.zone_for(pickup_x, pickup_y)
    surge = (
        w.surge_zones.get(pickup_zone.id, SURGE_MIN) if pickup_zone else SURGE_MIN
    )
    fare = compute_fare(
        distance_km=distance_km, duration_min=duration_min,
        surge=surge, vehicle_type=vehicle_type,  # type: ignore[arg-type]
    )
    return {
        "vehicle_type": vehicle_type,
        "distance_km": distance_km,
        "duration_min": duration_min,
        "surge": str(surge),
        "fare": {
            "base_fare": str(fare.base_fare),
            "distance_fare": str(fare.distance_fare),
            "time_fare": str(fare.time_fare),
            "surge_multiplier": str(fare.surge_multiplier),
            "booking_fee": str(fare.booking_fee),
            "safety_fee": str(fare.safety_fee),
            "subtotal": str(fare.subtotal),
            "tax": str(fare.tax),
            "total": str(fare.total),
        },
    }


@router.get("/zones")
def list_zones(request: Request) -> dict[str, Any]:
    w = get_world(get_tenant_id(request))
    return {
        "zones": [
            {
                "id": z.id, "name": z.name,
                "centroid": list(z.centroid), "radius_km": z.radius_km,
                "current_surge": str(w.surge_zones.get(z.id, SURGE_MIN)),
            } for z in w.city.zones
        ],
    }
