"""FastAPI middleware for ride-sharing-specific perturbations.

Honours the per-tenant `world.perturbations` list. Standard kinds (latency,
rate_limit, malformed_response) plus ride-sharing-specific ones (gps_dropout,
payment_3ds, driver_offline_mid_trip, eta_variance, traffic_event,
fraud_pattern_drift, messaging_delay, system_outage_partial).
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.middleware.base import BaseHTTPMiddleware

from rideshare_gym.mock_server.store import get_world

TENANT_HEADER = "X-Rideshare-Tenant"


def get_tenant_id(request: Request) -> str:
    return request.headers.get(TENANT_HEADER, "default")


def perturbations_for(request: Request) -> list[dict[str, Any]]:
    return list(get_world(get_tenant_id(request)).perturbations)


class PerturbationMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Gym-internal admin endpoints are never perturbed.
        if request.url.path.startswith("/admin/") and "/admin/api/" not in request.url.path:
            return await call_next(request)
        if request.url.path == "/health":
            return await call_next(request)

        perts = perturbations_for(request)

        # Rate limit — return early.
        for p in perts:
            if p.get("kind") != "rate_limit":
                continue
            params = p.get("params") or {}
            if random.random() < float(params.get("p", 0.0)):
                return JSONResponse(
                    {"errors": "rate_limited"},
                    status_code=int(params.get("status", 429)),
                    headers={"Retry-After": str(params.get("retry_after", 1))},
                )

        # System partial outage — selectively 503 a router family.
        for p in perts:
            if p.get("kind") != "system_outage_partial":
                continue
            params = p.get("params") or {}
            target = params.get("router", "")
            if target and target in request.url.path:
                if random.random() < float(params.get("p", 1.0)):
                    return JSONResponse(
                        {"errors": "service_unavailable", "retry_after": 5},
                        status_code=503,
                    )

        # Latency.
        for p in perts:
            if p.get("kind") != "latency":
                continue
            params = p.get("params") or {}
            if random.random() < float(params.get("p", 1.0)):
                ms = int(params.get("ms", 200))
                await asyncio.sleep(ms / 1000.0)

        response = await call_next(request)

        # Malformed response — corrupt body for some pct.
        for p in perts:
            if p.get("kind") != "malformed_response":
                continue
            params = p.get("params") or {}
            target = params.get("endpoint", "*")
            if target != "*" and target not in request.url.path:
                continue
            if random.random() < float(params.get("p", 0.05)):
                return PlainTextResponse(
                    "this is not valid json",
                    status_code=response.status_code,
                    media_type="application/json",
                )

        return response


def is_partial_failure_active(
    perturbations: list[dict[str, Any]], *, action: str, step: str,
) -> bool:
    """Routers call this when about to perform a multi-step write."""
    for p in perturbations:
        if p.get("kind") != "partial_failure":
            continue
        params = p.get("params") or {}
        if params.get("action") not in (None, "*", action):
            continue
        if params.get("step") not in (None, "*", step):
            continue
        if random.random() < float(params.get("p", 1.0)):
            return True
    return False
