"""FastAPI app composition."""

from __future__ import annotations

from fastapi import FastAPI

from rideshare_gym.mock_server.perturbations import PerturbationMiddleware
from rideshare_gym.mock_server.routers import (
    admin,
    disputes,
    drivers,
    fraud,
    lost_items,
    marketplace,
    messaging,
    pricing,
    refunds,
    riders,
    safety,
    trips,
)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Rideshare Mock Admin API (gym)",
        version="2026-04-mock",
        description=(
            "Self-hosted mock of a ride-sharing platform's Admin API used by "
            "the rideshare-gym RL benchmark. Per-tenant World instances; "
            "honours perturbations installed via /admin/perturb. "
            "Not intended for production use."
        ),
    )
    app.add_middleware(PerturbationMiddleware)
    app.include_router(admin.router)
    app.include_router(marketplace.router)
    app.include_router(trips.router)
    app.include_router(refunds.router)
    app.include_router(disputes.router)
    app.include_router(safety.router)
    app.include_router(fraud.router)
    app.include_router(lost_items.router)
    app.include_router(drivers.router)
    app.include_router(riders.router)
    app.include_router(pricing.router)
    app.include_router(messaging.router)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": "2026-04-mock"}

    return app


app = create_app()
