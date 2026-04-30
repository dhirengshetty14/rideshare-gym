"""RideshareSandbox — concrete Sandbox impl backed by an httpx/TestClient
to the mock ride-sharing server. Mirrors shopify-gym's ShopifySandbox shape.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RideshareClient:
    """Typed wrapper. Each method maps to one mock endpoint."""

    http: Any
    tenant_id: str

    def _get(self, path: str, **params: Any) -> dict[str, Any]:
        r = self.http.get(path, params={k: v for k, v in params.items() if v is not None},
                           headers={"X-Rideshare-Tenant": self.tenant_id})
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, json: Any | None = None, **params: Any) -> dict[str, Any]:
        r = self.http.post(path, json=json,
                            params={k: v for k, v in params.items() if v is not None},
                            headers={"X-Rideshare-Tenant": self.tenant_id})
        r.raise_for_status()
        return r.json()

    # ---- admin ---- #
    def reset(self) -> dict[str, Any]:
        return self._post("/admin/reset", tenant=self.tenant_id)

    def snapshot(self) -> dict[str, Any]:
        return self._post("/admin/snapshot", tenant=self.tenant_id)

    def perturb(self, perts: list[dict[str, Any]]) -> dict[str, Any]:
        return self._post("/admin/perturb", json=perts, tenant=self.tenant_id)

    def clear_perturbations(self) -> dict[str, Any]:
        return self._post("/admin/clear_perturbations", tenant=self.tenant_id)

    def tick(self, dt_seconds: float = 30.0) -> dict[str, Any]:
        return self._post("/admin/tick", json={"dt_seconds": dt_seconds},
                           tenant=self.tenant_id)

    def seed(self, *, n_drivers: int = 50, n_riders: int = 100, seed: int = 0,
             online_pct: float = 0.85) -> dict[str, Any]:
        return self._post("/admin/seed", json={
            "n_drivers": n_drivers, "n_riders": n_riders,
            "seed": seed, "online_pct": online_pct,
        }, tenant=self.tenant_id)

    def set_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return self._post("/admin/set_metadata", json=metadata, tenant=self.tenant_id)

    def get_metadata(self) -> dict[str, Any]:
        return self._get("/admin/metadata", tenant=self.tenant_id)

    def inject_event(self, **kwargs: Any) -> dict[str, Any]:
        return self._post("/admin/inject_event", json=kwargs, tenant=self.tenant_id)

    # ---- marketplace ---- #
    def list_pending_requests(self, **filters: Any) -> dict[str, Any]:
        return self._get("/api/v1/marketplace/pending_requests", **filters)

    def list_idle_drivers(self, **filters: Any) -> dict[str, Any]:
        return self._get("/api/v1/marketplace/idle_drivers", **filters)

    def match_ride(self, *, trip_id: int, driver_id: int) -> dict[str, Any]:
        return self._post("/api/v1/marketplace/match",
                           json={"trip_id": trip_id, "driver_id": driver_id})

    def cancel_trip_admin(self, *, trip_id: int, reason: str,
                            cancelled_by: str = "system") -> dict[str, Any]:
        return self._post("/api/v1/marketplace/cancel_trip", json={
            "trip_id": trip_id, "reason": reason, "cancelled_by": cancelled_by,
        })

    def auto_match_nearest(self, trip_id: int) -> dict[str, Any]:
        return self._post("/api/v1/marketplace/auto_match_nearest",
                           json={"trip_id": trip_id})

    def set_surge(self, *, zone_id: str, multiplier: float,
                   ttl_minutes: float = 5.0) -> dict[str, Any]:
        return self._post("/api/v1/marketplace/set_surge", json={
            "zone_id": zone_id, "multiplier": multiplier, "ttl_minutes": ttl_minutes,
        })

    def rebalance_driver(self, *, driver_id: int, target_zone_id: str) -> dict[str, Any]:
        return self._post("/api/v1/marketplace/rebalance_driver", json={
            "driver_id": driver_id, "target_zone_id": target_zone_id,
        })

    def offer_incentive(self, *, driver_id: int, type: str = "bonus",
                         amount: float = 10.0) -> dict[str, Any]:
        return self._post("/api/v1/marketplace/offer_incentive", json={
            "driver_id": driver_id, "type": type, "amount": amount,
        })

    def zone_supply_demand(self) -> dict[str, Any]:
        return self._get("/api/v1/marketplace/zone_supply_demand")

    def dispatch_log(self, **filters: Any) -> dict[str, Any]:
        return self._get("/api/v1/marketplace/dispatch_log", **filters)

    # ---- trips ---- #
    def list_trips(self, **filters: Any) -> dict[str, Any]:
        return self._get("/api/v1/trips", **filters)

    def get_trip(self, trip_id: int) -> dict[str, Any]:
        return self._get(f"/api/v1/trips/{trip_id}")

    def get_trip_gps_log(self, trip_id: int) -> dict[str, Any]:
        return self._get(f"/api/v1/trips/{trip_id}/gps_log")

    # ---- refunds ---- #
    def create_refund(self, *, trip_id: int, amount: float, reason: str,
                       notify_rider: bool = True) -> dict[str, Any]:
        return self._post("/api/v1/refunds", json={
            "trip_id": trip_id, "amount": amount, "reason": reason,
            "notify_rider": notify_rider,
        })

    def list_refunds(self, **filters: Any) -> dict[str, Any]:
        return self._get("/api/v1/refunds", **filters)

    # ---- disputes ---- #
    def list_disputes(self, **filters: Any) -> dict[str, Any]:
        return self._get("/api/v1/disputes", **filters)

    def get_dispute(self, dispute_id: int) -> dict[str, Any]:
        return self._get(f"/api/v1/disputes/{dispute_id}")

    def submit_dispute_response(self, dispute_id: int,
                                 response: dict[str, Any]) -> dict[str, Any]:
        return self._post(f"/api/v1/disputes/{dispute_id}/submit_response",
                           json=response)

    # ---- safety ---- #
    def list_safety_incidents(self, **filters: Any) -> dict[str, Any]:
        return self._get("/api/v1/safety/incidents", **filters)

    def get_incident(self, incident_id: int) -> dict[str, Any]:
        return self._get(f"/api/v1/safety/incidents/{incident_id}")

    def escalate_incident(self, *, incident_id: int, level: int,
                           notify_parties: bool = True) -> dict[str, Any]:
        return self._post("/api/v1/safety/escalate", json={
            "incident_id": incident_id, "level": level,
            "notify_parties": notify_parties,
        })

    def contact_emergency(self, *, incident_id: int,
                           kind: str = "911") -> dict[str, Any]:
        return self._post("/api/v1/safety/contact_emergency", json={
            "incident_id": incident_id, "kind": kind,
        })

    def attach_evidence(self, *, incident_id: int, evidence_kind: str,
                         note: str = "") -> dict[str, Any]:
        return self._post("/api/v1/safety/attach_evidence", json={
            "incident_id": incident_id, "evidence_kind": evidence_kind, "note": note,
        })

    # ---- fraud ---- #
    def list_flagged_trips(self) -> dict[str, Any]:
        return self._get("/api/v1/fraud/flagged_trips")

    def list_flagged_accounts(self) -> dict[str, Any]:
        return self._get("/api/v1/fraud/flagged_accounts")

    def freeze_account(self, *, target_kind: str, target_id: int,
                        reason: str) -> dict[str, Any]:
        return self._post("/api/v1/fraud/freeze_account", json={
            "target_kind": target_kind, "target_id": target_id, "reason": reason,
        })

    def ban_device(self, *, fingerprint: str, reason: str) -> dict[str, Any]:
        return self._post("/api/v1/fraud/ban_device", json={
            "fingerprint": fingerprint, "reason": reason,
        })

    def cluster_by_fingerprint(self, fingerprint: str) -> dict[str, Any]:
        return self._get("/api/v1/fraud/cluster_by_fingerprint",
                          fingerprint=fingerprint)

    # ---- lost items ---- #
    def list_lost_items(self, **filters: Any) -> dict[str, Any]:
        return self._get("/api/v1/lost_items", **filters)

    def create_lost_item(self, *, trip_id: int, description: str) -> dict[str, Any]:
        return self._post("/api/v1/lost_items", json={
            "trip_id": trip_id, "description": description,
        })

    def assign_lost_item(self, *, lost_item_id: int, driver_id: int,
                          return_method: str = "next_idle_window") -> dict[str, Any]:
        return self._post(f"/api/v1/lost_items/{lost_item_id}/assign", json={
            "driver_id": driver_id, "return_method": return_method,
        })

    def schedule_lost_item_pickup(self, *, lost_item_id: int, pickup_at: float,
                                   pickup_location: list[float],
                                   notify_rider: bool = True,
                                   notify_driver: bool = True) -> dict[str, Any]:
        return self._post(f"/api/v1/lost_items/{lost_item_id}/schedule_pickup", json={
            "pickup_at": pickup_at, "pickup_location": pickup_location,
            "notify_rider": notify_rider, "notify_driver": notify_driver,
        })

    def confirm_lost_item_handoff(self, *, lost_item_id: int,
                                    code: str) -> dict[str, Any]:
        return self._post(f"/api/v1/lost_items/{lost_item_id}/confirm_handoff",
                           json={"code": code})

    # ---- drivers ---- #
    def get_driver(self, driver_id: int) -> dict[str, Any]:
        return self._get(f"/api/v1/drivers/{driver_id}")

    def get_driver_documents(self, driver_id: int) -> dict[str, Any]:
        return self._get(f"/api/v1/drivers/{driver_id}/documents")

    def verify_driver_documents(self, driver_id: int) -> dict[str, Any]:
        return self._post(f"/api/v1/drivers/{driver_id}/verify_documents")

    def adjust_driver_payout(self, driver_id: int, *, amount: float,
                               reason: str,
                               related_trip_id: int | None = None) -> dict[str, Any]:
        return self._post(f"/api/v1/drivers/{driver_id}/payout_adjust", json={
            "amount": amount, "reason": reason, "related_trip_id": related_trip_id,
        })

    def get_driver_performance(self, driver_id: int) -> dict[str, Any]:
        return self._get(f"/api/v1/drivers/{driver_id}/performance")

    # ---- riders ---- #
    def get_rider(self, rider_id: int) -> dict[str, Any]:
        return self._get(f"/api/v1/riders/{rider_id}")

    def freeze_rider(self, rider_id: int, *, reason: str) -> dict[str, Any]:
        return self._post(f"/api/v1/riders/{rider_id}/freeze",
                           json={"reason": reason})

    def restore_rider(self, rider_id: int, *, reason: str = "") -> dict[str, Any]:
        return self._post(f"/api/v1/riders/{rider_id}/restore",
                           json={"reason": reason})

    def login_history(self, rider_id: int) -> dict[str, Any]:
        return self._get(f"/api/v1/riders/{rider_id}/login_history")

    # ---- pricing ---- #
    def get_pricing_quote(self, *, pickup_x: float, pickup_y: float,
                            dropoff_x: float, dropoff_y: float,
                            vehicle_type: str = "uberx") -> dict[str, Any]:
        return self._get("/api/v1/pricing/quote",
                          pickup_x=pickup_x, pickup_y=pickup_y,
                          dropoff_x=dropoff_x, dropoff_y=dropoff_y,
                          vehicle_type=vehicle_type)

    def list_surge_zones(self) -> dict[str, Any]:
        return self._get("/api/v1/pricing/zones")

    # ---- messaging ---- #
    def send_to_rider(self, *, rider_id: int, template: str,
                       variables: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._post("/api/v1/messaging/send_to_rider", json={
            "rider_id": rider_id, "template": template,
            "variables": variables or {},
        })

    def send_to_driver(self, *, driver_id: int, template: str,
                        variables: dict[str, Any] | None = None) -> dict[str, Any]:
        return self._post("/api/v1/messaging/send_to_driver", json={
            "driver_id": driver_id, "template": template,
            "variables": variables or {},
        })

    def list_messages(self, **filters: Any) -> dict[str, Any]:
        return self._get("/api/v1/messaging/messages", **filters)


@dataclass
class RideshareSandbox:
    """Concrete sandbox. Implements the `Sandbox` protocol."""

    http: Any
    tenant_id: str = field(default_factory=lambda: f"shop_{uuid.uuid4().hex[:10]}")
    rs: RideshareClient = field(init=False)

    def __post_init__(self) -> None:
        self.rs = RideshareClient(http=self.http, tenant_id=self.tenant_id)

    # Sandbox protocol
    def reset(self) -> None:
        self.rs.reset()

    def snapshot(self) -> dict[str, Any]:
        return self.rs.snapshot()

    def teardown(self) -> None:
        self.rs.reset()

    # Perturbation hooks (FixtureMutator looks for these)
    def inject_perturbations(self, perts: list[dict[str, Any]]) -> None:
        self.rs.perturb(perts)

    def clear_perturbations(self) -> None:
        self.rs.clear_perturbations()


# --------------------------------------------------------------------------- #
# Factory helpers
# --------------------------------------------------------------------------- #

def in_process_sandbox_factory(*, tenant_prefix: str = "test") -> Callable[[], RideshareSandbox]:
    from fastapi.testclient import TestClient

    from rideshare_gym.mock_server.app import create_app

    app = create_app()
    client = TestClient(app, raise_server_exceptions=True)

    def factory() -> RideshareSandbox:
        return RideshareSandbox(
            http=client, tenant_id=f"{tenant_prefix}_{uuid.uuid4().hex[:10]}")
    factory._client = client  # type: ignore[attr-defined]
    return factory


def remote_sandbox_factory(
    base_url: str = "http://rideshare-mock:8000",
    *,
    tenant_prefix: str = "shop",
    timeout: float = 30.0,
) -> Callable[[], RideshareSandbox]:
    import httpx
    client = httpx.Client(base_url=base_url, timeout=timeout)

    def factory() -> RideshareSandbox:
        return RideshareSandbox(
            http=client, tenant_id=f"{tenant_prefix}_{uuid.uuid4().hex[:10]}")
    factory._client = client  # type: ignore[attr-defined]
    return factory
