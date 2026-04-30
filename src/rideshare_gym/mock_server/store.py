"""Per-tenant World registry. Same shape as shopify-gym's TenantStore."""

from __future__ import annotations

import threading

from rideshare_gym.world.world import World


class WorldRegistry:
    """Process-wide registry of per-tenant Worlds."""

    def __init__(self) -> None:
        self._worlds: dict[str, World] = {}
        self._lock = threading.Lock()

    def get(self, tenant_id: str) -> World:
        with self._lock:
            if tenant_id not in self._worlds:
                self._worlds[tenant_id] = World(tenant_id=tenant_id)
            return self._worlds[tenant_id]

    def reset(self, tenant_id: str) -> World:
        with self._lock:
            self._worlds[tenant_id] = World(tenant_id=tenant_id)
            return self._worlds[tenant_id]

    def drop(self, tenant_id: str) -> None:
        with self._lock:
            self._worlds.pop(tenant_id, None)

    def all_tenants(self) -> list[str]:
        with self._lock:
            return list(self._worlds.keys())


_registry = WorldRegistry()


def get_world(tenant_id: str) -> World:
    return _registry.get(tenant_id)


def reset_world(tenant_id: str) -> World:
    return _registry.reset(tenant_id)


def drop_world(tenant_id: str) -> None:
    _registry.drop(tenant_id)


def all_tenants() -> list[str]:
    return _registry.all_tenants()
