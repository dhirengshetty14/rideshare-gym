"""Sandbox protocol — the opaque object handlers receive in `handler(sandbox, args)`.

A real Shopify Sandbox wraps an HTTP client to the mock server plus a tenant id.
The `StubSandbox` is a no-op for unit tests of the core abstractions.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol


class Sandbox(Protocol):
    """What every concrete sandbox exposes to tools and verifiers.

    Concrete impls (e.g. `ShopifySandbox`) add domain-specific clients —
    `sandbox.shopify.list_orders(...)` — but every sandbox can:
      * be reset to a clean state per episode
      * snapshot its full state for the verifier / trajectory recorder
      * be torn down at episode end
    """

    tenant_id: str

    def reset(self) -> None: ...
    def snapshot(self) -> dict[str, Any]: ...
    def teardown(self) -> None: ...


@dataclass
class StubSandbox:
    """In-memory sandbox for unit tests. Acts as a tiny key/value store.

    Tasks/tests can read and write `self.state` directly. Useful for testing
    verifiers and the env loop without spinning up the mock server.
    """

    tenant_id: str = field(default_factory=lambda: f"stub_{uuid.uuid4().hex[:8]}")
    state: dict[str, Any] = field(default_factory=dict)
    call_log: list[dict[str, Any]] = field(default_factory=list)
    perturbations: list[dict[str, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.state.clear()
        self.call_log.clear()
        self.perturbations.clear()

    def snapshot(self) -> dict[str, Any]:
        import copy
        return copy.deepcopy(self.state)

    def teardown(self) -> None:
        self.state.clear()
        self.call_log.clear()
        self.perturbations.clear()

    # Convenience for tests
    def record_call(self, name: str, args: dict[str, Any]) -> None:
        self.call_log.append({"name": name, "args": args})

    def set(self, key: str, value: Any) -> None:
        self.state[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.state.get(key, default)

    # Perturbation hooks (FixtureMutator looks for these)
    def inject_perturbations(self, perts: list[dict[str, Any]]) -> None:
        self.perturbations.extend(perts)

    def clear_perturbations(self) -> None:
        self.perturbations.clear()


@dataclass
class SandboxPool:
    """A simple lease/release pool sized to `--parallel`. Concrete sandboxes
    register themselves at startup; episodes lease one and release on teardown.

    For the StubSandbox this is overkill but the contract matches what the
    real ShopifySandbox pool will use in Phase 2.
    """

    factory: Any  # Callable[[], Sandbox]
    size: int = 4
    _free: list[Sandbox] = field(default_factory=list)
    _in_use: set[str] = field(default_factory=set)

    def warm(self) -> None:
        while len(self._free) < self.size:
            self._free.append(self.factory())

    def lease(self) -> Sandbox:
        if not self._free:
            sb = self.factory()
        else:
            sb = self._free.pop()
        sb.reset()
        self._in_use.add(sb.tenant_id)
        return sb

    def release(self, sb: Sandbox) -> None:
        self._in_use.discard(sb.tenant_id)
        self._free.append(sb)

    def shutdown(self) -> None:
        for sb in self._free:
            sb.teardown()
        self._free.clear()
        self._in_use.clear()
