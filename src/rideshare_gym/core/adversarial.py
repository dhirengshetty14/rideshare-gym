"""Adversarial perturbations — declarative chaos for any base task.

A `Perturbation` describes one form of misbehaviour to inject. A
`FixtureMutator` wraps an `AbstractTask` and applies its perturbations to
the sandbox during `setup()`. The Sandbox itself is responsible for
honouring the perturbations during tool dispatch — for the Shopify mock
that means the `inject_perturbation` admin endpoint plus a Toxiproxy
sidecar for network-level effects.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from rideshare_gym.core.task import AbstractTask, InitialState
from rideshare_gym.core.tools import ToolSpec
from rideshare_gym.core.verifier import Verifier

if TYPE_CHECKING:
    from rideshare_gym.core.sandbox import Sandbox


PerturbationKind = Literal[
    "latency",
    "rate_limit",
    "webhook_drop",
    "stale_get",
    "partial_failure",
    "malformed_response",
    "schema_drift",
    "deadline_pressure",
    "fraud_pattern_drift",
]


class Perturbation(BaseModel):
    """One injected failure mode. Each gets its own RNG seed for repro."""

    kind: PerturbationKind
    params: dict[str, Any] = Field(default_factory=dict)
    seed: int = 0

    def signature(self) -> str:
        """Short stable id usable in task variant naming."""
        body = self.model_dump_json()
        return hashlib.sha1(body.encode()).hexdigest()[:8]


class FixtureMutator(AbstractTask):
    """Wraps a base task and applies perturbations on setup.

    The wrapper is itself a Task (so the gym treats it identically) but
    delegates `tools()`, `verifier()`, `goal_prompt()`, and `setup()` to
    the inner task. After the inner setup runs, this class asks the
    sandbox to install perturbations via `sandbox.inject_perturbations(list)`.
    A StubSandbox doesn't honour them — it just records them — which is
    fine for unit tests of the mutator wiring.
    """

    def __init__(
        self,
        base: AbstractTask,
        perturbations: list[Perturbation],
    ) -> None:
        super().__init__(seed=base.seed, perturbations=perturbations)
        self.base = base
        self.task_id = f"{base.task_id}__{self._suffix(perturbations)}"
        self.difficulty = base.difficulty
        self.max_steps = base.max_steps
        self.step_penalty = base.step_penalty

    @staticmethod
    def _suffix(perts: list[Perturbation]) -> str:
        if not perts:
            return "clean"
        kinds = sorted({p.kind for p in perts})
        return "+".join(kinds)

    def tools(self) -> list[ToolSpec]:
        return self.base.tools()

    def setup(self, sandbox: Sandbox) -> InitialState:
        state = self.base.setup(sandbox)
        if self.perturbations:
            inject = getattr(sandbox, "inject_perturbations", None)
            if callable(inject):
                inject([p.model_dump() for p in self.perturbations])
        return state

    def goal_prompt(self, state: InitialState) -> str:
        return self.base.goal_prompt(state)

    def verifier(self) -> Verifier:
        return self.base.verifier()

    def teardown(self, sandbox: Sandbox) -> None:
        clear = getattr(sandbox, "clear_perturbations", None)
        if callable(clear):
            clear()
        self.base.teardown(sandbox)


def adversarial_variants(
    base_task_factory: Callable[[], AbstractTask],
    perturbation_sets: list[list[Perturbation]],
) -> list[FixtureMutator]:
    """Create one variant per perturbation-set. Useful for combinatorial coverage."""
    return [FixtureMutator(base_task_factory(), perts) for perts in perturbation_sets]
