"""AbstractTask — the unit of work in the gym.

Every task defines:
  * what tools the agent has access to
  * how the sandbox is initialised at episode start
  * the natural-language goal prompt the agent receives
  * the verifier that judges success at every step

Subclasses implement `setup` and `verifier`. Concrete domain tasks (e.g.
`ShopifyTask`) add domain-specific helpers on top.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from rideshare_gym.core.tools import ToolRegistry, ToolSpec
from rideshare_gym.core.verifier import Verifier

if TYPE_CHECKING:
    from rideshare_gym.core.adversarial import Perturbation
    from rideshare_gym.core.sandbox import Sandbox


Difficulty = Literal["easy", "medium", "hard"]


@dataclass
class InitialState:
    """What the task hands back from `setup()`. Goes into the first observation."""

    summary: str
    """Human-readable description of the situation, embedded in the goal prompt."""
    snapshot: dict[str, Any] = field(default_factory=dict)
    """Structured state the agent should be aware of from step 0 (e.g. open orders)."""
    ground_truth: dict[str, Any] = field(default_factory=dict)
    """Hidden labels the verifier compares against. NEVER shown to the agent."""


class AbstractTask(ABC):
    """Base task. Subclass and implement `setup` + `verifier`."""

    task_id: str = ""
    difficulty: Difficulty = "medium"
    max_steps: int = 30
    step_penalty: float = 0.0
    """Subtracted from reward each step to encourage efficiency."""

    def __init__(
        self,
        seed: int = 0,
        perturbations: list[Perturbation] | None = None,
    ) -> None:
        self.seed = seed
        self.perturbations: list[Perturbation] = list(perturbations or [])
        self._tool_registry: ToolRegistry | None = None

    # ------------------------------------------------------------------ #
    # Required overrides
    # ------------------------------------------------------------------ #

    @abstractmethod
    def tools(self) -> list[ToolSpec]:
        """Tools available to the agent during this task. Filter the global
        Shopify tool list down to what's actually needed — fewer tools
        usually means better agent performance."""

    @abstractmethod
    def setup(self, sandbox: Sandbox) -> InitialState:
        """Seed the sandbox tenant with the initial situation. Called on `reset()`.
        Must be deterministic given `self.seed`."""

    @abstractmethod
    def verifier(self) -> Verifier:
        """Return the verifier evaluated after every step.
        Should be a fresh instance — verifiers may carry per-episode state."""

    # ------------------------------------------------------------------ #
    # Optional overrides
    # ------------------------------------------------------------------ #

    def goal_prompt(self, state: InitialState) -> str:
        """The user message handed to the agent at episode start.
        Default: just the situation summary."""
        return state.summary

    def teardown(self, sandbox: Sandbox) -> None:
        """Cleanup. Default: rely on the SandboxPool to release the tenant."""
        return None

    # ------------------------------------------------------------------ #
    # Cached registry
    # ------------------------------------------------------------------ #

    @property
    def tool_registry(self) -> ToolRegistry:
        if self._tool_registry is None:
            self._tool_registry = ToolRegistry.from_specs(self.tools())
        return self._tool_registry

    def __repr__(self) -> str:
        n_perts = len(self.perturbations)
        suffix = f", +{n_perts} perturbations" if n_perts else ""
        return f"<{type(self).__name__} task_id={self.task_id} seed={self.seed}{suffix}>"
