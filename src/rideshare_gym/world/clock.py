"""Simulator clock — discrete time stepping, configurable seconds-per-tick."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SimClock:
    """Tracks simulated time. `now` is seconds since episode start."""

    now: float = 0.0
    step_seconds: float = 30.0
    """Default tick size. Tasks may override per call."""

    def tick(self, dt_seconds: float | None = None) -> float:
        dt = self.step_seconds if dt_seconds is None else float(dt_seconds)
        self.now += dt
        return self.now

    def epoch(self) -> int:
        if self.step_seconds <= 0:
            return 0
        return int(self.now // self.step_seconds)

    def reset(self, now: float = 0.0) -> None:
        self.now = float(now)

    @property
    def now_minutes(self) -> float:
        return self.now / 60.0
