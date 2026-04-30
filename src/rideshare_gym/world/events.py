"""World events that affect demand, traffic, or driver availability."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


WorldEventKind = Literal[
    "traffic_jam",
    "accident",
    "weather",
    "concert_let_out",
    "rush_hour",
    "fraud_attack_start",
    "system_outage_partial",
]


@dataclass
class WorldEvent:
    kind: WorldEventKind
    started_at: float
    duration_seconds: float
    affected_zones: list[str]
    severity: float = 0.5
    """0..1 — used to scale demand/traffic effects."""
    metadata: dict[str, Any] = field(default_factory=dict)

    def is_active(self, now_seconds: float) -> bool:
        return self.started_at <= now_seconds < self.started_at + self.duration_seconds

    def demand_multiplier_for(self, zone_id: str, now_seconds: float) -> float:
        """How much this event amplifies demand in the given zone, if active."""
        if not self.is_active(now_seconds):
            return 1.0
        if zone_id not in self.affected_zones:
            return 1.0
        if self.kind == "concert_let_out":
            return 1.0 + 6.0 * self.severity
        if self.kind == "weather":
            return 1.0 + 1.5 * self.severity
        if self.kind == "rush_hour":
            return 1.0 + 1.0 * self.severity
        return 1.0

    def traffic_factor_for(self, zone_id: str, now_seconds: float) -> float:
        if not self.is_active(now_seconds):
            return 1.0
        if zone_id not in self.affected_zones:
            return 1.0
        if self.kind == "traffic_jam":
            return 1.0 + 1.5 * self.severity
        if self.kind == "accident":
            return 1.0 + 1.0 * self.severity
        if self.kind == "weather":
            return 1.0 + 0.4 * self.severity
        return 1.0
