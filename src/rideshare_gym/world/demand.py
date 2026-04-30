"""Poisson-style demand generator with diurnal patterns."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from rideshare_gym.world.city import City


def time_of_day_multiplier(now_seconds: float, episode_start_hour: int = 17) -> float:
    """Morning + evening rush peaks. `now_seconds` is sim-time since episode start.
    `episode_start_hour` is the wall-clock hour the episode is set in (default 5pm).
    """
    hour = (episode_start_hour + now_seconds / 3600.0) % 24.0
    # Twin Gaussians centred on 8am (1.6×) and 6pm (1.8×).
    morn = 1.6 * math.exp(-((hour - 8) ** 2) / 2.0)
    even = 1.8 * math.exp(-((hour - 18) ** 2) / 2.0)
    base = 0.6   # baseline activity
    return base + morn + even


@dataclass
class DemandGenerator:
    rng: random.Random = field(default_factory=lambda: random.Random(0))
    episode_start_hour: int = 17

    def expected_arrivals(
        self, *, city: City, dt_seconds: float, now_seconds: float,
        zone_event_multipliers: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """Per-zone expected number of new requests in `dt_seconds`."""
        zone_event_multipliers = zone_event_multipliers or {}
        tod = time_of_day_multiplier(now_seconds, self.episode_start_hour)
        out: dict[str, float] = {}
        dt_min = dt_seconds / 60.0
        for z in city.zones:
            ev = zone_event_multipliers.get(z.id, 1.0)
            out[z.id] = z.base_demand_rate * tod * ev * dt_min
        return out

    def sample_arrivals(
        self, *, city: City, dt_seconds: float, now_seconds: float,
        zone_event_multipliers: dict[str, float] | None = None,
    ) -> dict[str, int]:
        """Sample per-zone request counts via Poisson."""
        expected = self.expected_arrivals(
            city=city, dt_seconds=dt_seconds, now_seconds=now_seconds,
            zone_event_multipliers=zone_event_multipliers,
        )
        out: dict[str, int] = {}
        for zone_id, mu in expected.items():
            out[zone_id] = _poisson(self.rng, mu)
        return out

    def sample_destination(self, city: City, origin_zone_id: str) -> str:
        """Origin-destination matrix — biased toward different zones (most riders go elsewhere)."""
        zones = [z.id for z in city.zones]
        if not zones:
            return origin_zone_id
        weights = [
            (3.0 if zid != origin_zone_id else 1.0) for zid in zones
        ]
        return self.rng.choices(zones, weights=weights, k=1)[0]


def _poisson(rng: random.Random, mu: float) -> int:
    """Knuth's Poisson sampler. Adequate for small mu (< ~30)."""
    if mu <= 0:
        return 0
    L = math.exp(-mu)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1
