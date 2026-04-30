"""2D city plane — zones, distances, travel times.

We use km in a flat coordinate system. Manhattan distance approximates
real-city travel (closer to truth than Euclidean for grid streets).
"""

from __future__ import annotations

from dataclasses import dataclass, field

ZONE_NAMES = ("downtown", "airport", "university", "stadium", "suburb_n", "suburb_s")


@dataclass
class Zone:
    """One geographic zone with surge dynamics."""

    id: str
    centroid: tuple[float, float]   # (x_km, y_km)
    radius_km: float
    base_demand_rate: float = 0.5   # rides per minute under typical conditions
    base_traffic_factor: float = 1.0
    name: str = ""

    def contains(self, x: float, y: float) -> bool:
        dx = x - self.centroid[0]
        dy = y - self.centroid[1]
        return (dx * dx + dy * dy) <= (self.radius_km * self.radius_km)

    def random_point(self, rng) -> tuple[float, float]:
        """Sample a point inside the zone."""
        import math
        r = self.radius_km * (rng.random() ** 0.5)
        theta = 2 * math.pi * rng.random()
        return (
            self.centroid[0] + r * math.cos(theta),
            self.centroid[1] + r * math.sin(theta),
        )


@dataclass
class City:
    """The whole 2D world."""

    name: str
    zones: list[Zone] = field(default_factory=list)
    bounds: tuple[float, float, float, float] = (0.0, 0.0, 30.0, 30.0)
    """(xmin, ymin, xmax, ymax) in km."""
    avg_speed_kmh: float = 30.0
    """Free-flow average speed; modified by traffic factor + events."""

    def zone_for(self, x: float, y: float) -> Zone | None:
        """Return the zone containing (x, y) or None."""
        candidates = [z for z in self.zones if z.contains(x, y)]
        if not candidates:
            return None
        # If overlapping, return the smallest-radius zone (most specific).
        return min(candidates, key=lambda z: z.radius_km)

    def zone_by_id(self, zone_id: str) -> Zone | None:
        return next((z for z in self.zones if z.id == zone_id), None)

    def distance_km(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        """Manhattan distance in km."""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def travel_time_minutes(
        self,
        a: tuple[float, float],
        b: tuple[float, float],
        traffic_factor: float = 1.0,
        avg_speed_kmh: float | None = None,
    ) -> float:
        """Manhattan distance / effective speed. traffic_factor >= 1 slows things down."""
        speed = (avg_speed_kmh or self.avg_speed_kmh) / max(traffic_factor, 0.5)
        if speed <= 0:
            return float("inf")
        return self.distance_km(a, b) / speed * 60.0  # km / kmh * 60 = minutes


def default_city() -> City:
    """A stylized "Bayview" city with 6 zones."""
    return City(
        name="Bayview",
        bounds=(0.0, 0.0, 30.0, 30.0),
        avg_speed_kmh=30.0,
        zones=[
            Zone(id="downtown",   centroid=(15.0, 15.0), radius_km=3.0,
                 base_demand_rate=2.0, base_traffic_factor=1.4, name="Downtown"),
            Zone(id="airport",    centroid=(27.0,  3.0), radius_km=2.5,
                 base_demand_rate=1.2, base_traffic_factor=1.0, name="Bayview International Airport"),
            Zone(id="university", centroid=( 6.0, 22.0), radius_km=2.5,
                 base_demand_rate=0.8, base_traffic_factor=1.1, name="State University"),
            Zone(id="stadium",    centroid=(23.0, 24.0), radius_km=2.0,
                 base_demand_rate=0.4, base_traffic_factor=1.2, name="Memorial Stadium"),
            Zone(id="suburb_n",   centroid=( 8.0, 10.0), radius_km=4.0,
                 base_demand_rate=0.6, base_traffic_factor=0.9, name="North Suburb"),
            Zone(id="suburb_s",   centroid=(20.0,  8.0), radius_km=4.0,
                 base_demand_rate=0.5, base_traffic_factor=0.9, name="South Suburb"),
        ],
    )
