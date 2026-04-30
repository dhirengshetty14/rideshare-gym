"""Driver model + state machine + movement logic."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rideshare_gym.world.world import World


class DriverStatus(str, Enum):
    OFFLINE = "offline"
    IDLE = "idle"
    DISPATCHED = "dispatched"   # accepted, driving to pickup
    PICKING_UP = "picking_up"   # at pickup waiting for rider
    IN_TRIP = "in_trip"
    BREAK = "break"


@dataclass
class Driver:
    id: int
    name: str
    location: tuple[float, float]
    status: DriverStatus = DriverStatus.OFFLINE
    rating: float = 4.85
    vehicle_type: str = "uberx"

    online_since: float | None = None
    current_trip_id: int | None = None
    target_location: tuple[float, float] | None = None
    """When dispatched / in-trip, the driver moves toward this point."""

    docs_verified: bool = True
    docs_expiry: str | None = None       # ISO date string

    cumulative_earnings_today: Decimal = Decimal("0")
    completed_trips_today: int = 0
    cancellation_count_today: int = 0

    device_fingerprint: str | None = None
    home_zone_id: str | None = None

    speed_kmh: float = 30.0
    accept_rate_baseline: float = 0.85

    flags: list[str] = field(default_factory=list)
    """e.g. ['fraud_suspected', 'frozen', 'gps_anomaly']"""


def step_driver(driver: Driver, world: "World", dt_seconds: float) -> None:
    """Advance the driver's state by `dt_seconds`.

    For driving statuses (DISPATCHED, PICKING_UP, IN_TRIP) we move toward
    `target_location` at `speed_kmh`. When we arrive, the trip's state machine
    transitions accordingly.
    """
    if driver.status not in (DriverStatus.DISPATCHED, DriverStatus.IN_TRIP):
        return
    if driver.target_location is None:
        return

    # How far can we move in dt seconds?
    speed_kmpm = driver.speed_kmh / 60.0       # km per minute
    max_km = speed_kmpm * (dt_seconds / 60.0)  # km in dt
    dx = driver.target_location[0] - driver.location[0]
    dy = driver.target_location[1] - driver.location[1]
    dist = abs(dx) + abs(dy)
    if dist <= max_km:
        driver.location = driver.target_location
    else:
        # Move proportionally along Manhattan path: x first, then y.
        if abs(dx) > 0:
            step = min(abs(dx), max_km)
            driver.location = (
                driver.location[0] + step * (1 if dx > 0 else -1),
                driver.location[1],
            )
            max_km -= step
        if max_km > 0 and abs(dy) > 0:
            step = min(abs(dy), max_km)
            driver.location = (
                driver.location[0],
                driver.location[1] + step * (1 if dy > 0 else -1),
            )
