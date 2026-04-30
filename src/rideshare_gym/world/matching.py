"""Matching helpers — used by gold oracle and tasks that need ETA computations."""

from __future__ import annotations

from rideshare_gym.world.city import City
from rideshare_gym.world.drivers import Driver, DriverStatus


def eta_minutes(city: City, driver: Driver, pickup: tuple[float, float],
                 traffic_factor: float = 1.0) -> float:
    return city.travel_time_minutes(driver.location, pickup, traffic_factor=traffic_factor)


def nearest_driver_for(
    city: City,
    drivers: list[Driver] | dict[int, Driver],
    pickup: tuple[float, float],
    *,
    vehicle_type: str | None = None,
    traffic_factor: float = 1.0,
) -> Driver | None:
    """Pick the IDLE driver with the smallest ETA to the pickup point."""
    pool = drivers.values() if isinstance(drivers, dict) else drivers
    candidates = [
        d for d in pool
        if d.status == DriverStatus.IDLE
        and d.docs_verified
        and "frozen" not in d.flags
        and (vehicle_type is None or d.vehicle_type == vehicle_type)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda d: eta_minutes(city, d, pickup, traffic_factor))
