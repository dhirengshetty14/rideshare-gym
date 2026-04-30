"""Canned populations for tasks: drivers spread across zones, baseline riders."""

from __future__ import annotations

import random

from faker import Faker

from rideshare_gym.world.drivers import Driver, DriverStatus
from rideshare_gym.world.riders import Rider
from rideshare_gym.world.world import World


def seed_drivers(
    world: World,
    *,
    n_drivers: int = 50,
    seed: int = 0,
    online_pct: float = 0.85,
) -> list[int]:
    faker = Faker()
    Faker.seed(seed)
    rng = random.Random(seed)
    out: list[int] = []
    for _ in range(n_drivers):
        zone = rng.choice(world.city.zones)
        loc = zone.random_point(rng)
        did = world.next_id()
        d = Driver(
            id=did,
            name=faker.name(),
            location=loc,
            status=DriverStatus.IDLE if rng.random() < online_pct else DriverStatus.OFFLINE,
            rating=round(rng.uniform(4.5, 5.0), 2),
            vehicle_type=rng.choice(["uberx", "uberx", "uberx", "uberxl", "uberblack"]),
            online_since=world.clock.now,
            docs_verified=True,
            home_zone_id=zone.id,
            speed_kmh=rng.uniform(28.0, 35.0),
            device_fingerprint=f"dfp_{did}_{rng.randint(1000, 9999)}",
        )
        world.drivers[did] = d
        out.append(did)
    return out


def seed_riders(world: World, *, n_riders: int = 100, seed: int = 0) -> list[int]:
    faker = Faker()
    Faker.seed(seed + 1)
    rng = random.Random(seed + 1)
    out: list[int] = []
    for _ in range(n_riders):
        zone = rng.choice(world.city.zones)
        rid = world.next_id()
        r = Rider(
            id=rid,
            name=faker.name(),
            email=faker.email(),
            phone=faker.phone_number(),
            rating=round(rng.uniform(4.5, 5.0), 2),
            payment_method_id=f"pm_{rid}_{rng.randint(1000, 9999)}",
            payment_bin=rng.choice(["411111", "424242", "455555", "511234", "601100"]),
            device_fingerprint=f"rfp_{rid}_{rng.randint(1000, 9999)}",
            home_zone_id=zone.id,
            typical_login_zone_id=zone.id,
            typical_login_hour_window=(7, 23),
            created_at=faker.date_between(start_date="-2y").isoformat(),
        )
        world.riders[rid] = r
        out.append(rid)
    return out
