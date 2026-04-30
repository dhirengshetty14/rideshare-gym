"""Coordinated fraud-ring patterns. Modelled on real-world ride-share fraud.

Three archetypes:
  * account_farm        — many rider accounts share payment-method fingerprint
                          and BIN range; book trips, then chargeback
  * gps_spoofing        — driver appears to drive long distances but GPS log
                          shows static or impossible velocity
  * collusion_ring      — same operator runs multiple driver + rider accounts,
                          books fake trips to extract surge incentives
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from decimal import Decimal


@dataclass
class FraudRing:
    """A coordinated fraud cluster planted in the world."""

    kind: str                          # "account_farm" | "collusion_ring" | "gps_spoofing"
    fingerprint: str
    bin: str
    rider_ids: list[int] = field(default_factory=list)
    driver_ids: list[int] = field(default_factory=list)
    trip_ids: list[int] = field(default_factory=list)


def _fp(seed: int, idx: int, kind: str) -> str:
    return hashlib.sha1(f"{kind}_{seed}_{idx}".encode()).hexdigest()[:16]


def plant_fraud_ring(
    world,
    *,
    kind: str = "account_farm",
    n_riders: int = 5,
    seed: int = 0,
    obvious_count: int | None = None,
) -> FraudRing:
    """Plant a fraud ring of `kind` into `world`. Returns the ring metadata.

    `obvious_count` controls how many of the fraud accounts also carry
    obvious flags (high risk score, blacklisted email domains). The remaining
    ones are subtle — same fingerprint and BIN but otherwise normal-looking.
    """
    obvious_count = n_riders - 2 if obvious_count is None else obvious_count
    rng = random.Random(seed)
    fp = _fp(seed, 0, kind)
    ring = FraudRing(kind=kind, fingerprint=fp, bin="411111")

    if kind == "account_farm":
        for i in range(n_riders):
            obvious = i < obvious_count
            rid = world.next_id()
            from rideshare_gym.world.riders import Rider
            r = Rider(
                id=rid,
                name=f"farm_account_{i}_{rng.randint(1000, 9999)}",
                email=f"ring{seed}_{i}@temp-mail.com",
                phone="+15555550000",
                rating=4.5 if obvious else 4.85,
                payment_method_id=fp,
                payment_bin=ring.bin,
                device_fingerprint=fp,
                created_at="2026-04-15",
                flags=(["high_risk_signup"] if obvious else []),
            )
            world.riders[rid] = r
            ring.rider_ids.append(rid)
    elif kind == "collusion_ring":
        # 3 drivers + 5 riders share the operator
        from rideshare_gym.world.drivers import Driver, DriverStatus
        from rideshare_gym.world.riders import Rider
        for i in range(3):
            did = world.next_id()
            d = Driver(
                id=did,
                name=f"collusion_driver_{i}",
                location=(20.0 + i, 8.0),
                status=DriverStatus.IDLE,
                vehicle_type="uberx",
                rating=4.7,
                device_fingerprint=fp,
                docs_verified=True,
                home_zone_id="suburb_s",
                flags=["collusion_suspect"] if i == 0 else [],
            )
            world.drivers[did] = d
            ring.driver_ids.append(did)
        for i in range(n_riders):
            rid = world.next_id()
            r = Rider(
                id=rid,
                name=f"collusion_rider_{i}",
                email=f"col{seed}_{i}@temp-mail.com",
                phone="+15555550000",
                rating=4.85,
                payment_method_id=fp,
                payment_bin=ring.bin,
                device_fingerprint=fp,
            )
            world.riders[rid] = r
            ring.rider_ids.append(rid)

    return ring


def detect_account_farm_signals(rider) -> list[str]:
    """For documentation / debugging: which signals on a rider would surface this fraud?"""
    sigs: list[str] = []
    if rider.payment_bin in ("411111", "424242", "455555"):
        sigs.append("disposable_bin")
    if "temp-mail.com" in (rider.email or "") or "10minute" in (rider.email or ""):
        sigs.append("disposable_email")
    if "high_risk_signup" in rider.flags:
        sigs.append("high_risk_signup_flag")
    return sigs
