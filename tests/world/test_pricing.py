"""Fare + surge tests."""

from __future__ import annotations

from decimal import Decimal

from rideshare_gym.world.pricing import RATE_CARDS, compute_fare
from rideshare_gym.world.surge import compute_surge


def test_uberx_fare_no_surge():
    fare = compute_fare(distance_km=10.0, duration_min=20.0,
                          surge=Decimal("1.0"), vehicle_type="uberx")
    rc = RATE_CARDS["uberx"]
    expected_distance = rc.per_km * Decimal("10.0")
    expected_time = rc.per_min * Decimal("20.0")
    assert fare.distance_fare == expected_distance
    assert fare.time_fare == expected_time
    assert fare.surge_multiplier == Decimal("1.0")
    assert fare.total > Decimal("0")


def test_surge_increases_total():
    no_surge = compute_fare(distance_km=10.0, duration_min=20.0,
                              surge=Decimal("1.0"))
    with_surge = compute_fare(distance_km=10.0, duration_min=20.0,
                                surge=Decimal("2.0"))
    assert with_surge.total > no_surge.total


def test_uberblack_minimum_fare_applies():
    """Black has $15 minimum — short trip should hit it."""
    short = compute_fare(distance_km=0.5, duration_min=2.0,
                          surge=Decimal("1.0"), vehicle_type="uberblack")
    assert short.subtotal >= Decimal("15.00")


def test_driver_payout_is_share_of_fare_minus_service_fee():
    fare = compute_fare(distance_km=10.0, duration_min=20.0,
                          surge=Decimal("2.0"), vehicle_type="uberx")
    payout = fare.driver_payout()
    # Payout should be > 0 and < total (platform takes a cut + tax+fees).
    assert payout > Decimal("0")
    assert payout < fare.total


def test_surge_basic_levels():
    assert compute_surge(demand=0, supply=10) == Decimal("1.00")
    assert compute_surge(demand=5, supply=10) == Decimal("1.00")
    high = compute_surge(demand=20, supply=5)
    assert high > Decimal("1.0")
    assert high <= Decimal("5.0")


def test_surge_rounds_to_quarter():
    s = compute_surge(demand=15, supply=5)
    quarter = (s * 4) % 1
    assert quarter == 0
