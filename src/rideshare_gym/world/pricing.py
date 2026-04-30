"""Fare computation — modelled on real Uber rate cards."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from typing import Literal


VehicleType = Literal["uberx", "uberxl", "uberblack"]


@dataclass(frozen=True)
class RateCard:
    base_fare: Decimal
    per_km: Decimal
    per_min: Decimal
    booking_fee: Decimal = Decimal("2.50")
    safety_fee: Decimal = Decimal("0.55")
    minimum_fare: Decimal = Decimal("4.00")
    service_fee_pct: Decimal = Decimal("0.25")  # what platform takes


# Modelled on published Uber rate cards (USD).
RATE_CARDS: dict[str, RateCard] = {
    "uberx":     RateCard(base_fare=Decimal("2.55"), per_km=Decimal("0.95"),
                          per_min=Decimal("0.34")),
    "uberxl":    RateCard(base_fare=Decimal("3.85"), per_km=Decimal("1.45"),
                          per_min=Decimal("0.45")),
    "uberblack": RateCard(base_fare=Decimal("7.00"), per_km=Decimal("2.55"),
                          per_min=Decimal("0.50"),
                          booking_fee=Decimal("4.50"), minimum_fare=Decimal("15.00")),
}


@dataclass
class FareBreakdown:
    vehicle_type: VehicleType
    base_fare: Decimal
    distance_fare: Decimal
    time_fare: Decimal
    surge_multiplier: Decimal
    booking_fee: Decimal
    safety_fee: Decimal
    subtotal: Decimal
    tax: Decimal
    total: Decimal
    distance_km: float
    duration_min: float

    def driver_payout(self, service_fee_pct: Decimal | None = None) -> Decimal:
        """Driver earnings (before tip + bonuses) — distance + time × surge × (1 - service fee)."""
        rate = RATE_CARDS[self.vehicle_type]
        pct = service_fee_pct if service_fee_pct is not None else rate.service_fee_pct
        gross = (self.distance_fare + self.time_fare) * self.surge_multiplier
        return _round_money(gross * (Decimal("1.0") - pct))


def _round_money(d: Decimal) -> Decimal:
    return d.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def compute_fare(
    *,
    distance_km: float,
    duration_min: float,
    surge: Decimal,
    vehicle_type: VehicleType = "uberx",
    tax_rate: Decimal = Decimal("0.08"),
) -> FareBreakdown:
    """Compute a fare breakdown matching the real Uber structure."""
    rc = RATE_CARDS[vehicle_type]
    surge = max(Decimal("1.0"), Decimal(str(surge)))
    distance_fare = _round_money(rc.per_km * Decimal(str(distance_km)))
    time_fare = _round_money(rc.per_min * Decimal(str(duration_min)))
    subtotal = _round_money(
        (rc.base_fare + distance_fare + time_fare) * surge
        + rc.booking_fee + rc.safety_fee
    )
    if subtotal < rc.minimum_fare:
        subtotal = rc.minimum_fare
    tax = _round_money(subtotal * tax_rate)
    total = _round_money(subtotal + tax)
    return FareBreakdown(
        vehicle_type=vehicle_type,
        base_fare=rc.base_fare,
        distance_fare=distance_fare,
        time_fare=time_fare,
        surge_multiplier=surge,
        booking_fee=rc.booking_fee,
        safety_fee=rc.safety_fee,
        subtotal=subtotal,
        tax=tax,
        total=total,
        distance_km=distance_km,
        duration_min=duration_min,
    )
