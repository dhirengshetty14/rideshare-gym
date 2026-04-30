"""Surge pricing — supply/demand driven multiplier per zone."""

from __future__ import annotations

from decimal import Decimal


SURGE_MIN = Decimal("1.00")
SURGE_MAX = Decimal("5.00")
SURGE_STEP = Decimal("0.25")


def round_to_step(d: Decimal, step: Decimal = SURGE_STEP) -> Decimal:
    """Round to nearest 0.25 increment."""
    n = (d / step).to_integral_value(rounding="ROUND_HALF_UP")
    return n * step


def compute_surge(*, demand: int, supply: int) -> Decimal:
    """Default surge rule: ratio = demand / max(supply, 1).
    1.0 if demand <= supply; otherwise scales up to SURGE_MAX.
    Rounds to 0.25 increments."""
    if demand <= 0 or supply >= demand:
        return SURGE_MIN
    ratio = Decimal(demand) / Decimal(max(supply, 1))
    raw = SURGE_MIN + (ratio - Decimal("1.0")) * Decimal("0.5")
    raw = max(SURGE_MIN, min(SURGE_MAX, raw))
    return round_to_step(raw)


def apply_hysteresis(current: Decimal, target: Decimal,
                      delta_step: Decimal = SURGE_STEP) -> Decimal:
    """Move surge towards target by at most one step. Avoids thrashing."""
    if target == current:
        return current
    if target > current:
        return min(target, current + delta_step)
    return max(target, current - delta_step)
