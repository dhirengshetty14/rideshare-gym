"""Trip lifecycle — state machine + GPS log + side records (refunds, disputes, lost items)."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from enum import Enum
from typing import Any


class TripStatus(str, Enum):
    REQUESTED = "requested"
    MATCHED = "matched"
    DRIVER_ARRIVING = "driver_arriving"
    DRIVER_ARRIVED = "driver_arrived"
    IN_TRIP = "in_trip"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


CancellationReason = str
"""One of: customer_no_show, customer_changed_mind, driver_too_far,
driver_no_show, vehicle_issue, safety_concern, fraud_suspected, other."""


@dataclass
class GpsPoint:
    t: float                   # sim seconds since episode start
    location: tuple[float, float]
    speed_kmh: float = 0.0


@dataclass
class Trip:
    id: int
    rider_id: int
    driver_id: int | None = None
    pickup: tuple[float, float] = (0.0, 0.0)
    dropoff: tuple[float, float] = (0.0, 0.0)
    pickup_zone_id: str = ""
    dropoff_zone_id: str = ""
    vehicle_type: str = "uberx"

    status: TripStatus = TripStatus.REQUESTED

    requested_at: float = 0.0
    matched_at: float | None = None
    pickup_arrived_at: float | None = None
    picked_up_at: float | None = None
    completed_at: float | None = None
    cancelled_at: float | None = None
    cancelled_by: str | None = None     # "rider" | "driver" | "system"
    cancel_reason: CancellationReason | None = None

    # Pricing.
    surge_at_request: Decimal = Decimal("1.0")
    fare: dict[str, Any] | None = None
    """Snapshot of FareBreakdown.model_dump() / dict at completion."""

    # GPS waypoints sampled per tick.
    gps_log: list[GpsPoint] = field(default_factory=list)

    # Ratings (post-completion).
    rider_rating_of_driver: int | None = None
    driver_rating_of_rider: int | None = None
    tip_amount: Decimal = Decimal("0")

    # Side records (populated by ops actions).
    refund_id: int | None = None
    dispute_id: int | None = None
    lost_item_id: int | None = None
    incident_id: int | None = None

    flags: list[str] = field(default_factory=list)
    """e.g. ['flagged_fraud', 'gps_anomaly', 'long_wait']"""

    metadata: dict[str, Any] = field(default_factory=dict)
