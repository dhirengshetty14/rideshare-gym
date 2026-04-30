"""Rider model."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal


@dataclass
class Rider:
    id: int
    name: str
    email: str
    phone: str = ""
    rating: float = 4.9
    payment_method_id: str = ""
    """Hashed/tokenised ID of saved card; same hash for shared cards (fraud signal)."""
    payment_bin: str = ""
    """First 6 digits of card — used to identify disposable BINs."""
    device_fingerprint: str = ""
    """Browser/device fingerprint hash — used to identify shared devices."""
    home_zone_id: str | None = None
    work_zone_id: str | None = None

    completed_trips: int = 0
    chargeback_count: int = 0
    lifetime_spent: Decimal = Decimal("0")
    created_at: str = ""

    flags: list[str] = field(default_factory=list)
    """e.g. ['frozen', 'fraud_suspected', 'verified']"""

    typical_login_zone_id: str | None = None
    typical_login_hour_window: tuple[int, int] = (7, 23)
    """(start_hour, end_hour) in 24h. Anomalous logins outside this window."""
