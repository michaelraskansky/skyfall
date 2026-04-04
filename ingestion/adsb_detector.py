"""
ADS-B Anomaly Detector
=======================

Stateful detector that processes batches of aircraft states and emits
RawEvent objects for anomalies: emergency squawks, watched hex codes,
and sudden heading changes.

Used by the POST /api/v1/adsb push endpoint.
"""

from __future__ import annotations

import logging
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from models import EventSource, RawEvent

logger = logging.getLogger(__name__)

# ── Emergency squawk codes ───────────────────────────────────────────────────

EMERGENCY_SQUAWKS: dict[str, str] = {
    "7700": "MAYDAY",
    "7600": "LOST COMM",
    "7500": "HIJACK",
}

# ── Default heading change threshold ─────────────────────────────────────────

HEADING_CHANGE_THRESHOLD_DEG: float = 45.0


# ── Aircraft state model ─────────────────────────────────────────────────────


class AircraftState(BaseModel):
    """A single aircraft state from an ADS-B push batch."""

    hex: str
    callsign: str = ""
    lat: Optional[float] = None
    lon: Optional[float] = None
    track: Optional[float] = None
    alt_m: Optional[float] = None
    velocity_m_s: Optional[float] = None
    origin_country: str = ""
    on_ground: bool = False
    squawk: Optional[str] = None

    @field_validator("hex", mode="before")
    @classmethod
    def lowercase_hex(cls, v: str) -> str:
        return v.lower() if isinstance(v, str) else v


# ── Utilities ────────────────────────────────────────────────────────────────


def heading_delta(h1: float, h2: float) -> float:
    """Compute the smallest angular difference between two headings (0-360)."""
    d = abs(h1 - h2) % 360
    return d if d <= 180 else 360 - d
