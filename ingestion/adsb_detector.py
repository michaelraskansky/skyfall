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


# ── Detector ─────────────────────────────────────────────────────────────────

_HEADING_CACHE_LIMIT = 50_000


class AircraftDetector:
    """
    Stateful anomaly detector for ADS-B aircraft state batches.

    Evaluates three rules in priority order per aircraft:
      1. Emergency squawk (7700/7600/7500)
      2. Watched hex code
      3. Heading reroute (>= threshold since last batch)

    An aircraft matching an earlier rule is emitted and skips later rules.
    """

    def __init__(
        self,
        watch_hex_codes: set[str],
        heading_threshold_deg: float = HEADING_CHANGE_THRESHOLD_DEG,
    ) -> None:
        self._watch_hexes = {h.lower() for h in watch_hex_codes}
        self._heading_threshold = heading_threshold_deg
        self._prev_headings: dict[str, float] = {}

    def process_batch(self, states: list[AircraftState]) -> list[RawEvent]:
        """
        Process a batch of aircraft states and return anomaly events.

        Updates internal heading state for every airborne aircraft.
        """
        events: list[RawEvent] = []

        for state in states:
            if state.on_ground:
                continue

            payload = {
                "hex": state.hex,
                "callsign": state.callsign,
                "lat": state.lat,
                "lon": state.lon,
                "track": state.track,
                "alt_baro_m": state.alt_m,
                "velocity_m_s": state.velocity_m_s,
                "origin_country": state.origin_country,
                "squawk": state.squawk,
            }

            label = state.callsign or state.hex
            coord_str = f"({state.lat}, {state.lon})" if state.lat is not None else "(unknown)"

            # ── Rule 1: Emergency squawk ─────────────────────────
            if state.squawk and state.squawk in EMERGENCY_SQUAWKS:
                meaning = EMERGENCY_SQUAWKS[state.squawk]
                events.append(RawEvent(
                    source=EventSource.ADSB,
                    latitude=state.lat,
                    longitude=state.lon,
                    raw_payload=payload,
                    description=(
                        f"SQUAWK {state.squawk} ({meaning}): "
                        f"{label} @ {coord_str}, alt={state.alt_m}m"
                    ),
                ))
                logger.warning(
                    "Emergency squawk %s (%s): %s @ %s",
                    state.squawk, meaning, label, coord_str,
                )
                if state.track is not None:
                    self._prev_headings[state.hex] = state.track
                continue

            # ── Rule 2: Watch hex ────────────────────────────────
            if state.hex in self._watch_hexes:
                events.append(RawEvent(
                    source=EventSource.ADSB,
                    latitude=state.lat,
                    longitude=state.lon,
                    raw_payload=payload,
                    description=(
                        f"Watched aircraft {state.hex} ({label}) "
                        f"detected @ {coord_str}, alt={state.alt_m}m"
                    ),
                ))
                logger.info("ADS-B watch hit: %s", label)
                if state.track is not None:
                    self._prev_headings[state.hex] = state.track
                continue

            # ── Rule 3: Heading reroute ──────────────────────────
            if state.lat is None or state.lon is None or state.track is None:
                continue

            track = float(state.track)

            if state.hex in self._prev_headings:
                delta = heading_delta(self._prev_headings[state.hex], track)
                if delta >= self._heading_threshold:
                    events.append(RawEvent(
                        source=EventSource.ADSB,
                        latitude=state.lat,
                        longitude=state.lon,
                        raw_payload=payload,
                        description=(
                            f"Flight {label} rerouted by {delta:.0f}° "
                            f"@ {coord_str}"
                        ),
                    ))
                    logger.info("ADS-B reroute: %s by %.0f°", label, delta)

            self._prev_headings[state.hex] = track

        # Prune heading cache if too large
        if len(self._prev_headings) > _HEADING_CACHE_LIMIT:
            self._prev_headings.clear()

        return events
