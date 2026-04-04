"""
ADS-B Airspace Disruption Poller (OpenSky Network)
====================================================

Polls the OpenSky Network REST API every 30 seconds to detect:

  1. **Sudden rerouting** – aircraft whose ground track changes by >45°
     between consecutive polls, suggesting ATC-directed avoidance of an
     airspace hazard.
  2. **Survey aircraft appearance** – specific hex (ICAO24) codes belonging
     to government / high-altitude survey planes suddenly appearing in a
     region of interest.

OpenSky API reference: https://openskynetwork.github.io/opensky-api/rest.html
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import aiohttp

if TYPE_CHECKING:
    from asyncio import Queue

from config import settings
from models import EventSource, RawEvent

logger = logging.getLogger(__name__)

# ── OpenSky response field indices ───────────────────────────────────────────
_ICAO24 = 0
_CALLSIGN = 1
_ORIGIN_COUNTRY = 2
_LONGITUDE = 5
_LATITUDE = 6
_BARO_ALT = 7
_ON_GROUND = 8
_VELOCITY = 9
_TRUE_TRACK = 10
_GEO_ALT = 13

_OPENSKY_URL = "https://opensky-network.org/api/states/all"

from ingestion.adsb_detector import (
    HEADING_CHANGE_THRESHOLD_DEG,
    heading_delta as _heading_delta,
)


def _parse_watch_hex_codes(raw: str) -> set[str]:
    """Return lowercased set of hex ICAO24 codes to watch for."""
    return {h.strip().lower() for h in raw.split(",") if h.strip()}


async def poll_adsb(event_queue: Queue[RawEvent]) -> None:
    """
    Long-running coroutine that polls OpenSky for aircraft states and pushes
    anomalous airspace observations into *event_queue*.
    """
    # OpenSky blocks AWS IP ranges — disabled until we have a cloud-friendly
    # ADS-B source (ADS-B Exchange RapidAPI, AeroAPI, or authenticated OpenSky).
    logger.warning("ADS-B poller disabled – OpenSky blocks cloud IPs.")
    await asyncio.Event().wait()

    interval = settings.adsb_poll_interval_sec
    watch_hexes = _parse_watch_hex_codes(settings.adsb_watch_hex_codes)

    logger.info(
        "ADS-B poller starting (OpenSky) – interval=%ds, watching %d hex codes",
        interval,
        len(watch_hexes),
    )

    # Previous-poll state: hex -> last known heading.
    prev_headings: dict[str, float] = {}

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                # Build query params — use bounding box if configured
                params = {}
                bbox = settings.adsb_bounding_box
                if bbox:
                    parts = [p.strip() for p in bbox.split(",")]
                    if len(parts) == 4:
                        params = {
                            "lamin": parts[0],
                            "lomin": parts[1],
                            "lamax": parts[2],
                            "lomax": parts[3],
                        }

                async with session.get(
                    _OPENSKY_URL,
                    params=params if params else None,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status != 200:
                        logger.error("OpenSky API returned HTTP %d", resp.status)
                        await asyncio.sleep(interval)
                        continue

                    data = await resp.json()

                states = data.get("states") or []

                for ac in states:
                    if len(ac) < 17:
                        continue

                    hex_code = (ac[_ICAO24] or "").lower()
                    callsign = (ac[_CALLSIGN] or "").strip()
                    lat = ac[_LATITUDE]
                    lon = ac[_LONGITUDE]
                    heading = ac[_TRUE_TRACK]
                    alt_m = ac[_BARO_ALT]
                    on_ground = ac[_ON_GROUND]

                    if not hex_code or on_ground:
                        continue

                    ac_payload = {
                        "hex": hex_code,
                        "callsign": callsign,
                        "lat": lat,
                        "lon": lon,
                        "track": heading,
                        "alt_baro_m": alt_m,
                        "geo_alt_m": ac[_GEO_ALT],
                        "velocity_m_s": ac[_VELOCITY],
                        "origin_country": ac[_ORIGIN_COUNTRY],
                    }

                    # ── Check 1: survey aircraft appearance ──────────────
                    if hex_code in watch_hexes:
                        event = RawEvent(
                            source=EventSource.ADSB,
                            latitude=float(lat) if lat else None,
                            longitude=float(lon) if lon else None,
                            raw_payload=ac_payload,
                            description=(
                                f"Watched survey aircraft {hex_code} ({callsign}) "
                                f"detected @ ({lat}, {lon}), alt={alt_m}m"
                            ),
                        )
                        logger.info("ADS-B watch hit: %s", event.description)
                        await event_queue.put(event)
                        continue

                    # ── Check 2: sudden heading change (rerouting) ───────
                    if heading is None or lat is None or lon is None:
                        continue

                    heading = float(heading)

                    if hex_code in prev_headings:
                        delta = _heading_delta(prev_headings[hex_code], heading)
                        if delta >= HEADING_CHANGE_THRESHOLD_DEG:
                            event = RawEvent(
                                source=EventSource.ADSB,
                                latitude=float(lat),
                                longitude=float(lon),
                                raw_payload=ac_payload,
                                description=(
                                    f"Flight {callsign or hex_code} "
                                    f"rerouted by {delta:.0f}° "
                                    f"@ ({lat}, {lon})"
                                ),
                            )
                            logger.info("ADS-B reroute: %s", event.description)
                            await event_queue.put(event)

                    prev_headings[hex_code] = heading

            except Exception:
                logger.exception("ADS-B poll error")

            # Prune heading cache if it grows too large
            if len(prev_headings) > 50_000:
                prev_headings.clear()

            await asyncio.sleep(interval)
