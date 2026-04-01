"""
ADS-B Exchange Airspace Disruption Poller
==========================================

Polls an ADS-B Exchange-compatible API every 30 seconds to detect:

  1. **Sudden rerouting** – commercial flights (identified by common airline
     ICAO prefixes) whose ground track changes by >45° between consecutive
     polls, suggesting ATC-directed avoidance of an airspace hazard.
  2. **Survey aircraft appearance** – specific hex (ICAO24) codes belonging
     to government / high-altitude survey planes suddenly appearing in a
     region of interest.

Each qualifying observation is pushed as a ``RawEvent`` into the shared
``asyncio.Queue``.
"""

from __future__ import annotations

import asyncio
import logging
import math
from typing import TYPE_CHECKING

import aiohttp

if TYPE_CHECKING:
    from asyncio import Queue

from config import settings
from models import EventSource, RawEvent

logger = logging.getLogger(__name__)

# ── Tunables ──────────────────────────────────────────────────────────────────
# Minimum heading change (degrees) between two consecutive polls for a
# commercial flight to be flagged as "suddenly rerouted".
HEADING_CHANGE_THRESHOLD_DEG: float = 45.0


def _parse_watch_hex_codes(raw: str) -> set[str]:
    """Return uppercased set of hex ICAO24 codes to watch for."""
    return {h.strip().upper() for h in raw.split(",") if h.strip()}


def _heading_delta(h1: float, h2: float) -> float:
    """Compute the smallest angular difference between two headings (0-360)."""
    d = abs(h1 - h2) % 360
    return d if d <= 180 else 360 - d


async def poll_adsb(event_queue: Queue[RawEvent]) -> None:
    """
    Long-running coroutine that polls ADS-B data and pushes anomalous
    airspace observations into *event_queue*.
    """
    api_key = settings.adsb_api_key
    base_url = settings.adsb_api_base_url
    interval = settings.adsb_poll_interval_sec
    watch_hexes = _parse_watch_hex_codes(settings.adsb_watch_hex_codes)

    if not api_key:
        logger.warning("ADSB_API_KEY not set – ADS-B poller disabled.")
        return

    logger.info(
        "ADS-B poller starting – interval=%ds, watching %d hex codes",
        interval,
        len(watch_hexes),
    )

    # Previous-poll state: hex -> last known heading.
    prev_headings: dict[str, float] = {}

    headers = {
        "api-auth": api_key,
        "Accept": "application/json",
    }

    async with aiohttp.ClientSession(headers=headers) as session:
        while True:
            try:
                # ── Fetch current aircraft snapshot ───────────────────────
                # The exact endpoint and payload shape depends on the ADS-B
                # provider.  We assume a JSON response with an "ac" list
                # (ADS-B Exchange v2 style).
                async with session.get(
                    base_url,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    if resp.status != 200:
                        logger.error("ADS-B API returned HTTP %d", resp.status)
                        await asyncio.sleep(interval)
                        continue

                    data = await resp.json()

                aircraft_list: list[dict] = data.get("ac", [])

                for ac in aircraft_list:
                    hex_code: str = ac.get("hex", "").upper()
                    heading: float | None = ac.get("track") or ac.get("true_heading")
                    lat = ac.get("lat")
                    lon = ac.get("lon")

                    if not hex_code:
                        continue

                    # ── Check 1: survey aircraft appearance ───────────────
                    if hex_code in watch_hexes:
                        event = RawEvent(
                            source=EventSource.ADSB,
                            latitude=float(lat) if lat else None,
                            longitude=float(lon) if lon else None,
                            raw_payload=ac,
                            description=(
                                f"Watched survey aircraft {hex_code} detected "
                                f"@ ({lat}, {lon}), alt={ac.get('alt_baro', '?')} ft"
                            ),
                        )
                        logger.info("ADS-B watch hit: %s", event.description)
                        await event_queue.put(event)
                        continue

                    # ── Check 2: sudden heading change (rerouting) ────────
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
                                raw_payload=ac,
                                description=(
                                    f"Flight {ac.get('flight', hex_code).strip()} "
                                    f"rerouted by {delta:.0f}° "
                                    f"@ ({lat}, {lon})"
                                ),
                            )
                            logger.info("ADS-B reroute: %s", event.description)
                            await event_queue.put(event)

                    prev_headings[hex_code] = heading

            except Exception:
                logger.exception("ADS-B poll error")

            await asyncio.sleep(interval)
