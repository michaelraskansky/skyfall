"""
NASA FIRMS Thermal Anomaly Poller
=================================

Polls the NASA Fire Information for Resource Management System (FIRMS) API
for VIIRS 375m active-fire / thermal-anomaly data.

Strategy:
  1. Every ``FIRMS_POLL_INTERVAL_SEC`` seconds, query the FIRMS CSV endpoint
     for the configured bounding box(es).
  2. Filter for anomalously high brightness temperatures (FRP – Fire Radiative
     Power) that indicate sudden, massive thermal spikes rather than ordinary
     wildfires.
  3. Emit a ``RawEvent`` for each qualifying hotspot into the shared
     ``asyncio.Queue`` consumed by the correlation engine.

Reference:
  https://firms.modaps.eosdis.nasa.gov/api/area/
"""

from __future__ import annotations

import asyncio
import csv
import io
import logging
from typing import TYPE_CHECKING

import aiohttp

if TYPE_CHECKING:
    from asyncio import Queue

from config import settings
from models import EventSource, RawEvent

logger = logging.getLogger(__name__)

# ── Tunables ──────────────────────────────────────────────────────────────────
# Fire Radiative Power threshold (MW).  Normal wildfires sit below ~200 MW;
# industrial explosions or re-entry events easily exceed 500 MW.
FRP_THRESHOLD_MW: float = 500.0

# Brightness temperature threshold (Kelvin).  VIIRS band I4 saturates around
# 367 K for low-temp fires; we want only extreme outliers.
BRIGHTNESS_THRESHOLD_K: float = 400.0


def _parse_bounding_boxes(raw: str) -> list[str]:
    """Parse semicolon-separated bounding-box strings."""
    return [bb.strip() for bb in raw.split(";") if bb.strip()]


async def poll_firms(event_queue: Queue[RawEvent]) -> None:
    """
    Long-running coroutine that polls FIRMS on a timer and pushes
    qualifying thermal events into *event_queue*.
    """
    api_key = settings.firms_api_key
    if not api_key:
        logger.warning("FIRMS_API_KEY not set – FIRMS poller disabled.")
        return

    bboxes = _parse_bounding_boxes(settings.firms_bounding_boxes)
    interval = settings.firms_poll_interval_sec

    # FIRMS CSV endpoint pattern for VIIRS NOAA-20 (NOAA-21) 375m NRT data.
    # Area query: /api/area/csv/{key}/VIIRS_NOAA20_NRT/{bbox}/1
    # The trailing "1" means "last 1 day".  We de-duplicate by timestamp.
    base_url = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

    logger.info(
        "FIRMS poller starting – %d bbox(es), interval=%ds, FRP>%.0f MW",
        len(bboxes),
        interval,
        FRP_THRESHOLD_MW,
    )

    seen_ids: set[str] = set()  # simple de-dup by (lat, lon, acq_date, acq_time)

    async with aiohttp.ClientSession() as session:
        while True:
            for bbox in bboxes:
                url = f"{base_url}/{api_key}/VIIRS_NOAA20_NRT/{bbox}/1"
                try:
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                        if resp.status != 200:
                            logger.error("FIRMS API returned HTTP %d", resp.status)
                            continue

                        text = await resp.text()
                        reader = csv.DictReader(io.StringIO(text))

                        for row in reader:
                            # ── Build a dedup key ─────────────────────────
                            dedup = f"{row.get('latitude')}_{row.get('longitude')}_{row.get('acq_date')}_{row.get('acq_time')}"
                            if dedup in seen_ids:
                                continue
                            seen_ids.add(dedup)

                            # ── Filter for extreme thermal events ─────────
                            frp = float(row.get("frp", 0))
                            bright = float(row.get("bright_ti4", 0))

                            if frp < FRP_THRESHOLD_MW and bright < BRIGHTNESS_THRESHOLD_K:
                                continue

                            lat = float(row["latitude"])
                            lon = float(row["longitude"])

                            event = RawEvent(
                                source=EventSource.FIRMS,
                                latitude=lat,
                                longitude=lon,
                                raw_payload=dict(row),
                                description=(
                                    f"FIRMS thermal spike: FRP={frp:.1f} MW, "
                                    f"bright_ti4={bright:.1f} K @ ({lat:.4f}, {lon:.4f})"
                                ),
                            )
                            logger.info("FIRMS event: %s", event.description)
                            await event_queue.put(event)

                except Exception:
                    logger.exception("FIRMS poll error for bbox=%s", bbox)

            # Prune dedup set periodically to avoid unbounded growth.
            if len(seen_ids) > 50_000:
                seen_ids.clear()

            await asyncio.sleep(interval)
