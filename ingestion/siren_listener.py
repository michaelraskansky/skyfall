"""
Pikud HaOref Siren Listener
=============================

Polls the Pikud HaOref (Israel Home Front Command) real-time alert API
every second for rocket/missile siren activations.

Filters for configured watch zones and differentiates between active
alerts and "event ended" clearance messages.

API endpoint: https://www.oref.org.il/warningMessages/alert/Alerts.json
"""

from __future__ import annotations

import asyncio
import logging
import random
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import aiohttp
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from asyncio import Queue

from config import settings
from models import EventSource, RawEvent
from output.alerter import send_system_warning

logger = logging.getLogger(__name__)

_ALERTS_URL = "https://www.oref.org.il/warningMessages/alert/Alerts.json"

# Required headers — the Oref API rejects requests without a proper Referer.
_HEADERS = {
    "Referer": "https://www.oref.org.il/",
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "application/json",
}

# Title strings for alert state detection
_TITLE_EVENT_ENDED = "האירוע הסתיים"
_TITLE_ACTIVE_PREFIX = "בדקות הקרובות"

# Resilience: consecutive failure tracking
_CONSECUTIVE_FAILURE_THRESHOLD = 3
_BACKOFF_BASE_SEC = 5.0
_BACKOFF_JITTER_SEC = 2.0

# Alert categories we react to (trajectory-relevant threats).
# Based on the Pikud HaOref categories API schema.
ALERT_CATEGORIES: set[str] = {
    "missilealert",      # Rocket/missile — primary use case
    "uav",               # UAV/drone alerts
    "nonconventional",   # CBRN threats
    "warning",           # General warning
}

# Categories to explicitly reject (drills, informational).
# matrix_id >= 100 are drills; update/flash are informational.
DRILL_CATEGORIES: set[str] = {
    "missilealertdrill", "uavdrill", "nonconventionaldrill", "warningdrill",
    "memorialdaydrill1", "memorialdaydrill2",
    "earthquakedrill1", "earthquakedrill2",
    "cbrnedrill", "terrorattackdrill", "tsunamidrill", "hazmatdrill",
    "updatedrill", "flashdrill",
}

# Human-readable labels for alert categories in Slack/Discord messages.
CATEGORY_LABELS: dict[str, str] = {
    "missilealert": "Missile",
    "uav": "UAV/Drone",
    "nonconventional": "Non-Conventional",
    "warning": "General Warning",
}

# Watch zones — only emit events if these zones appear in the data array.
WATCH_ZONES: set[str] = {
    "רמת השרון",
    "מתחם גלילות",
    "הוד השרון",
}

# Approximate coordinates for watch zones (for correlation with trajectories).
ZONE_COORDINATES: dict[str, tuple[float, float]] = {
    "רמת השרון": (32.1461, 34.8394),
    "מתחם גלילות": (32.1580, 34.8020),
    "הוד השרון": (32.1500, 34.8880),
}


class SirenEvent(BaseModel):
    """Parsed siren alert from Pikud HaOref."""

    alert_id: str
    title: str
    category: str = ""
    zones: list[str] = Field(default_factory=list)
    matched_watch_zones: list[str] = Field(default_factory=list)
    description: str = ""
    is_active: bool = True
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


async def poll_sirens(
    event_queue: Queue[RawEvent],
    siren_callback=None,
) -> None:
    """
    Long-running coroutine that polls Pikud HaOref every second.

    Parameters
    ----------
    event_queue : Queue[RawEvent]
        Shared pipeline queue for raw events.
    siren_callback : callable, optional
        Async callback invoked with a SirenEvent when a watch zone is hit.
        Used by the triage loop for trajectory correlation.
    """
    logger.info("Siren listener starting – watching zones: %s", WATCH_ZONES)

    seen_ids: set[str] = set()
    consecutive_failures: int = 0
    blindness_alerted: bool = False

    async with aiohttp.ClientSession(headers=_HEADERS) as session:
        while True:
            try:
                async with session.get(
                    _ALERTS_URL,
                    timeout=aiohttp.ClientTimeout(total=5),
                ) as resp:
                    if resp.status != 200:
                        consecutive_failures += 1
                        logger.warning(
                            "Siren poll HTTP %d (failure %d/%d)",
                            resp.status, consecutive_failures,
                            _CONSECUTIVE_FAILURE_THRESHOLD,
                        )

                        if (
                            consecutive_failures >= _CONSECUTIVE_FAILURE_THRESHOLD
                            and not blindness_alerted
                        ):
                            blindness_alerted = True
                            await send_system_warning(
                                f"Siren Feed Blocked: {consecutive_failures} consecutive "
                                f"failures (last HTTP {resp.status}). "
                                f"The siren listener may be blind."
                            )

                        if consecutive_failures >= _CONSECUTIVE_FAILURE_THRESHOLD:
                            backoff = _BACKOFF_BASE_SEC + random.uniform(
                                0, _BACKOFF_JITTER_SEC,
                            )
                            await asyncio.sleep(backoff)
                        else:
                            await asyncio.sleep(1)
                        continue

                    # Success — reset failure state
                    if consecutive_failures > 0:
                        logger.info(
                            "Siren feed recovered after %d failures",
                            consecutive_failures,
                        )
                    consecutive_failures = 0
                    blindness_alerted = False

                    text = await resp.text()

                    # The API returns empty string or empty array when no alerts
                    if not text or text.strip() in ("", "[]"):
                        await asyncio.sleep(1)
                        continue

                    alerts = await resp.json(content_type=None)

                # Handle single alert (dict) or multiple (list)
                if isinstance(alerts, dict):
                    alerts = [alerts]

                for alert in alerts:
                    alert_id = alert.get("id", "")
                    if not alert_id or alert_id in seen_ids:
                        continue
                    seen_ids.add(alert_id)

                    title = alert.get("title", "")
                    zones = alert.get("data", [])
                    category = str(alert.get("cat", ""))
                    desc = alert.get("desc", "")

                    # Filter by category — only react to trajectory-relevant threats
                    if category in DRILL_CATEGORIES:
                        logger.debug(
                            "Skipping drill alert %s (cat=%s)", alert_id, category,
                        )
                        continue
                    if category and category not in ALERT_CATEGORIES:
                        logger.debug(
                            "Skipping non-trajectory alert %s (cat=%s)",
                            alert_id, category,
                        )
                        continue

                    # Check if any watch zones are in the alert
                    matched = [z for z in zones if z in WATCH_ZONES]
                    if not matched:
                        continue

                    is_active = title != _TITLE_EVENT_ENDED

                    siren_event = SirenEvent(
                        alert_id=alert_id,
                        title=title,
                        category=category,
                        zones=zones,
                        matched_watch_zones=matched,
                        description=desc,
                        is_active=is_active,
                    )

                    # Use the centroid of matched watch zones for coordinates
                    lats = [ZONE_COORDINATES[z][0] for z in matched if z in ZONE_COORDINATES]
                    lons = [ZONE_COORDINATES[z][1] for z in matched if z in ZONE_COORDINATES]
                    lat = sum(lats) / len(lats) if lats else None
                    lon = sum(lons) / len(lons) if lons else None

                    cat_label = CATEGORY_LABELS.get(category, category or "unknown")
                    status = "ACTIVE SIREN" if is_active else "EVENT ENDED"
                    logger.warning(
                        "[SIREN] %s %s in %s (alert %s)",
                        cat_label, status, ", ".join(matched), alert_id,
                    )

                    # Push to the shared event queue
                    raw_event = RawEvent(
                        source=EventSource.SIREN,
                        latitude=lat,
                        longitude=lon,
                        raw_payload=alert,
                        description=(
                            f"[SIREN {status}] {cat_label} | "
                            f"Zones: {', '.join(matched)} | Title: {title}"
                        ),
                    )
                    await event_queue.put(raw_event)

                    # Invoke callback for trajectory correlation
                    if siren_callback:
                        await siren_callback(siren_event)

            except asyncio.CancelledError:
                raise
            except Exception:
                consecutive_failures += 1
                logger.exception(
                    "Siren poll error (failure %d/%d)",
                    consecutive_failures, _CONSECUTIVE_FAILURE_THRESHOLD,
                )

                if (
                    consecutive_failures >= _CONSECUTIVE_FAILURE_THRESHOLD
                    and not blindness_alerted
                ):
                    blindness_alerted = True
                    await send_system_warning(
                        f"Siren Feed Blocked: {consecutive_failures} consecutive "
                        f"failures (exception). The siren listener may be blind."
                    )

                if consecutive_failures >= _CONSECUTIVE_FAILURE_THRESHOLD:
                    backoff = _BACKOFF_BASE_SEC + random.uniform(
                        0, _BACKOFF_JITTER_SEC,
                    )
                    await asyncio.sleep(backoff)
                else:
                    await asyncio.sleep(1)

            # Prune seen IDs periodically
            if len(seen_ids) > 10_000:
                seen_ids.clear()

            await asyncio.sleep(1)
