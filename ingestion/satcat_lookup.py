"""
SATCAT Enrichment Lookup
=========================

On-demand queries to Space-Track's Satellite Catalog (SATCAT) to enrich
trajectory alerts with object identity: name, country, launch date, type,
and radar cross-section size.

Results are cached in-memory for the process lifetime — SATCAT data is
static (a satellite's name and launch date never change).
"""

from __future__ import annotations

import logging

import aiohttp
from pydantic import BaseModel, Field

from config import settings

logger = logging.getLogger(__name__)

_LOGIN_URL = "https://www.space-track.org/ajaxauth/login"
_SATCAT_URL = (
    "https://www.space-track.org/basicspacedata/query"
    "/class/satcat"
    "/NORAD_CAT_ID/{norad_id}"
    "/format/json"
)


class SatcatInfo(BaseModel):
    """Identity data for a catalogued space object."""

    norad_cat_id: str
    object_name: str = Field(default="UNKNOWN")
    country: str = Field(default="UNKNOWN")
    launch_date: str = Field(default="", description="ISO date, e.g. 2022-08-12")
    object_type: str = Field(default="", description="PAYLOAD, ROCKET BODY, DEBRIS, UNKNOWN")
    rcs_size: str = Field(default="", description="SMALL, MEDIUM, LARGE")


class SatcatLookup:
    """Async Space-Track SATCAT client with in-memory cache."""

    def __init__(self) -> None:
        self._session: aiohttp.ClientSession | None = None
        self._authenticated: bool = False
        self._cache: dict[str, SatcatInfo] = {}

    async def connect(self) -> None:
        """Create the HTTP session and authenticate."""
        self._session = aiohttp.ClientSession()
        await self._authenticate()
        logger.info("SatcatLookup connected to Space-Track")

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session:
            await self._session.close()
            self._session = None

    async def _authenticate(self) -> bool:
        """Authenticate against Space-Track. Returns True on success."""
        identity = settings.spacetrack_identity
        password = settings.spacetrack_password
        if not identity or not password:
            logger.warning("Space-Track credentials not set — SATCAT lookup disabled")
            return False

        try:
            async with self._session.post(
                _LOGIN_URL,
                data={"identity": identity, "password": password},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                if resp.status == 200:
                    self._authenticated = True
                    return True
                logger.error("Space-Track SATCAT auth failed: HTTP %d", resp.status)
                return False
        except Exception:
            logger.exception("Space-Track SATCAT auth error")
            return False

    async def get_info(self, norad_cat_id: str) -> SatcatInfo | None:
        """
        Look up SATCAT data for a NORAD catalog ID.

        Returns cached result if available. On cache miss, queries
        Space-Track. Returns None on any failure.
        """
        # Cache hit
        if norad_cat_id in self._cache:
            return self._cache[norad_cat_id]

        if not self._session or not self._authenticated:
            return None

        url = _SATCAT_URL.format(norad_id=norad_cat_id)

        try:
            async with self._session.get(
                url, timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                # Re-auth on 401 and retry once
                if resp.status == 401:
                    logger.info("SATCAT query got 401, re-authenticating")
                    if not await self._authenticate():
                        return None
                    async with self._session.get(
                        url, timeout=aiohttp.ClientTimeout(total=15),
                    ) as retry_resp:
                        if retry_resp.status != 200:
                            return None
                        data = await retry_resp.json()
                elif resp.status != 200:
                    logger.warning(
                        "SATCAT query failed for %s: HTTP %d",
                        norad_cat_id, resp.status,
                    )
                    return None
                else:
                    data = await resp.json()

            if not data:
                logger.debug("SATCAT returned empty for %s", norad_cat_id)
                return None

            record = data[0]
            info = SatcatInfo(
                norad_cat_id=norad_cat_id,
                object_name=record.get("OBJECT_NAME", "UNKNOWN"),
                country=record.get("COUNTRY", "UNKNOWN"),
                launch_date=record.get("LAUNCH", ""),
                object_type=record.get("OBJECT_TYPE", ""),
                rcs_size=record.get("RCSVALUE", ""),
            )
            self._cache[norad_cat_id] = info
            logger.info(
                "SATCAT enrichment for %s: %s (%s, %s)",
                norad_cat_id, info.object_name, info.country, info.object_type,
            )
            return info

        except Exception:
            logger.exception("SATCAT lookup error for %s", norad_cat_id)
            return None
