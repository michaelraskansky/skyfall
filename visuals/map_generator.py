"""
Ground Track Map Generator
===========================

Renders an ImpactPrediction's trajectory data as a static PNG map
with ground track line, confidence corridor, and impact marker.

Uses the ``staticmap`` library (pure Python, Pillow + requests).
This module is CPU-bound and should be called via ``asyncio.to_thread()``.
"""

from __future__ import annotations

import logging
import math
import tempfile
from typing import Optional

from staticmap import CircleMarker, Line, Polygon, StaticMap

from ingestion.satcat_lookup import SatcatInfo
from trajectory.models import ImpactPrediction

logger = logging.getLogger(__name__)

# Map dimensions
MAP_WIDTH = 720
MAP_HEIGHT = 480

# Metres per degree of latitude (approximate)
_M_PER_DEG_LAT = 111_320.0


def _m_per_deg_lon(lat_deg: float) -> float:
    """Metres per degree of longitude at a given latitude."""
    return _M_PER_DEG_LAT * math.cos(math.radians(lat_deg))


def _compute_corridor(
    waypoints: list[dict],
    semi_major_m: float,
) -> list[tuple[float, float]]:
    """
    Compute a confidence corridor polygon around the ground track.

    For each waypoint, offset perpendicular to the track direction by
    ``semi_major_m`` metres (converted to degrees at that latitude).
    Returns a closed polygon as [(lon, lat), ...].
    """
    if len(waypoints) < 2 or semi_major_m <= 0:
        return []

    left_edge: list[tuple[float, float]] = []
    right_edge: list[tuple[float, float]] = []

    for i, wp in enumerate(waypoints):
        lat, lon = wp["lat"], wp["lon"]

        # Compute track direction from neighboring waypoints
        if i == 0:
            dlat = waypoints[1]["lat"] - lat
            dlon = waypoints[1]["lon"] - lon
        elif i == len(waypoints) - 1:
            dlat = lat - waypoints[-2]["lat"]
            dlon = lon - waypoints[-2]["lon"]
        else:
            dlat = waypoints[i + 1]["lat"] - waypoints[i - 1]["lat"]
            dlon = waypoints[i + 1]["lon"] - waypoints[i - 1]["lon"]

        # Perpendicular direction (rotate 90°)
        track_len = math.sqrt(dlat ** 2 + dlon ** 2)
        if track_len < 1e-10:
            continue
        perp_dlat = -dlon / track_len
        perp_dlon = dlat / track_len

        # Convert buffer from metres to degrees
        buf_lat = semi_major_m / _M_PER_DEG_LAT
        m_per_lon = _m_per_deg_lon(lat)
        buf_lon = semi_major_m / m_per_lon if m_per_lon > 0 else buf_lat

        # Offset in the perpendicular direction
        left_edge.append((
            lon + perp_dlon * buf_lon,
            lat + perp_dlat * buf_lat,
        ))
        right_edge.append((
            lon - perp_dlon * buf_lon,
            lat - perp_dlat * buf_lat,
        ))

    # Close the polygon: left edge forward, right edge reversed
    if not left_edge:
        return []
    return left_edge + list(reversed(right_edge))


def render_ground_track(
    prediction: ImpactPrediction,
    satcat_info: Optional[SatcatInfo] = None,
) -> str:
    """
    Render a ground track map as a PNG file.

    Parameters
    ----------
    prediction : ImpactPrediction
        Must have ``trajectory_points`` with at least 2 entries.
    satcat_info : SatcatInfo, optional
        Used for the title annotation (not rendered on map, used by caller).

    Returns
    -------
    str : Path to the temporary PNG file. Caller must delete after use.
    """
    m = StaticMap(MAP_WIDTH, MAP_HEIGHT)
    waypoints = prediction.trajectory_points

    if len(waypoints) < 2:
        logger.warning("Not enough trajectory points for map (%d)", len(waypoints))
        # Still produce a map with just the impact marker
        m.add_marker(CircleMarker(
            (prediction.impact_longitude, prediction.impact_latitude),
            "red", 12,
        ))
        return _save_map(m)

    # ── 1. Confidence corridor (semi-transparent polygon) ────────────────
    cov = prediction.covariance_position_enu
    cov_ee = cov[0][0]
    cov_en = cov[0][1]
    cov_nn = cov[1][1]
    trace = cov_ee + cov_nn
    det = cov_ee * cov_nn - cov_en * cov_en
    discriminant = max((trace / 2) ** 2 - det, 0)
    semi_major_m = 2.0 * math.sqrt(max(trace / 2 + math.sqrt(discriminant), 0))

    corridor = _compute_corridor(waypoints, semi_major_m)
    if corridor:
        m.add_polygon(Polygon(corridor, fill_color="#FF000030", outline_color="#FF000060"))

    # ── 2. Ground track line (red) ───────────────────────────────────────
    track_coords = [(wp["lon"], wp["lat"]) for wp in waypoints]
    m.add_line(Line(track_coords, "red", 3))

    # ── 3. Origin marker (blue) ──────────────────────────────────────────
    origin = waypoints[0]
    m.add_marker(CircleMarker((origin["lon"], origin["lat"]), "blue", 8))

    # ── 4. Impact marker (red, larger) ───────────────────────────────────
    m.add_marker(CircleMarker(
        (prediction.impact_longitude, prediction.impact_latitude),
        "red", 12,
    ))

    return _save_map(m)


def _save_map(m: StaticMap) -> str:
    """Render the map to a temporary PNG and return the path."""
    image = m.render()
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    image.save(tmp.name)
    tmp.close()
    logger.info("Ground track map rendered: %s", tmp.name)
    return tmp.name
