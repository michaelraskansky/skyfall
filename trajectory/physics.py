"""
Trajectory Engine – Physics & Coordinate Utilities
====================================================

Pure-Python (+ numpy) implementations of:

  1. **Geodetic ↔ ECEF ↔ local-ENU coordinate transforms.**
     All EKF math happens in a local East-North-Up (ENU) Cartesian frame
     anchored at the first observation.  We convert geodetic (lat/lon/alt)
     inputs into ENU on ingestion and convert ENU results back to geodetic
     for output.

  2. **US Standard Atmosphere 1976 – exponential approximation.**
     Provides air density ρ(h) as a function of altitude.

  3. **Aerodynamic drag acceleration.**
     Given the ballistic coefficient β = m/(C_d·A) and the current state
     vector, returns the drag deceleration opposing the velocity vector.

  4. **Gravity.**
     Simple constant-g model (9.81 m/s² downward in ENU).

All functions are stateless and strictly typed.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
# WGS-84 ellipsoid constants
# ═══════════════════════════════════════════════════════════════════════════════
WGS84_A: float = 6_378_137.0           # semi-major axis (m)
WGS84_F: float = 1.0 / 298.257223563   # flattening
WGS84_B: float = WGS84_A * (1.0 - WGS84_F)  # semi-minor axis (m)
WGS84_E2: float = 1.0 - (WGS84_B ** 2) / (WGS84_A ** 2)  # first eccentricity²

# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════
GRAVITY: float = 9.80665  # m/s² (standard gravity)


# ═══════════════════════════════════════════════════════════════════════════════
# Coordinate transforms
# ═══════════════════════════════════════════════════════════════════════════════


def geodetic_to_ecef(
    lat_deg: float, lon_deg: float, alt_m: float
) -> Tuple[float, float, float]:
    """
    Convert geodetic (WGS-84) coordinates to Earth-Centred Earth-Fixed (ECEF).

    Parameters
    ----------
    lat_deg : Geodetic latitude in **degrees**.
    lon_deg : Geodetic longitude in **degrees**.
    alt_m   : Height above the WGS-84 ellipsoid in **metres**.

    Returns
    -------
    (x, y, z) in metres (ECEF).
    """
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    # Radius of curvature in the prime vertical
    N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat ** 2)

    x = (N + alt_m) * cos_lat * cos_lon
    y = (N + alt_m) * cos_lat * sin_lon
    z = (N * (1.0 - WGS84_E2) + alt_m) * sin_lat
    return (x, y, z)


def ecef_to_enu(
    x: float,
    y: float,
    z: float,
    ref_lat_deg: float,
    ref_lon_deg: float,
    ref_alt_m: float,
) -> Tuple[float, float, float]:
    """
    Convert an ECEF point to local East-North-Up (ENU) relative to a
    reference geodetic origin.

    Parameters
    ----------
    x, y, z           : ECEF coordinates of the target point (m).
    ref_lat_deg, ref_lon_deg, ref_alt_m : Geodetic origin of the ENU frame.

    Returns
    -------
    (e, n, u) in metres – East, North, Up offsets from the reference origin.
    """
    # Reference point in ECEF
    rx, ry, rz = geodetic_to_ecef(ref_lat_deg, ref_lon_deg, ref_alt_m)
    dx, dy, dz = x - rx, y - ry, z - rz

    lat = math.radians(ref_lat_deg)
    lon = math.radians(ref_lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    # Rotation matrix ECEF → ENU
    e = -sin_lon * dx + cos_lon * dy
    n = -sin_lat * cos_lon * dx - sin_lat * sin_lon * dy + cos_lat * dz
    u = cos_lat * cos_lon * dx + cos_lat * sin_lon * dy + sin_lat * dz
    return (e, n, u)


def geodetic_to_enu(
    lat_deg: float,
    lon_deg: float,
    alt_m: float,
    ref_lat_deg: float,
    ref_lon_deg: float,
    ref_alt_m: float,
) -> Tuple[float, float, float]:
    """Convenience: geodetic → ECEF → ENU in one call."""
    ecef = geodetic_to_ecef(lat_deg, lon_deg, alt_m)
    return ecef_to_enu(*ecef, ref_lat_deg, ref_lon_deg, ref_alt_m)


def enu_to_geodetic(
    e: float,
    n: float,
    u: float,
    ref_lat_deg: float,
    ref_lon_deg: float,
    ref_alt_m: float,
) -> Tuple[float, float, float]:
    """
    Convert local ENU offsets back to geodetic (lat_deg, lon_deg, alt_m).

    Uses the inverse of the ENU rotation to get ECEF, then iterative
    Bowring method for ECEF → geodetic.
    """
    lat = math.radians(ref_lat_deg)
    lon = math.radians(ref_lon_deg)
    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    sin_lon = math.sin(lon)
    cos_lon = math.cos(lon)

    # Inverse rotation: ENU → ECEF delta
    dx = -sin_lon * e - sin_lat * cos_lon * n + cos_lat * cos_lon * u
    dy = cos_lon * e - sin_lat * sin_lon * n + cos_lat * sin_lon * u
    dz = cos_lat * n + sin_lat * u

    # Add reference ECEF
    rx, ry, rz = geodetic_to_ecef(ref_lat_deg, ref_lon_deg, ref_alt_m)
    x, y, z = rx + dx, ry + dy, rz + dz

    return ecef_to_geodetic(x, y, z)


def ecef_to_geodetic(
    x: float, y: float, z: float
) -> Tuple[float, float, float]:
    """
    Convert ECEF to geodetic coordinates using Bowring's iterative method.

    Returns (lat_deg, lon_deg, alt_m).
    """
    lon = math.atan2(y, x)
    p = math.sqrt(x ** 2 + y ** 2)

    # Initial estimate using Bowring's formula
    theta = math.atan2(z * WGS84_A, p * WGS84_B)
    lat = math.atan2(
        z + WGS84_E2 / (1 - WGS84_E2) * WGS84_B * math.sin(theta) ** 3,
        p - WGS84_E2 * WGS84_A * math.cos(theta) ** 3,
    )

    # One Newton-Raphson iteration is usually sufficient
    for _ in range(5):
        sin_lat = math.sin(lat)
        N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat ** 2)
        lat = math.atan2(z + WGS84_E2 * N * sin_lat, p)

    sin_lat = math.sin(lat)
    cos_lat = math.cos(lat)
    N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat ** 2)

    if abs(cos_lat) > 1e-10:
        alt = p / cos_lat - N
    else:
        alt = abs(z) - WGS84_B

    return (math.degrees(lat), math.degrees(lon), alt)


# ═══════════════════════════════════════════════════════════════════════════════
# Atmosphere model – US Standard 1976 (exponential approximation)
# ═══════════════════════════════════════════════════════════════════════════════

# Sea-level air density (kg/m³)
RHO_0: float = 1.225

# Scale height (m) – the altitude at which density drops by factor e.
# ~8500 m is a good single-layer approximation for the troposphere /
# lower stratosphere where most re-entry heating occurs.
SCALE_HEIGHT: float = 8_500.0


def air_density(altitude_m: float) -> float:
    """
    Compute approximate atmospheric density at *altitude_m* metres above
    sea level using the exponential atmosphere model.

    Returns density in kg/m³.  Returns 0 for altitudes above ~100 km
    (effectively vacuum – negligible drag).
    """
    if altitude_m < 0:
        return RHO_0  # at or below sea level
    if altitude_m > 100_000:
        return 0.0  # Kármán line – treat as vacuum
    return RHO_0 * math.exp(-altitude_m / SCALE_HEIGHT)


# ═══════════════════════════════════════════════════════════════════════════════
# Drag & gravity accelerations
# ═══════════════════════════════════════════════════════════════════════════════


def drag_acceleration(
    vx: float,
    vy: float,
    vz: float,
    altitude_m: float,
    ballistic_coeff: float,
) -> Tuple[float, float, float]:
    """
    Compute aerodynamic drag deceleration opposing the velocity vector.

    The drag force per unit mass is:

        a_drag = -½ · ρ(h) · |v|² / β  ·  v̂

    where β = m / (C_d · A) is the ballistic coefficient and v̂ is the
    unit velocity vector.

    Parameters
    ----------
    vx, vy, vz       : Velocity components in ENU (m/s).
    altitude_m        : Current altitude above sea level (m).
    ballistic_coeff   : β in kg/m².

    Returns
    -------
    (ax, ay, az) drag deceleration in m/s² (ENU frame).
    """
    speed = math.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    if speed < 1e-6:
        return (0.0, 0.0, 0.0)

    rho = air_density(altitude_m)
    # Drag magnitude (deceleration, always opposes motion)
    drag_mag = 0.5 * rho * speed ** 2 / ballistic_coeff

    # Decompose along the velocity unit vector (opposing direction)
    ax = -drag_mag * (vx / speed)
    ay = -drag_mag * (vy / speed)
    az = -drag_mag * (vz / speed)
    return (ax, ay, az)


def gravity_acceleration() -> Tuple[float, float, float]:
    """
    Return gravitational acceleration in ENU frame.

    In ENU, "Up" is the +z axis, so gravity is (0, 0, −g).
    """
    return (0.0, 0.0, -GRAVITY)
