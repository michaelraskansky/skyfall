"""
Trajectory Engine – Data Models
================================

Strictly-typed Pydantic models for the trajectory prediction pipeline.
These are independent from the ingestion-layer models in the parent
``models.py``; the orchestrator bridges them.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
# Input models
# ═══════════════════════════════════════════════════════════════════════════════


class SensorObservation(BaseModel):
    """
    A single noisy geographic observation of a debris object.

    Sources range from satellite thermal pings (moderately accurate) to
    unverified social-media reports (very noisy).  The ``noise_profile``
    field lets the caller declare which reliability tier this observation
    belongs to – the EKF will inflate measurement noise accordingly.
    """

    timestamp: datetime = Field(
        ..., description="UTC timestamp of the observation."
    )
    latitude: float = Field(
        ..., ge=-90, le=90, description="Geodetic latitude (degrees)."
    )
    longitude: float = Field(
        ..., ge=-180, le=180, description="Geodetic longitude (degrees)."
    )
    altitude_m: float = Field(
        ..., description="Estimated altitude above sea level (metres). Highly noisy."
    )
    noise_profile: str = Field(
        default="default",
        description=(
            "Reliability tier: 'satellite' (±200 m), 'thermal' (±1 km), "
            "'social_media' (±5 km), or 'default'."
        ),
    )


class TrajectoryRequest(BaseModel):
    """Top-level request: an ordered list of observations for one debris object."""

    object_id: str = Field(default="unknown", description="Opaque debris object identifier.")
    observations: list[SensorObservation] = Field(
        ..., min_length=2, description="At least 2 observations, ordered by time."
    )
    ballistic_coefficient: float = Field(
        default=50.0,
        gt=0,
        description=(
            "Ballistic coefficient β = m / (C_d · A) in kg/m².  "
            "Higher → less drag (denser/smaller object).  "
            "Typical space debris: 20–150 kg/m²."
        ),
    )
    propagation_dt: float = Field(
        default=1.0,
        gt=0,
        description="Time step (seconds) for forward trajectory propagation.",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Output models
# ═══════════════════════════════════════════════════════════════════════════════


class ImpactPrediction(BaseModel):
    """Predicted ground-impact point with uncertainty."""

    object_id: str
    impact_latitude: float = Field(..., description="Predicted impact latitude (degrees).")
    impact_longitude: float = Field(..., description="Predicted impact longitude (degrees).")
    impact_altitude_m: float = Field(
        default=0.0, description="Should be ≈ 0 (sea level)."
    )
    time_of_impact_utc: datetime = Field(
        ..., description="Estimated UTC time of ground impact."
    )
    seconds_until_impact: float = Field(
        ..., description="Seconds from the last observation to predicted impact."
    )
    terminal_velocity_m_s: float = Field(
        ..., description="Estimated speed at impact (m/s)."
    )
    covariance_position_enu: list[list[float]] = Field(
        ...,
        description=(
            "3×3 position-block of the EKF covariance matrix at impact, "
            "in the local ENU frame (metres²).  Use to plot a probability "
            "ellipse on a map."
        ),
    )
    trajectory_points: list[dict] = Field(
        default_factory=list,
        description=(
            "Sampled trajectory waypoints [{lat, lon, alt_m, t_sec}, ...] "
            "for visualisation."
        ),
    )
    filter_state_at_impact: list[float] = Field(
        default_factory=list,
        description="Final 6-element state vector [x,y,z,vx,vy,vz] in ENU (m, m/s).",
    )
