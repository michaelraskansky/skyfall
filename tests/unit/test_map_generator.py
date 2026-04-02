"""Unit tests for visuals/map_generator.py."""

import os
from datetime import datetime, timezone

import pytest

from trajectory.models import ImpactPrediction
from visuals.map_generator import (
    _compute_corridor,
    _m_per_deg_lon,
    render_ground_track,
)


def _make_prediction(n_waypoints=10) -> ImpactPrediction:
    """Build an ImpactPrediction with a realistic trajectory."""
    waypoints = []
    for i in range(n_waypoints):
        waypoints.append({
            "lat": 37.0 - i * 0.05,
            "lon": -115.0 - i * 0.03,
            "alt_m": max(100_000.0 - i * 12_000, 0),
            "t_sec": i * 10.0,
            "speed_m_s": 2000.0 - i * 100,
        })
    return ImpactPrediction(
        object_id="TEST-001",
        impact_latitude=waypoints[-1]["lat"],
        impact_longitude=waypoints[-1]["lon"],
        impact_altitude_m=0.0,
        time_of_impact_utc=datetime(2026, 4, 1, 15, 0, 0, tzinfo=timezone.utc),
        seconds_until_impact=90.0,
        terminal_velocity_m_s=300.0,
        covariance_position_enu=[
            [40_000.0, 100.0, 0.0],
            [100.0, 35_000.0, 0.0],
            [0.0, 0.0, 90_000.0],
        ],
        trajectory_points=waypoints,
    )


class TestMPerDegLon:
    """Tests for longitude degree-to-metre conversion."""

    def test_equator(self):
        result = _m_per_deg_lon(0.0)
        assert result == pytest.approx(111_320.0, rel=0.01)

    def test_higher_latitude_is_smaller(self):
        assert _m_per_deg_lon(60.0) < _m_per_deg_lon(0.0)

    def test_pole_is_near_zero(self):
        assert _m_per_deg_lon(89.9) < 500


class TestComputeCorridor:
    """Tests for confidence corridor polygon generation."""

    def test_returns_empty_for_single_waypoint(self):
        result = _compute_corridor([{"lat": 37.0, "lon": -115.0}], 1000.0)
        assert result == []

    def test_returns_empty_for_zero_buffer(self):
        wps = [{"lat": 37.0, "lon": -115.0}, {"lat": 36.9, "lon": -115.1}]
        result = _compute_corridor(wps, 0.0)
        assert result == []

    def test_corridor_has_correct_shape(self):
        wps = [
            {"lat": 37.0, "lon": -115.0},
            {"lat": 36.9, "lon": -115.1},
            {"lat": 36.8, "lon": -115.2},
        ]
        result = _compute_corridor(wps, 5000.0)
        # Should be a closed polygon: left_edge (3) + right_edge reversed (3)
        assert len(result) == 6
        # Each point is (lon, lat)
        for point in result:
            assert len(point) == 2

    def test_corridor_encloses_track(self):
        """The corridor polygon should be wider than the track itself."""
        wps = [
            {"lat": 37.0, "lon": -115.0},
            {"lat": 36.5, "lon": -115.5},
        ]
        result = _compute_corridor(wps, 10_000.0)
        assert len(result) > 0
        lons = [p[0] for p in result]
        lats = [p[1] for p in result]
        # The corridor should extend beyond the track bounds
        assert min(lons) < -115.5
        assert max(lons) > -115.0


class TestRenderGroundTrack:
    """Tests for the full map rendering pipeline."""

    def test_produces_png_file(self):
        prediction = _make_prediction()
        path = render_ground_track(prediction)
        try:
            assert os.path.exists(path)
            assert path.endswith(".png")
            # File should have content (not empty)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_handles_minimal_waypoints(self):
        """Two waypoints should still produce a valid map."""
        prediction = _make_prediction(n_waypoints=2)
        path = render_ground_track(prediction)
        try:
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)

    def test_handles_no_waypoints(self):
        """Zero waypoints should still produce a map (impact marker only)."""
        prediction = _make_prediction(n_waypoints=2)
        prediction.trajectory_points = []
        path = render_ground_track(prediction)
        try:
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.unlink(path)
