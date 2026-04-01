"""Unit tests for trajectory/predictor.py DebrisTrajectoryPredictor."""

from datetime import datetime, timedelta, timezone

import pytest

from trajectory.models import SensorObservation, TrajectoryRequest
from trajectory.predictor import DebrisTrajectoryPredictor


def _make_observations(count=5, interval_sec=10.0, start_alt=50000.0):
    """Generate synthetic observations for a descending object."""
    base_time = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
    obs = []
    for i in range(count):
        obs.append(
            SensorObservation(
                timestamp=base_time + timedelta(seconds=i * interval_sec),
                latitude=30.0 - i * 0.01,
                longitude=-95.0,
                altitude_m=start_alt - i * 2000.0,
                noise_profile="satellite",
            )
        )
    return obs


class TestDebrisTrajectoryPredictor:
    """Tests for DebrisTrajectoryPredictor.predict()."""

    def _predict(self, ballistic_coefficient=50.0, **kwargs):
        obs = _make_observations(**kwargs)
        request = TrajectoryRequest(
            object_id="TEST-001",
            observations=obs,
            ballistic_coefficient=ballistic_coefficient,
        )
        predictor = DebrisTrajectoryPredictor(
            ballistic_coefficient=ballistic_coefficient
        )
        return predictor.predict(request)

    def test_returns_correct_object_id(self):
        result = self._predict()
        assert result.object_id == "TEST-001"

    def test_impact_altitude_near_zero(self):
        result = self._predict()
        assert result.impact_altitude_m == pytest.approx(0.0, abs=500.0)

    def test_impact_time_positive(self):
        result = self._predict()
        assert result.seconds_until_impact > 0

    def test_terminal_velocity_positive(self):
        result = self._predict()
        assert result.terminal_velocity_m_s > 0

    def test_covariance_matrix_is_3x3(self):
        result = self._predict()
        cov = result.covariance_position_enu
        assert len(cov) == 3
        for row in cov:
            assert len(row) == 3

    def test_trajectory_points_populated(self):
        result = self._predict()
        assert len(result.trajectory_points) > 0
        point = result.trajectory_points[0]
        assert "lat" in point
        assert "lon" in point
        assert "alt_m" in point

    def test_higher_ballistic_coefficient_higher_terminal_velocity(self):
        result_low = self._predict(ballistic_coefficient=30.0)
        result_high = self._predict(ballistic_coefficient=150.0)
        # Higher ballistic coefficient means less drag, so higher terminal velocity
        assert result_high.terminal_velocity_m_s > result_low.terminal_velocity_m_s
