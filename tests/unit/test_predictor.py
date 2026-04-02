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


def _make_ground_launch_observations(count=3, interval_sec=15.0):
    """Generate synthetic observations for a ground-launched vehicle ascending."""
    base_time = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
    obs = []
    for i in range(count):
        obs.append(
            SensorObservation(
                timestamp=base_time + timedelta(seconds=i * interval_sec),
                latitude=30.0 + i * 0.005,
                longitude=-95.0 + i * 0.002,
                altitude_m=i * 5000.0,  # 0, 5000, 10000, ...
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

    def test_high_altitude_detects_ballistic_phase(self):
        result = self._predict(start_alt=50000.0)
        assert result.flight_phase_detected == "ballistic"


class TestBoostPhase:
    """Tests for the two-phase boost → ballistic state machine."""

    def _predict_ground_launch(self, **request_kwargs):
        obs = _make_ground_launch_observations()
        # Conservative defaults: short burn, low thrust → stays sub-orbital
        # and returns to ground well within 3600s propagation limit.
        defaults = {
            "burn_time_seconds": 15.0,
            "thrust_to_mass_ratio": 12.0,
            "pitch_angle_deg": 75.0,
            "propagation_dt": 0.5,
        }
        defaults.update(request_kwargs)
        request = TrajectoryRequest(
            object_id="BOOST-001",
            observations=obs,
            ballistic_coefficient=30.0,
            **defaults,
        )
        predictor = DebrisTrajectoryPredictor(ballistic_coefficient=30.0)
        return predictor.predict(request)

    def test_ground_launch_detects_boost_phase(self):
        """Initial altitude 0m should auto-detect as boost phase."""
        result = self._predict_ground_launch()
        assert result.flight_phase_detected == "boost"

    def test_ground_launch_does_not_terminate_immediately(self):
        """The ground-launch bug: alt=0 at t=0 must NOT terminate the prediction."""
        result = self._predict_ground_launch()
        assert result.seconds_until_impact > 0
        assert result.terminal_velocity_m_s > 0

    def test_ground_launch_gains_altitude(self):
        """During boost, the vehicle should ascend — trajectory must show altitude gain."""
        result = self._predict_ground_launch()
        # Find the peak altitude in trajectory points
        alts = [wp["alt_m"] for wp in result.trajectory_points]
        peak_alt = max(alts)
        # With 15s burn at 12 m/s² thrust, the vehicle should reach
        # meaningful altitude before ballistic descent
        assert peak_alt > 1_000, f"Peak altitude {peak_alt}m is too low for a boost phase"

    def test_ground_launch_eventually_impacts(self):
        """After boost ends, the vehicle must descend and impact ground."""
        result = self._predict_ground_launch()
        assert result.impact_altitude_m == pytest.approx(0.0, abs=500.0)

    def test_custom_burn_time_override(self):
        """Longer burn time should produce a longer total flight."""
        short_burn = self._predict_ground_launch(burn_time_seconds=5.0)
        long_burn = self._predict_ground_launch(burn_time_seconds=20.0)
        assert long_burn.seconds_until_impact > short_burn.seconds_until_impact

    def test_custom_thrust_override(self):
        """Higher thrust should produce a longer total flight time."""
        low_thrust = self._predict_ground_launch(thrust_to_mass_ratio=11.0)
        high_thrust = self._predict_ground_launch(thrust_to_mass_ratio=13.0)
        # Higher thrust imparts more energy → longer flight time to impact
        assert high_thrust.seconds_until_impact > low_thrust.seconds_until_impact

    def test_zero_velocity_safeguard(self):
        """Thrust decomposition must not crash when velocity is zero."""
        # At t=0 the vehicle has zero velocity — during clear-off,
        # thrust should be pure vertical without raising ZeroDivisionError.
        thrust_enu = DebrisTrajectoryPredictor._compute_thrust_enu(
            ve=0.0, vn=0.0, vu=0.0, thrust=25.0, pitch_kick_deg=2.0, elapsed=0.0,
        )
        assert len(thrust_enu) == 3
        # During clear-off (elapsed=0), thrust is pure vertical
        assert thrust_enu[0] == 0.0
        assert thrust_enu[1] == 0.0
        assert thrust_enu[2] == 25.0

    def test_gravity_turn_follows_velocity(self):
        """After clear-off, thrust should align with velocity vector."""
        # Vehicle moving northeast and up at >50 m/s — gravity turn mode
        thrust_enu = DebrisTrajectoryPredictor._compute_thrust_enu(
            ve=100.0, vn=100.0, vu=200.0, thrust=25.0,
            pitch_kick_deg=2.0, elapsed=10.0,
        )
        speed = (100**2 + 100**2 + 200**2) ** 0.5
        # Thrust should be proportional to velocity direction
        assert thrust_enu[0] == pytest.approx(25.0 * 100.0 / speed, abs=0.1)
        assert thrust_enu[1] == pytest.approx(25.0 * 100.0 / speed, abs=0.1)
        assert thrust_enu[2] == pytest.approx(25.0 * 200.0 / speed, abs=0.1)

    def test_clear_off_is_vertical(self):
        """During first 5 seconds, thrust should be pure vertical regardless of velocity."""
        thrust_enu = DebrisTrajectoryPredictor._compute_thrust_enu(
            ve=10.0, vn=5.0, vu=30.0, thrust=25.0,
            pitch_kick_deg=2.0, elapsed=3.0,
        )
        # Speed is ~33 m/s (< 50), still in clear-off
        assert thrust_enu[0] == 0.0
        assert thrust_enu[1] == 0.0
        assert thrust_enu[2] == 25.0
