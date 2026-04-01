"""Unit tests for trajectory/physics.py."""

import math

import pytest

from trajectory.physics import (
    GRAVITY,
    RHO_0,
    SCALE_HEIGHT,
    WGS84_B,
    air_density,
    drag_acceleration,
    ecef_to_geodetic,
    enu_to_geodetic,
    geodetic_to_ecef,
    geodetic_to_enu,
    gravity_acceleration,
)


class TestGeodeticToECEF:
    """Tests for geodetic_to_ecef conversion."""

    def test_equator_prime_meridian(self):
        x, y, z = geodetic_to_ecef(0.0, 0.0, 0.0)
        assert x == pytest.approx(6_378_137.0, rel=1e-6)
        assert y == pytest.approx(0.0, abs=1.0)
        assert z == pytest.approx(0.0, abs=1.0)

    def test_north_pole(self):
        x, y, z = geodetic_to_ecef(90.0, 0.0, 0.0)
        assert x == pytest.approx(0.0, abs=1.0)
        assert y == pytest.approx(0.0, abs=1.0)
        assert z == pytest.approx(WGS84_B, rel=1e-6)


class TestRoundTripECEF:
    """Round-trip: geodetic -> ECEF -> geodetic recovers original coords."""

    @pytest.mark.parametrize(
        "lat, lon, alt",
        [
            (0.0, 0.0, 0.0),
            (45.0, 90.0, 1000.0),
            (-33.8688, 151.2093, 58.0),  # Sydney
            (90.0, 0.0, 0.0),            # North pole
            (30.0, -95.0, 50000.0),      # Houston at altitude
        ],
    )
    def test_round_trip(self, lat, lon, alt):
        x, y, z = geodetic_to_ecef(lat, lon, alt)
        lat2, lon2, alt2 = ecef_to_geodetic(x, y, z)
        assert lat2 == pytest.approx(lat, abs=1e-6)
        assert lon2 == pytest.approx(lon, abs=1e-6)
        assert alt2 == pytest.approx(alt, abs=0.01)


class TestRoundTripENU:
    """Round-trip: geodetic -> ENU -> geodetic recovers original coords."""

    @pytest.mark.parametrize(
        "lat, lon, alt",
        [
            (30.1, -95.1, 10000.0),
            (30.5, -94.5, 40000.0),
        ],
    )
    def test_round_trip(self, lat, lon, alt):
        ref_lat, ref_lon, ref_alt = 30.0, -95.0, 0.0
        e, n, u = geodetic_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt)
        lat2, lon2, alt2 = enu_to_geodetic(e, n, u, ref_lat, ref_lon, ref_alt)
        assert lat2 == pytest.approx(lat, abs=1e-5)
        assert lon2 == pytest.approx(lon, abs=1e-5)
        assert alt2 == pytest.approx(alt, abs=0.1)


class TestAirDensity:
    """Tests for air_density()."""

    def test_sea_level(self):
        assert air_density(0.0) == RHO_0

    def test_decreases_with_altitude(self):
        rho_1k = air_density(1000.0)
        rho_10k = air_density(10000.0)
        assert rho_1k < RHO_0
        assert rho_10k < rho_1k

    def test_below_sea_level_returns_rho0(self):
        assert air_density(-100.0) == RHO_0

    def test_above_karman_line_returns_zero(self):
        assert air_density(100_001.0) == 0.0

    def test_at_scale_height(self):
        expected = RHO_0 / math.e
        assert air_density(SCALE_HEIGHT) == pytest.approx(expected, rel=1e-10)


class TestDragAcceleration:
    """Tests for drag_acceleration()."""

    def test_zero_velocity_no_drag(self):
        ax, ay, az = drag_acceleration(0.0, 0.0, 0.0, 10000.0, 50.0)
        assert ax == 0.0
        assert ay == 0.0
        assert az == 0.0

    def test_drag_opposes_motion(self):
        # Moving in +east direction
        ax, ay, az = drag_acceleration(100.0, 0.0, 0.0, 10000.0, 50.0)
        assert ax < 0.0  # drag opposes +east motion
        assert ay == pytest.approx(0.0, abs=1e-10)
        assert az == pytest.approx(0.0, abs=1e-10)

    def test_higher_speed_more_drag(self):
        ax_slow, _, _ = drag_acceleration(100.0, 0.0, 0.0, 10000.0, 50.0)
        ax_fast, _, _ = drag_acceleration(200.0, 0.0, 0.0, 10000.0, 50.0)
        assert abs(ax_fast) > abs(ax_slow)


class TestGravityAcceleration:
    """Tests for gravity_acceleration()."""

    def test_gravity_in_enu(self):
        gx, gy, gz = gravity_acceleration()
        assert gx == 0.0
        assert gy == 0.0
        assert gz == -GRAVITY
