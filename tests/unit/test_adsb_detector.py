"""Unit tests for ingestion/adsb_detector.py."""

import pytest

from ingestion.adsb_detector import AircraftState, heading_delta


class TestAircraftState:
    """Tests for the AircraftState Pydantic model."""

    def test_minimal_valid(self):
        state = AircraftState(hex="abc123")
        assert state.hex == "abc123"
        assert state.lat is None
        assert state.lon is None
        assert state.track is None
        assert state.squawk is None
        assert state.callsign == ""
        assert state.on_ground is False

    def test_full_fields(self):
        state = AircraftState(
            hex="abc123",
            callsign="SVA123",
            lat=32.05,
            lon=34.78,
            track=180.0,
            alt_m=10000,
            velocity_m_s=250.0,
            origin_country="Saudi Arabia",
            on_ground=False,
            squawk="1200",
        )
        assert state.callsign == "SVA123"
        assert state.squawk == "1200"
        assert state.alt_m == 10000

    def test_hex_required(self):
        with pytest.raises(Exception):
            AircraftState()

    def test_hex_lowercased(self):
        state = AircraftState(hex="ABC123")
        assert state.hex == "abc123"


class TestHeadingDelta:
    """Tests for the heading_delta utility function."""

    def test_same_heading(self):
        assert heading_delta(90.0, 90.0) == pytest.approx(0.0)

    def test_simple_delta(self):
        assert heading_delta(10.0, 55.0) == pytest.approx(45.0)

    def test_wraparound(self):
        assert heading_delta(350.0, 10.0) == pytest.approx(20.0)

    def test_opposite(self):
        assert heading_delta(0.0, 180.0) == pytest.approx(180.0)

    def test_symmetry(self):
        assert heading_delta(30.0, 300.0) == heading_delta(300.0, 30.0)
