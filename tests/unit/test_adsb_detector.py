"""Unit tests for ingestion/adsb_detector.py."""

import pytest

from ingestion.adsb_detector import AircraftDetector, AircraftState, EMERGENCY_SQUAWKS, heading_delta
from models import EventSource


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


def _make_state(**kwargs) -> AircraftState:
    """Factory with sensible defaults for an airborne aircraft."""
    defaults = {
        "hex": "aabbcc",
        "callsign": "TEST1",
        "lat": 32.0,
        "lon": 34.8,
        "track": 90.0,
        "alt_m": 10000,
        "on_ground": False,
    }
    defaults.update(kwargs)
    return AircraftState(**defaults)


class TestSquawkDetection:
    """Rule 1: Emergency squawk codes."""

    def test_squawk_7700_emits_event(self):
        detector = AircraftDetector(watch_hex_codes=set())
        events = detector.process_batch([_make_state(squawk="7700")])
        assert len(events) == 1
        assert "MAYDAY" in events[0].description
        assert "SQUAWK 7700" in events[0].description

    def test_squawk_7500_emits_hijack(self):
        detector = AircraftDetector(watch_hex_codes=set())
        events = detector.process_batch([_make_state(squawk="7500")])
        assert len(events) == 1
        assert "HIJACK" in events[0].description

    def test_squawk_7600_emits_lost_comm(self):
        detector = AircraftDetector(watch_hex_codes=set())
        events = detector.process_batch([_make_state(squawk="7600")])
        assert len(events) == 1
        assert "LOST COMM" in events[0].description

    def test_normal_squawk_no_event(self):
        detector = AircraftDetector(watch_hex_codes=set())
        events = detector.process_batch([_make_state(squawk="1200")])
        assert len(events) == 0

    def test_squawk_none_no_event(self):
        detector = AircraftDetector(watch_hex_codes=set())
        events = detector.process_batch([_make_state(squawk=None)])
        assert len(events) == 0

    def test_squawk_fires_even_without_position(self):
        detector = AircraftDetector(watch_hex_codes=set())
        events = detector.process_batch([
            _make_state(squawk="7700", lat=None, lon=None, track=None),
        ])
        assert len(events) == 1


class TestWatchHexDetection:
    """Rule 2: Watch hex codes."""

    def test_watched_hex_emits_event(self):
        detector = AircraftDetector(watch_hex_codes={"aabbcc"})
        events = detector.process_batch([_make_state()])
        assert len(events) == 1
        assert "aabbcc" in events[0].description

    def test_unwatched_hex_no_event(self):
        detector = AircraftDetector(watch_hex_codes={"ffffff"})
        events = detector.process_batch([_make_state()])
        assert len(events) == 0

    def test_watch_hex_case_insensitive(self):
        detector = AircraftDetector(watch_hex_codes={"aabbcc"})
        events = detector.process_batch([_make_state(hex="AABBCC")])
        assert len(events) == 1


class TestHeadingRerouteDetection:
    """Rule 3: Sudden heading change."""

    def test_first_observation_no_event(self):
        detector = AircraftDetector(watch_hex_codes=set())
        events = detector.process_batch([_make_state(track=90.0)])
        assert len(events) == 0

    def test_small_heading_change_no_event(self):
        detector = AircraftDetector(watch_hex_codes=set())
        detector.process_batch([_make_state(track=90.0)])
        events = detector.process_batch([_make_state(track=100.0)])
        assert len(events) == 0

    def test_large_heading_change_emits_event(self):
        detector = AircraftDetector(watch_hex_codes=set())
        detector.process_batch([_make_state(track=90.0)])
        events = detector.process_batch([_make_state(track=180.0)])
        assert len(events) == 1
        assert "rerouted" in events[0].description

    def test_heading_requires_position(self):
        detector = AircraftDetector(watch_hex_codes=set())
        detector.process_batch([_make_state(track=90.0)])
        events = detector.process_batch([
            _make_state(track=180.0, lat=None),
        ])
        assert len(events) == 0

    def test_custom_threshold(self):
        detector = AircraftDetector(watch_hex_codes=set(), heading_threshold_deg=10.0)
        detector.process_batch([_make_state(track=90.0)])
        events = detector.process_batch([_make_state(track=105.0)])
        assert len(events) == 1


class TestDetectorGeneral:
    """General detector behavior."""

    def test_on_ground_skipped(self):
        detector = AircraftDetector(watch_hex_codes={"aabbcc"})
        events = detector.process_batch([_make_state(on_ground=True)])
        assert len(events) == 0

    def test_squawk_takes_priority_over_watch_hex(self):
        detector = AircraftDetector(watch_hex_codes={"aabbcc"})
        events = detector.process_batch([_make_state(squawk="7700")])
        assert len(events) == 1
        assert "SQUAWK" in events[0].description

    def test_batch_multiple_anomalies(self):
        detector = AircraftDetector(watch_hex_codes={"watched1"})
        states = [
            _make_state(hex="watched1"),
            _make_state(hex="normal1", squawk="7700"),
            _make_state(hex="normal2"),
        ]
        events = detector.process_batch(states)
        assert len(events) == 2

    def test_event_source_is_adsb(self):
        detector = AircraftDetector(watch_hex_codes=set())
        events = detector.process_batch([_make_state(squawk="7700")])
        assert events[0].source == EventSource.ADSB

    def test_heading_cache_pruned_at_limit(self):
        detector = AircraftDetector(watch_hex_codes=set())
        detector._prev_headings = {f"hex{i}": 0.0 for i in range(50_001)}
        detector.process_batch([_make_state()])
        assert len(detector._prev_headings) <= 50_001
