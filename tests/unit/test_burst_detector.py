"""Unit tests for the temporal burst detector."""

import time
from unittest.mock import patch

from models import EventClassification, EventSeverity, EventSource, RawEvent
from processing.burst_detector import BurstDetector


def _make_event(source: EventSource, lat=None, lon=None):
    return RawEvent(
        source=source,
        latitude=lat,
        longitude=lon,
        description=f"Test {source.value}",
        raw_payload={},
    )


class TestBurstDetector:
    def test_below_event_threshold(self):
        """4 events from 3 sources → no burst."""
        bd = BurstDetector(min_events=5, min_sources=3)
        assert bd.check(_make_event(EventSource.FIRMS)) is None
        assert bd.check(_make_event(EventSource.SPACETRACK)) is None
        assert bd.check(_make_event(EventSource.SOCIAL_MEDIA)) is None
        assert bd.check(_make_event(EventSource.FIRMS)) is None

    def test_below_source_threshold(self):
        """5 events from 2 sources → no burst."""
        bd = BurstDetector(min_events=5, min_sources=3)
        for _ in range(3):
            bd.check(_make_event(EventSource.FIRMS))
        for _ in range(2):
            result = bd.check(_make_event(EventSource.SPACETRACK))
        assert result is None

    def test_at_threshold_fires(self):
        """5 events from 3 sources → burst."""
        bd = BurstDetector(min_events=5, min_sources=3)
        bd.check(_make_event(EventSource.FIRMS))
        bd.check(_make_event(EventSource.FIRMS))
        bd.check(_make_event(EventSource.SPACETRACK))
        bd.check(_make_event(EventSource.SOCIAL_MEDIA))
        result = bd.check(_make_event(EventSource.SIREN))
        # 5 events, 4 sources — should fire
        assert result is not None
        assert result.severity == EventSeverity.CRITICAL
        assert result.classification == EventClassification.REGIONAL_EVENT
        assert len(result.contributing_events) == 5
        assert len(result.corroborating_sources) >= 3

    def test_cooldown_suppresses_second_burst(self):
        """After burst fires, suppress for cooldown period."""
        bd = BurstDetector(min_events=5, min_sources=3, cooldown_sec=300)
        # Fire first burst
        bd.check(_make_event(EventSource.FIRMS))
        bd.check(_make_event(EventSource.SPACETRACK))
        bd.check(_make_event(EventSource.SOCIAL_MEDIA))
        bd.check(_make_event(EventSource.SIREN))
        first = bd.check(_make_event(EventSource.ADSB))
        assert first is not None

        # More events within cooldown — should not fire again
        bd.check(_make_event(EventSource.FIRMS))
        bd.check(_make_event(EventSource.SPACETRACK))
        second = bd.check(_make_event(EventSource.SOCIAL_MEDIA))
        assert second is None

    def test_cooldown_expires(self):
        """After cooldown, burst can fire again."""
        bd = BurstDetector(min_events=5, min_sources=3, cooldown_sec=0.1, window_sec=300)
        # Fire first burst
        bd.check(_make_event(EventSource.FIRMS))
        bd.check(_make_event(EventSource.SPACETRACK))
        bd.check(_make_event(EventSource.SOCIAL_MEDIA))
        bd.check(_make_event(EventSource.SIREN))
        first = bd.check(_make_event(EventSource.ADSB))
        assert first is not None

        # Wait for cooldown to expire
        time.sleep(0.15)

        # Next event should fire — cooldown elapsed, threshold still met
        second = bd.check(_make_event(EventSource.FIRMS))
        assert second is not None

    def test_window_expiry(self):
        """Events older than window are pruned."""
        bd = BurstDetector(min_events=5, min_sources=3, window_sec=10)

        # Add 3 events "in the past"
        old_time = time.time() - 15
        bd._events.append((old_time, "firms", _make_event(EventSource.FIRMS)))
        bd._events.append((old_time, "spacetrack", _make_event(EventSource.SPACETRACK)))
        bd._events.append((old_time, "social_media", _make_event(EventSource.SOCIAL_MEDIA)))

        # Add 2 more now — total 5 but only 2 in window
        bd.check(_make_event(EventSource.SIREN))
        result = bd.check(_make_event(EventSource.ADSB))
        assert result is None

    def test_coordinates_from_first_event_with_location(self):
        """Burst picks first event with lat/lon."""
        bd = BurstDetector(min_events=5, min_sources=3)
        bd.check(_make_event(EventSource.FIRMS))  # no coords
        bd.check(_make_event(EventSource.SPACETRACK, lat=31.5, lon=34.47))
        bd.check(_make_event(EventSource.SOCIAL_MEDIA))
        bd.check(_make_event(EventSource.SIREN, lat=32.08, lon=34.78))
        result = bd.check(_make_event(EventSource.ADSB))
        assert result is not None
        assert result.latitude == 31.5
        assert result.longitude == 34.47

    def test_summary_contains_source_names(self):
        """Summary lists all contributing sources."""
        bd = BurstDetector(min_events=5, min_sources=3)
        bd.check(_make_event(EventSource.FIRMS))
        bd.check(_make_event(EventSource.SPACETRACK))
        bd.check(_make_event(EventSource.SOCIAL_MEDIA))
        bd.check(_make_event(EventSource.SIREN))
        result = bd.check(_make_event(EventSource.ADSB))
        assert "REGIONAL EVENT" in result.summary
        assert "5 events" in result.summary
        assert "firms" in result.summary
