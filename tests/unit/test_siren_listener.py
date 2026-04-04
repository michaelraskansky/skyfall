"""Unit tests for ingestion/siren_listener.py and siren-trajectory correlation."""

import pytest

from ingestion.siren_listener import (
    ALERT_CATEGORIES,
    CATEGORY_LABELS,
    DRILL_CATEGORIES,
    SirenEvent,
    WATCH_ZONES,
    ZONE_COORDINATES,
    _TITLE_EVENT_ENDED,
)


class TestSirenEvent:
    """Tests for the SirenEvent model."""

    def test_active_alert(self):
        event = SirenEvent(
            alert_id="123",
            title="בדקות הקרובות צפויות להתקבל התרעות באזורך",
            zones=["רמת השרון", "תל אביב"],
            matched_watch_zones=["רמת השרון"],
            is_active=True,
        )
        assert event.is_active is True
        assert event.matched_watch_zones == ["רמת השרון"]

    def test_event_ended(self):
        event = SirenEvent(
            alert_id="456",
            title=_TITLE_EVENT_ENDED,
            zones=["רמת השרון"],
            matched_watch_zones=["רמת השרון"],
            is_active=False,
        )
        assert event.is_active is False

    def test_title_detection_logic(self):
        """The event ended title must match exactly."""
        assert _TITLE_EVENT_ENDED == "האירוע הסתיים"


class TestWatchZones:
    """Tests for watch zone configuration."""

    def test_watch_zones_configured(self):
        assert "רמת השרון" in WATCH_ZONES
        assert "מתחם גלילות" in WATCH_ZONES
        assert "הוד השרון" in WATCH_ZONES

    def test_all_watch_zones_have_coordinates(self):
        for zone in WATCH_ZONES:
            assert zone in ZONE_COORDINATES, f"Missing coordinates for {zone}"
            lat, lon = ZONE_COORDINATES[zone]
            assert 31.0 < lat < 33.0, f"Bad latitude for {zone}: {lat}"
            assert 34.0 < lon < 36.0, f"Bad longitude for {zone}: {lon}"

    def test_zone_filtering(self):
        """Only watch zones should match, not arbitrary zone names."""
        alert_zones = ["תל אביב", "רמת השרון", "חיפה", "מתחם גלילות"]
        matched = [z for z in alert_zones if z in WATCH_ZONES]
        assert matched == ["רמת השרון", "מתחם גלילות"]

    def test_no_match_for_unrelated_zones(self):
        alert_zones = ["אשקלון", "באר שבע", "חיפה"]
        matched = [z for z in alert_zones if z in WATCH_ZONES]
        assert matched == []


class TestAlertCategories:
    """Tests for alert category filtering."""

    def test_missile_alert_is_accepted(self):
        assert "missilealert" in ALERT_CATEGORIES

    def test_uav_is_accepted(self):
        assert "uav" in ALERT_CATEGORIES

    def test_nonconventional_is_accepted(self):
        assert "nonconventional" in ALERT_CATEGORIES

    def test_drill_categories_are_rejected(self):
        assert "missilealertdrill" in DRILL_CATEGORIES
        assert "uavdrill" in DRILL_CATEGORIES
        assert "earthquakedrill1" in DRILL_CATEGORIES

    def test_no_overlap_between_alert_and_drill(self):
        assert ALERT_CATEGORIES.isdisjoint(DRILL_CATEGORIES)

    def test_earthquake_not_in_alert_categories(self):
        """Earthquakes are not trajectory-relevant."""
        assert "earthquakealert1" not in ALERT_CATEGORIES
        assert "earthquakealert2" not in ALERT_CATEGORIES

    def test_all_alert_categories_have_labels(self):
        for cat in ALERT_CATEGORIES:
            assert cat in CATEGORY_LABELS, f"Missing label for {cat}"

    def test_category_label_values(self):
        assert CATEGORY_LABELS["missilealert"] == "Missile"
        assert CATEGORY_LABELS["uav"] == "UAV/Drone"


class TestHaversine:
    """Tests for the haversine distance function used in trajectory correlation."""

    def test_same_point_is_zero(self):
        from main import _haversine_km
        assert _haversine_km(32.15, 34.84, 32.15, 34.84) == pytest.approx(0.0, abs=0.01)

    def test_known_distance(self):
        from main import _haversine_km
        # Ramat HaSharon to Hod HaSharon is ~5 km
        dist = _haversine_km(32.1461, 34.8394, 32.1500, 34.8880)
        assert 3.0 < dist < 7.0

    def test_within_50km_threshold(self):
        from main import _haversine_km
        # Tel Aviv to Ramat HaSharon is ~10 km — within 50km
        dist = _haversine_km(32.07, 34.77, 32.15, 34.84)
        assert dist < 50.0

    def test_far_away_exceeds_threshold(self):
        from main import _haversine_km
        # Eilat to Ramat HaSharon is ~300 km
        dist = _haversine_km(29.56, 34.95, 32.15, 34.84)
        assert dist > 50.0
