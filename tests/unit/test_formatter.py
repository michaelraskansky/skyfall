"""Unit tests for output/formatter.py format_alert_payload()."""

from datetime import datetime, timezone

from models import (
    CorrelatedEvent,
    EventClassification,
    EventSeverity,
    EventSource,
    LLMParsedEvent,
    RawEvent,
)
from output.formatter import format_alert_payload


def _make_correlated_event(
    *,
    lat=30.0,
    lon=-95.0,
    with_llm=False,
    with_contributing=True,
) -> CorrelatedEvent:
    """Build a CorrelatedEvent for testing."""
    contributing = []
    if with_contributing:
        contributing = [
            RawEvent(
                source=EventSource.FIRMS,
                latitude=30.0,
                longitude=-95.0,
                description="Thermal anomaly detected by FIRMS",
            ),
            RawEvent(
                source=EventSource.ADSB,
                latitude=30.1,
                longitude=-95.1,
                description="ADS-B signal lost",
            ),
        ]

    llm = None
    if with_llm:
        llm = LLMParsedEvent(
            is_valid_anomaly=True,
            approximate_origin="ISS debris",
            debris_trajectory_or_blast_radius="SE trajectory, 50km radius",
            event_classification="debris_reentry",
            confidence_score=8,
        )

    return CorrelatedEvent(
        severity=EventSeverity.HIGH,
        classification=EventClassification.DEBRIS_REENTRY,
        latitude=lat,
        longitude=lon,
        contributing_events=contributing,
        llm_analysis=llm,
        summary="Possible debris reentry over Houston",
        corroborating_sources=["FIRMS", "ADS-B"],
    )


class TestFormatAlertPayload:
    """Tests for format_alert_payload()."""

    def test_returns_dict_with_alert_key(self):
        event = _make_correlated_event()
        result = format_alert_payload(event)
        assert isinstance(result, dict)
        assert "alert" in result

    def test_alert_contains_required_fields(self):
        event = _make_correlated_event()
        alert = format_alert_payload(event)["alert"]
        required = [
            "correlation_id",
            "severity",
            "classification",
            "timestamp_utc",
            "coordinates",
            "summary",
            "corroborating_sources",
        ]
        for field in required:
            assert field in alert, f"Missing required field: {field}"

    def test_coordinates_present_when_lat_lon_set(self):
        event = _make_correlated_event(lat=30.0, lon=-95.0)
        alert = format_alert_payload(event)["alert"]
        assert alert["coordinates"] is not None
        assert alert["coordinates"]["lat"] == 30.0
        assert alert["coordinates"]["lon"] == -95.0

    def test_no_coordinates_when_lat_lon_none(self):
        event = _make_correlated_event(lat=None, lon=None)
        alert = format_alert_payload(event)["alert"]
        assert alert["coordinates"] is None

    def test_llm_analysis_present_when_provided(self):
        event = _make_correlated_event(with_llm=True)
        alert = format_alert_payload(event)["alert"]
        assert alert["llm_analysis"] is not None
        assert alert["llm_analysis"]["is_valid_anomaly"] is True
        assert alert["llm_analysis"]["confidence_score"] == 8

    def test_no_llm_analysis_when_not_provided(self):
        event = _make_correlated_event(with_llm=False)
        alert = format_alert_payload(event)["alert"]
        assert alert["llm_analysis"] is None

    def test_contributing_events_included(self):
        event = _make_correlated_event(with_contributing=True)
        alert = format_alert_payload(event)["alert"]
        assert len(alert["contributing_events"]) == 2
        ce = alert["contributing_events"][0]
        assert "source" in ce
        assert "event_id" in ce
        assert "timestamp" in ce
        assert "description" in ce
        assert "coordinates" in ce

    def test_contributing_event_source_is_string_value(self):
        event = _make_correlated_event(with_contributing=True)
        alert = format_alert_payload(event)["alert"]
        assert alert["contributing_events"][0]["source"] == "firms"
        assert alert["contributing_events"][1]["source"] == "adsb"
