"""Unit tests for output/formatter.py format_alert_payload()."""

from datetime import datetime, timezone

from ingestion.satcat_lookup import SatcatInfo
from models import (
    CorrelatedEvent,
    EventClassification,
    EventSeverity,
    EventSource,
    LLMParsedEvent,
    RawEvent,
)
from output.formatter import format_alert_payload
from trajectory.models import ImpactPrediction


def _make_impact_prediction() -> ImpactPrediction:
    """Build a minimal ImpactPrediction for testing."""
    return ImpactPrediction(
        object_id="99999",
        impact_latitude=35.123,
        impact_longitude=-115.456,
        impact_altitude_m=0.0,
        time_of_impact_utc=datetime(2026, 4, 1, 15, 0, 0, tzinfo=timezone.utc),
        seconds_until_impact=120.0,
        terminal_velocity_m_s=250.5,
        covariance_position_enu=[
            [40000.0, 100.0, 0.0],
            [100.0, 35000.0, 0.0],
            [0.0, 0.0, 90000.0],
        ],
    )


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

    def test_impact_prediction_included_when_present(self):
        event = _make_correlated_event()
        event.impact_prediction = _make_impact_prediction()
        alert = format_alert_payload(event)["alert"]
        ip = alert["impact_prediction"]
        assert ip is not None
        assert ip["object_id"] == "99999"
        assert ip["impact_latitude"] == 35.123
        assert ip["impact_longitude"] == -115.456
        assert ip["terminal_velocity_m_s"] == 250.5
        assert ip["seconds_until_impact"] == 120.0

    def test_impact_prediction_has_confidence_ellipse(self):
        event = _make_correlated_event()
        event.impact_prediction = _make_impact_prediction()
        alert = format_alert_payload(event)["alert"]
        ip = alert["impact_prediction"]
        assert "confidence_ellipse_95pct_m" in ip
        ellipse = ip["confidence_ellipse_95pct_m"]
        assert "semi_major" in ellipse
        assert "semi_minor" in ellipse
        assert ellipse["semi_major"] > 0
        assert ellipse["semi_minor"] > 0

    def test_no_impact_prediction_when_absent(self):
        event = _make_correlated_event()
        assert event.impact_prediction is None
        alert = format_alert_payload(event)["alert"]
        assert alert.get("impact_prediction") is None

    def test_object_info_included_when_satcat_present(self):
        event = _make_correlated_event()
        event.satcat_info = SatcatInfo(
            norad_cat_id="53493",
            object_name="STARLINK-4361",
            country="US",
            launch_date="2022-08-12",
            object_type="PAYLOAD",
            rcs_size="LARGE",
        )
        alert = format_alert_payload(event)["alert"]
        oi = alert["object_info"]
        assert oi is not None
        assert oi["object_name"] == "STARLINK-4361"
        assert oi["country"] == "US"
        assert oi["launch_date"] == "2022-08-12"
        assert oi["object_type"] == "PAYLOAD"
        assert oi["norad_cat_id"] == "53493"

    def test_no_object_info_when_satcat_absent(self):
        event = _make_correlated_event()
        assert event.satcat_info is None
        alert = format_alert_payload(event)["alert"]
        assert alert.get("object_info") is None
