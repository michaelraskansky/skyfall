"""Unit tests for Pydantic models in models.py."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from models import (
    CorrelatedEvent,
    EventClassification,
    EventSeverity,
    EventSource,
    LLMParsedEvent,
    RawEvent,
)
from trajectory.models import ImpactPrediction


class TestRawEvent:
    """Tests for RawEvent model."""

    def test_default_event_id_is_generated(self):
        event = RawEvent(source=EventSource.FIRMS)
        assert isinstance(event.event_id, str)
        assert len(event.event_id) == 12

    def test_default_event_ids_are_unique(self):
        e1 = RawEvent(source=EventSource.FIRMS)
        e2 = RawEvent(source=EventSource.FIRMS)
        assert e1.event_id != e2.event_id

    def test_all_event_sources_valid(self):
        for source in EventSource:
            event = RawEvent(source=source)
            assert event.source == source

    def test_event_source_values(self):
        assert EventSource.FIRMS.value == "firms"
        assert EventSource.ADSB.value == "adsb"
        assert EventSource.SOCIAL_MEDIA.value == "social_media"
        assert EventSource.EMERGENCY_WEBHOOK.value == "emergency_webhook"


class TestLLMParsedEvent:
    """Tests for LLMParsedEvent confidence_score bounds."""

    def test_confidence_score_min_valid(self):
        event = LLMParsedEvent(is_valid_anomaly=True, confidence_score=1)
        assert event.confidence_score == 1

    def test_confidence_score_max_valid(self):
        event = LLMParsedEvent(is_valid_anomaly=True, confidence_score=10)
        assert event.confidence_score == 10

    def test_confidence_score_zero_rejected(self):
        with pytest.raises(ValidationError):
            LLMParsedEvent(is_valid_anomaly=True, confidence_score=0)

    def test_confidence_score_eleven_rejected(self):
        with pytest.raises(ValidationError):
            LLMParsedEvent(is_valid_anomaly=True, confidence_score=11)

    def test_confidence_score_negative_rejected(self):
        with pytest.raises(ValidationError):
            LLMParsedEvent(is_valid_anomaly=True, confidence_score=-1)


class TestCorrelatedEvent:
    """Tests for CorrelatedEvent defaults and contributing_events."""

    def test_default_severity_is_low(self):
        event = CorrelatedEvent()
        assert event.severity == EventSeverity.LOW

    def test_default_classification_is_unknown(self):
        event = CorrelatedEvent()
        assert event.classification == EventClassification.UNKNOWN

    def test_contributing_events_list_works(self):
        raw1 = RawEvent(source=EventSource.FIRMS, description="fire detected")
        raw2 = RawEvent(source=EventSource.ADSB, description="aircraft anomaly")
        event = CorrelatedEvent(contributing_events=[raw1, raw2])
        assert len(event.contributing_events) == 2
        assert event.contributing_events[0].source == EventSource.FIRMS
        assert event.contributing_events[1].source == EventSource.ADSB

    def test_default_contributing_events_empty(self):
        event = CorrelatedEvent()
        assert event.contributing_events == []

    def test_impact_prediction_default_none(self):
        """CorrelatedEvent.impact_prediction defaults to None."""
        event = CorrelatedEvent()
        assert event.impact_prediction is None

    def test_accepts_impact_prediction(self):
        """CorrelatedEvent can hold an ImpactPrediction."""
        prediction = ImpactPrediction(
            object_id="99999",
            impact_latitude=35.0,
            impact_longitude=-115.0,
            impact_altitude_m=0.0,
            time_of_impact_utc=datetime(2026, 4, 1, 15, 0, 0, tzinfo=timezone.utc),
            seconds_until_impact=120.0,
            terminal_velocity_m_s=200.0,
            covariance_position_enu=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        )
        event = CorrelatedEvent(impact_prediction=prediction)
        assert event.impact_prediction is not None
        assert event.impact_prediction.object_id == "99999"
        assert event.impact_prediction.impact_latitude == 35.0
