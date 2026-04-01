"""
Shared Pydantic models used across every layer of the pipeline.

These models enforce schema consistency from ingestion through alerting.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ═══════════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════════


class EventSource(str, Enum):
    """Identifies which sensor produced the raw event."""
    FIRMS = "firms"
    ADSB = "adsb"
    SOCIAL_MEDIA = "social_media"
    EMERGENCY_WEBHOOK = "emergency_webhook"


class EventSeverity(str, Enum):
    """Severity after correlation triage."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class EventClassification(str, Enum):
    """High-level event type assigned by the LLM or correlation engine."""
    DEBRIS_REENTRY = "debris_reentry"
    INDUSTRIAL_ACCIDENT = "industrial_accident"
    METEOR = "meteor"
    MILITARY_ACTIVITY = "military_activity"
    UNKNOWN = "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
# Raw sensor events
# ═══════════════════════════════════════════════════════════════════════════════


class RawEvent(BaseModel):
    """Lowest-level event emitted by any ingestion sensor."""
    event_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    source: EventSource
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    raw_payload: dict = Field(default_factory=dict)
    description: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# LLM-parsed event
# ═══════════════════════════════════════════════════════════════════════════════


class LLMParsedEvent(BaseModel):
    """Structured output returned by the LLM triage step."""
    is_valid_anomaly: bool
    approximate_origin: str = ""
    debris_trajectory_or_blast_radius: str = ""
    event_classification: str = ""
    confidence_score: int = Field(ge=1, le=10)


# ═══════════════════════════════════════════════════════════════════════════════
# Correlated / elevated event
# ═══════════════════════════════════════════════════════════════════════════════


class CorrelatedEvent(BaseModel):
    """
    An event that has been cross-referenced across multiple sensors.

    When two or more independent sensors agree within the correlation window,
    severity is elevated to CRITICAL.
    """
    correlation_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    severity: EventSeverity = EventSeverity.LOW
    classification: EventClassification = EventClassification.UNKNOWN
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    contributing_events: list[RawEvent] = Field(default_factory=list)
    llm_analysis: Optional[LLMParsedEvent] = None
    summary: str = ""
    corroborating_sources: list[str] = Field(default_factory=list)
