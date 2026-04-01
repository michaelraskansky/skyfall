"""
Output Formatter
=================

Takes a ``CorrelatedEvent`` and produces a clean, human-readable JSON
payload suitable for webhook delivery (Slack, Discord, PagerDuty, etc.).
"""

from __future__ import annotations

from datetime import timezone

from models import CorrelatedEvent


def format_alert_payload(event: CorrelatedEvent) -> dict:
    """
    Build a structured alert payload from a correlated event.

    The returned dict is JSON-serialisable and contains everything an
    on-call responder needs at a glance:

      - severity & classification
      - location
      - LLM summary
      - list of contributing sensor sources with excerpts
      - timestamp
    """
    contributing_summaries = []
    for raw in event.contributing_events:
        contributing_summaries.append(
            {
                "source": raw.source.value,
                "event_id": raw.event_id,
                "timestamp": raw.timestamp.isoformat(),
                "description": raw.description[:300],
                "coordinates": (
                    {"lat": raw.latitude, "lon": raw.longitude}
                    if raw.latitude is not None
                    else None
                ),
            }
        )

    llm_block = None
    if event.llm_analysis:
        llm_block = {
            "is_valid_anomaly": event.llm_analysis.is_valid_anomaly,
            "approximate_origin": event.llm_analysis.approximate_origin,
            "trajectory_or_blast_radius": event.llm_analysis.debris_trajectory_or_blast_radius,
            "event_classification": event.llm_analysis.event_classification,
            "confidence_score": event.llm_analysis.confidence_score,
        }

    return {
        "alert": {
            "correlation_id": event.correlation_id,
            "severity": event.severity.value,
            "classification": event.classification.value,
            "timestamp_utc": event.timestamp.astimezone(timezone.utc).isoformat(),
            "coordinates": (
                {"lat": event.latitude, "lon": event.longitude}
                if event.latitude is not None
                else None
            ),
            "summary": event.summary,
            "corroborating_sources": event.corroborating_sources,
            "llm_analysis": llm_block,
            "contributing_events": contributing_summaries,
        }
    }
