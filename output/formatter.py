"""
Output Formatter
=================

Takes a ``CorrelatedEvent`` and produces a clean, human-readable JSON
payload suitable for webhook delivery (Slack, Discord, PagerDuty, etc.).
"""

from __future__ import annotations

import math
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

    impact_block = None
    if event.impact_prediction:
        ip = event.impact_prediction
        # Compute 95% confidence ellipse semi-axes from the 2x2 horizontal covariance
        cov_ee = ip.covariance_position_enu[0][0]
        cov_en = ip.covariance_position_enu[0][1]
        cov_nn = ip.covariance_position_enu[1][1]
        trace = cov_ee + cov_nn
        det = cov_ee * cov_nn - cov_en * cov_en
        discriminant = max((trace / 2) ** 2 - det, 0)
        lam1 = trace / 2 + math.sqrt(discriminant)
        lam2 = trace / 2 - math.sqrt(discriminant)
        semi_major = 2.0 * math.sqrt(max(lam1, 0))  # 95% ≈ 2σ
        semi_minor = 2.0 * math.sqrt(max(lam2, 0))

        impact_block = {
            "object_id": ip.object_id,
            "impact_latitude": ip.impact_latitude,
            "impact_longitude": ip.impact_longitude,
            "time_of_impact_utc": ip.time_of_impact_utc.isoformat(),
            "seconds_until_impact": ip.seconds_until_impact,
            "terminal_velocity_m_s": ip.terminal_velocity_m_s,
            "confidence_ellipse_95pct_m": {
                "semi_major": round(semi_major, 1),
                "semi_minor": round(semi_minor, 1),
            },
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
            "impact_prediction": impact_block,
        }
    }
