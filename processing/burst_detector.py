"""
Temporal Burst Detector
========================

Detects regional-scale events by identifying bursts of activity across
multiple sensor sources within a short time window, regardless of
geographic location.

Complements the geohash-based point correlation engine. While the
correlation engine matches events in the same ~20km cell, the burst
detector catches events like missile barrages that generate signals
across thousands of kilometers (launches, aircraft rerouting, sirens,
social media reports).

Threshold: 5+ events from 3+ distinct sources within 5 minutes.
"""

from __future__ import annotations

import time
from collections import deque

from models import (
    CorrelatedEvent,
    EventClassification,
    EventSeverity,
    RawEvent,
)


class BurstDetector:
    """
    Sliding-window burst detector.

    Call ``check(event)`` on every ingested event. Returns a
    CorrelatedEvent when the burst threshold is met, None otherwise.
    """

    def __init__(
        self,
        window_sec: float = 300.0,
        min_events: int = 5,
        min_sources: int = 3,
        cooldown_sec: float = 300.0,
    ) -> None:
        self._window_sec = window_sec
        self._min_events = min_events
        self._min_sources = min_sources
        self._cooldown_sec = cooldown_sec

        self._events: deque[tuple[float, str, RawEvent]] = deque()
        self._last_burst_time: float = 0.0

    def check(self, event: RawEvent) -> CorrelatedEvent | None:
        """
        Record an event and check if a burst threshold is met.

        Returns a CRITICAL CorrelatedEvent bundling all window events
        if the threshold is reached and cooldown has elapsed.
        """
        now = time.time()

        # Append new event
        self._events.append((now, event.source.value, event))

        # Prune events outside the window
        cutoff = now - self._window_sec
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

        # Check thresholds
        if len(self._events) < self._min_events:
            return None

        sources = {source for _, source, _ in self._events}
        if len(sources) < self._min_sources:
            return None

        # Check cooldown
        if now - self._last_burst_time < self._cooldown_sec:
            return None

        # Burst detected — fire alert
        self._last_burst_time = now

        contributing = [ev for _, _, ev in self._events]

        # Pick best coordinates from contributing events
        lat, lon = None, None
        for ev in contributing:
            if ev.latitude is not None and ev.longitude is not None:
                lat, lon = ev.latitude, ev.longitude
                break

        return CorrelatedEvent(
            severity=EventSeverity.CRITICAL,
            classification=EventClassification.REGIONAL_EVENT,
            latitude=lat,
            longitude=lon,
            contributing_events=contributing,
            summary=(
                f"REGIONAL EVENT: {len(contributing)} events from "
                f"{len(sources)} sources in {int(self._window_sec / 60)} min: "
                f"{', '.join(sorted(sources))}"
            ),
            corroborating_sources=sorted(sources),
        )
