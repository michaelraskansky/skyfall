"""Integration tests for the webhook alerter output loop."""

import asyncio
from datetime import datetime, timezone
from unittest.mock import patch

import httpx
import pytest
import respx

from models import (
    CorrelatedEvent,
    EventClassification,
    EventSeverity,
    EventSource,
    LLMParsedEvent,
    RawEvent,
)
from output.alerter import alert_loop


def _make_critical_event() -> CorrelatedEvent:
    raw = RawEvent(
        event_id="test_raw",
        source=EventSource.FIRMS,
        latitude=29.76,
        longitude=-95.37,
        description="Thermal anomaly",
        timestamp=datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc),
    )
    llm = LLMParsedEvent(
        is_valid_anomaly=True,
        approximate_origin="Houston",
        debris_trajectory_or_blast_radius="unknown",
        event_classification="industrial_accident",
        confidence_score=9,
    )
    return CorrelatedEvent(
        correlation_id="corr_test",
        severity=EventSeverity.CRITICAL,
        classification=EventClassification.INDUSTRIAL_ACCIDENT,
        latitude=29.76,
        longitude=-95.37,
        contributing_events=[raw],
        llm_analysis=llm,
        summary="CRITICAL alert test",
        corroborating_sources=["firms", "adsb"],
        timestamp=datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc),
    )


@pytest.mark.asyncio
class TestAlerter:
    """Tests for the alert_loop webhook delivery."""

    @respx.mock
    async def test_slack_webhook_called(self):
        slack_url = "https://hooks.slack.com/services/test"
        slack_route = respx.post(slack_url).mock(
            return_value=httpx.Response(200)
        )

        with patch("output.alerter.settings") as mock_settings:
            mock_settings.slack_webhook_url = slack_url
            mock_settings.discord_webhook_url = ""

            queue: asyncio.Queue[CorrelatedEvent] = asyncio.Queue()
            await queue.put(_make_critical_event())

            task = asyncio.create_task(alert_loop(queue))
            await asyncio.sleep(0.5)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert slack_route.called

    @respx.mock
    async def test_discord_webhook_called(self):
        discord_url = "https://discord.com/api/webhooks/test"
        discord_route = respx.post(discord_url).mock(
            return_value=httpx.Response(200)
        )

        with patch("output.alerter.settings") as mock_settings:
            mock_settings.slack_webhook_url = ""
            mock_settings.discord_webhook_url = discord_url

            queue: asyncio.Queue[CorrelatedEvent] = asyncio.Queue()
            await queue.put(_make_critical_event())

            task = asyncio.create_task(alert_loop(queue))
            await asyncio.sleep(0.5)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert discord_route.called

    @respx.mock
    async def test_non_critical_event_skipped(self):
        slack_url = "https://hooks.slack.com/services/test"
        slack_route = respx.post(slack_url).mock(
            return_value=httpx.Response(200)
        )

        with patch("output.alerter.settings") as mock_settings:
            mock_settings.slack_webhook_url = slack_url
            mock_settings.discord_webhook_url = ""

            event = _make_critical_event()
            event.severity = EventSeverity.HIGH  # Not CRITICAL

            queue: asyncio.Queue[CorrelatedEvent] = asyncio.Queue()
            await queue.put(event)

            task = asyncio.create_task(alert_loop(queue))
            await asyncio.sleep(0.5)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        assert not slack_route.called
