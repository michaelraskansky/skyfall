"""
Webhook Alerter
================

Consumes ``CorrelatedEvent`` objects from the output queue and delivers
formatted alert payloads to Slack and/or Discord webhooks.

Both platforms accept incoming-webhook POST requests with a JSON body.
We use ``httpx.AsyncClient`` for non-blocking delivery with automatic
retries via ``tenacity``.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

if TYPE_CHECKING:
    from asyncio import Queue

from config import settings
from models import CorrelatedEvent, EventSeverity
from output.formatter import format_alert_payload

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Webhook delivery helpers
# ═══════════════════════════════════════════════════════════════════════════════


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def _post_slack(client: httpx.AsyncClient, payload: dict) -> None:
    """
    POST a Slack incoming-webhook message.

    Slack expects:  {"text": "...", "blocks": [...]}
    We send a simple ``text`` field with the full JSON payload as a
    code block so it's easy to read in the channel.
    """
    url = settings.slack_webhook_url
    if not url:
        return

    summary = payload["alert"]["summary"]
    severity = payload["alert"]["severity"]

    slack_body = {
        "text": f":rotating_light: *{severity} ALERT* :rotating_light:\n{summary}",
        "blocks": [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{severity}: {payload['alert']['classification']}",
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": summary,
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*Correlation ID:* `{payload['alert']['correlation_id']}`\n"
                        f"*Sources:* {', '.join(payload['alert']['corroborating_sources'])}\n"
                        f"*Coordinates:* {payload['alert'].get('coordinates', 'N/A')}\n"
                        f"*Timestamp:* {payload['alert']['timestamp_utc']}"
                    ),
                },
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"```{json.dumps(payload, indent=2, default=str)[:2900]}```",
                },
            },
        ],
    }

    resp = await client.post(url, json=slack_body)
    resp.raise_for_status()
    logger.info("Slack alert sent for %s", payload["alert"]["correlation_id"])


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
async def _post_discord(client: httpx.AsyncClient, payload: dict) -> None:
    """
    POST a Discord incoming-webhook message.

    Discord expects:  {"content": "...", "embeds": [...]}
    """
    url = settings.discord_webhook_url
    if not url:
        return

    summary = payload["alert"]["summary"]
    severity = payload["alert"]["severity"]

    discord_body = {
        "content": f"**{severity} ALERT**",
        "embeds": [
            {
                "title": f"{severity}: {payload['alert']['classification']}",
                "description": summary,
                "color": 0xFF0000 if severity == "CRITICAL" else 0xFFA500,
                "fields": [
                    {
                        "name": "Correlation ID",
                        "value": f"`{payload['alert']['correlation_id']}`",
                        "inline": True,
                    },
                    {
                        "name": "Sources",
                        "value": ", ".join(payload["alert"]["corroborating_sources"]),
                        "inline": True,
                    },
                    {
                        "name": "Coordinates",
                        "value": str(payload["alert"].get("coordinates", "N/A")),
                        "inline": True,
                    },
                    {
                        "name": "Full Payload",
                        "value": f"```json\n{json.dumps(payload, indent=2, default=str)[:900]}\n```",
                    },
                ],
            }
        ],
    }

    resp = await client.post(url, json=discord_body)
    resp.raise_for_status()
    logger.info("Discord alert sent for %s", payload["alert"]["correlation_id"])


# ═══════════════════════════════════════════════════════════════════════════════
# Main alerting loop
# ═══════════════════════════════════════════════════════════════════════════════


async def alert_loop(alert_queue: Queue[CorrelatedEvent]) -> None:
    """
    Long-running coroutine that pulls CRITICAL events from *alert_queue*,
    formats them, and pushes to all configured webhooks.
    """
    logger.info("Alerter loop starting.")

    async with httpx.AsyncClient(timeout=15.0) as client:
        while True:
            event = await alert_queue.get()

            if event.severity != EventSeverity.CRITICAL:
                # Safety guard – only CRITICAL events should reach here,
                # but skip gracefully if something else slips through.
                alert_queue.task_done()
                continue

            payload = format_alert_payload(event)

            # Fire Slack and Discord webhooks concurrently.
            results = await asyncio.gather(
                _post_slack(client, payload),
                _post_discord(client, payload),
                return_exceptions=True,
            )

            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    target = "Slack" if i == 0 else "Discord"
                    logger.error(
                        "%s webhook failed for %s: %s",
                        target,
                        event.correlation_id,
                        result,
                    )

            alert_queue.task_done()
