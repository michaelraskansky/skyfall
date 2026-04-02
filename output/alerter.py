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

    blocks = [
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
    ]

    ip = payload["alert"].get("impact_prediction")
    if ip:
        ellipse = ip.get("confidence_ellipse_95pct_m", {})
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f":dart: *Impact Prediction*\n"
                    f"*Object:* `{ip['object_id']}`\n"
                    f"*Impact:* ({ip['impact_latitude']}, {ip['impact_longitude']})\n"
                    f"*ETA:* {ip['seconds_until_impact']:.0f}s ({ip['time_of_impact_utc']})\n"
                    f"*Terminal velocity:* {ip['terminal_velocity_m_s']:.0f} m/s\n"
                    f"*95% ellipse:* {ellipse.get('semi_major', '?')}m x {ellipse.get('semi_minor', '?')}m"
                ),
            },
        })

    oi = payload["alert"].get("object_info")
    if oi:
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f":satellite: *Object Identity*\n"
                    f"*Name:* {oi['object_name']} (NORAD {oi['norad_cat_id']})\n"
                    f"*Origin:* {oi['country']} | Launched {oi['launch_date'] or 'N/A'}\n"
                    f"*Type:* {oi['object_type'] or 'N/A'} | RCS: {oi['rcs_size'] or 'N/A'}"
                ),
            },
        })

    blocks.append({
        "type": "section",
        "text": {
            "type": "mrkdwn",
            "text": f"```{json.dumps(payload, indent=2, default=str)[:2900]}```",
        },
    })

    slack_body = {
        "text": f":rotating_light: *{severity} ALERT* :rotating_light:\n{summary}",
        "blocks": blocks,
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

    fields = [
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
    ]

    ip = payload["alert"].get("impact_prediction")
    if ip:
        ellipse = ip.get("confidence_ellipse_95pct_m", {})
        fields.append({
            "name": "Impact Prediction",
            "value": (
                f"**Object:** `{ip['object_id']}`\n"
                f"**Impact:** ({ip['impact_latitude']}, {ip['impact_longitude']})\n"
                f"**ETA:** {ip['seconds_until_impact']:.0f}s\n"
                f"**Terminal velocity:** {ip['terminal_velocity_m_s']:.0f} m/s\n"
                f"**95% ellipse:** {ellipse.get('semi_major', '?')}m x {ellipse.get('semi_minor', '?')}m"
            ),
        })

    oi = payload["alert"].get("object_info")
    if oi:
        fields.append({
            "name": "Object Identity",
            "value": (
                f"**{oi['object_name']}** (NORAD {oi['norad_cat_id']})\n"
                f"**Origin:** {oi['country']} | Launched {oi['launch_date'] or 'N/A'}\n"
                f"**Type:** {oi['object_type'] or 'N/A'} | RCS: {oi['rcs_size'] or 'N/A'}"
            ),
        })

    fields.append({
        "name": "Full Payload",
        "value": f"```json\n{json.dumps(payload, indent=2, default=str)[:900]}\n```",
    })

    discord_body = {
        "content": f"**{severity} ALERT**",
        "embeds": [
            {
                "title": f"{severity}: {payload['alert']['classification']}",
                "description": summary,
                "color": 0xFF0000 if severity == "CRITICAL" else 0xFFA500,
                "fields": fields,
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
