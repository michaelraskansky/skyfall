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
import time
from typing import TYPE_CHECKING

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

if TYPE_CHECKING:
    from asyncio import Queue

from config import settings
from models import CorrelatedEvent, EventSeverity
from output.formatter import format_alert_payload

logger = logging.getLogger(__name__)

# Rate-limit state for Slack map uploads (min 5s between uploads)
_last_map_upload_ts: float = 0.0
_MAP_UPLOAD_COOLDOWN_SEC: float = 5.0


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
# Slack map upload (Bot API — separate from webhook text alerts)
# ═══════════════════════════════════════════════════════════════════════════════


async def upload_slack_map(file_path: str, caption: str) -> None:
    """
    Upload a ground track map PNG to Slack via the Bot API.

    Requires ``slack_bot_token`` and ``slack_channel_id`` in settings.
    Rate-limited to one upload per 5 seconds to respect Slack API limits.
    Silently skips if credentials are missing or cooldown hasn't elapsed.
    """
    global _last_map_upload_ts

    token = settings.slack_bot_token
    channel = settings.slack_channel_id
    if not token or not channel:
        return

    # Rate-limit: skip if too soon after last upload
    now = time.monotonic()
    if now - _last_map_upload_ts < _MAP_UPLOAD_COOLDOWN_SEC:
        logger.debug("Slack map upload skipped (rate limit cooldown)")
        return

    try:
        from slack_sdk.web.async_client import AsyncWebClient

        client = AsyncWebClient(token=token)
        await client.files_upload_v2(
            file=file_path,
            channel=channel,
            title="Ground Track Map",
            initial_comment=caption,
        )
        _last_map_upload_ts = time.monotonic()
        logger.info("Slack map uploaded to channel %s", channel)
    except Exception:
        logger.exception("Slack map upload failed")


# ═══════════════════════════════════════════════════════════════════════════════
# Siren alert helpers
# ═══════════════════════════════════════════════════════════════════════════════


async def send_siren_alert(
    zones: list[str],
    trajectory_match: bool = False,
    prediction_summary: str = "",
) -> None:
    """Send a CRITICAL siren alert to Slack and Discord."""
    zone_str = ", ".join(zones)

    if trajectory_match:
        text = (
            f":red_circle: *OFFICIAL SIREN CONFIRMED for {zone_str}. "
            f"Trajectory match detected!*\n{prediction_summary}"
        )
    else:
        text = f":red_circle: *SIREN ALERT: {zone_str}*\nNo active trajectory match."

    async with httpx.AsyncClient(timeout=15.0) as client:
        if settings.slack_webhook_url:
            try:
                slack_body = {
                    "text": text,
                    "blocks": [
                        {
                            "type": "section",
                            "text": {"type": "mrkdwn", "text": text},
                        }
                    ],
                }
                resp = await client.post(settings.slack_webhook_url, json=slack_body)
                resp.raise_for_status()
                logger.info("Siren alert sent to Slack")
            except Exception:
                logger.exception("Siren Slack alert failed")

        if settings.discord_webhook_url:
            try:
                discord_body = {
                    "content": text.replace("*", "**").replace(":red_circle:", ""),
                }
                resp = await client.post(settings.discord_webhook_url, json=discord_body)
                resp.raise_for_status()
                logger.info("Siren alert sent to Discord")
            except Exception:
                logger.exception("Siren Discord alert failed")


async def send_siren_clearance(zones: list[str]) -> None:
    """Send a clearance message when the siren event ends."""
    zone_str = ", ".join(zones)
    text = f":large_green_circle: *EVENT ENDED for {zone_str}.* Residents may leave shelters."

    async with httpx.AsyncClient(timeout=15.0) as client:
        if settings.slack_webhook_url:
            try:
                slack_body = {"text": text}
                resp = await client.post(settings.slack_webhook_url, json=slack_body)
                resp.raise_for_status()
                logger.info("Siren clearance sent to Slack")
            except Exception:
                logger.exception("Siren clearance Slack failed")

        if settings.discord_webhook_url:
            try:
                discord_body = {
                    "content": text.replace("*", "**").replace(":large_green_circle:", ""),
                }
                resp = await client.post(settings.discord_webhook_url, json=discord_body)
                resp.raise_for_status()
                logger.info("Siren clearance sent to Discord")
            except Exception:
                logger.exception("Siren clearance Discord failed")


async def send_system_warning(message: str) -> None:
    """Send a system health warning to Slack (not a siren or event alert)."""
    async with httpx.AsyncClient(timeout=15.0) as client:
        if settings.slack_webhook_url:
            try:
                slack_body = {
                    "text": f":warning: *SYSTEM WARNING*\n{message}",
                }
                resp = await client.post(settings.slack_webhook_url, json=slack_body)
                resp.raise_for_status()
                logger.info("System warning sent to Slack: %s", message[:80])
            except Exception:
                logger.exception("Failed to send system warning to Slack")


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
