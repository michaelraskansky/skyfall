"""
Social Media Listener – Telegram / Generic Scraper
====================================================

Monitors unstructured social media channels for keywords that may indicate
aerospace debris re-entries or industrial anomalies.

Two backends are supported:

  1. **Telegram** (via Telethon) – listens to configured public channels
     in real-time using the Telegram MTProto API.
  2. **Generic keyword filter** – a placeholder coroutine you can swap out
     for any X/Twitter scraper, Bluesky firehose, or RSS poller.

Matching messages are wrapped in ``RawEvent`` and placed on the shared queue.
Because social media is *noisy*, these events are always sent through the
LLM triage step before reaching the correlation engine.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from asyncio import Queue

from config import settings
from models import EventSource, RawEvent

logger = logging.getLogger(__name__)

# ── Keywords / patterns that suggest an anomalous event ───────────────────────
# Case-insensitive.  Extend as needed.
KEYWORDS: list[str] = [
    # English
    "sonic boom",
    "bright flash",
    "fireball",
    "meteor",
    "factory explosion",
    "industrial explosion",
    "refinery fire",
    "chemical plant",
    "sirens",
    "mushroom cloud",
    "debris falling",
    "space debris",
    "re-entry",
    "reentry",
    "sky lit up",
    "missile",
    "airstrike",
    "air strike",
    "rocket attack",
    "shelling",
    "bombardment",
    # Arabic
    "انفجار",        # explosion
    "صاروخ",         # missile/rocket
    "غارة",          # airstrike/raid
    "قصف",           # shelling/bombardment
    "حريق",          # fire
    "صافرات",        # sirens
    "كرة نار",       # fireball
    "حطام",          # debris
    "صاروخ باليستي",  # ballistic missile
    "دمار",          # destruction
    "اشتعال",        # ignition/blaze
    # Hebrew
    "פיצוץ",         # explosion
    "טיל",           # missile
    "רקטה",          # rocket
    "תקיפה",         # strike/attack
    "אזעקה",         # alarm/siren
    "שריפה",         # fire
    "הפצצה",         # bombardment
]

_KEYWORD_PATTERN = re.compile(
    "|".join(re.escape(kw) for kw in KEYWORDS),
    re.IGNORECASE,
)


def _matches_keywords(text: str) -> bool:
    """Return True if *text* contains at least one tracked keyword."""
    return bool(_KEYWORD_PATTERN.search(text))


# ═══════════════════════════════════════════════════════════════════════════════
# Backend 1: Telegram via Telethon
# ═══════════════════════════════════════════════════════════════════════════════


async def listen_telegram(event_queue: Queue[RawEvent]) -> None:
    """
    Connect to Telegram, join configured channels, and stream messages
    through the keyword filter into *event_queue*.
    """
    api_id = settings.telegram_api_id
    api_hash = settings.telegram_api_hash
    channels_raw = settings.telegram_channels

    if not api_id or not api_hash or not channels_raw:
        logger.warning("Telegram credentials not set – Telegram listener disabled.")
        await asyncio.Event().wait()

    # Import Telethon lazily so the rest of the app works without it.
    try:
        from telethon import TelegramClient, events as tg_events
        from telethon.sessions import StringSession
    except ImportError:
        logger.error("telethon is not installed – Telegram listener disabled.")
        await asyncio.Event().wait()
        return

    session_str = settings.telegram_session
    channels = [c.strip() for c in channels_raw.split(",") if c.strip()]
    print(f"[TELEGRAM] Starting listener for channels: {channels}", flush=True)

    # Use StringSession if available (no filesystem session file needed).
    session = StringSession(session_str) if session_str else "debris_tracker_session"
    client = TelegramClient(session, api_id, api_hash)

    @client.on(tg_events.NewMessage(chats=channels))
    async def _on_message(tg_event):
        text: str = tg_event.raw_text or ""
        if not _matches_keywords(text):
            return

        event = RawEvent(
            source=EventSource.SOCIAL_MEDIA,
            raw_payload={
                "platform": "telegram",
                "channel": str(tg_event.chat_id),
                "text": text,
                "message_id": tg_event.id,
            },
            description=f"Telegram keyword match: {text[:200]}",
        )
        print(f"[TELEGRAM] Keyword hit: {event.description[:120]}", flush=True)
        await event_queue.put(event)

    try:
        await client.start()
        print(f"[TELEGRAM] Client connected, listening to {channels}", flush=True)
        await client.run_until_disconnected()
    except Exception as e:
        print(f"[TELEGRAM] Connection error: {e}", flush=True)
        raise


# ═══════════════════════════════════════════════════════════════════════════════
# Backend 2: Generic keyword scraper (placeholder)
# ═══════════════════════════════════════════════════════════════════════════════


async def listen_generic_scraper(event_queue: Queue[RawEvent]) -> None:
    """
    Placeholder coroutine for a generic social-media scraper (X/Twitter,
    Bluesky, RSS, etc.).

    Replace the inner loop with your preferred scraping library.  The
    contract is simple: yield text blobs, run them through
    ``_matches_keywords``, and push matching ones onto *event_queue*.
    """
    logger.info("Generic social scraper starting (placeholder loop).")

    while True:
        # ── Replace this block with real scraping logic ───────────────────
        # Example: poll an RSS feed, call a scraper API, read from a
        #          message broker topic, etc.
        #
        # async with aiohttp.ClientSession() as session:
        #     async with session.get(SOME_FEED_URL) as resp:
        #         items = parse_feed(await resp.text())
        #         for item in items:
        #             if _matches_keywords(item.text):
        #                 event = RawEvent(...)
        #                 await event_queue.put(event)
        #
        await asyncio.sleep(60)  # stub – sleep and retry
