"""
Main Orchestrator
==================

This is the single entry-point for the Aerospace Debris & Industrial Anomaly
Tracker.  It wires together every layer of the pipeline:

  1. **Ingestion** – FIRMS poller, ADS-B poller, Telegram listener, and the
     FastAPI emergency-webhook server all run as concurrent asyncio tasks.
  2. **Processing** – A triage loop reads raw events from the shared queue,
     routes social-media / emergency text through the LLM parser, and then
     asks the correlation engine whether the event should be elevated.
  3. **Output** – An alerter loop consumes CRITICAL correlated events and
     pushes formatted payloads to Slack / Discord webhooks.

Architecture diagram (simplified)::

    ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐
    │  FIRMS   │  │  ADS-B   │  │ Telegram │  │  Emergency  │
    │  Poller  │  │  Poller  │  │ Listener │  │  Webhook    │
    └────┬─────┘  └────┬─────┘  └────┬─────┘  └──────┬──────┘
         │             │             │               │
         └─────────────┴──────┬──────┴───────────────┘
                              ▼
                    ┌───────────────────┐
                    │  Raw Event Queue  │  (asyncio.Queue)
                    └────────┬──────────┘
                             ▼
                    ┌───────────────────┐
                    │   Triage Loop     │
                    │  (LLM + Correlate)│
                    └────────┬──────────┘
                             ▼
                    ┌───────────────────┐
                    │  Alert Queue      │  (asyncio.Queue)
                    └────────┬──────────┘
                             ▼
                    ┌───────────────────┐
                    │   Alerter Loop    │
                    │ (Slack / Discord) │
                    └───────────────────┘

Run with::

    python main.py

Or via Docker::

    docker compose up --build
"""

from __future__ import annotations

import asyncio
import logging
import sys

import uvicorn

from config import settings
from ingestion.adsb_poller import poll_adsb
from ingestion.emergency_webhook import app as webhook_app, set_event_queue
from ingestion.firms_poller import poll_firms
from ingestion.social_listener import listen_generic_scraper, listen_telegram
from models import CorrelatedEvent, EventSource, RawEvent
from output.alerter import alert_loop
from processing.correlation_engine import CorrelationEngine
from processing.llm_parser import parse_with_llm

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)-28s  %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("orchestrator")


# ═══════════════════════════════════════════════════════════════════════════════
# Triage loop – the central processing coroutine
# ═══════════════════════════════════════════════════════════════════════════════


async def triage_loop(
    raw_queue: asyncio.Queue[RawEvent],
    alert_queue: asyncio.Queue[CorrelatedEvent],
    engine: CorrelationEngine,
) -> None:
    """
    Continuously dequeue raw sensor events, triage them through the LLM
    (if they are unstructured text), and attempt correlation.

    Events from structured sensors (FIRMS, ADS-B) are ingested into the
    correlation engine directly.  Events from noisy sources (social media,
    emergency webhooks) are first parsed by the LLM before correlation.
    """
    logger.info("Triage loop starting.")

    while True:
        event = await raw_queue.get()

        try:
            # Always ingest into the correlation engine for geo-indexing.
            await engine.ingest(event)

            # Decide whether LLM parsing is needed.
            needs_llm = event.source in (
                EventSource.SOCIAL_MEDIA,
                EventSource.EMERGENCY_WEBHOOK,
            )

            if needs_llm:
                # Extract the text blob for LLM analysis.
                text = event.raw_payload.get("text") or event.description
                llm_result = await parse_with_llm(text)

                if llm_result is None:
                    logger.warning(
                        "LLM parse returned None for event %s – skipping correlation.",
                        event.event_id,
                    )
                    continue

                await engine.try_correlate(event, llm_result, output_queue=alert_queue)

            else:
                # Structured sensors (FIRMS, ADS-B) are implicitly valid.
                # We still attempt correlation in case another sensor has
                # already flagged the same area.
                from models import LLMParsedEvent

                synthetic = LLMParsedEvent(
                    is_valid_anomaly=True,
                    approximate_origin=event.description[:100],
                    debris_trajectory_or_blast_radius="unknown",
                    event_classification="unknown",
                    confidence_score=7,  # structured sensors get baseline 7
                )
                await engine.try_correlate(event, synthetic, output_queue=alert_queue)

        except Exception:
            logger.exception("Triage error for event %s", event.event_id)
        finally:
            raw_queue.task_done()


# ═══════════════════════════════════════════════════════════════════════════════
# Uvicorn server wrapper (runs FastAPI in the same event loop)
# ═══════════════════════════════════════════════════════════════════════════════


async def run_webhook_server() -> None:
    """Launch the FastAPI emergency webhook server inside the shared loop."""
    config = uvicorn.Config(
        app=webhook_app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
    server = uvicorn.Server(config)
    await server.serve()


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════


async def main() -> None:
    """Wire everything together and run forever."""

    # ── Shared queues ─────────────────────────────────────────────────────
    raw_queue: asyncio.Queue[RawEvent] = asyncio.Queue(maxsize=10_000)
    alert_queue: asyncio.Queue[CorrelatedEvent] = asyncio.Queue(maxsize=1_000)

    # Inject the queue into the FastAPI webhook app.
    set_event_queue(raw_queue)

    # ── Correlation engine ────────────────────────────────────────────────
    engine = CorrelationEngine()
    await engine.connect()

    # ── Launch all tasks concurrently ─────────────────────────────────────
    tasks = [
        # Ingestion layer
        asyncio.create_task(poll_firms(raw_queue), name="firms_poller"),
        asyncio.create_task(poll_adsb(raw_queue), name="adsb_poller"),
        asyncio.create_task(listen_telegram(raw_queue), name="telegram_listener"),
        asyncio.create_task(listen_generic_scraper(raw_queue), name="generic_scraper"),
        asyncio.create_task(run_webhook_server(), name="webhook_server"),
        # Processing layer
        asyncio.create_task(triage_loop(raw_queue, alert_queue, engine), name="triage"),
        # Output layer
        asyncio.create_task(alert_loop(alert_queue), name="alerter"),
    ]

    logger.info("All %d tasks launched.  Pipeline is live.", len(tasks))

    # Run until one task crashes (fail-fast).
    done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)

    for task in done:
        if task.exception():
            logger.critical(
                "Task %s crashed: %s", task.get_name(), task.exception()
            )

    # Cancel remaining tasks gracefully.
    for task in pending:
        task.cancel()

    await engine.close()


if __name__ == "__main__":
    asyncio.run(main())
