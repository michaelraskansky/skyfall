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
import math
import os
import signal
import sys
from datetime import datetime, timezone

import structlog
import uvicorn

from config import settings
from ingestion.adsb_poller import poll_adsb
from ingestion.emergency_webhook import app as webhook_app, set_event_queue
from ingestion.firms_poller import poll_firms
from ingestion.satcat_lookup import SatcatLookup
from ingestion.siren_listener import SirenEvent, poll_sirens, ZONE_COORDINATES
from ingestion.social_listener import listen_generic_scraper, listen_telegram
from ingestion.spacetrack_poller import poll_spacetrack
from models import CorrelatedEvent, EventClassification, EventSeverity, EventSource, RawEvent
from output.alerter import alert_loop, send_siren_alert, send_siren_clearance, upload_slack_map
from processing.object_tracker import ObjectTracker
from trajectory.models import TrajectoryRequest
from trajectory.predictor import DebrisTrajectoryPredictor
from visuals.map_generator import render_ground_track
from processing.correlation_engine import CorrelationEngine
from processing.llm_parser import parse_with_llm, close_client as close_llm_client

# ── Structured logging ────────────────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
        if settings.log_format == "console"
        else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger("orchestrator")


# ═══════════════════════════════════════════════════════════════════════════════
# Triage loop – the central processing coroutine
# ═══════════════════════════════════════════════════════════════════════════════


async def triage_loop(
    raw_queue: asyncio.Queue[RawEvent],
    alert_queue: asyncio.Queue[CorrelatedEvent],
    engine: CorrelationEngine,
    tracker: ObjectTracker,
    satcat: SatcatLookup,
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

                classification = (
                    "debris_reentry" if event.source == EventSource.SPACETRACK
                    else "unknown"
                )
                synthetic = LLMParsedEvent(
                    is_valid_anomaly=True,
                    approximate_origin=event.description[:100],
                    debris_trajectory_or_blast_radius="unknown",
                    event_classification=classification,
                    confidence_score=7,  # structured sensors get baseline 7
                )
                await engine.try_correlate(event, synthetic, output_queue=alert_queue)

            # ── Trajectory tracking for objects with NORAD_CAT_ID ────────
            norad_id = event.raw_payload.get("NORAD_CAT_ID")
            if norad_id:
                await tracker.track_observation(event)

                # Enrich with SATCAT identity data (cached after first lookup)
                sat_info = await satcat.get_info(str(norad_id))

                # Only trigger the EKF on real sensor observations (FIRMS,
                # ADS-B, social media), not TIP predictions.  TIP messages
                # are revised predictions of a future position, not actual
                # observations of where the object is now — feeding them
                # into the EKF produces nonsensical velocities.
                observations = await tracker.get_observations(
                    str(norad_id), exclude_source="spacetrack",
                )

                if len(observations) >= 2:
                    try:
                        obj_label = (
                            f"{sat_info.object_name} (NORAD {norad_id}, {sat_info.country})"
                            if sat_info
                            else f"NORAD {norad_id}"
                        )
                        logger.info(
                            "Triggering trajectory prediction for %s (%d observations)",
                            obj_label, len(observations),
                        )
                        request = TrajectoryRequest(
                            object_id=str(norad_id),
                            observations=observations,
                        )
                        predictor = DebrisTrajectoryPredictor()
                        prediction = await asyncio.to_thread(predictor.predict, request)

                        trajectory_event = CorrelatedEvent(
                            severity=EventSeverity.CRITICAL,
                            classification=EventClassification.DEBRIS_REENTRY,
                            latitude=prediction.impact_latitude,
                            longitude=prediction.impact_longitude,
                            contributing_events=[event],
                            summary=(
                                f"TRAJECTORY PREDICTION: {obj_label} "
                                f"impact at ({prediction.impact_latitude}, {prediction.impact_longitude}) "
                                f"in {prediction.seconds_until_impact:.0f}s, "
                                f"terminal velocity {prediction.terminal_velocity_m_s:.0f} m/s"
                            ),
                            corroborating_sources=["spacetrack"],
                            impact_prediction=prediction,
                            satcat_info=sat_info,
                        )
                        await alert_queue.put(trajectory_event)

                        # Store for siren-trajectory correlation
                        _recent_predictions.append(
                            (prediction, obj_label, datetime.now(timezone.utc))
                        )

                        # Reverse-lookup: check pending sirens
                        await _check_prediction_against_pending_sirens(
                            prediction, obj_label,
                        )

                        logger.info(
                            "Trajectory prediction queued for %s: impact at (%.4f, %.4f)",
                            obj_label, prediction.impact_latitude, prediction.impact_longitude,
                        )

                        # Generate and upload ground track map (non-blocking)
                        if prediction.trajectory_points:
                            try:
                                map_path = await asyncio.to_thread(
                                    render_ground_track, prediction, sat_info,
                                )
                                ellipse = prediction.covariance_position_enu
                                cov_ee, cov_en, cov_nn = ellipse[0][0], ellipse[0][1], ellipse[1][1]
                                tr = cov_ee + cov_nn
                                det = cov_ee * cov_nn - cov_en ** 2
                                disc = max((tr / 2) ** 2 - det, 0)
                                sm = 2.0 * math.sqrt(max(tr / 2 + math.sqrt(disc), 0))
                                sn = 2.0 * math.sqrt(max(tr / 2 - math.sqrt(disc), 0))
                                caption = (
                                    f"Ground Track: {obj_label}\n"
                                    f"Impact: ({prediction.impact_latitude}, {prediction.impact_longitude}) "
                                    f"| ETA: {prediction.seconds_until_impact:.0f}s "
                                    f"| Terminal: {prediction.terminal_velocity_m_s:.0f} m/s\n"
                                    f"95% ellipse: {sm / 1000:.1f}km x {sn / 1000:.1f}km"
                                )
                                await upload_slack_map(map_path, caption)
                                os.unlink(map_path)
                            except Exception:
                                logger.warning("Map generation/upload failed for %s", obj_label, exc_info=True)
                    except Exception:
                        logger.warning(
                            "Trajectory prediction failed for NORAD %s, will retry on next observation",
                            norad_id, exc_info=True,
                        )

        except Exception as exc:
            logger.error("Triage error for event %s: %s", event.event_id, exc, exc_info=True)
        finally:
            raw_queue.task_done()


# ═══════════════════════════════════════════════════════════════════════════════
# Siren-trajectory correlation
# ═══════════════════════════════════════════════════════════════════════════════

# Recent trajectory predictions, accessible to the siren callback.
# Each entry: (ImpactPrediction, satcat_label, timestamp)
_recent_predictions: list[tuple] = []
_PREDICTION_RETENTION_SEC = 300  # Keep predictions for 5 minutes

# Pending sirens awaiting a trajectory match (reverse-lookup).
# Zone name -> unix timestamp of siren activation.
_pending_sirens: dict[str, float] = {}
_PENDING_SIREN_RETENTION_SEC = 60  # Keep pending sirens for 60 seconds


def _prune_pending_sirens() -> None:
    """Remove pending sirens older than the retention window."""
    cutoff = datetime.now(timezone.utc).timestamp() - _PENDING_SIREN_RETENTION_SEC
    expired = [z for z, ts in _pending_sirens.items() if ts < cutoff]
    for z in expired:
        del _pending_sirens[z]


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate distance in km between two lat/lon points."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return 6371.0 * 2 * math.asin(math.sqrt(a))


async def _check_prediction_against_pending_sirens(
    prediction, obj_label: str,
) -> None:
    """
    Reverse-lookup: after an EKF prediction completes, check if any
    pending (unmatched) sirens are within 50km of the predicted impact.

    If a match is found, fire a retroactive OFFICIAL SIREN CONFIRMED alert
    and remove the siren from pending (so it doesn't fire twice).
    """
    _prune_pending_sirens()

    matched_zones: list[str] = []
    match_summary = ""

    for zone_name, ts in list(_pending_sirens.items()):
        zone_coords = ZONE_COORDINATES.get(zone_name)
        if not zone_coords:
            continue
        dist_km = _haversine_km(
            prediction.impact_latitude, prediction.impact_longitude,
            zone_coords[0], zone_coords[1],
        )
        if dist_km <= 50.0:
            matched_zones.append(zone_name)
            match_summary = (
                f"Object: {obj_label}\n"
                f"Predicted impact: ({prediction.impact_latitude}, {prediction.impact_longitude})\n"
                f"Distance to {zone_name}: {dist_km:.1f} km\n"
                f"Terminal velocity: {prediction.terminal_velocity_m_s:.0f} m/s\n"
                f"ETA: {prediction.seconds_until_impact:.0f}s"
            )

    if matched_zones:
        logger.warning(
            "Retroactive siren-trajectory match: %s near %s",
            obj_label, ", ".join(matched_zones),
        )
        for z in matched_zones:
            del _pending_sirens[z]
        await send_siren_alert(matched_zones, True, match_summary)


async def _on_siren(siren_event: SirenEvent) -> None:
    """
    Callback invoked by the siren listener when a watch zone is hit.

    Checks recent trajectory predictions for proximity to siren zones
    and sends the appropriate Slack/Discord alert.
    """
    zones = siren_event.matched_watch_zones

    if not siren_event.is_active:
        await send_siren_clearance(zones)
        return

    # Prune old predictions
    now = datetime.now(timezone.utc)
    cutoff = now.timestamp() - _PREDICTION_RETENTION_SEC
    _recent_predictions[:] = [
        (pred, label, ts) for pred, label, ts in _recent_predictions
        if ts.timestamp() > cutoff
    ]

    # Check each recent prediction against the siren zone coordinates
    trajectory_match = False
    match_summary = ""

    for pred, label, ts in _recent_predictions:
        for zone_name in zones:
            zone_coords = ZONE_COORDINATES.get(zone_name)
            if not zone_coords:
                continue
            dist_km = _haversine_km(
                pred.impact_latitude, pred.impact_longitude,
                zone_coords[0], zone_coords[1],
            )
            if dist_km <= 50.0:
                trajectory_match = True
                match_summary = (
                    f"Object: {label}\n"
                    f"Predicted impact: ({pred.impact_latitude}, {pred.impact_longitude})\n"
                    f"Distance to {zone_name}: {dist_km:.1f} km\n"
                    f"Terminal velocity: {pred.terminal_velocity_m_s:.0f} m/s\n"
                    f"ETA: {pred.seconds_until_impact:.0f}s"
                )
                break
        if trajectory_match:
            break

    await send_siren_alert(zones, trajectory_match, match_summary)

    # If no match was found, store in pending for reverse-lookup
    if not trajectory_match:
        now_ts = now.timestamp()
        for zone_name in zones:
            _pending_sirens[zone_name] = now_ts
        logger.info(
            "Siren stored in pending for reverse-lookup: %s",
            ", ".join(zones),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Uvicorn server wrapper (runs FastAPI in the same event loop)
# ═══════════════════════════════════════════════════════════════════════════════


async def run_webhook_server() -> None:
    """Launch the FastAPI emergency webhook server inside the shared loop."""
    config = uvicorn.Config(
        app=webhook_app,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8000")),
        log_level="info",
    )
    server = uvicorn.Server(config)
    await server.serve()


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════


async def main() -> None:
    """Wire everything together and run forever."""

    raw_queue: asyncio.Queue[RawEvent] = asyncio.Queue(maxsize=10_000)
    alert_queue: asyncio.Queue[CorrelatedEvent] = asyncio.Queue(maxsize=1_000)

    set_event_queue(raw_queue)

    engine = CorrelationEngine()
    await engine.connect()

    tracker = ObjectTracker()
    await tracker.connect()

    satcat = SatcatLookup()
    await satcat.connect()

    tasks = [
        asyncio.create_task(poll_firms(raw_queue), name="firms_poller"),
        asyncio.create_task(poll_adsb(raw_queue), name="adsb_poller"),
        asyncio.create_task(listen_telegram(raw_queue), name="telegram_listener"),
        asyncio.create_task(listen_generic_scraper(raw_queue), name="generic_scraper"),
        asyncio.create_task(poll_spacetrack(raw_queue), name="spacetrack_poller"),
        asyncio.create_task(poll_sirens(raw_queue, siren_callback=_on_siren), name="siren_listener"),
        asyncio.create_task(run_webhook_server(), name="webhook_server"),
        asyncio.create_task(triage_loop(raw_queue, alert_queue, engine, tracker, satcat), name="triage"),
        asyncio.create_task(alert_loop(alert_queue), name="alerter"),
    ]

    logger.info("All tasks launched", task_count=len(tasks))

    shutdown_event = asyncio.Event()

    def _handle_signal(sig, frame):
        logger.info("Received signal, initiating graceful shutdown", signal=sig)
        shutdown_event.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    shutdown_task = asyncio.create_task(shutdown_event.wait(), name="shutdown_watcher")
    all_tasks = tasks + [shutdown_task]
    done, pending = await asyncio.wait(all_tasks, return_when=asyncio.FIRST_COMPLETED)

    for task in done:
        if task == shutdown_task:
            logger.info("Graceful shutdown initiated")
        elif task.exception():
            logger.critical(
                "Task crashed",
                task_name=task.get_name(),
                error=str(task.exception()),
            )

    for task in pending:
        task.cancel()

    logger.info("Draining queues...")
    try:
        await asyncio.wait_for(raw_queue.join(), timeout=10.0)
        await asyncio.wait_for(alert_queue.join(), timeout=10.0)
    except asyncio.TimeoutError:
        logger.warning("Queue drain timed out")

    await engine.close()
    await tracker.close()
    await satcat.close()
    await close_llm_client()
    logger.info("Shutdown complete")


if __name__ == "__main__":
    asyncio.run(main())
