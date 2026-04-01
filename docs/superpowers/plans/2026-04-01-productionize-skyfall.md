# Productionize Skyfall Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate Skyfall from prototype to production-ready: swap Redis→DynamoDB, OpenAI/Gemini→Bedrock, add full test suite, operational hardening, and CloudFormation deployment.

**Architecture:** Bottom-up approach. Migrate service backends first (DynamoDB, Bedrock), update config/deps, write tests at three layers (unit, integration, e2e), harden operations (structured logging, health checks, graceful shutdown), then write CloudFormation for ECS/Fargate deployment.

**Tech Stack:** Python 3.12+, aioboto3 (DynamoDB + Bedrock), FastAPI, pytest + moto + respx, structlog, CloudFormation (ECS/Fargate)

---

## File Structure

### New Files

- `processing/geohash.py` — Pure Python geohash encoder + neighbor computation
- `tests/unit/__init__.py` — Package marker
- `tests/unit/test_geohash.py` — Geohash encode/decode/neighbors tests
- `tests/unit/test_physics.py` — Coordinate transform and atmosphere model tests
- `tests/unit/test_predictor.py` — EKF trajectory predictor tests
- `tests/unit/test_formatter.py` — Alert payload formatting tests
- `tests/unit/test_models.py` — Pydantic model validation tests
- `tests/integration/__init__.py` — Package marker
- `tests/integration/test_correlation_engine.py` — Correlation engine with moto DynamoDB
- `tests/integration/test_llm_parser.py` — LLM parser with moto Bedrock
- `tests/integration/test_emergency_webhook.py` — FastAPI TestClient tests
- `tests/integration/test_alerter.py` — Alerter with respx mocked webhooks
- `tests/e2e/__init__.py` — Package marker
- `tests/e2e/test_pipeline.py` — Full pipeline end-to-end test
- `tests/conftest.py` — Shared fixtures (moto DynamoDB, moto Bedrock, event factories)
- `infra/template.yaml` — CloudFormation template
- `pytest.ini` — Pytest configuration

### Modified Files

- `processing/correlation_engine.py` — Rewrite from Redis to DynamoDB + geohash
- `processing/llm_parser.py` — Rewrite from OpenAI/Gemini to Bedrock
- `config.py` — Remove Redis/OpenAI/Gemini settings, add AWS/Bedrock/DynamoDB settings
- `models.py` — No changes needed
- `main.py` — Update imports, add signal handlers, structlog setup, health check wiring
- `output/alerter.py` — Add tenacity retries to Bedrock (already has them for webhooks)
- `ingestion/emergency_webhook.py` — Enhance /health endpoint
- `pyproject.toml` — Swap dependencies
- `Dockerfile` — Switch to uv-based install
- `.gitignore` — Add moto/pytest artifacts

---

### Task 1: Geohash Helper

**Files:**
- Create: `processing/geohash.py`
- Create: `tests/unit/__init__.py`
- Create: `tests/unit/test_geohash.py`

- [ ] **Step 1: Create pytest config**

Create `pytest.ini` at the project root:

```ini
[pytest]
asyncio_mode = auto
testpaths = tests
```

- [ ] **Step 2: Write failing tests for geohash**

Create `tests/unit/__init__.py` (empty file) and `tests/unit/test_geohash.py`:

```python
"""Tests for the pure-Python geohash implementation."""

from processing.geohash import encode, neighbors


class TestEncode:
    def test_known_location_houston(self):
        """Houston, TX (29.76, -95.37) should produce geohash starting with '9vk1'."""
        result = encode(29.76, -95.37, precision=4)
        assert result == "9vk1"

    def test_known_location_london(self):
        """London (51.5074, -0.1278) should produce geohash starting with 'gcpv'."""
        result = encode(51.5074, -0.1278, precision=4)
        assert result == "gcpv"

    def test_known_location_tokyo(self):
        """Tokyo (35.6762, 139.6503) should produce geohash starting with 'xn76'."""
        result = encode(35.6762, 139.6503, precision=4)
        assert result == "xn76"

    def test_precision_5(self):
        """5-char geohash for Houston should be '9vk1m' or similar."""
        result = encode(29.76, -95.37, precision=5)
        assert len(result) == 5
        assert result.startswith("9vk1")

    def test_edge_case_north_pole(self):
        result = encode(90.0, 0.0, precision=4)
        assert len(result) == 4

    def test_edge_case_south_pole(self):
        result = encode(-90.0, 0.0, precision=4)
        assert len(result) == 4

    def test_edge_case_dateline(self):
        result = encode(0.0, 180.0, precision=4)
        assert len(result) == 4


class TestNeighbors:
    def test_returns_eight_neighbors(self):
        """Every geohash has exactly 8 neighbors."""
        result = neighbors("9vk1")
        assert len(result) == 8

    def test_neighbors_are_unique(self):
        result = neighbors("9vk1")
        assert len(set(result)) == 8

    def test_neighbors_same_precision(self):
        result = neighbors("9vk1")
        for n in result:
            assert len(n) == 4

    def test_neighbors_do_not_include_self(self):
        result = neighbors("9vk1")
        assert "9vk1" not in result

    def test_nearby_locations_share_prefix_or_are_neighbors(self):
        """Two points ~1km apart should either share a geohash or be neighbors."""
        gh1 = encode(29.760, -95.370, precision=4)
        gh2 = encode(29.765, -95.370, precision=4)
        assert gh1 == gh2 or gh2 in neighbors(gh1)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_geohash.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'processing.geohash'`

- [ ] **Step 4: Implement geohash module**

Create `processing/geohash.py`:

```python
"""
Pure-Python Geohash
====================

Encodes latitude/longitude into geohash strings and computes the 8
neighboring cells. Used by the correlation engine for geo-bucketing
events in DynamoDB.

Reference: https://en.wikipedia.org/wiki/Geohash
"""

from __future__ import annotations

_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_DECODE_MAP = {c: i for i, c in enumerate(_BASE32)}

# Neighbor lookup tables (direction → even/odd row character mapping).
_NEIGHBORS = {
    "n": {"even": "p0r21436x8zb9dcf5h7kjnmqesgutwvy", "odd": "bc01fg45telefonías89telefonías"},
    "s": {"even": "14365h7k9dcfesgujnmqp0r2twvyx8zb", "odd": "238967debc01telefonías45telefoníasfg"},
    "e": {"even": "bc01fg45238967deuvhjyznpkmstqrwx", "odd": "p0r21436x8zb9dcf5h7kjnmqesgutwvy"},
    "w": {"even": "238967debc01fg45uvhjyznpkmstqrwx", "odd": "14365h7k9dcfesgujnmqp0r2twvyx8zb"},
}

_BORDERS = {
    "n": {"even": "prxz", "odd": "bcfguvyz"},
    "s": {"even": "028b", "odd": "0145hjnp"},
    "e": {"even": "bcfguvyz", "odd": "prxz"},
    "w": {"even": "0145hjnp", "odd": "028b"},
}


def encode(latitude: float, longitude: float, precision: int = 4) -> str:
    """
    Encode a (latitude, longitude) pair into a geohash string.

    Parameters
    ----------
    latitude : float
        Latitude in degrees (-90 to 90).
    longitude : float
        Longitude in degrees (-180 to 180).
    precision : int
        Length of the returned geohash string. Default 4 (~20km cells).
    """
    lat_range = (-90.0, 90.0)
    lon_range = (-180.0, 180.0)
    is_lon = True  # longitude bit comes first
    bit = 0
    ch_index = 0
    result = []

    while len(result) < precision:
        if is_lon:
            mid = (lon_range[0] + lon_range[1]) / 2.0
            if longitude >= mid:
                ch_index = ch_index * 2 + 1
                lon_range = (mid, lon_range[1])
            else:
                ch_index = ch_index * 2
                lon_range = (lon_range[0], mid)
        else:
            mid = (lat_range[0] + lat_range[1]) / 2.0
            if latitude >= mid:
                ch_index = ch_index * 2 + 1
                lat_range = (mid, lat_range[1])
            else:
                ch_index = ch_index * 2
                lat_range = (lat_range[0], mid)

        is_lon = not is_lon
        bit += 1

        if bit == 5:
            result.append(_BASE32[ch_index])
            bit = 0
            ch_index = 0

    return "".join(result)


def _adjacent(geohash: str, direction: str) -> str:
    """Return the geohash of the cell adjacent in the given direction."""
    last_char = geohash[-1]
    parent = geohash[:-1]
    parity = "odd" if len(geohash) % 2 == 0 else "even"

    if last_char in _BORDERS[direction][parity] and parent:
        parent = _adjacent(parent, direction)

    return parent + _BASE32[_NEIGHBORS[direction][parity].index(last_char)]


def neighbors(geohash: str) -> list[str]:
    """
    Return the 8 neighboring geohash cells (N, NE, E, SE, S, SW, W, NW).
    """
    n = _adjacent(geohash, "n")
    s = _adjacent(geohash, "s")
    e = _adjacent(geohash, "e")
    w = _adjacent(geohash, "w")
    ne = _adjacent(n, "e")
    nw = _adjacent(n, "w")
    se = _adjacent(s, "e")
    sw = _adjacent(s, "w")
    return [n, ne, e, se, s, sw, w, nw]
```

**Important:** The `_NEIGHBORS` and `_BORDERS` lookup tables above are placeholders for illustration. The actual implementation must use the standard geohash neighbor algorithm. Here are the correct tables:

```python
_NEIGHBORS = {
    "n": {"even": "p0r21436x8zb9dcf5h7kjnmqesgutwvy", "odd": "bc01fg45238967deuvhjyznpkmstqrwx"},
    "s": {"even": "14365h7k9dcfesgujnmqp0r2twvyx8zb", "odd": "238967debc01fg45uvhjyznpkmstqrwx"},
    "e": {"even": "bc01fg45238967deuvhjyznpkmstqrwx", "odd": "p0r21436x8zb9dcf5h7kjnmqesgutwvy"},
    "w": {"even": "238967debc01fg45uvhjyznpkmstqrwx", "odd": "14365h7k9dcfesgujnmqp0r2twvyx8zb"},
}

_BORDERS = {
    "n": {"even": "prxz", "odd": "bcfguvyz"},
    "s": {"even": "028b", "odd": "0145hjnp"},
    "e": {"even": "bcfguvyz", "odd": "prxz"},
    "w": {"even": "0145hjnp", "odd": "028b"},
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_geohash.py -v`
Expected: All 11 tests PASS

- [ ] **Step 6: Commit**

```bash
git add processing/geohash.py tests/unit/__init__.py tests/unit/test_geohash.py pytest.ini
git commit -m "feat: add pure-Python geohash encoder with neighbor computation"
```

---

### Task 2: Update Dependencies and Config

**Files:**
- Modify: `pyproject.toml`
- Modify: `config.py`

- [ ] **Step 1: Update pyproject.toml**

Replace the current dependencies block. Remove `openai`, `google-generativeai`, `redis[hiredis]`. Add `aioboto3`. Move `telethon` to optional. Add `moto` and `respx` to dev deps:

```toml
[project]
name = "skyfall"
version = "0.1.0"
description = "Real-time aerospace debris & industrial anomaly tracker"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "aiohttp>=3.9.0",
    "fastapi>=0.110.0",
    "uvicorn[standard]>=0.29.0",
    "pydantic>=2.6.0",
    "pydantic-settings>=2.2.0",
    "aioboto3>=13.0.0",
    "httpx>=0.27.0",
    "numpy>=1.26.0",
    "scipy>=1.12.0",
    "geopy>=2.4.0",
    "python-dotenv>=1.0.0",
    "structlog>=24.1.0",
    "tenacity>=8.2.0",
]

[project.optional-dependencies]
telegram = [
    "telethon>=1.36.0",
]
dev = [
    "pytest>=8.0.0",
    "pytest-asyncio>=0.23.0",
    "moto[dynamodb,bedrock]>=5.0.0",
    "respx>=0.22.0",
]

[tool.hatch.build.targets.wheel]
packages = ["ingestion", "processing", "output", "trajectory"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

- [ ] **Step 2: Update config.py**

Replace the entire `Settings` class body. Remove Redis, OpenAI, Gemini settings. Add AWS, Bedrock, DynamoDB settings:

```python
"""
Centralized configuration loaded from environment variables.

All secrets (API keys, webhook URLs) are read from the environment so that
nothing sensitive is committed to source control.  A .env file is supported
for local development via pydantic-settings.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application-wide settings - populated from env vars or .env file."""

    # -- NASA FIRMS --
    firms_api_key: str = ""
    firms_poll_interval_sec: int = 180
    firms_bounding_boxes: str = "-125,24,-66,50"

    # -- ADS-B Exchange --
    adsb_api_key: str = ""
    adsb_api_base_url: str = "https://adsbexchange.com/api/aircraft/v2"
    adsb_poll_interval_sec: int = 30
    adsb_watch_hex_codes: str = ""

    # -- Telegram Listener --
    telegram_api_id: int = 0
    telegram_api_hash: str = ""
    telegram_channels: str = ""

    # -- AWS --
    aws_region: str = "us-east-1"

    # -- Amazon Bedrock (LLM) --
    bedrock_model_id: str = "anthropic.claude-sonnet-4-20250514"

    # -- DynamoDB --
    dynamodb_table_name: str = "skyfall-events"
    dynamodb_endpoint_url: str = ""  # Override for local testing

    # -- Alerting --
    slack_webhook_url: str = ""
    discord_webhook_url: str = ""

    # -- Correlation tuning --
    correlation_window_sec: int = 300
    min_confidence_score: int = 6
    geohash_precision: int = 4

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Singleton - import this everywhere.
settings = Settings()
```

- [ ] **Step 3: Run uv sync to install new dependencies**

Run: `rm -rf .venv && uv sync --extra dev`
Expected: All packages install successfully, `aioboto3`, `moto`, `respx` present in output.

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml config.py
git commit -m "chore: swap redis/openai/gemini deps for aioboto3, update config"
```

---

### Task 3: Rewrite Correlation Engine for DynamoDB

**Files:**
- Modify: `processing/correlation_engine.py`
- Create: `tests/integration/__init__.py`
- Create: `tests/integration/test_correlation_engine.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Write conftest.py with shared fixtures**

Create `tests/conftest.py`:

```python
"""Shared test fixtures for the Skyfall test suite."""

import os
from datetime import datetime, timezone
from unittest.mock import patch

import boto3
import pytest
from moto import mock_aws

from models import EventSource, RawEvent


@pytest.fixture
def aws_credentials():
    """Mock AWS credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
    yield
    for key in [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SECURITY_TOKEN",
        "AWS_SESSION_TOKEN",
        "AWS_DEFAULT_REGION",
    ]:
        os.environ.pop(key, None)


@pytest.fixture
def dynamodb_table(aws_credentials):
    """Create a moto DynamoDB table matching the skyfall-events schema."""
    with mock_aws():
        client = boto3.client("dynamodb", region_name="us-east-1")
        client.create_table(
            TableName="skyfall-events",
            KeySchema=[
                {"AttributeName": "pk", "KeyType": "HASH"},
                {"AttributeName": "sk", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "pk", "AttributeType": "S"},
                {"AttributeName": "sk", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        # Enable TTL
        client.update_time_to_live(
            TableName="skyfall-events",
            TimeToLiveSpecification={
                "Enabled": True,
                "AttributeName": "expires_at",
            },
        )
        yield client


def make_raw_event(
    source: EventSource = EventSource.FIRMS,
    lat: float = 29.76,
    lon: float = -95.37,
    description: str = "Test event",
    event_id: str | None = None,
) -> RawEvent:
    """Factory for creating RawEvent instances in tests."""
    kwargs = {
        "source": source,
        "latitude": lat,
        "longitude": lon,
        "description": description,
        "timestamp": datetime.now(timezone.utc),
        "raw_payload": {"text": description},
    }
    if event_id:
        kwargs["event_id"] = event_id
    return RawEvent(**kwargs)
```

- [ ] **Step 2: Write failing integration tests for correlation engine**

Create `tests/integration/__init__.py` (empty file) and `tests/integration/test_correlation_engine.py`:

```python
"""Integration tests for the DynamoDB-backed correlation engine."""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest
from moto import mock_aws

from models import (
    CorrelatedEvent,
    EventSeverity,
    EventSource,
    LLMParsedEvent,
    RawEvent,
)
from processing.correlation_engine import CorrelationEngine
from tests.conftest import make_raw_event


@pytest.fixture
async def engine(dynamodb_table):
    """Create a CorrelationEngine connected to moto DynamoDB."""
    with mock_aws():
        # Recreate table inside mock_aws context for async operations
        import boto3

        client = boto3.client("dynamodb", region_name="us-east-1")
        client.create_table(
            TableName="skyfall-events",
            KeySchema=[
                {"AttributeName": "pk", "KeyType": "HASH"},
                {"AttributeName": "sk", "KeyType": "RANGE"},
            ],
            AttributeDefinitions=[
                {"AttributeName": "pk", "AttributeType": "S"},
                {"AttributeName": "sk", "AttributeType": "S"},
            ],
            BillingMode="PAY_PER_REQUEST",
        )

        with patch("processing.correlation_engine.settings") as mock_settings:
            mock_settings.aws_region = "us-east-1"
            mock_settings.dynamodb_table_name = "skyfall-events"
            mock_settings.dynamodb_endpoint_url = ""
            mock_settings.correlation_window_sec = 300
            mock_settings.min_confidence_score = 6
            mock_settings.geohash_precision = 4

            eng = CorrelationEngine()
            await eng.connect()
            yield eng
            await eng.close()


class TestIngest:
    async def test_ingest_stores_event(self, engine):
        event = make_raw_event(source=EventSource.FIRMS, lat=29.76, lon=-95.37)
        await engine.ingest(event)
        # Verify event is retrievable — query the geohash cell
        from processing.geohash import encode

        gh = encode(29.76, -95.37, precision=4)
        items = await engine._query_cell(f"geohash#{gh}")
        assert len(items) == 1
        assert items[0]["event_id"] == event.event_id

    async def test_ingest_sets_ttl(self, engine):
        event = make_raw_event()
        await engine.ingest(event)
        from processing.geohash import encode

        gh = encode(29.76, -95.37, precision=4)
        items = await engine._query_cell(f"geohash#{gh}")
        assert "expires_at" in items[0]

    async def test_ingest_without_coordinates(self, engine):
        event = RawEvent(
            source=EventSource.SOCIAL_MEDIA,
            description="No location",
            raw_payload={"text": "something happened"},
        )
        await engine.ingest(event)
        # Should not raise — events without coordinates are stored with a fallback


class TestCorrelation:
    async def test_two_sources_same_area_elevates_to_critical(self, engine):
        """Two events from different sources in the same geohash cell → CRITICAL."""
        event1 = make_raw_event(
            source=EventSource.FIRMS, lat=29.76, lon=-95.37, event_id="evt_firms"
        )
        event2 = make_raw_event(
            source=EventSource.ADSB, lat=29.761, lon=-95.371, event_id="evt_adsb"
        )
        await engine.ingest(event1)
        await engine.ingest(event2)

        llm_result = LLMParsedEvent(
            is_valid_anomaly=True,
            approximate_origin="Houston, TX",
            debris_trajectory_or_blast_radius="unknown",
            event_classification="industrial_accident",
            confidence_score=8,
        )

        alert_queue: asyncio.Queue[CorrelatedEvent] = asyncio.Queue()
        result = await engine.try_correlate(event2, llm_result, output_queue=alert_queue)

        assert result is not None
        assert result.severity == EventSeverity.CRITICAL
        assert len(result.corroborating_sources) >= 2
        assert not alert_queue.empty()

    async def test_single_source_does_not_elevate(self, engine):
        """One source alone should not produce CRITICAL."""
        event1 = make_raw_event(
            source=EventSource.FIRMS, lat=29.76, lon=-95.37, event_id="evt_1"
        )
        event2 = make_raw_event(
            source=EventSource.FIRMS, lat=29.761, lon=-95.371, event_id="evt_2"
        )
        await engine.ingest(event1)
        await engine.ingest(event2)

        llm_result = LLMParsedEvent(
            is_valid_anomaly=True,
            approximate_origin="Houston",
            debris_trajectory_or_blast_radius="unknown",
            event_classification="unknown",
            confidence_score=8,
        )

        result = await engine.try_correlate(event2, llm_result)
        assert result is not None
        assert result.severity != EventSeverity.CRITICAL

    async def test_low_confidence_skipped(self, engine):
        """Events below min_confidence_score should return None."""
        event = make_raw_event()
        await engine.ingest(event)

        llm_result = LLMParsedEvent(
            is_valid_anomaly=True,
            approximate_origin="somewhere",
            debris_trajectory_or_blast_radius="unknown",
            event_classification="unknown",
            confidence_score=3,
        )

        result = await engine.try_correlate(event, llm_result)
        assert result is None

    async def test_invalid_anomaly_skipped(self, engine):
        """Events marked as not valid anomalies should return None."""
        event = make_raw_event()
        await engine.ingest(event)

        llm_result = LLMParsedEvent(
            is_valid_anomaly=False,
            approximate_origin="",
            debris_trajectory_or_blast_radius="",
            event_classification="unknown",
            confidence_score=8,
        )

        result = await engine.try_correlate(event, llm_result)
        assert result is None

    async def test_distant_events_not_correlated(self, engine):
        """Events far apart (different geohash cells, not neighbors) should not correlate."""
        event1 = make_raw_event(
            source=EventSource.FIRMS, lat=29.76, lon=-95.37, event_id="evt_houston"
        )
        # London — totally different geohash
        event2 = make_raw_event(
            source=EventSource.ADSB, lat=51.5, lon=-0.12, event_id="evt_london"
        )
        await engine.ingest(event1)
        await engine.ingest(event2)

        llm_result = LLMParsedEvent(
            is_valid_anomaly=True,
            approximate_origin="London",
            debris_trajectory_or_blast_radius="unknown",
            event_classification="debris_reentry",
            confidence_score=9,
        )

        result = await engine.try_correlate(event2, llm_result)
        assert result is not None
        assert result.severity != EventSeverity.CRITICAL

    async def test_neighbor_cell_events_are_correlated(self, engine):
        """Events in adjacent geohash cells should still correlate."""
        # These two points are close but might fall in different 4-char geohash cells
        # near a cell boundary. We use two points that are ~15km apart.
        event1 = make_raw_event(
            source=EventSource.FIRMS, lat=29.80, lon=-95.37, event_id="evt_north"
        )
        event2 = make_raw_event(
            source=EventSource.ADSB, lat=29.60, lon=-95.37, event_id="evt_south"
        )
        await engine.ingest(event1)
        await engine.ingest(event2)

        llm_result = LLMParsedEvent(
            is_valid_anomaly=True,
            approximate_origin="Houston area",
            debris_trajectory_or_blast_radius="unknown",
            event_classification="industrial_accident",
            confidence_score=8,
        )

        result = await engine.try_correlate(event2, llm_result)
        # These should be in same or neighboring cells at precision 4
        # If they are, severity should be CRITICAL
        if result and len(result.corroborating_sources) >= 2:
            assert result.severity == EventSeverity.CRITICAL
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/integration/test_correlation_engine.py -v`
Expected: FAIL — `CorrelationEngine` still uses Redis, no `_query_cell` method, no geohash imports.

- [ ] **Step 4: Rewrite correlation_engine.py for DynamoDB**

Replace the entire contents of `processing/correlation_engine.py`:

```python
"""
Stateful Correlation Engine (DynamoDB)
=======================================

Cross-references sensor events using DynamoDB with geohash-based
spatial indexing.

Algorithm
---------
1. Every RawEvent is stored in DynamoDB keyed by geohash cell + timestamp,
   with a TTL for automatic cleanup.
2. When the LLM parser flags a high-confidence event, the engine queries
   the event's geohash cell plus its 8 neighbors for corroborating signals.
3. If two or more distinct sensor sources agree within the time window,
   the event is elevated to CRITICAL.

DynamoDB layout
---------------
- PK: ``geohash#<4-char>`` — groups events by ~20km geographic cell
- SK: ``<ISO-timestamp>#<event_id>`` — enables time-range queries
- TTL: ``expires_at`` — auto-cleanup after correlation window
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import aioboto3

if TYPE_CHECKING:
    from asyncio import Queue

from config import settings
from models import (
    CorrelatedEvent,
    EventClassification,
    EventSeverity,
    LLMParsedEvent,
    RawEvent,
)
from processing.geohash import encode, neighbors

logger = logging.getLogger(__name__)

# Fallback geohash for events without coordinates.
_NO_COORDS_GEOHASH = "none"


class CorrelationEngine:
    """
    DynamoDB-backed engine that cross-references sensor events by
    geographic proximity and time window.
    """

    def __init__(self) -> None:
        self._session: aioboto3.Session | None = None
        self._table_name = settings.dynamodb_table_name
        self._window = settings.correlation_window_sec
        self._min_confidence = settings.min_confidence_score
        self._precision = settings.geohash_precision
        self._table = None

    # -- Lifecycle --

    async def connect(self) -> None:
        """Open the DynamoDB connection."""
        self._session = aioboto3.Session()
        kwargs = {"region_name": settings.aws_region}
        if settings.dynamodb_endpoint_url:
            kwargs["endpoint_url"] = settings.dynamodb_endpoint_url
        self._resource = await self._session.resource("dynamodb", **kwargs).__aenter__()
        self._table = await self._resource.Table(self._table_name)
        logger.info("Correlation engine connected to DynamoDB table %s", self._table_name)

    async def close(self) -> None:
        """Shut down the DynamoDB connection."""
        if self._resource:
            await self._resource.__aexit__(None, None, None)

    # -- Ingestion --

    async def ingest(self, event: RawEvent) -> None:
        """Store a raw sensor event in DynamoDB with geohash key and TTL."""
        if event.latitude is not None and event.longitude is not None:
            gh = encode(event.latitude, event.longitude, precision=self._precision)
        else:
            gh = _NO_COORDS_GEOHASH

        pk = f"geohash#{gh}"
        sk = f"{event.timestamp.isoformat()}#{event.event_id}"
        expires_at = int(time.time()) + self._window + 60  # buffer

        item = {
            "pk": pk,
            "sk": sk,
            "event_id": event.event_id,
            "source": event.source.value,
            "latitude": str(event.latitude) if event.latitude is not None else None,
            "longitude": str(event.longitude) if event.longitude is not None else None,
            "description": event.description,
            "raw_payload": json.dumps(event.raw_payload),
            "timestamp": event.timestamp.isoformat(),
            "expires_at": expires_at,
        }
        # Remove None values (DynamoDB doesn't accept them)
        item = {k: v for k, v in item.items() if v is not None}

        await self._table.put_item(Item=item)
        logger.debug("Ingested event %s from %s into cell %s", event.event_id, event.source.value, pk)

    # -- Query helper --

    async def _query_cell(self, pk: str) -> list[dict]:
        """Query all items in a single geohash cell within the time window."""
        from datetime import timedelta

        cutoff = (datetime.now(timezone.utc) - timedelta(seconds=self._window)).isoformat()

        response = await self._table.query(
            KeyConditionExpression="pk = :pk AND sk >= :cutoff",
            ExpressionAttributeValues={
                ":pk": pk,
                ":cutoff": cutoff,
            },
        )
        return response.get("Items", [])

    # -- Correlation --

    async def try_correlate(
        self,
        trigger_event: RawEvent,
        llm_result: LLMParsedEvent,
        output_queue: Queue[CorrelatedEvent] | None = None,
    ) -> CorrelatedEvent | None:
        """
        Attempt to correlate trigger_event with other recent sensor data.

        Queries the trigger event's geohash cell plus its 8 neighbors.
        If 2+ distinct sources are found, elevates to CRITICAL.
        """
        if not llm_result.is_valid_anomaly:
            return None
        if llm_result.confidence_score < self._min_confidence:
            logger.debug(
                "Event %s below confidence threshold (%d < %d)",
                trigger_event.event_id,
                llm_result.confidence_score,
                self._min_confidence,
            )
            return None

        # -- Find nearby events across geohash cells --
        contributing: list[RawEvent] = [trigger_event]
        sources_seen: set[str] = {trigger_event.source.value}

        if trigger_event.latitude is not None and trigger_event.longitude is not None:
            gh = encode(
                trigger_event.latitude, trigger_event.longitude,
                precision=self._precision,
            )
            cells_to_query = [f"geohash#{gh}"] + [
                f"geohash#{n}" for n in neighbors(gh)
            ]

            for cell_pk in cells_to_query:
                items = await self._query_cell(cell_pk)
                for item in items:
                    eid = item.get("event_id")
                    if eid == trigger_event.event_id:
                        continue

                    source_val = item.get("source", "")
                    if source_val not in sources_seen:
                        sources_seen.add(source_val)
                        # Reconstruct a minimal RawEvent for the contributing list
                        lat = float(item["latitude"]) if item.get("latitude") else None
                        lon = float(item["longitude"]) if item.get("longitude") else None
                        neighbour = RawEvent(
                            event_id=eid,
                            source=EventSource(source_val),
                            latitude=lat,
                            longitude=lon,
                            description=item.get("description", ""),
                            timestamp=datetime.fromisoformat(item["timestamp"]),
                            raw_payload=json.loads(item.get("raw_payload", "{}")),
                        )
                        contributing.append(neighbour)

        # -- Decide severity --
        if len(sources_seen) >= 2:
            severity = EventSeverity.CRITICAL
        elif llm_result.confidence_score >= 8:
            severity = EventSeverity.HIGH
        elif llm_result.confidence_score >= 5:
            severity = EventSeverity.MEDIUM
        else:
            severity = EventSeverity.LOW

        try:
            classification = EventClassification(llm_result.event_classification)
        except ValueError:
            classification = EventClassification.UNKNOWN

        correlated = CorrelatedEvent(
            severity=severity,
            classification=classification,
            latitude=trigger_event.latitude,
            longitude=trigger_event.longitude,
            contributing_events=contributing,
            llm_analysis=llm_result,
            summary=(
                f"{severity.value} - {classification.value}: "
                f"{llm_result.approximate_origin}. "
                f"Corroborated by {len(sources_seen)} source(s): "
                f"{', '.join(sorted(sources_seen))}."
            ),
            corroborating_sources=sorted(sources_seen),
        )

        logger.info(
            "Correlation result: %s (sources=%s)",
            correlated.severity.value,
            correlated.corroborating_sources,
        )

        if severity == EventSeverity.CRITICAL and output_queue is not None:
            await output_queue.put(correlated)

        return correlated
```

- [ ] **Step 5: Run integration tests**

Run: `uv run pytest tests/integration/test_correlation_engine.py -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add processing/correlation_engine.py tests/conftest.py tests/integration/__init__.py tests/integration/test_correlation_engine.py
git commit -m "feat: rewrite correlation engine from Redis to DynamoDB with geohash"
```

---

### Task 4: Rewrite LLM Parser for Bedrock

**Files:**
- Modify: `processing/llm_parser.py`
- Create: `tests/integration/test_llm_parser.py`

- [ ] **Step 1: Write failing integration tests**

Create `tests/integration/test_llm_parser.py`:

```python
"""Integration tests for the Bedrock-backed LLM parser."""

import json
from unittest.mock import patch, AsyncMock

import pytest

from models import LLMParsedEvent
from processing.llm_parser import parse_with_llm, SYSTEM_PROMPT


class TestParseWithLLM:
    async def test_valid_response_returns_parsed_event(self):
        """A well-formed Bedrock response should parse into LLMParsedEvent."""
        mock_response = {
            "is_valid_anomaly": True,
            "approximate_origin": "Houston, TX",
            "debris_trajectory_or_blast_radius": "NW to SE, ~3 km",
            "event_classification": "industrial_accident",
            "confidence_score": 8,
        }
        response_body = json.dumps({"content": [{"text": json.dumps(mock_response)}]})

        with patch("processing.llm_parser._get_bedrock_client") as mock_client:
            mock_invoke = AsyncMock()
            mock_invoke.return_value = {"body": AsyncMock(read=AsyncMock(return_value=response_body.encode()))}
            mock_client.return_value.invoke_model = mock_invoke

            result = await parse_with_llm("Large explosion reported downtown Houston")

        assert result is not None
        assert isinstance(result, LLMParsedEvent)
        assert result.is_valid_anomaly is True
        assert result.confidence_score == 8
        assert result.approximate_origin == "Houston, TX"

    async def test_malformed_json_returns_none(self):
        """If Bedrock returns non-JSON, parse_with_llm should return None."""
        response_body = json.dumps({"content": [{"text": "This is not JSON at all"}]})

        with patch("processing.llm_parser._get_bedrock_client") as mock_client:
            mock_invoke = AsyncMock()
            mock_invoke.return_value = {"body": AsyncMock(read=AsyncMock(return_value=response_body.encode()))}
            mock_client.return_value.invoke_model = mock_invoke

            result = await parse_with_llm("some text")

        assert result is None

    async def test_api_error_returns_none(self):
        """If the Bedrock call raises an exception, return None gracefully."""
        with patch("processing.llm_parser._get_bedrock_client") as mock_client:
            mock_client.return_value.invoke_model = AsyncMock(
                side_effect=Exception("Bedrock throttled")
            )

            result = await parse_with_llm("some text")

        assert result is None

    async def test_system_prompt_exists(self):
        """The system prompt should contain key instructions."""
        assert "disaster-response" in SYSTEM_PROMPT.lower()
        assert "JSON" in SYSTEM_PROMPT
        assert "confidence_score" in SYSTEM_PROMPT
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/integration/test_llm_parser.py -v`
Expected: FAIL — `_get_bedrock_client` doesn't exist yet, module still imports `openai`.

- [ ] **Step 3: Rewrite llm_parser.py for Bedrock**

Replace the entire contents of `processing/llm_parser.py`:

```python
"""
LLM Triage Parser (Amazon Bedrock)
====================================

Takes unstructured text and passes it to Claude on Bedrock with a
disaster-response analyst system prompt. Returns a structured
LLMParsedEvent or None if the response is unparseable.
"""

from __future__ import annotations

import json
import logging

import aioboto3

from config import settings
from models import LLMParsedEvent

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are a disaster-response intelligence analyst embedded in a real-time
anomaly detection pipeline.  Your job is to read raw social-media posts,
emergency broadcast fragments, or sensor descriptions and determine whether
they describe a genuine aerospace debris re-entry, industrial explosion,
or other catastrophic thermal / kinetic event.

You MUST respond with ONLY a valid JSON object - no markdown fences, no
explanation text.  The JSON schema is:

{
  "is_valid_anomaly": <bool>,
  "approximate_origin": "<city, region, or lat/lon if mentioned>",
  "debris_trajectory_or_blast_radius": "<direction, radius, or 'unknown'>",
  "event_classification": "<debris_reentry | industrial_accident | meteor | military_activity | unknown>",
  "confidence_score": <int 1-10>
}

Rules:
- If the text is clearly a joke, meme, movie reference, or unrelated, set
  is_valid_anomaly=false and confidence_score=1.
- If the text is ambiguous but *could* be real, set is_valid_anomaly=true
  with a low confidence_score (3-5).
- Only assign confidence_score >= 7 when the text contains specific details
  like location, time, physical descriptions (flash, boom, heat, smoke).
"""

# Module-level client (lazy-initialized)
_client = None


async def _get_bedrock_client():
    """Get or create a Bedrock runtime client."""
    global _client
    if _client is None:
        session = aioboto3.Session()
        _client = await session.client(
            "bedrock-runtime",
            region_name=settings.aws_region,
        ).__aenter__()
    return _client


async def parse_with_llm(text: str) -> LLMParsedEvent | None:
    """
    Send text to Bedrock Claude and return a validated LLMParsedEvent,
    or None if the response is unparseable.
    """
    try:
        raw_json = await _call_bedrock(text)
        parsed = json.loads(raw_json)
        return LLMParsedEvent(**parsed)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.error("LLM returned unparseable response: %s", exc)
        return None
    except Exception:
        logger.exception("LLM call failed")
        return None


async def _call_bedrock(text: str) -> str:
    """Call Claude on Bedrock and return the raw response text."""
    client = await _get_bedrock_client()

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 1024,
        "temperature": 0.1,
        "system": SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": f"Analyze the following text:\n\n{text}"},
        ],
    })

    response = await client.invoke_model(
        modelId=settings.bedrock_model_id,
        body=body,
        contentType="application/json",
        accept="application/json",
    )

    response_bytes = await response["body"].read()
    response_json = json.loads(response_bytes)

    # Bedrock Claude response: {"content": [{"type": "text", "text": "..."}], ...}
    return response_json["content"][0]["text"]
```

- [ ] **Step 4: Run integration tests**

Run: `uv run pytest tests/integration/test_llm_parser.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add processing/llm_parser.py tests/integration/test_llm_parser.py
git commit -m "feat: rewrite LLM parser from OpenAI/Gemini to Amazon Bedrock"
```

---

### Task 5: Unit Tests — Models, Formatter, Physics, Predictor

**Files:**
- Create: `tests/unit/test_models.py`
- Create: `tests/unit/test_formatter.py`
- Create: `tests/unit/test_physics.py`
- Create: `tests/unit/test_predictor.py`

- [ ] **Step 1: Write model validation tests**

Create `tests/unit/test_models.py`:

```python
"""Unit tests for Pydantic data models."""

import pytest
from pydantic import ValidationError

from models import (
    CorrelatedEvent,
    EventClassification,
    EventSeverity,
    EventSource,
    LLMParsedEvent,
    RawEvent,
)


class TestRawEvent:
    def test_default_event_id_generated(self):
        event = RawEvent(source=EventSource.FIRMS, description="test")
        assert len(event.event_id) == 12

    def test_all_sources_valid(self):
        for source in EventSource:
            event = RawEvent(source=source, description="test")
            assert event.source == source


class TestLLMParsedEvent:
    def test_valid_confidence_bounds(self):
        event = LLMParsedEvent(
            is_valid_anomaly=True,
            confidence_score=1,
        )
        assert event.confidence_score == 1

        event = LLMParsedEvent(
            is_valid_anomaly=True,
            confidence_score=10,
        )
        assert event.confidence_score == 10

    def test_confidence_below_min_rejected(self):
        with pytest.raises(ValidationError):
            LLMParsedEvent(is_valid_anomaly=True, confidence_score=0)

    def test_confidence_above_max_rejected(self):
        with pytest.raises(ValidationError):
            LLMParsedEvent(is_valid_anomaly=True, confidence_score=11)


class TestCorrelatedEvent:
    def test_default_severity_is_low(self):
        event = CorrelatedEvent()
        assert event.severity == EventSeverity.LOW

    def test_default_classification_is_unknown(self):
        event = CorrelatedEvent()
        assert event.classification == EventClassification.UNKNOWN

    def test_contributing_events_list(self):
        raw = RawEvent(source=EventSource.FIRMS, description="test")
        event = CorrelatedEvent(contributing_events=[raw])
        assert len(event.contributing_events) == 1
```

- [ ] **Step 2: Write formatter tests**

Create `tests/unit/test_formatter.py`:

```python
"""Unit tests for the output formatter."""

from datetime import datetime, timezone

from models import (
    CorrelatedEvent,
    EventClassification,
    EventSeverity,
    EventSource,
    LLMParsedEvent,
    RawEvent,
)
from output.formatter import format_alert_payload


class TestFormatAlertPayload:
    def _make_correlated_event(self) -> CorrelatedEvent:
        raw = RawEvent(
            event_id="test123",
            source=EventSource.FIRMS,
            latitude=29.76,
            longitude=-95.37,
            description="Large thermal anomaly detected",
            timestamp=datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc),
        )
        llm = LLMParsedEvent(
            is_valid_anomaly=True,
            approximate_origin="Houston, TX",
            debris_trajectory_or_blast_radius="3 km radius",
            event_classification="industrial_accident",
            confidence_score=8,
        )
        return CorrelatedEvent(
            correlation_id="corr_abc",
            severity=EventSeverity.CRITICAL,
            classification=EventClassification.INDUSTRIAL_ACCIDENT,
            latitude=29.76,
            longitude=-95.37,
            contributing_events=[raw],
            llm_analysis=llm,
            summary="CRITICAL industrial accident near Houston",
            corroborating_sources=["firms", "adsb"],
            timestamp=datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc),
        )

    def test_returns_dict_with_alert_key(self):
        event = self._make_correlated_event()
        result = format_alert_payload(event)
        assert "alert" in result

    def test_alert_contains_required_fields(self):
        event = self._make_correlated_event()
        result = format_alert_payload(event)
        alert = result["alert"]
        assert alert["correlation_id"] == "corr_abc"
        assert alert["severity"] == "CRITICAL"
        assert alert["classification"] == "industrial_accident"
        assert "timestamp_utc" in alert
        assert "coordinates" in alert
        assert "summary" in alert
        assert "corroborating_sources" in alert

    def test_coordinates_present(self):
        event = self._make_correlated_event()
        result = format_alert_payload(event)
        coords = result["alert"]["coordinates"]
        assert coords["lat"] == 29.76
        assert coords["lon"] == -95.37

    def test_llm_analysis_present(self):
        event = self._make_correlated_event()
        result = format_alert_payload(event)
        llm = result["alert"]["llm_analysis"]
        assert llm["is_valid_anomaly"] is True
        assert llm["confidence_score"] == 8

    def test_contributing_events_included(self):
        event = self._make_correlated_event()
        result = format_alert_payload(event)
        contributing = result["alert"]["contributing_events"]
        assert len(contributing) == 1
        assert contributing[0]["event_id"] == "test123"

    def test_no_coordinates_returns_none(self):
        event = CorrelatedEvent(summary="test")
        result = format_alert_payload(event)
        assert result["alert"]["coordinates"] is None

    def test_no_llm_analysis_returns_none(self):
        event = CorrelatedEvent(summary="test")
        result = format_alert_payload(event)
        assert result["alert"]["llm_analysis"] is None
```

- [ ] **Step 3: Write physics tests**

Create `tests/unit/test_physics.py`:

```python
"""Unit tests for trajectory physics utilities."""

import math

import pytest

from trajectory.physics import (
    GRAVITY,
    RHO_0,
    air_density,
    drag_acceleration,
    ecef_to_geodetic,
    enu_to_geodetic,
    geodetic_to_ecef,
    geodetic_to_enu,
    gravity_acceleration,
)


class TestGeodeticToECEF:
    def test_equator_prime_meridian(self):
        """(0, 0, 0) should give x ≈ WGS84_A, y ≈ 0, z ≈ 0."""
        x, y, z = geodetic_to_ecef(0.0, 0.0, 0.0)
        assert abs(x - 6_378_137.0) < 1.0
        assert abs(y) < 1.0
        assert abs(z) < 1.0

    def test_north_pole(self):
        """(90, 0, 0) should give x ≈ 0, y ≈ 0, z ≈ WGS84_B."""
        x, y, z = geodetic_to_ecef(90.0, 0.0, 0.0)
        assert abs(x) < 1.0
        assert abs(y) < 1.0
        assert abs(z - 6_356_752.3) < 1.0


class TestRoundTrip:
    def test_geodetic_ecef_roundtrip(self):
        """Geodetic → ECEF → Geodetic should recover original coordinates."""
        lat, lon, alt = 29.76, -95.37, 100.0
        x, y, z = geodetic_to_ecef(lat, lon, alt)
        lat2, lon2, alt2 = ecef_to_geodetic(x, y, z)
        assert abs(lat2 - lat) < 1e-6
        assert abs(lon2 - lon) < 1e-6
        assert abs(alt2 - alt) < 0.1

    def test_enu_roundtrip(self):
        """Geodetic → ENU → Geodetic should recover original coordinates."""
        ref_lat, ref_lon, ref_alt = 29.76, -95.37, 0.0
        target_lat, target_lon, target_alt = 29.77, -95.36, 500.0

        e, n, u = geodetic_to_enu(
            target_lat, target_lon, target_alt,
            ref_lat, ref_lon, ref_alt,
        )
        lat2, lon2, alt2 = enu_to_geodetic(
            e, n, u,
            ref_lat, ref_lon, ref_alt,
        )
        assert abs(lat2 - target_lat) < 1e-5
        assert abs(lon2 - target_lon) < 1e-5
        assert abs(alt2 - target_alt) < 1.0


class TestAirDensity:
    def test_sea_level(self):
        assert air_density(0.0) == pytest.approx(RHO_0)

    def test_decreases_with_altitude(self):
        assert air_density(5000.0) < air_density(0.0)
        assert air_density(10000.0) < air_density(5000.0)

    def test_below_sea_level_returns_rho0(self):
        assert air_density(-100.0) == RHO_0

    def test_above_karman_line_returns_zero(self):
        assert air_density(100_001.0) == 0.0

    def test_scale_height(self):
        """At one scale height, density should be ≈ RHO_0 / e."""
        rho = air_density(8500.0)
        assert rho == pytest.approx(RHO_0 / math.e, rel=0.01)


class TestDragAcceleration:
    def test_zero_velocity_no_drag(self):
        ax, ay, az = drag_acceleration(0.0, 0.0, 0.0, 5000.0, 50.0)
        assert ax == 0.0
        assert ay == 0.0
        assert az == 0.0

    def test_drag_opposes_motion(self):
        ax, ay, az = drag_acceleration(100.0, 0.0, 0.0, 5000.0, 50.0)
        assert ax < 0  # opposing positive vx
        assert ay == 0.0

    def test_higher_speed_more_drag(self):
        ax1, _, _ = drag_acceleration(100.0, 0.0, 0.0, 5000.0, 50.0)
        ax2, _, _ = drag_acceleration(200.0, 0.0, 0.0, 5000.0, 50.0)
        assert abs(ax2) > abs(ax1)


class TestGravity:
    def test_gravity_downward_in_enu(self):
        gx, gy, gz = gravity_acceleration()
        assert gx == 0.0
        assert gy == 0.0
        assert gz == pytest.approx(-GRAVITY)
```

- [ ] **Step 4: Write predictor tests**

Create `tests/unit/test_predictor.py`:

```python
"""Unit tests for the EKF trajectory predictor."""

from datetime import datetime, timedelta, timezone

import pytest

from trajectory.models import ImpactPrediction, SensorObservation, TrajectoryRequest
from trajectory.predictor import DebrisTrajectoryPredictor


def _make_observations(
    start_lat: float = 30.0,
    start_lon: float = -95.0,
    start_alt: float = 50_000.0,
    count: int = 5,
    interval_sec: float = 10.0,
    descent_rate: float = 2000.0,
    south_drift: float = 0.01,
) -> list[SensorObservation]:
    """Generate a series of synthetic observations for a descending object."""
    base_time = datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc)
    obs = []
    for i in range(count):
        obs.append(
            SensorObservation(
                timestamp=base_time + timedelta(seconds=i * interval_sec),
                latitude=start_lat - i * south_drift,
                longitude=start_lon,
                altitude_m=start_alt - i * descent_rate,
                noise_profile="satellite",
            )
        )
    return obs


class TestDebrisTrajectoryPredictor:
    def test_returns_impact_prediction(self):
        obs = _make_observations()
        request = TrajectoryRequest(
            object_id="test-debris",
            observations=obs,
            ballistic_coefficient=50.0,
        )
        predictor = DebrisTrajectoryPredictor(ballistic_coefficient=50.0)
        result = predictor.predict(request)

        assert isinstance(result, ImpactPrediction)
        assert result.object_id == "test-debris"

    def test_impact_altitude_near_zero(self):
        obs = _make_observations()
        request = TrajectoryRequest(observations=obs)
        predictor = DebrisTrajectoryPredictor()
        result = predictor.predict(request)

        assert result.impact_altitude_m < 500  # should be near ground

    def test_impact_time_in_future(self):
        obs = _make_observations()
        request = TrajectoryRequest(observations=obs)
        predictor = DebrisTrajectoryPredictor()
        result = predictor.predict(request)

        assert result.seconds_until_impact > 0

    def test_terminal_velocity_positive(self):
        obs = _make_observations()
        request = TrajectoryRequest(observations=obs)
        predictor = DebrisTrajectoryPredictor()
        result = predictor.predict(request)

        assert result.terminal_velocity_m_s > 0

    def test_covariance_matrix_shape(self):
        obs = _make_observations()
        request = TrajectoryRequest(observations=obs)
        predictor = DebrisTrajectoryPredictor()
        result = predictor.predict(request)

        cov = result.covariance_position_enu
        assert len(cov) == 3
        assert all(len(row) == 3 for row in cov)

    def test_trajectory_points_populated(self):
        obs = _make_observations()
        request = TrajectoryRequest(observations=obs)
        predictor = DebrisTrajectoryPredictor()
        result = predictor.predict(request)

        assert len(result.trajectory_points) > 0
        point = result.trajectory_points[0]
        assert "lat" in point
        assert "lon" in point
        assert "alt_m" in point

    def test_higher_ballistic_coeff_travels_further(self):
        """Higher β (less drag) should result in a more distant impact."""
        obs = _make_observations()

        predictor_low = DebrisTrajectoryPredictor(ballistic_coefficient=30.0)
        result_low = predictor_low.predict(TrajectoryRequest(observations=obs, ballistic_coefficient=30.0))

        predictor_high = DebrisTrajectoryPredictor(ballistic_coefficient=150.0)
        result_high = predictor_high.predict(TrajectoryRequest(observations=obs, ballistic_coefficient=150.0))

        # Higher β → less drag → higher terminal velocity
        assert result_high.terminal_velocity_m_s > result_low.terminal_velocity_m_s
```

- [ ] **Step 5: Run all unit tests**

Run: `uv run pytest tests/unit/ -v`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add tests/unit/test_models.py tests/unit/test_formatter.py tests/unit/test_physics.py tests/unit/test_predictor.py
git commit -m "test: add unit tests for models, formatter, physics, and predictor"
```

---

### Task 6: Integration Tests — Webhook and Alerter

**Files:**
- Create: `tests/integration/test_emergency_webhook.py`
- Create: `tests/integration/test_alerter.py`

- [ ] **Step 1: Write webhook integration tests**

Create `tests/integration/test_emergency_webhook.py`:

```python
"""Integration tests for the emergency webhook endpoint."""

import asyncio

import pytest
from fastapi.testclient import TestClient

from ingestion.emergency_webhook import app, set_event_queue
from models import EventSource, RawEvent


@pytest.fixture
def webhook_client():
    """Create a test client with an injected event queue."""
    queue: asyncio.Queue[RawEvent] = asyncio.Queue()
    set_event_queue(queue)
    client = TestClient(app)
    yield client, queue
    set_event_queue(None)


class TestEmergencyWebhook:
    def test_valid_payload_accepted(self, webhook_client):
        client, queue = webhook_client
        response = client.post(
            "/api/v1/emergency",
            json={
                "source_system": "NWS",
                "event_type": "explosion",
                "latitude": 29.76,
                "longitude": -95.37,
                "description": "Large fireball reported",
            },
        )
        assert response.status_code == 202
        body = response.json()
        assert body["status"] == "accepted"
        assert "event_id" in body

    def test_event_pushed_to_queue(self, webhook_client):
        client, queue = webhook_client
        client.post(
            "/api/v1/emergency",
            json={
                "source_system": "test",
                "event_type": "test_event",
                "description": "test",
            },
        )
        assert not queue.empty()
        event = queue.get_nowait()
        assert event.source == EventSource.EMERGENCY_WEBHOOK

    def test_missing_required_field_rejected(self, webhook_client):
        client, _ = webhook_client
        response = client.post(
            "/api/v1/emergency",
            json={"description": "missing source_system and event_type"},
        )
        assert response.status_code == 422

    def test_queue_not_initialized_returns_503(self):
        set_event_queue(None)
        client = TestClient(app)
        response = client.post(
            "/api/v1/emergency",
            json={
                "source_system": "test",
                "event_type": "test",
            },
        )
        assert response.status_code == 503

    def test_health_endpoint(self, webhook_client):
        client, _ = webhook_client
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
```

- [ ] **Step 2: Write alerter integration tests**

Create `tests/integration/test_alerter.py`:

```python
"""Integration tests for the webhook alerter using respx."""

import asyncio
from datetime import datetime, timezone

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


class TestAlerter:
    @respx.mock
    async def test_slack_webhook_called(self):
        """Alert loop should POST to Slack when a CRITICAL event is queued."""
        slack_url = "https://hooks.slack.com/services/test"
        slack_route = respx.post(slack_url).mock(return_value=httpx.Response(200))

        from unittest.mock import patch

        with patch("output.alerter.settings") as mock_settings:
            mock_settings.slack_webhook_url = slack_url
            mock_settings.discord_webhook_url = ""

            queue: asyncio.Queue[CorrelatedEvent] = asyncio.Queue()
            await queue.put(_make_critical_event())

            # Run the alert loop with a timeout so it doesn't hang
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
        """Alert loop should POST to Discord when a CRITICAL event is queued."""
        discord_url = "https://discord.com/api/webhooks/test"
        discord_route = respx.post(discord_url).mock(return_value=httpx.Response(200))

        from unittest.mock import patch

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
        """Non-CRITICAL events should not trigger webhooks."""
        slack_url = "https://hooks.slack.com/services/test"
        slack_route = respx.post(slack_url).mock(return_value=httpx.Response(200))

        from unittest.mock import patch

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
```

- [ ] **Step 3: Run integration tests**

Run: `uv run pytest tests/integration/test_emergency_webhook.py tests/integration/test_alerter.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_emergency_webhook.py tests/integration/test_alerter.py
git commit -m "test: add integration tests for webhook endpoint and alerter"
```

---

### Task 7: End-to-End Pipeline Test

**Files:**
- Create: `tests/e2e/__init__.py`
- Create: `tests/e2e/test_pipeline.py`

- [ ] **Step 1: Write e2e pipeline test**

Create `tests/e2e/__init__.py` (empty file) and `tests/e2e/test_pipeline.py`:

```python
"""End-to-end test: full pipeline from ingestion to alert output."""

import asyncio
import json
import os
from datetime import datetime, timezone
from unittest.mock import patch, AsyncMock

import boto3
import pytest
import respx
import httpx
from moto import mock_aws

from models import (
    CorrelatedEvent,
    EventSeverity,
    EventSource,
    LLMParsedEvent,
    RawEvent,
)
from processing.correlation_engine import CorrelationEngine
from output.alerter import alert_loop
from output.formatter import format_alert_payload


class TestFullPipeline:
    """
    Simulates the full pipeline:
    1. Ingest events from two different sources in the same area
    2. LLM parser returns a high-confidence result
    3. Correlation engine elevates to CRITICAL
    4. Alerter dispatches to Slack webhook
    """

    @respx.mock
    async def test_multi_source_corroboration_triggers_alert(self):
        slack_url = "https://hooks.slack.com/services/e2e-test"
        slack_route = respx.post(slack_url).mock(return_value=httpx.Response(200))

        # Mock Bedrock LLM response
        llm_response = {
            "is_valid_anomaly": True,
            "approximate_origin": "Houston, TX area",
            "debris_trajectory_or_blast_radius": "NW to SE, ~3 km radius",
            "event_classification": "industrial_accident",
            "confidence_score": 9,
        }
        bedrock_response = json.dumps(
            {"content": [{"text": json.dumps(llm_response)}]}
        )

        with mock_aws():
            # Setup DynamoDB
            os.environ["AWS_ACCESS_KEY_ID"] = "testing"
            os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
            os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

            ddb = boto3.client("dynamodb", region_name="us-east-1")
            ddb.create_table(
                TableName="skyfall-events",
                KeySchema=[
                    {"AttributeName": "pk", "KeyType": "HASH"},
                    {"AttributeName": "sk", "KeyType": "RANGE"},
                ],
                AttributeDefinitions=[
                    {"AttributeName": "pk", "AttributeType": "S"},
                    {"AttributeName": "sk", "AttributeType": "S"},
                ],
                BillingMode="PAY_PER_REQUEST",
            )

            with patch("processing.correlation_engine.settings") as mock_ce_settings, \
                 patch("processing.llm_parser._get_bedrock_client") as mock_bedrock, \
                 patch("output.alerter.settings") as mock_alert_settings:

                # Configure correlation engine settings
                mock_ce_settings.aws_region = "us-east-1"
                mock_ce_settings.dynamodb_table_name = "skyfall-events"
                mock_ce_settings.dynamodb_endpoint_url = ""
                mock_ce_settings.correlation_window_sec = 300
                mock_ce_settings.min_confidence_score = 6
                mock_ce_settings.geohash_precision = 4

                # Configure alerter settings
                mock_alert_settings.slack_webhook_url = slack_url
                mock_alert_settings.discord_webhook_url = ""

                # Configure Bedrock mock
                mock_invoke = AsyncMock()
                mock_invoke.return_value = {
                    "body": AsyncMock(
                        read=AsyncMock(return_value=bedrock_response.encode())
                    )
                }
                mock_bedrock.return_value.invoke_model = mock_invoke

                # --- Pipeline execution ---

                # 1. Setup engine
                engine = CorrelationEngine()
                await engine.connect()

                # 2. Ingest FIRMS event (structured sensor)
                firms_event = RawEvent(
                    event_id="e2e_firms",
                    source=EventSource.FIRMS,
                    latitude=29.76,
                    longitude=-95.37,
                    description="VIIRS thermal anomaly FRP=800MW",
                    timestamp=datetime(2026, 4, 1, 12, 0, 0, tzinfo=timezone.utc),
                    raw_payload={"frp": 800, "brightness": 450},
                )
                await engine.ingest(firms_event)

                # 3. Ingest emergency webhook event (needs LLM)
                webhook_event = RawEvent(
                    event_id="e2e_webhook",
                    source=EventSource.EMERGENCY_WEBHOOK,
                    latitude=29.761,
                    longitude=-95.371,
                    description="Large fireball reported in Houston industrial district",
                    timestamp=datetime(2026, 4, 1, 12, 1, 0, tzinfo=timezone.utc),
                    raw_payload={"text": "Large fireball reported in Houston industrial district"},
                )
                await engine.ingest(webhook_event)

                # 4. LLM parses the webhook event
                from processing.llm_parser import parse_with_llm

                llm_result = await parse_with_llm(webhook_event.description)
                assert llm_result is not None
                assert llm_result.is_valid_anomaly is True

                # 5. Correlate — should find FIRMS event nearby
                alert_queue: asyncio.Queue[CorrelatedEvent] = asyncio.Queue()
                correlated = await engine.try_correlate(
                    webhook_event, llm_result, output_queue=alert_queue
                )

                assert correlated is not None
                assert correlated.severity == EventSeverity.CRITICAL
                assert len(correlated.corroborating_sources) >= 2
                assert not alert_queue.empty()

                # 6. Alerter sends to Slack
                task = asyncio.create_task(alert_loop(alert_queue))
                await asyncio.sleep(0.5)
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

                assert slack_route.called

                await engine.close()

            # Cleanup env
            for key in ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION"]:
                os.environ.pop(key, None)
```

- [ ] **Step 2: Run e2e test**

Run: `uv run pytest tests/e2e/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest -v`
Expected: All tests across unit, integration, and e2e PASS

- [ ] **Step 4: Commit**

```bash
git add tests/e2e/__init__.py tests/e2e/test_pipeline.py
git commit -m "test: add end-to-end pipeline test"
```

---

### Task 8: Operational Hardening

**Files:**
- Modify: `main.py`
- Modify: `ingestion/emergency_webhook.py`

- [ ] **Step 1: Add structlog configuration**

Create a logging setup that replaces `logging.basicConfig` in `main.py`. Add this near the top of the file, replacing the existing logging setup:

```python
import signal
import structlog

# -- Structured logging --
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
```

Add `log_format: str = "json"` to the Settings class in `config.py`.

- [ ] **Step 2: Enhance health check endpoint**

Update `ingestion/emergency_webhook.py` — replace the `/health` endpoint:

```python
@app.get("/health")
async def health():
    """
    Readiness probe for load balancers / container orchestrators.
    Checks that critical dependencies are reachable.
    """
    checks = {"queue": _event_queue is not None}
    all_ok = all(checks.values())
    return {
        "status": "ok" if all_ok else "degraded",
        "checks": checks,
    }
```

- [ ] **Step 3: Add graceful shutdown to main.py**

Add signal handling to the `main()` function. Replace the existing `main()` function:

```python
async def main() -> None:
    """Wire everything together and run forever."""

    # -- Shared queues --
    raw_queue: asyncio.Queue[RawEvent] = asyncio.Queue(maxsize=10_000)
    alert_queue: asyncio.Queue[CorrelatedEvent] = asyncio.Queue(maxsize=1_000)

    set_event_queue(raw_queue)

    # -- Correlation engine --
    engine = CorrelationEngine()
    await engine.connect()

    # -- Launch all tasks concurrently --
    tasks = [
        asyncio.create_task(poll_firms(raw_queue), name="firms_poller"),
        asyncio.create_task(poll_adsb(raw_queue), name="adsb_poller"),
        asyncio.create_task(listen_telegram(raw_queue), name="telegram_listener"),
        asyncio.create_task(listen_generic_scraper(raw_queue), name="generic_scraper"),
        asyncio.create_task(run_webhook_server(), name="webhook_server"),
        asyncio.create_task(triage_loop(raw_queue, alert_queue, engine), name="triage"),
        asyncio.create_task(alert_loop(alert_queue), name="alerter"),
    ]

    logger.info("All tasks launched", task_count=len(tasks))

    # -- Graceful shutdown on SIGTERM (ECS sends this on deploy/scale-down) --
    shutdown_event = asyncio.Event()

    def _handle_signal(sig, frame):
        logger.info("Received signal, initiating graceful shutdown", signal=sig)
        shutdown_event.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    # Run until a task crashes or shutdown is requested
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

    # Cancel remaining tasks
    for task in pending:
        task.cancel()

    # Wait for queue drain
    logger.info("Draining queues...")
    try:
        await asyncio.wait_for(raw_queue.join(), timeout=10.0)
        await asyncio.wait_for(alert_queue.join(), timeout=10.0)
    except asyncio.TimeoutError:
        logger.warning("Queue drain timed out")

    await engine.close()
    logger.info("Shutdown complete")
```

- [ ] **Step 4: Update imports in main.py**

Update the imports at the top of `main.py`:

```python
from __future__ import annotations

import asyncio
import logging
import signal
import sys

import structlog
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
```

- [ ] **Step 5: Commit**

```bash
git add main.py config.py ingestion/emergency_webhook.py
git commit -m "feat: add structured logging, health checks, and graceful shutdown"
```

---

### Task 9: Update Dockerfile for uv

**Files:**
- Modify: `Dockerfile`

- [ ] **Step 1: Rewrite Dockerfile**

Replace the entire `Dockerfile`:

```dockerfile
FROM python:3.12-slim AS base

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Install dependencies first (cache-friendly)
COPY pyproject.toml ./
RUN uv sync --no-dev --no-install-project

# Copy application code
COPY . .

# Install the project itself
RUN uv sync --no-dev

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["uv", "run", "python", "-c", "import httpx; r = httpx.get('http://localhost:8000/health'); r.raise_for_status()"]

CMD ["uv", "run", "python", "main.py"]
```

- [ ] **Step 2: Commit**

```bash
git add Dockerfile
git commit -m "chore: update Dockerfile to use uv instead of pip"
```

---

### Task 10: CloudFormation Template

**Files:**
- Create: `infra/template.yaml`

- [ ] **Step 1: Write CloudFormation template**

Create `infra/template.yaml`:

```yaml
AWSTemplateFormatVersion: "2010-09-09"
Description: "Skyfall - Aerospace Debris & Industrial Anomaly Tracker"

Parameters:
  VpcId:
    Type: AWS::EC2::VPC::Id
    Description: VPC to deploy into

  SubnetIds:
    Type: List<AWS::EC2::Subnet::Id>
    Description: Subnets for ECS tasks and ALB (must be in at least 2 AZs)

  ImageUri:
    Type: String
    Description: ECR image URI (e.g. 123456789.dkr.ecr.us-east-1.amazonaws.com/skyfall:latest)

  FirmsApiKey:
    Type: String
    NoEcho: true
    Default: ""

  AdsbApiKey:
    Type: String
    NoEcho: true
    Default: ""

  SlackWebhookUrl:
    Type: String
    NoEcho: true
    Default: ""

  DiscordWebhookUrl:
    Type: String
    NoEcho: true
    Default: ""

Resources:
  # -- DynamoDB --
  EventsTable:
    Type: AWS::DynamoDB::Table
    Properties:
      TableName: skyfall-events
      BillingMode: PAY_PER_REQUEST
      AttributeDefinitions:
        - AttributeName: pk
          AttributeType: S
        - AttributeName: sk
          AttributeType: S
      KeySchema:
        - AttributeName: pk
          KeyType: HASH
        - AttributeName: sk
          KeyType: RANGE
      TimeToLiveSpecification:
        AttributeName: expires_at
        Enabled: true

  # -- ECR Repository --
  ECRRepository:
    Type: AWS::ECR::Repository
    Properties:
      RepositoryName: skyfall
      ImageScanningConfiguration:
        ScanOnPush: true

  # -- CloudWatch Logs --
  LogGroup:
    Type: AWS::Logs::LogGroup
    Properties:
      LogGroupName: /ecs/skyfall
      RetentionInDays: 30

  # -- Secrets --
  AppSecrets:
    Type: AWS::SecretsManager::Secret
    Properties:
      Name: skyfall/app-secrets
      SecretString: !Sub |
        {
          "FIRMS_API_KEY": "${FirmsApiKey}",
          "ADSB_API_KEY": "${AdsbApiKey}",
          "SLACK_WEBHOOK_URL": "${SlackWebhookUrl}",
          "DISCORD_WEBHOOK_URL": "${DiscordWebhookUrl}"
        }

  # -- IAM --
  TaskExecutionRole:
    Type: AWS::IAM::Role
    Properties:
      RoleName: skyfall-task-execution
      AssumeRolePolicyDocument:
        Version: "2012-10-17"
        Statement:
          - Effect: Allow
            Principal:
              Service: ecs-tasks.amazonaws.com
            Action: sts:AssumeRole
      ManagedPolicyArns:
        - arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy
      Policies:
        - PolicyName: secrets-access
          PolicyDocument:
            Version: "2012-10-17"
            Statement:
              - Effect: Allow
                Action:
                  - secretsmanager:GetSecretValue
                Resource: !Ref AppSecrets

  TaskRole:
    Type: AWS::IAM::Role
    Properties:
      RoleName: skyfall-task
      AssumeRolePolicyDocument:
        Version: "2012-10-17"
        Statement:
          - Effect: Allow
            Principal:
              Service: ecs-tasks.amazonaws.com
            Action: sts:AssumeRole
      Policies:
        - PolicyName: bedrock-access
          PolicyDocument:
            Version: "2012-10-17"
            Statement:
              - Effect: Allow
                Action:
                  - bedrock:InvokeModel
                Resource: "*"
        - PolicyName: dynamodb-access
          PolicyDocument:
            Version: "2012-10-17"
            Statement:
              - Effect: Allow
                Action:
                  - dynamodb:PutItem
                  - dynamodb:GetItem
                  - dynamodb:Query
                  - dynamodb:DeleteItem
                Resource: !GetAtt EventsTable.Arn

  # -- Security Group --
  ServiceSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Skyfall ECS service
      VpcId: !Ref VpcId
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 8000
          ToPort: 8000
          SourceSecurityGroupId: !Ref ALBSecurityGroup

  ALBSecurityGroup:
    Type: AWS::EC2::SecurityGroup
    Properties:
      GroupDescription: Skyfall ALB
      VpcId: !Ref VpcId
      SecurityGroupIngress:
        - IpProtocol: tcp
          FromPort: 80
          ToPort: 80
          CidrIp: 0.0.0.0/0

  # -- ALB --
  ALB:
    Type: AWS::ElasticLoadBalancingV2::LoadBalancer
    Properties:
      Name: skyfall-alb
      Scheme: internet-facing
      SecurityGroups:
        - !Ref ALBSecurityGroup
      Subnets: !Ref SubnetIds

  TargetGroup:
    Type: AWS::ElasticLoadBalancingV2::TargetGroup
    Properties:
      Name: skyfall-tg
      Port: 8000
      Protocol: HTTP
      TargetType: ip
      VpcId: !Ref VpcId
      HealthCheckPath: /health
      HealthCheckIntervalSeconds: 30
      HealthyThresholdCount: 2
      UnhealthyThresholdCount: 3

  Listener:
    Type: AWS::ElasticLoadBalancingV2::Listener
    Properties:
      LoadBalancerArn: !Ref ALB
      Port: 80
      Protocol: HTTP
      DefaultActions:
        - Type: forward
          TargetGroupArn: !Ref TargetGroup

  # -- ECS --
  ECSCluster:
    Type: AWS::ECS::Cluster
    Properties:
      ClusterName: skyfall
      CapacityProviders:
        - FARGATE

  TaskDefinition:
    Type: AWS::ECS::TaskDefinition
    Properties:
      Family: skyfall
      NetworkMode: awsvpc
      RequiresCompatibilities:
        - FARGATE
      Cpu: "512"
      Memory: "1024"
      ExecutionRoleArn: !GetAtt TaskExecutionRole.Arn
      TaskRoleArn: !GetAtt TaskRole.Arn
      ContainerDefinitions:
        - Name: skyfall
          Image: !Ref ImageUri
          Essential: true
          PortMappings:
            - ContainerPort: 8000
              Protocol: tcp
          Environment:
            - Name: AWS_REGION
              Value: !Ref "AWS::Region"
            - Name: DYNAMODB_TABLE_NAME
              Value: !Ref EventsTable
            - Name: LOG_FORMAT
              Value: json
          Secrets:
            - Name: FIRMS_API_KEY
              ValueFrom: !Sub "${AppSecrets}:FIRMS_API_KEY::"
            - Name: ADSB_API_KEY
              ValueFrom: !Sub "${AppSecrets}:ADSB_API_KEY::"
            - Name: SLACK_WEBHOOK_URL
              ValueFrom: !Sub "${AppSecrets}:SLACK_WEBHOOK_URL::"
            - Name: DISCORD_WEBHOOK_URL
              ValueFrom: !Sub "${AppSecrets}:DISCORD_WEBHOOK_URL::"
          LogConfiguration:
            LogDriver: awslogs
            Options:
              awslogs-group: !Ref LogGroup
              awslogs-region: !Ref "AWS::Region"
              awslogs-stream-prefix: skyfall

  ECSService:
    Type: AWS::ECS::Service
    DependsOn: Listener
    Properties:
      ServiceName: skyfall
      Cluster: !Ref ECSCluster
      TaskDefinition: !Ref TaskDefinition
      DesiredCount: 1
      LaunchType: FARGATE
      NetworkConfiguration:
        AwsvpcConfiguration:
          AssignPublicIp: ENABLED
          Subnets: !Ref SubnetIds
          SecurityGroups:
            - !Ref ServiceSecurityGroup
      LoadBalancers:
        - ContainerName: skyfall
          ContainerPort: 8000
          TargetGroupArn: !Ref TargetGroup

Outputs:
  WebhookEndpoint:
    Description: Emergency webhook URL
    Value: !Sub "http://${ALB.DNSName}/api/v1/emergency"

  DynamoDBTable:
    Description: DynamoDB table name
    Value: !Ref EventsTable

  ECRRepository:
    Description: ECR repository URI
    Value: !GetAtt ECRRepository.RepositoryUri
```

- [ ] **Step 2: Commit**

```bash
git add infra/template.yaml
git commit -m "infra: add CloudFormation template for ECS/Fargate deployment"
```

---

### Task 11: Final Cleanup and Full Test Run

**Files:**
- Modify: `docker-compose.yml` (update for DynamoDB Local)

- [ ] **Step 1: Update docker-compose.yml for local development**

Replace the contents of `docker-compose.yml`:

```yaml
version: "3.9"

services:
  dynamodb-local:
    image: amazon/dynamodb-local:latest
    ports:
      - "8001:8000"
    command: "-jar DynamoDBLocal.jar -inMemory"
    healthcheck:
      test: ["CMD-SHELL", "curl -sf http://localhost:8000 || exit 1"]
      interval: 5s
      timeout: 3s
      retries: 5

  setup-dynamodb:
    image: amazon/aws-cli:latest
    depends_on:
      dynamodb-local:
        condition: service_healthy
    environment:
      AWS_ACCESS_KEY_ID: local
      AWS_SECRET_ACCESS_KEY: local
      AWS_DEFAULT_REGION: us-east-1
    entrypoint: >
      aws dynamodb create-table
        --endpoint-url http://dynamodb-local:8000
        --table-name skyfall-events
        --key-schema
          AttributeName=pk,KeyType=HASH
          AttributeName=sk,KeyType=RANGE
        --attribute-definitions
          AttributeName=pk,AttributeType=S
          AttributeName=sk,AttributeType=S
        --billing-mode PAY_PER_REQUEST

  tracker:
    build: .
    depends_on:
      setup-dynamodb:
        condition: service_completed_successfully
    ports:
      - "8000:8000"
    env_file: .env
    environment:
      DYNAMODB_ENDPOINT_URL: http://dynamodb-local:8000
      DYNAMODB_TABLE_NAME: skyfall-events
      AWS_ACCESS_KEY_ID: local
      AWS_SECRET_ACCESS_KEY: local
      AWS_REGION: us-east-1
```

- [ ] **Step 2: Run the full test suite**

Run: `uv run pytest -v --tb=short`
Expected: All tests PASS (unit + integration + e2e)

- [ ] **Step 3: Commit**

```bash
git add docker-compose.yml
git commit -m "chore: update docker-compose for DynamoDB Local, remove Redis"
```
