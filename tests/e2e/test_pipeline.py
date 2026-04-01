"""End-to-end pipeline test: ingestion -> correlation -> LLM -> alerting."""

import asyncio
import json
import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import boto3
import httpx
import pytest
import respx
from moto import mock_aws

from models import CorrelatedEvent, EventSeverity, EventSource, RawEvent
from processing.correlation_engine import CorrelationEngine
from processing.llm_parser import parse_with_llm
from output.alerter import alert_loop


HOUSTON_LAT = 29.76
HOUSTON_LON = -95.37
SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/TEST/E2E/pipeline"


def _make_settings_mock():
    """Build a mock settings object with all required attributes."""
    s = MagicMock()
    s.dynamodb_table_name = "skyfall-events"
    s.aws_region = "us-east-1"
    s.dynamodb_endpoint_url = ""
    s.correlation_window_sec = 300
    s.min_confidence_score = 6
    s.geohash_precision = 4
    s.slack_webhook_url = SLACK_WEBHOOK_URL
    s.discord_webhook_url = ""
    s.bedrock_model_id = "anthropic.claude-sonnet-4-20250514"
    return s


class TestFullPipeline:
    """Exercises the entire pipeline from ingestion through Slack alert."""

    @respx.mock
    @pytest.mark.asyncio
    async def test_full_pipeline_critical_alert(self):
        # -- 0. Environment setup ------------------------------------------------
        os.environ["AWS_ACCESS_KEY_ID"] = "testing"
        os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
        os.environ["AWS_SECURITY_TOKEN"] = "testing"
        os.environ["AWS_SESSION_TOKEN"] = "testing"
        os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

        try:
            with mock_aws():
                # -- 1. Create DynamoDB table ------------------------------------
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

                # -- 2. Prepare mocked Bedrock response --------------------------
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

                mock_bedrock_client = AsyncMock()
                body_mock = AsyncMock()
                body_mock.read = AsyncMock(return_value=bedrock_response.encode())
                mock_bedrock_client.invoke_model = AsyncMock(
                    return_value={"body": body_mock}
                )

                # -- 3. Set up respx to capture Slack webhook calls --------------
                slack_route = respx.post(SLACK_WEBHOOK_URL).mock(
                    return_value=httpx.Response(200, text="ok")
                )

                # -- 4. Reset global LLM client and patch everything -------------
                import processing.llm_parser as llm_mod
                llm_mod._client = None

                mock_settings = _make_settings_mock()

                with (
                    patch(
                        "processing.correlation_engine.settings",
                        mock_settings,
                    ),
                    patch(
                        "output.alerter.settings",
                        mock_settings,
                    ),
                    patch(
                        "processing.llm_parser._get_bedrock_client",
                        new_callable=AsyncMock,
                        return_value=mock_bedrock_client,
                    ),
                ):
                    # -- 5. Create and connect engine ----------------------------
                    engine = CorrelationEngine()
                    await engine.connect()

                    try:
                        # -- 6. Ingest FIRMS event (structured sensor) -----------
                        firms_event = RawEvent(
                            source=EventSource.FIRMS,
                            latitude=HOUSTON_LAT,
                            longitude=HOUSTON_LON,
                            description="FIRMS thermal anomaly detected near Houston",
                            timestamp=datetime.now(timezone.utc),
                            raw_payload={"brightness": 400, "confidence": "high"},
                        )
                        await engine.ingest(firms_event)

                        # -- 7. Ingest EMERGENCY_WEBHOOK event nearby ------------
                        webhook_event = RawEvent(
                            source=EventSource.EMERGENCY_WEBHOOK,
                            latitude=HOUSTON_LAT + 0.01,
                            longitude=HOUSTON_LON + 0.01,
                            description="Large explosion reported near Houston Ship Channel, multiple fires visible",
                            timestamp=datetime.now(timezone.utc),
                            raw_payload={"reporter": "Houston FD dispatch"},
                        )
                        await engine.ingest(webhook_event)

                        # -- 8. Parse webhook event description with LLM --------
                        llm_result = await parse_with_llm(
                            webhook_event.description
                        )
                        assert llm_result is not None, "LLM parser returned None"
                        assert llm_result.is_valid_anomaly is True
                        assert llm_result.confidence_score == 9

                        # -- 9. Correlate ----------------------------------------
                        alert_queue: asyncio.Queue[CorrelatedEvent] = (
                            asyncio.Queue()
                        )
                        result = await engine.try_correlate(
                            webhook_event,
                            llm_result,
                            output_queue=alert_queue,
                        )

                        assert result is not None, "Correlation returned None"
                        assert result.severity == EventSeverity.CRITICAL
                        assert len(result.corroborating_sources) >= 2
                        assert not alert_queue.empty()

                        # -- 10. Run alert_loop briefly --------------------------
                        task = asyncio.create_task(alert_loop(alert_queue))
                        await asyncio.sleep(0.5)
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

                        # -- 11. Assert Slack webhook was called -----------------
                        assert (
                            slack_route.called
                        ), "Slack webhook was never called"

                    finally:
                        await engine.close()

        finally:
            # Clean up AWS env vars
            for key in [
                "AWS_ACCESS_KEY_ID",
                "AWS_SECRET_ACCESS_KEY",
                "AWS_SECURITY_TOKEN",
                "AWS_SESSION_TOKEN",
                "AWS_DEFAULT_REGION",
            ]:
                os.environ.pop(key, None)
