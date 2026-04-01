"""Integration tests for the Bedrock-backed LLM parser."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest

from processing.llm_parser import SYSTEM_PROMPT, parse_with_llm


@pytest.fixture(autouse=True)
def reset_llm_client():
    """Reset the global LLM client between tests."""
    import processing.llm_parser as mod

    mod._client = None
    yield
    mod._client = None


def _mock_bedrock(llm_output: dict | str):
    """Return a patch context that mocks _get_bedrock_client.

    *llm_output* is either a dict (will be JSON-encoded inside the Claude
    response envelope) or a raw string (used as-is for malformed-response
    tests).
    """
    if isinstance(llm_output, dict):
        text_payload = json.dumps(llm_output)
    else:
        text_payload = llm_output

    mock_response_body = json.dumps({"content": [{"text": text_payload}]})

    mock_client = AsyncMock()
    body_mock = AsyncMock()
    body_mock.read = AsyncMock(return_value=mock_response_body.encode())
    mock_client.invoke_model = AsyncMock(return_value={"body": body_mock})

    return patch(
        "processing.llm_parser._get_bedrock_client",
        new_callable=AsyncMock,
        return_value=mock_client,
    )


# ── Tests ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_valid_response_returns_parsed_event():
    """Mock Bedrock returning well-formed JSON; verify LLMParsedEvent fields."""
    llm_output = {
        "is_valid_anomaly": True,
        "approximate_origin": "Houston, TX",
        "debris_trajectory_or_blast_radius": "NW to SE, ~3 km radius",
        "event_classification": "industrial_accident",
        "confidence_score": 8,
    }

    with _mock_bedrock(llm_output):
        result = await parse_with_llm("Big explosion near the refinery district")

    assert result is not None
    assert result.is_valid_anomaly is True
    assert result.approximate_origin == "Houston, TX"
    assert result.debris_trajectory_or_blast_radius == "NW to SE, ~3 km radius"
    assert result.event_classification == "industrial_accident"
    assert result.confidence_score == 8


@pytest.mark.asyncio
async def test_malformed_json_returns_none():
    """Mock Bedrock returning non-JSON text; verify None is returned."""
    with _mock_bedrock("This is not JSON at all, sorry!"):
        result = await parse_with_llm("Something happened somewhere")

    assert result is None


@pytest.mark.asyncio
async def test_api_error_returns_none():
    """Mock Bedrock raising an exception; verify None is returned."""
    with patch("processing.llm_parser._get_bedrock_client") as mock_get:
        mock_client = AsyncMock()
        mock_client.invoke_model = AsyncMock(
            side_effect=RuntimeError("Bedrock is down")
        )
        mock_get.return_value = mock_client

        result = await parse_with_llm("test text")

    assert result is None


@pytest.mark.asyncio
async def test_system_prompt_exists():
    """Verify SYSTEM_PROMPT contains key instructions for the LLM."""
    assert "disaster-response" in SYSTEM_PROMPT
    assert "is_valid_anomaly" in SYSTEM_PROMPT
    assert "confidence_score" in SYSTEM_PROMPT
    assert "JSON" in SYSTEM_PROMPT
    assert "event_classification" in SYSTEM_PROMPT
