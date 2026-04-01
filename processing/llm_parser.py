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
from tenacity import retry, stop_after_attempt, wait_exponential

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
    """Send text to Bedrock Claude and return a validated LLMParsedEvent, or None."""
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


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
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
    return response_json["content"][0]["text"]


async def close_client() -> None:
    """Close the Bedrock client. Called during shutdown."""
    global _client
    if _client is not None:
        await _client.__aexit__(None, None, None)
        _client = None
