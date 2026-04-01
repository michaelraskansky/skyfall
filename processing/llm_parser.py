"""
LLM Triage Parser
==================

Takes unstructured text (typically from the social-media listener) and
passes it to an LLM with a strict *disaster-response analyst* system
prompt.  The LLM must return a structured JSON object matching the
``LLMParsedEvent`` schema:

    {
        "is_valid_anomaly": true,
        "approximate_origin": "Houston, TX area",
        "debris_trajectory_or_blast_radius": "NW to SE, ~3 km radius",
        "event_classification": "industrial_accident",
        "confidence_score": 8
    }

Two providers are supported (selected via ``LLM_PROVIDER`` env var):
  - **openai** – any OpenAI-compatible chat endpoint (GPT-4o, etc.)
  - **gemini** – Google Generative AI (Gemini Pro, etc.)

The function is intentionally synchronous-looking but uses ``asyncio``
under the hood so it can be ``await``-ed from the main event loop.
"""

from __future__ import annotations

import json
import logging

from config import settings
from models import LLMParsedEvent

logger = logging.getLogger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────
# The LLM is instructed to act as a disaster-response analyst and ONLY
# return valid JSON.  We repeat the schema inside the prompt so the model
# can self-validate.

SYSTEM_PROMPT = """\
You are a disaster-response intelligence analyst embedded in a real-time
anomaly detection pipeline.  Your job is to read raw social-media posts,
emergency broadcast fragments, or sensor descriptions and determine whether
they describe a genuine aerospace debris re-entry, industrial explosion,
or other catastrophic thermal / kinetic event.

You MUST respond with ONLY a valid JSON object – no markdown fences, no
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


async def parse_with_llm(text: str) -> LLMParsedEvent | None:
    """
    Send *text* to the configured LLM and return a validated
    ``LLMParsedEvent``, or ``None`` if the LLM response is unparseable.
    """
    provider = settings.llm_provider.lower()

    try:
        raw_json = await _call_llm(provider, text)
        parsed = json.loads(raw_json)
        return LLMParsedEvent(**parsed)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        logger.error("LLM returned unparseable response: %s", exc)
        return None
    except Exception:
        logger.exception("LLM call failed")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Provider implementations
# ═══════════════════════════════════════════════════════════════════════════════


async def _call_llm(provider: str, text: str) -> str:
    """Dispatch to the appropriate LLM backend and return the raw JSON string."""
    if provider == "openai":
        return await _call_openai(text)
    elif provider == "gemini":
        return await _call_gemini(text)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


async def _call_openai(text: str) -> str:
    """
    Call the OpenAI Chat Completions API (works with any compatible
    endpoint, including Azure OpenAI and local vLLM servers).
    """
    import openai

    client = openai.AsyncOpenAI(api_key=settings.openai_api_key)

    response = await client.chat.completions.create(
        model=settings.llm_model,
        temperature=0.1,  # near-deterministic for structured output
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze the following text:\n\n{text}"},
        ],
        # Ask for JSON mode where supported (GPT-4o, GPT-4-turbo).
        response_format={"type": "json_object"},
    )

    return response.choices[0].message.content or "{}"


async def _call_gemini(text: str) -> str:
    """
    Call the Google Generative AI (Gemini) API.

    ``google-generativeai`` is synchronous, so we run it in a thread
    executor to avoid blocking the event loop.
    """
    import asyncio
    import google.generativeai as genai

    genai.configure(api_key=settings.gemini_api_key)

    model = genai.GenerativeModel(
        model_name=settings.llm_model,
        system_instruction=SYSTEM_PROMPT,
    )

    # GenerativeModel.generate_content is sync – offload to a thread.
    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,
        lambda: model.generate_content(
            f"Analyze the following text:\n\n{text}",
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        ),
    )

    return response.text or "{}"
