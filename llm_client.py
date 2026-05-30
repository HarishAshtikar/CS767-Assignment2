"""Gemini client wrapper."""

import json
import re
import time
import ast

from google import genai
from google.genai import errors
from google.genai import types

from config import GEMINI_API_KEY, GEMINI_FALLBACK_MODELS, GEMINI_MAX_RETRIES, GEMINI_MODEL


class GeminiQuotaError(RuntimeError):
    """Raised when Gemini quota is exhausted across all configured models."""


EXHAUSTED_MODELS = set()


def build_client() -> genai.Client:
    """Create a Gemini API client."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not set. Copy .env.example to .env and add your key.")
    return genai.Client(api_key=GEMINI_API_KEY)


def generate_agent_step(client: genai.Client, messages: list[dict], system_prompt: str) -> dict:
    """Ask Gemini for the next structured agent action."""
    contents = _messages_to_text(messages)
    response = _generate_with_retries(client, contents, system_prompt)
    return _parse_json_response(response.text or "")


def _generate_with_retries(client: genai.Client, contents: str, system_prompt: str):
    models = [model for model in dict.fromkeys([GEMINI_MODEL, *GEMINI_FALLBACK_MODELS]) if model not in EXHAUSTED_MODELS]
    last_error = None

    for model in models:
        for attempt in range(GEMINI_MAX_RETRIES + 1):
            try:
                if model != GEMINI_MODEL:
                    print(f"  Primary model quota exhausted. Trying fallback model: {model}")
                return client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        temperature=0.2,
                        response_mime_type="application/json",
                    ),
                )
            except errors.ClientError as exc:
                last_error = exc
                if not _is_quota_error(exc):
                    raise

                if _has_zero_quota(exc):
                    print(f"  Gemini quota is unavailable for {model}. Skipping to fallback.")
                    EXHAUSTED_MODELS.add(model)
                    break

                retry_delay = _retry_delay_seconds(exc)
                if attempt < GEMINI_MAX_RETRIES and retry_delay > 0:
                    print(f"  Gemini quota/rate limit hit for {model}. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    continue
                EXHAUSTED_MODELS.add(model)
                break

    raise GeminiQuotaError(
        "Gemini quota is exhausted for all configured models. "
        "Try again later, enable billing, or set GEMINI_FALLBACK_MODELS to a model with quota."
    ) from last_error


def _is_quota_error(exc: errors.ClientError) -> bool:
    return getattr(exc, "code", None) == 429 or "RESOURCE_EXHAUSTED" in str(exc)


def _has_zero_quota(exc: errors.ClientError) -> bool:
    return "limit: 0" in str(exc)


def _retry_delay_seconds(exc: errors.ClientError) -> int:
    match = re.search(r"Please retry in ([\d.]+)s", str(exc))
    if match:
        return max(1, int(float(match.group(1))))

    match = re.search(r"'retryDelay': '(\d+)s'", str(exc))
    if match:
        return max(1, int(match.group(1)))

    return 0


def _messages_to_text(messages: list[dict]) -> str:
    lines = []
    for message in messages:
        role = message["role"].upper()
        lines.append(f"{role}:\n{message['content']}")
    return "\n\n".join(lines)


def _parse_json_response(text: str) -> dict:
    clean = text.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```(?:json)?\s*", "", clean)
        clean = re.sub(r"\s*```$", "", clean)

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", clean, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Gemini did not return JSON: {text[:500]}")
        json_like = match.group(0)
        try:
            return json.loads(json_like)
        except json.JSONDecodeError:
            parsed = ast.literal_eval(json_like)
            if not isinstance(parsed, dict):
                raise ValueError(f"Gemini JSON response was not an object: {text[:500]}")
            return parsed
