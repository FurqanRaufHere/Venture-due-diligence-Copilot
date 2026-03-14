"""
utils/llm_client.py
────────────────────
Updated to support Groq as the primary provider.

Groq uses an OpenAI-compatible API, so we use the `groq` SDK
which has the exact same interface as the openai SDK.

MODEL CHOICE: llama-3.3-70b-versatile
  - Best general-purpose model on Groq right now
  - 128k context window (handles long pitch decks)
  - Strong at JSON schema enforcement
  - Free tier: 14,400 requests/day, 500,000 tokens/minute
  - Fast: ~280 tokens/second on Groq's LPU hardware
"""

import os
import json
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL    = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")


def call_llm(
    prompt: str,
    system: str = "You are a helpful AI assistant.",
    json_mode: bool = False,
    max_tokens: int = 2000,
    temperature: float = 0.1,
) -> str:
    """
    Call the configured LLM and return the response as a string.
    Supports: groq, openai, anthropic
    """
    if json_mode:
        system += "\n\nCRITICAL: Respond ONLY with valid JSON. No preamble, no explanation, no markdown code fences. Pure JSON only."

    try:
        if LLM_PROVIDER == "groq":
            return _call_groq(prompt, system, max_tokens, temperature, json_mode)
        elif LLM_PROVIDER == "openai":
            return _call_openai(prompt, system, max_tokens, temperature)
        elif LLM_PROVIDER == "anthropic":
            return _call_anthropic(prompt, system, max_tokens, temperature)
        else:
            raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}. Use 'groq', 'openai', or 'anthropic'.")
    except Exception as e:
        logger.error(f"LLM call failed [{LLM_PROVIDER}]: {e}")
        raise


def _call_groq(prompt: str, system: str, max_tokens: int, temperature: float, json_mode: bool) -> str:
    """Call Groq API — OpenAI-compatible interface."""
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)

    kwargs = dict(
        model=LLM_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ]
    )

    # Groq supports native JSON mode — more reliable than prompting alone
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def _call_openai(prompt: str, system: str, max_tokens: int, temperature: float) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": prompt},
        ]
    )
    return response.choices[0].message.content


def _call_anthropic(prompt: str, system: str, max_tokens: int, temperature: float) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=LLM_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content[0].text


def call_llm_json(prompt: str, system: str, max_tokens: int = 2000) -> dict:
    """
    Call the LLM in JSON mode and parse the result into a Python dict.
    With Groq's native json_object response format, this is very reliable.
    """
    raw = call_llm(prompt, system, json_mode=True, max_tokens=max_tokens)

    # Strip accidental markdown fences just in case
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("```")[1]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM JSON: {e}\nRaw output: {raw[:500]}")
        raise ValueError(f"LLM returned non-JSON output: {e}")