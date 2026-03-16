"""
agents/claim_extraction_agent.py
─────────────────────────────────
WHAT THIS AGENT DOES:
  Takes the raw text from a pitch deck PDF and uses an LLM with
  strict JSON schema enforcement to extract structured business claims.

WHY STRUCTURED OUTPUT MATTERS:
  Without schema enforcement, the LLM might return:
    "The startup targets the $45B logistics market..."
  With schema enforcement, we get:
    {"tam_claim": "$45B logistics market", ...}
  
  Structured output = deterministic downstream processing.
  The rest of the pipeline doesn't parse free text — it reads dicts.

THE PROMPT STRATEGY:
  1. Role-playing: "You are an expert VC analyst"
  2. Chain-of-thought hint: "Carefully read..." 
  3. Strict JSON schema provided in the prompt
  4. Low temperature (0.1) for consistency
  5. json_mode=True adds a final hard instruction to return pure JSON

HYPE DETECTION:
  We also extract "hype_indicators" — phrases that signal narrative
  inflation ("revolutionize", "AI-powered everything", "10x better",
  "blue ocean market", etc.). These feed into the narrative_inflation_score.

CONFIDENCE SCORE:
  Based on how many fields were successfully extracted.
  0.9+ = very clean pitch deck
  0.5–0.9 = partial extraction
  < 0.5 = pitch deck may be poorly structured or image-based
"""

import json
import logging
from typing import Dict, Any, Optional
from utils.llm_client import call_llm_json
from utils.pdf_parser import get_pdf_summary_chunk

logger = logging.getLogger(__name__)

# Hype phrases that indicate narrative inflation
HYPE_PHRASES = [
    "revolutionize", "disrupt", "10x", "100x", "game changer", "game-changer",
    "blue ocean", "first mover", "no competition", "patented technology",
    "proprietary ai", "ai-powered", "blockchain-based", "exponential growth",
    "viral growth", "zero to one", "trillion dollar", "massive market",
    "untapped market", "infinite scale", "guaranteed", "risk-free",
    "fastest growing", "world's first", "only solution"
]

EXTRACTION_SYSTEM_PROMPT = """You are a senior venture capital analyst with 15 years of experience 
evaluating startup pitch decks. You are precise, skeptical, and extract only what is explicitly 
stated — never invent or assume information not present in the text.

Your job is to extract structured claims from a startup pitch deck.
Return ONLY valid JSON matching the schema exactly. No commentary, no markdown, no explanation.

JSON Schema:
{
  "problem_statement": "string — the core problem the startup claims to solve",
  "solution_claim": "string — what their product/service does",
  "target_market": "string — who their customers are",
  "tam_claim": "string — Total Addressable Market size claim (exact quote or null)",
  "revenue_model": "string — how they make money",
  "growth_claims": ["list of strings — specific growth metrics or projections claimed"],
  "competitive_advantage_claims": ["list of strings — why they claim to win vs competitors"],
  "key_risks_acknowledged": ["list of strings — any risks the founders themselves mention"],
  "funding_ask": "string — how much they are raising and at what valuation, or null",
  "stage": "string — pre-seed / seed / Series A / etc., or null"
}

Rules:
- If a field is not found in the text, use null (not empty string, not "not mentioned")
- Extract growth_claims as individual claims: ["3x YoY revenue", "40% MoM user growth"]
- Be a skeptic: extract what they say, not what you think is true
"""


def run_claim_extraction(pitch_deck_text: str) -> Dict[str, Any]:
    """
    Main entry point for the Business Claim Extraction Agent.

    Args:
        pitch_deck_text: Full extracted text from pitch deck PDF

    Returns:
        {
          "problem_statement": ...,
          "solution_claim": ...,
          "tam_claim": ...,
          "revenue_model": ...,
          "growth_claims": [...],
          "competitive_advantage_claims": [...],
          "hype_indicators": [...],  # detected from text
          "confidence_score": 0.0–1.0,
          "raw_llm_output": {...}    # full LLM response for auditing
        }
    """
    # Use the first ~8000 chars which captures the key slides
    summary_chunk = get_pdf_summary_chunk(pitch_deck_text, max_chars=8000)

    prompt = f"""Analyze this startup pitch deck text and extract structured claims.

PITCH DECK TEXT:
{summary_chunk}

Extract all fields from the JSON schema. Return only the JSON object."""

    logger.info("Running claim extraction agent...")

    try:
        extracted = call_llm_json(prompt=prompt, system=EXTRACTION_SYSTEM_PROMPT, max_tokens=2000)
    except ValueError as e:
        logger.error(f"Claim extraction LLM failed: {e}")
        extracted = {}

    # Detect hype indicators in the original text
    hype_found = _detect_hype_phrases(pitch_deck_text)

    # Compute confidence: proportion of key fields that were extracted
    key_fields = ["problem_statement", "solution_claim", "tam_claim", "revenue_model", "growth_claims"]
    filled = sum(1 for f in key_fields if extracted.get(f))
    confidence = filled / len(key_fields)

    result = {
        "problem_statement": extracted.get("problem_statement"),
        "solution_claim": extracted.get("solution_claim"),
        "target_market": extracted.get("target_market"),
        "tam_claim": extracted.get("tam_claim"),
        "revenue_model": extracted.get("revenue_model"),
        "growth_claims": extracted.get("growth_claims") or [],
        "competitive_advantage_claims": extracted.get("competitive_advantage_claims") or [],
        "key_risks_acknowledged": extracted.get("key_risks_acknowledged") or [],
        "funding_ask": extracted.get("funding_ask"),
        "stage": extracted.get("stage"),
        "hype_indicators": hype_found,
        "confidence_score": round(confidence, 2),
        "raw_llm_output": extracted,
    }

    logger.info(f"Claim extraction complete. Confidence: {confidence:.0%}. Hype phrases: {len(hype_found)}")
    return result


def _detect_hype_phrases(text: str) -> list:
    """
    Scan the pitch deck text for known hype/inflation phrases.
    Case-insensitive. Returns list of matched phrases.
    """
    text_lower = text.lower()
    found = []
    for phrase in HYPE_PHRASES:
        if phrase in text_lower:
            found.append(phrase)
    return found