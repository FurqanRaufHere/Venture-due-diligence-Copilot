"""
agents/founder_agent.py
────────────────────────────────────────────────────────────────
STEP 12 — Founder Risk Profiling Agent

WHAT THIS DOES:
  Reads the founder bio text and extracts structured signals about
  execution credibility and risk.

WHY NOT JUST ASK THE LLM TO SCORE IT?
  "Rate this founder 0-100" → random number, no justification
  
  Instead we:
    1. LLM extracts STRUCTURED FACTS from the bio (JSON schema)
    2. Rule-based logic computes a score FROM those facts
    3. LLM explains what those facts mean in context
  
  This is hybrid: structured extraction + deterministic scoring
  + contextual explanation. Same philosophy as the financial engine.

WHAT WE EXTRACT:
  • Prior exits: Did any founder previously sell a company?
  • Domain expertise: Is the team's background relevant to this problem?
  • Role coverage: Are the critical roles (CEO, CTO, Sales) filled?
  • Relevant experience: Years in the specific industry
  • Red flags: Serial failures, gaps, or mismatches

SCORING:
  Starts at 50 (neutral).
  DEDUCTS points for risk signals:
    - First-time founder, no exits: no deduction (just neutral)
    - Obvious role mismatch: -15
    - No domain expertise: -20
    - Serial failures without learning signals: -15
  
  ADDS points for positive signals:
    - Prior successful exit: +25
    - Deep domain expertise (10+ years): +20
    - Full team coverage: +10
    - Prior failed startup (shows experience): +5

  Final score is INVERTED for risk: higher score = more risky
  (to match the convention of all other agents)

OUTPUT RISK SCORE:
  Low risk founder (serial founder, domain expert) → 10–25
  Average founder → 40–60
  High risk founder (no experience, mismatch) → 65–85
"""

import logging
import json
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

FOUNDER_EXTRACTION_SYSTEM = """You are a venture capital analyst evaluating founder backgrounds.
Extract structured information from founder bios. Be precise and factual.
Only extract what is explicitly stated — do not infer or assume.
Return ONLY valid JSON matching the schema exactly.

JSON Schema:
{
  "founders": [
    {
      "name": "string",
      "role": "string (CEO/CTO/VP Sales/etc)",
      "prior_exits": ["list of company names sold/acquired, empty if none"],
      "prior_failures": ["list of failed ventures if mentioned"],
      "years_industry_experience": number or null,
      "industry_relevance": "high/medium/low (how relevant is their background to this startup)",
      "key_credentials": ["list of 2-3 most impressive credentials"],
      "red_flags": ["any concerning gaps, mismatches, or issues"]
    }
  ],
  "team_coverage": {
    "has_technical_founder": true/false,
    "has_sales_founder": true/false,
    "has_domain_expert": true/false,
    "missing_critical_roles": ["list of critical missing roles"]
  },
  "overall_team_assessment": "1-2 sentence factual summary"
}"""


def run_founder_agent(founder_bio_text: str) -> Dict[str, Any]:
    """
    Main entry point for the Founder Risk Profiling Agent.
    
    Args:
        founder_bio_text: Raw text from founder bio file or pitch deck team section
    
    Returns:
        {
          "founder_credibility_score": 0–100 (higher = more credible),
          "founder_risk_score": 0–100 (higher = more risky),
          "execution_risk_level": "low" / "medium" / "high",
          "prior_exits": int,
          "domain_expertise_level": "high" / "medium" / "low",
          "team_coverage_complete": bool,
          "red_flags": [str],
          "positive_signals": [str],
          "extracted_founders": [...],
          "risk_explanation": str,
        }
    """
    from utils.llm_client import call_llm_json, call_llm
    
    logger.info("Running founder risk profiling agent...")
    
    if not founder_bio_text or len(founder_bio_text.strip()) < 50:
        logger.warning("Founder bio too short for analysis")
        return _empty_result()
    
    # ── Step 1: LLM extracts structured founder data ───────────────────
    prompt = f"""Extract structured founder information from these bios.

FOUNDER BIOS:
{founder_bio_text[:4000]}

Extract all founders and team coverage information per the JSON schema."""
    
    try:
        extracted = call_llm_json(
            prompt=prompt,
            system=FOUNDER_EXTRACTION_SYSTEM,
            max_tokens=1500
        )
    except (ValueError, Exception) as e:
        logger.error(f"Founder extraction failed: {e}")
        extracted = {"founders": [], "team_coverage": {}, "overall_team_assessment": ""}
    
    founders = extracted.get("founders", [])
    team_coverage = extracted.get("team_coverage", {})
    
    # ── Step 2: Rule-based credibility scoring ─────────────────────────
    credibility_score, positive_signals, red_flags = _compute_credibility_score(
        founders=founders,
        team_coverage=team_coverage,
    )
    
    # Convert credibility (higher=better) to risk (higher=worse)
    founder_risk_score = round(100 - credibility_score, 1)
    
    # Execution risk level
    if founder_risk_score < 30:
        execution_risk = "low"
    elif founder_risk_score < 60:
        execution_risk = "medium"
    else:
        execution_risk = "high"
    
    # ── Step 3: Domain expertise assessment ───────────────────────────
    domain_levels = [f.get("industry_relevance", "medium") for f in founders if founders]
    domain_expertise = _aggregate_domain_level(domain_levels)
    
    # ── Step 4: Count prior exits ──────────────────────────────────────
    prior_exits = sum(len(f.get("prior_exits", [])) for f in founders)
    
    # ── Step 5: LLM writes explanation ────────────────────────────────
    risk_explanation = _generate_founder_explanation(
        founders=founders,
        credibility_score=credibility_score,
        founder_risk_score=founder_risk_score,
        positive_signals=positive_signals,
        red_flags=red_flags,
        execution_risk=execution_risk,
    )
    
    result = {
        "founder_credibility_score": round(credibility_score, 1),
        "founder_risk_score": founder_risk_score,
        "execution_risk_level": execution_risk,
        "prior_exits": prior_exits,
        "domain_expertise_level": domain_expertise,
        "team_coverage_complete": _is_team_complete(team_coverage),
        "missing_roles": team_coverage.get("missing_critical_roles", []),
        "red_flags": red_flags,
        "positive_signals": positive_signals,
        "extracted_founders": founders,
        "team_coverage": team_coverage,
        "risk_explanation": risk_explanation,
        "overall_assessment": extracted.get("overall_team_assessment", ""),
    }
    
    logger.info(
        f"Founder agent complete. "
        f"Credibility: {credibility_score:.0f}/100, "
        f"Risk: {founder_risk_score:.0f}/100, "
        f"Execution risk: {execution_risk}"
    )
    return result


# ── Scoring Logic ─────────────────────────────────────────────────────────────

def _compute_credibility_score(
    founders: List[Dict],
    team_coverage: Dict,
) -> tuple:
    """
    Compute a 0–100 credibility score from extracted founder facts.
    Returns (score, positive_signals, red_flags)
    """
    score = 40.0  # baseline — neutral first-time founder
    positive_signals = []
    red_flags = []
    
    for founder in founders:
        name = founder.get("name", "A founder")
        role = founder.get("role", "")
        
        # ── Positive signals ──────────────────────────────────────
        exits = founder.get("prior_exits", [])
        if exits:
            score += 20
            positive_signals.append(
                f"{name} has {len(exits)} prior exit(s): {', '.join(exits[:2])}"
            )
        
        years_exp = founder.get("years_industry_experience")
        if years_exp:
            if years_exp >= 10:
                score += 15
                positive_signals.append(f"{name} has {years_exp}+ years of domain experience")
            elif years_exp >= 5:
                score += 8
                positive_signals.append(f"{name} has {years_exp} years of relevant experience")
        
        relevance = founder.get("industry_relevance", "medium")
        if relevance == "high":
            score += 10
            positive_signals.append(f"{name} has highly relevant industry background")
        
        # Prior failures (shows experience, not automatically bad)
        failures = founder.get("prior_failures", [])
        if failures:
            score += 5  # small positive — shows they've learned from failure
            positive_signals.append(f"{name} has prior startup experience (including failures — shows learning)")
        
        # ── Red flags ─────────────────────────────────────────────
        founder_red_flags = founder.get("red_flags", [])
        for flag in founder_red_flags:
            score -= 8
            red_flags.append(f"{name}: {flag}")
        
        if relevance == "low":
            score -= 15
            red_flags.append(f"{name} ({role}): Low industry relevance — background doesn't match this problem")
    
    # ── Team coverage bonuses/penalties ──────────────────────────
    if team_coverage.get("has_technical_founder"):
        score += 8
        positive_signals.append("Technical co-founder present")
    else:
        score -= 10
        red_flags.append("No technical founder identified — significant execution risk for a tech startup")
    
    if team_coverage.get("has_domain_expert"):
        score += 7
        positive_signals.append("Domain expert on founding team")
    
    if team_coverage.get("has_sales_founder"):
        score += 5
        positive_signals.append("Sales/GTM expertise on founding team")
    
    missing = team_coverage.get("missing_critical_roles", [])
    if missing:
        penalty = len(missing) * 8
        score -= penalty
        red_flags.append(f"Missing critical roles: {', '.join(missing)}")
    
    # If no founders extracted, it's a data issue — neutral score
    if not founders:
        score = 40.0
        red_flags.append("Could not extract founder profiles from provided bio")
    
    return round(max(0, min(100, score)), 1), positive_signals, red_flags


def _aggregate_domain_level(levels: List[str]) -> str:
    """Return the highest domain expertise level across all founders."""
    if "high" in levels:
        return "high"
    elif "medium" in levels:
        return "medium"
    return "low"


def _is_team_complete(team_coverage: Dict) -> bool:
    """Team is complete if it has technical + sales coverage and no missing roles."""
    has_tech = team_coverage.get("has_technical_founder", False)
    no_missing = len(team_coverage.get("missing_critical_roles", [])) == 0
    return has_tech and no_missing


def _generate_founder_explanation(
    founders: List[Dict],
    credibility_score: float,
    founder_risk_score: float,
    positive_signals: List[str],
    red_flags: List[str],
    execution_risk: str,
) -> str:
    """Use the LLM to write a professional founder risk assessment."""
    from utils.llm_client import call_llm
    
    if not founders:
        return "Insufficient founder data provided for detailed analysis."
    
    positives_text = "\n".join(f"+ {s}" for s in positive_signals[:4]) or "None identified"
    flags_text = "\n".join(f"- {f}" for f in red_flags[:4]) or "None identified"
    
    system = """You are a VC analyst writing the team section of an investment memo.
Be direct and factual. 2 paragraphs max. No headers. Plain prose."""
    
    prompt = f"""Write a concise founder and team risk assessment for an investment memo.

Team size: {len(founders)} identified founders
Credibility Score: {credibility_score:.0f}/100
Execution Risk Level: {execution_risk.upper()}

Positive signals:
{positives_text}

Risk flags:
{flags_text}

Write a 2-paragraph assessment of the team's execution capability and risk profile."""
    
    try:
        return call_llm(prompt=prompt, system=system, max_tokens=400, temperature=0.2)
    except Exception as e:
        logger.error(f"Founder explanation LLM failed: {e}")
        return (
            f"Team credibility score: {credibility_score:.0f}/100 "
            f"({execution_risk} execution risk). "
            f"Identified {len(positive_signals)} positive signals and "
            f"{len(red_flags)} risk flags."
        )


def _empty_result() -> Dict:
    return {
        "founder_credibility_score": 40.0,
        "founder_risk_score": 60.0,
        "execution_risk_level": "medium",
        "prior_exits": 0,
        "domain_expertise_level": "medium",
        "team_coverage_complete": False,
        "missing_roles": [],
        "red_flags": ["No founder bio provided"],
        "positive_signals": [],
        "extracted_founders": [],
        "team_coverage": {},
        "risk_explanation": "No founder bio was provided. Founder risk cannot be assessed.",
        "overall_assessment": "",
    }