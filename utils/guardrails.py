"""
utils/guardrails.py
────────────────────────────────────────────────────────────────
STEP 20 — Hallucination Guardrails

WHAT THIS DOES:
  Validates all LLM outputs before they reach the database or frontend.
  Catches hallucinations, empty responses, and format violations.

WHY THIS MATTERS:
  LLMs can return:
    - Empty strings or "I cannot..." refusals
    - Made-up company names or statistics
    - Responses that ignore the JSON schema
    - Overly confident language on uncertain data
    - Numbers outside valid ranges

THREE LAYERS OF PROTECTION:
  1. SCHEMA VALIDATION  — Pydantic checks structure + types
  2. CONTENT VALIDATION — Rule-based checks for hallucination signals
  3. CONFIDENCE FLAGGING — Marks low-quality outputs for review

USAGE:
  from utils.guardrails import validate_claims, validate_financial_explanation

  claims = validate_claims(raw_llm_dict)
  # Returns (validated_dict, issues_list)
  # issues_list is empty if clean, or contains warning strings
"""

import re
import logging
from typing import Dict, Any, Tuple, List, Optional

logger = logging.getLogger(__name__)

# ── Hallucination signal phrases ──────────────────────────────
# These phrases in LLM output suggest the model is making things up
# or refusing to answer rather than grounding its response in data.
HALLUCINATION_SIGNALS = [
    "i cannot", "i don't have", "i don't know", "as an ai",
    "as a language model", "i'm not able", "i am unable",
    "no information provided", "cannot determine", "not enough information",
    "based on my training", "i would need more", "insufficient data provided",
    "please provide", "could you provide", "i need more context",
]

# Phrases that suggest overconfidence / fabrication
OVERCONFIDENCE_SIGNALS = [
    "definitely will", "guaranteed to", "certainly will",
    "without a doubt", "100% certain", "absolutely will",
    "will definitely succeed", "will certainly fail",
]

# Valid ranges for numeric fields
NUMERIC_BOUNDS = {
    "confidence_score":            (0.0, 1.0),
    "financial_plausibility_score": (0.0, 100.0),
    "financial_risk_score":        (0.0, 100.0),
    "market_risk_score":           (0.0, 100.0),
    "founder_risk_score":          (0.0, 100.0),
    "narrative_inflation_score":   (0.0, 100.0),
    "pattern_similarity_score":    (0.0, 100.0),
    "overall_risk_score":          (0.0, 100.0),
    "revenue_cagr":                (-100.0, 10000.0),
    "gross_margin_avg":            (-500.0, 100.0),
}


def validate_llm_text(text: str, field_name: str = "output") -> Tuple[str, List[str]]:
    """
    Validate a free-text LLM output (explanation, summary, etc).

    Returns:
        (cleaned_text, issues_list)
        issues_list is empty if clean.
    """
    issues = []

    if not text or not text.strip():
        issues.append(f"{field_name}: Empty LLM response")
        return f"[{field_name} unavailable]", issues

    text_lower = text.lower()

    # Check for refusal / hallucination signals
    for signal in HALLUCINATION_SIGNALS:
        if signal in text_lower:
            issues.append(f"{field_name}: Contains refusal/hallucination signal: '{signal}'")
            break

    # Check for overconfidence signals
    for signal in OVERCONFIDENCE_SIGNALS:
        if signal in text_lower:
            issues.append(f"{field_name}: Contains overconfidence signal: '{signal}'")
            break

    # Check minimum length — too short means LLM didn't really engage
    if len(text.strip()) < 30:
        issues.append(f"{field_name}: Response too short ({len(text)} chars) — likely low quality")

    # Check for repeated phrases (copy-paste hallucination)
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
    if len(sentences) > 2:
        unique = set(sentences)
        if len(unique) < len(sentences) * 0.7:
            issues.append(f"{field_name}: High phrase repetition detected — possible hallucination")

    return text.strip(), issues


def validate_claims(claims: Dict) -> Tuple[Dict, List[str]]:
    """
    Validate the output of the claim extraction agent.
    Checks that key fields are present and not hallucinated.

    Returns (validated_claims, issues)
    """
    issues = []
    validated = dict(claims)

    required_fields = ['problem_statement', 'solution_claim', 'target_market']
    for field in required_fields:
        val = validated.get(field)
        if not val or not str(val).strip():
            issues.append(f"claims.{field}: Missing or empty — LLM may not have extracted it")
            validated[field] = None
        else:
            cleaned, field_issues = validate_llm_text(str(val), f"claims.{field}")
            validated[field] = cleaned
            issues.extend(field_issues)

    # Validate confidence score range
    conf = validated.get('confidence_score')
    if conf is not None:
        try:
            conf_float = float(conf)
            if not (0.0 <= conf_float <= 1.0):
                issues.append(f"claims.confidence_score: Out of range ({conf_float}) — clamping to [0,1]")
                validated['confidence_score'] = max(0.0, min(1.0, conf_float))
        except (TypeError, ValueError):
            issues.append(f"claims.confidence_score: Not a number ({conf})")
            validated['confidence_score'] = None

    # Validate hype_indicators is a list
    hype = validated.get('hype_indicators')
    if hype is not None and not isinstance(hype, list):
        issues.append("claims.hype_indicators: Not a list — converting")
        validated['hype_indicators'] = [str(hype)] if hype else []

    # Check TAM claim for wildly inflated values
    tam = validated.get('tam_claim', '') or ''
    if re.search(r'\$\d+\s*quadrillion', tam.lower()):
        issues.append(f"claims.tam_claim: Implausibly large claim detected: {tam[:80]}")

    return validated, issues


def validate_financial_explanation(explanation: str) -> Tuple[str, List[str]]:
    """Validate the LLM-generated financial anomaly explanation."""
    return validate_llm_text(explanation, "financial_explanation")


def validate_market_assessment(assessment: str) -> Tuple[str, List[str]]:
    """Validate the LLM-generated market assessment."""
    cleaned, issues = validate_llm_text(assessment, "market_assessment")

    # Market assessment should reference actual companies or markets
    if cleaned and len(cleaned) > 50:
        has_specific = any(c.isupper() for c in cleaned[10:])  # proper nouns
        if not has_specific:
            issues.append("market_assessment: May lack specific company/market references — possible generic output")

    return cleaned, issues


def validate_numeric_score(value: Any, field_name: str) -> Tuple[Optional[float], List[str]]:
    """
    Validate a numeric score field.
    Returns (validated_value, issues).
    """
    issues = []

    if value is None:
        return None, []

    try:
        v = float(value)
    except (TypeError, ValueError):
        issues.append(f"{field_name}: Not a number ({value}) — setting to None")
        return None, issues

    bounds = NUMERIC_BOUNDS.get(field_name)
    if bounds:
        lo, hi = bounds
        if not (lo <= v <= hi):
            issues.append(f"{field_name}: Out of expected range [{lo}, {hi}] (got {v}) — clamping")
            v = max(lo, min(hi, v))

    return round(v, 2), issues


def validate_risk_scores(scores: Dict) -> Tuple[Dict, List[str]]:
    """
    Validate all numeric risk scores in the aggregation output.
    """
    issues = []
    validated = dict(scores)

    score_fields = [
        'financial_risk_score', 'market_risk_score', 'founder_risk_score',
        'narrative_inflation_score', 'pattern_similarity_score', 'overall_risk_score'
    ]

    for field in score_fields:
        val = validated.get(field)
        cleaned, field_issues = validate_numeric_score(val, field)
        validated[field] = cleaned
        issues.extend(field_issues)

    # Grade must be A-F
    grade = validated.get('investment_grade')
    if grade and grade not in ('A', 'B', 'C', 'D', 'F'):
        issues.append(f"investment_grade: Invalid value '{grade}' — expected A-F")
        validated['investment_grade'] = 'C'  # safe default

    # Confidence level must be low/medium/high
    conf = validated.get('confidence_level')
    if conf and conf not in ('low', 'medium', 'high'):
        issues.append(f"confidence_level: Invalid value '{conf}'")
        validated['confidence_level'] = 'low'

    return validated, issues


def validate_founder_extraction(extracted: Dict) -> Tuple[Dict, List[str]]:
    """Validate the structured output from the founder extraction agent."""
    issues = []
    validated = dict(extracted)

    founders = validated.get('founders', [])
    if not isinstance(founders, list):
        issues.append("founder_extraction.founders: Not a list")
        validated['founders'] = []
        founders = []

    for i, founder in enumerate(founders):
        if not isinstance(founder, dict):
            issues.append(f"founder_extraction.founders[{i}]: Not a dict")
            continue

        # Name must be present
        if not founder.get('name'):
            issues.append(f"founder_extraction.founders[{i}]: Missing name")

        # Industry relevance must be valid
        rel = founder.get('industry_relevance')
        if rel and rel not in ('high', 'medium', 'low'):
            issues.append(f"founder_extraction.founders[{i}].industry_relevance: Invalid '{rel}' — defaulting to medium")
            founder['industry_relevance'] = 'medium'

        # Years experience must be positive number or None
        yrs = founder.get('years_industry_experience')
        if yrs is not None:
            try:
                yrs_float = float(yrs)
                if yrs_float < 0 or yrs_float > 60:
                    issues.append(f"founder_extraction.founders[{i}].years_experience: Implausible value {yrs}")
                    founder['years_industry_experience'] = None
            except (TypeError, ValueError):
                founder['years_industry_experience'] = None

    return validated, issues


def log_validation_issues(issues: List[str], context: str = ""):
    """Log all validation issues at appropriate log levels."""
    if not issues:
        return
    prefix = f"[GUARDRAIL{' ' + context if context else ''}]"
    for issue in issues:
        if any(sig in issue.lower() for sig in ['hallucination', 'refusal', 'implausible', 'fraud']):
            logger.warning(f"{prefix} {issue}")
        else:
            logger.info(f"{prefix} {issue}")


def get_output_quality_flag(all_issues: List[str]) -> str:
    """
    Given a list of all validation issues across all agents,
    return a quality flag: 'clean', 'warnings', or 'review_required'.
    """
    if not all_issues:
        return 'clean'
    high_severity = [i for i in all_issues if any(
        s in i.lower() for s in ['hallucination', 'refusal', 'implausible', 'empty', 'fraud']
    )]
    if high_severity:
        return 'review_required'
    return 'warnings'