"""
agents/risk_aggregation_engine.py  (UPDATED — Phase 2)
────────────────────────────────────────────────────────────────
STEP 13 — Full Risk Aggregation (Updated)

Phase 1: Only Financial Risk + Narrative Inflation (2 of 5)
Phase 2: All 5 dimensions active:
  Financial Risk      (30%) — financial_analysis_agent
  Market Risk         (25%) — market_agent         (NEW Phase 2)
  Founder Risk        (20%) — founder_agent        (NEW Phase 2)
  Narrative Inflation (15%) — claim_extraction_agent
  Pattern Similarity  (10%) — similarity_engine    (NEW Phase 2)

CONFIDENCE:
  2 agents = LOW | 3-4 agents = MEDIUM | 5 agents = HIGH
"""

import logging
from typing import Dict, Any, Optional, List
from utils.llm_client import call_llm

logger = logging.getLogger(__name__)

WEIGHTS = {
    "financial_risk":      0.30,
    "market_risk":         0.25,
    "founder_risk":        0.20,
    "narrative_inflation": 0.15,
    "pattern_similarity":  0.10,
}

GRADE_THRESHOLDS = [(20,"A"),(40,"B"),(60,"C"),(80,"D"),(100,"F")]


def run_risk_aggregation(
    financial_risk_score:      Optional[float] = None,
    market_risk_score:         Optional[float] = None,
    founder_risk_score:        Optional[float] = None,
    narrative_inflation_score: Optional[float] = None,
    pattern_similarity_score:  Optional[float] = None,
    extracted_claims:          Optional[Dict] = None,
    financial_metrics:         Optional[Dict] = None,
    market_analysis:           Optional[Dict] = None,
    founder_analysis:          Optional[Dict] = None,
    similarity_results:        Optional[Dict] = None,
) -> Dict[str, Any]:
    scores = {
        "financial_risk":      financial_risk_score,
        "market_risk":         market_risk_score,
        "founder_risk":        founder_risk_score,
        "narrative_inflation": narrative_inflation_score,
        "pattern_similarity":  pattern_similarity_score,
    }
    available = {k: v for k, v in scores.items() if v is not None}
    n_available = len(available)

    if not available:
        return {**{f"{k}_score": None for k in WEIGHTS},
                "overall_risk_score": None, "investment_grade": None,
                "confidence_level": "low", "due_diligence_memo": None}

    total_weight = sum(WEIGHTS[k] for k in available)
    overall_score = round(sum(v * (WEIGHTS[k]/total_weight) for k,v in available.items()), 1)
    grade = _score_to_grade(overall_score)
    confidence = _compute_confidence(n_available)

    memo = _build_due_diligence_memo(
        overall_score, grade, confidence, scores,
        extracted_claims, financial_metrics, market_analysis,
        founder_analysis, similarity_results
    )

    logger.info(f"Risk aggregation complete. Score:{overall_score}/100 Grade:{grade} Confidence:{confidence} ({n_available}/5 agents)")
    return {
        "financial_risk_score":      financial_risk_score,
        "market_risk_score":         market_risk_score,
        "founder_risk_score":        founder_risk_score,
        "narrative_inflation_score": narrative_inflation_score,
        "pattern_similarity_score":  pattern_similarity_score,
        "overall_risk_score":        overall_score,
        "investment_grade":          grade,
        "confidence_level":          confidence,
        "scores_available":          n_available,
        "due_diligence_memo":        memo,
    }


def compute_narrative_inflation_score(hype_indicators: list, growth_claims: list) -> float:
    score = len(hype_indicators) * 10.0
    if len(growth_claims) > 3:
        score += 15
    return min(100.0, round(score, 1))


def compute_financial_risk_score(financial_plausibility_score: float) -> float:
    return round(100.0 - financial_plausibility_score, 1)


def _score_to_grade(score: float) -> str:
    for threshold, grade in GRADE_THRESHOLDS:
        if score <= threshold:
            return grade
    return "F"


def _compute_confidence(n: int) -> str:
    return "high" if n >= 5 else "medium" if n >= 3 else "low"


def _build_due_diligence_memo(overall_score, grade, confidence, scores,
    extracted_claims, financial_metrics, market_analysis, founder_analysis, similarity_results):

    def fmt_claims(c):
        if not c: return ""
        return f"\nSTARTUP CLAIMS:\n  Problem: {c.get('problem_statement','N/A')}\n  Solution: {c.get('solution_claim','N/A')}\n  TAM: {c.get('tam_claim','N/A')}\n  Revenue: {c.get('revenue_model','N/A')}\n  Hype phrases: {len(c.get('hype_indicators',[]))}"
    def fmt_fin(f):
        if not f: return ""
        return f"\nFINANCIALS:\n  CAGR: {f.get('revenue_cagr','N/A')}%  Runway: {f.get('runway_months','N/A')}mo  Margin: {f.get('gross_margin_avg','N/A')}%  RedFlags: {len(f.get('red_flags',[]))}"
    def fmt_mkt(m):
        if not m: return ""
        return f"\nMARKET:\n  Competition Density: {m.get('competition_density','N/A')}/100  TAM Plausibility: {m.get('tam_plausibility','N/A').upper() if m.get('tam_plausibility') else 'N/A'}\n  Competitors found: {', '.join(m.get('identified_competitors',[])[:5]) or 'None'}"
    def fmt_fnd(f):
        if not f: return ""
        return f"\nFOUNDERS:\n  Credibility: {f.get('founder_credibility_score','N/A')}/100  Exits: {f.get('prior_exits',0)}  Domain: {str(f.get('domain_expertise_level','N/A')).upper()}  ExecRisk: {str(f.get('execution_risk_level','N/A')).upper()}"
    def fmt_sim(s):
        if not s: return ""
        return f"\nPATTERNS:\n  Failure similarity: {s.get('failed_similarity_pct','N/A')}%  Success similarity: {s.get('success_similarity_pct','N/A')}%\n  Archetype: {s.get('archetype_label','N/A')}"

    scores_txt = "\n".join(
        f"  {k.replace('_',' ').title()}: {v:.0f}/100" if v is not None else f"  {k.replace('_',' ').title()}: N/A"
        for k,v in scores.items()
    )

    system = "You are a senior VC analyst writing an investment due diligence executive summary. Be direct, factual, professional. 3–5 sentences. No lists or headers."
    prompt = f"""Write an executive summary for this startup due diligence report.

OVERALL: Risk Score {overall_score}/100 | Grade {grade} | Confidence {confidence.upper()} ({sum(1 for v in scores.values() if v is not None)}/5 dimensions)

RISK SCORES:
{scores_txt}
{fmt_claims(extracted_claims)}{fmt_fin(financial_metrics)}{fmt_mkt(market_analysis)}{fmt_fnd(founder_analysis)}{fmt_sim(similarity_results)}

Write a 3–5 sentence executive summary with key findings and investment stance."""

    try:
        executive_summary = call_llm(prompt=prompt, system=system, max_tokens=400, temperature=0.2)
    except Exception as e:
        logger.error(f"Memo LLM failed: {e}")
        executive_summary = f"Investment Grade {grade} | Risk Score {overall_score}/100 | Confidence: {confidence}."

    # Collect all red flags
    all_flags = []
    if financial_metrics: all_flags.extend(financial_metrics.get("red_flags", []))
    if market_analysis and market_analysis.get("competition_density", 0) > 70:
        all_flags.append(f"High market saturation ({market_analysis['competition_density']}/100)")
    if market_analysis and market_analysis.get("tam_plausibility") == "low":
        all_flags.append("TAM claim appears inflated")
    if founder_analysis: all_flags.extend(founder_analysis.get("red_flags", []))

    return {
        "executive_summary": executive_summary,
        "overall_risk_score": overall_score,
        "investment_grade": grade,
        "confidence_level": confidence,
        "risk_breakdown": {
            k: {"score": v, "weight": f"{WEIGHTS[k]*100:.0f}%", "status": "analyzed" if v is not None else "pending"}
            for k,v in scores.items()
        },
        "all_red_flags": all_flags,
        "comparable_startups": similarity_results.get("comparable_startups",[]) if similarity_results else [],
        "dominant_failure_archetype": similarity_results.get("dominant_failure_archetype") if similarity_results else None,
        "key_claims": {
            "problem": extracted_claims.get("problem_statement") if extracted_claims else None,
            "solution": extracted_claims.get("solution_claim") if extracted_claims else None,
            "tam": extracted_claims.get("tam_claim") if extracted_claims else None,
            "revenue_model": extracted_claims.get("revenue_model") if extracted_claims else None,
        },
        "recommendations": _generate_recommendations(grade, scores, financial_metrics, market_analysis, founder_analysis),
    }


def _generate_recommendations(grade, scores, financial_metrics, market_analysis, founder_analysis) -> List[str]:
    recs = []
    if (scores.get("financial_risk") or 0) > 60:
        recs.append("Request updated financial model with documented assumptions.")
    if (scores.get("narrative_inflation") or 0) > 50:
        recs.append("Require third-party validation of market size claims.")
    if (scores.get("market_risk") or 0) > 60:
        recs.append("Conduct competitive positioning deep-dive — differentiation story is weak.")
    if (scores.get("founder_risk") or 0) > 60:
        recs.append("Evaluate team gaps; consider requiring key hires before close.")
    if (scores.get("pattern_similarity") or 0) > 65:
        recs.append("High failure pattern similarity — review comparable failure cases with founders.")
    if financial_metrics and (financial_metrics.get("runway_months") or 99) < 12:
        recs.append("Cash runway critical — confirm bridge financing or adjust valuation.")
    if founder_analysis and founder_analysis.get("prior_exits", 0) == 0:
        recs.append("No prior exits on founding team — weight team risk higher in final decision.")
    if market_analysis and market_analysis.get("tam_plausibility") == "low":
        recs.append("Request bottom-up TAM analysis with specific customer segment sizing.")
    grade_recs = {
        "F": "PASS — do not invest without major restructuring of deal terms.",
        "D": "PASS or require significant restructuring before proceeding.",
        "C": "Deeper diligence required. Focus on customer validation and unit economics.",
        "B": "Strong candidate. Proceed to references and customer conversations.",
        "A": "Excellent profile. Fast-track to partner review and term sheet.",
    }
    recs.append(grade_recs.get(grade, "Proceed with standard diligence process."))
    return recs