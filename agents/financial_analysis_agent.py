"""
agents/financial_analysis_agent.py
────────────────────────────────────
WHAT THIS AGENT DOES:
  Takes the normalized financial data (from financial_parser.py) and:
    1. DETERMINISTICALLY computes financial metrics using Pandas/NumPy
       (no LLM involved here — numbers are numbers)
    2. DETECTS anomalies and red flags using rule-based logic
    3. Calls the LLM ONLY to write human-readable explanations of
       what the numbers mean and why they're flagged

WHY HYBRID (DETERMINISTIC + LLM)?
  Pure LLM: "The revenue looks aggressive" — vague, hallucination-prone
  Pure code: "CAGR=342%, spike in Y3=8x" — accurate but no context
  Hybrid:    Code detects the anomaly; LLM explains why it matters
             to a VC evaluating this specific startup.

METRICS COMPUTED:
  ┌────────────────────────────────┬────────────────────────────────────┐
  │ Metric                         │ What it tells you                  │
  ├────────────────────────────────┼────────────────────────────────────┤
  │ Revenue CAGR                   │ How fast revenue is claimed to grow │
  │ Monthly Burn Rate              │ Cash spent per month                │
  │ Runway (months)                │ How long before cash runs out       │
  │ CAC/LTV Ratio                  │ Unit economics health               │
  │ Gross Margin consistency       │ Margin stability across years       │
  │ Growth spike detection         │ Unrealistic single-year jumps       │
  └────────────────────────────────┴────────────────────────────────────┘

RED FLAG THRESHOLDS (tunable):
  - CAGR > 300%  → "Extremely aggressive growth assumption"
  - Runway < 6 months → "Critical cash position"
  - CAC > LTV → "Unit economics inverted (losing money per customer)"
  - Gross margin drops > 20pp year-over-year → "Margin compression"
  - Single year revenue jump > 5x → "Unrealistic spike"

FINANCIAL PLAUSIBILITY SCORE:
  Starts at 100. Each red flag deducts points.
  Returns 0–100 (higher = more financially plausible).
"""

import logging
import math
from typing import Dict, Any, List, Optional, Tuple

import numpy as np

from utils.llm_client import call_llm

logger = logging.getLogger(__name__)


# ── Thresholds ──────────────────────────────────────────────────────────────
CAGR_WARNING_THRESHOLD = 1.5     # 150% CAGR is aggressive
CAGR_RED_FLAG_THRESHOLD = 3.0    # 300% CAGR is a red flag
RUNWAY_WARNING_MONTHS = 12
RUNWAY_CRITICAL_MONTHS = 6
MAX_SINGLE_YEAR_GROWTH = 5.0     # 5x in one year = spike
MARGIN_DROP_THRESHOLD = 0.20     # 20 percentage point drop


def run_financial_analysis(financial_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for the Financial Analysis Engine.

    Args:
        financial_data: Normalized dict from financial_parser.parse_financial_file()

    Returns:
        {
          "revenue_cagr": float,
          "burn_rate_monthly": float,
          "runway_months": float,
          "cac_ltv_ratio": float,
          "gross_margin_avg": float,
          "gross_margin_consistent": bool,
          "red_flags": [str, ...],
          "unrealistic_growth_spikes": [{year, growth_pct}, ...],
          "financial_plausibility_score": 0–100,
          "anomaly_explanation": str   ← LLM-generated
        }
    """
    red_flags = []
    spikes = []

    revenue = _clean_series(financial_data.get("revenue"))
    costs = _clean_series(financial_data.get("costs"))
    gross_profit = _clean_series(financial_data.get("gross_profit"))
    cash_balance = _clean_series(financial_data.get("cash_balance"))
    burn_rate_series = _clean_series(financial_data.get("burn_rate"))
    years = financial_data.get("years", [])
    cac_raw = financial_data.get("cac")
    ltv_raw = financial_data.get("ltv")

    # ── 1. Revenue CAGR ────────────────────────────────────────────────
    revenue_cagr = None
    if revenue and len(revenue) >= 2:
        revenue_cagr, cagr_flags = _compute_cagr(revenue, years)
        red_flags.extend(cagr_flags)

    # ── 2. Growth Spikes ───────────────────────────────────────────────
    if revenue and len(revenue) >= 2:
        spikes, spike_flags = _detect_growth_spikes(revenue, years)
        red_flags.extend(spike_flags)

    # ── 3. Burn Rate & Runway ──────────────────────────────────────────
    burn_rate_monthly = None
    runway_months = None

    if burn_rate_series:
        burn_rate_monthly = abs(np.mean(burn_rate_series))
    elif costs and revenue:
        # Estimate: if costs > revenue, the difference is the burn
        latest_cost = costs[-1]
        latest_rev = revenue[-1]
        if latest_cost > latest_rev:
            burn_rate_monthly = (latest_cost - latest_rev) / 12
        else:
            burn_rate_monthly = latest_cost / 12  # use total costs as proxy

    if burn_rate_monthly and burn_rate_monthly > 0 and cash_balance:
        runway_months = cash_balance[-1] / burn_rate_monthly
        if runway_months < RUNWAY_CRITICAL_MONTHS:
            red_flags.append(f"CRITICAL: Only {runway_months:.1f} months of runway at current burn rate")
        elif runway_months < RUNWAY_WARNING_MONTHS:
            red_flags.append(f"WARNING: Runway is {runway_months:.1f} months — less than 12 months")

    # ── 4. CAC / LTV Ratio ─────────────────────────────────────────────
    cac_ltv_ratio = None
    if cac_raw and ltv_raw:
        try:
            cac = float(str(cac_raw).replace(',', '').replace('$', ''))
            ltv = float(str(ltv_raw).replace(',', '').replace('$', ''))
            if ltv > 0:
                cac_ltv_ratio = cac / ltv
                if cac_ltv_ratio > 1.0:
                    red_flags.append(f"Unit economics inverted: CAC (${cac:,.0f}) > LTV (${ltv:,.0f}). Losing money per customer.")
                elif cac_ltv_ratio > 0.5:
                    red_flags.append(f"CAC/LTV ratio of {cac_ltv_ratio:.2f} is high. LTV:CAC below the healthy 3:1 threshold.")
        except (ValueError, TypeError):
            pass

    # ── 5. Gross Margin ────────────────────────────────────────────────
    gross_margin_avg = None
    gross_margin_consistent = None
    if gross_profit and revenue and len(gross_profit) == len(revenue):
        margins = []
        for gp, rev in zip(gross_profit, revenue):
            if rev and rev > 0:
                margins.append(gp / rev)
        if margins:
            gross_margin_avg = float(np.mean(margins))
            # Check for drops > threshold
            drops = []
            for i in range(1, len(margins)):
                drop = margins[i - 1] - margins[i]
                if drop > MARGIN_DROP_THRESHOLD:
                    year_label = years[i] if i < len(years) else f"Year {i+1}"
                    drops.append(f"{year_label}: margin dropped {drop:.0%}")
            gross_margin_consistent = len(drops) == 0
            if drops:
                red_flags.append(f"Gross margin compression detected: {'; '.join(drops)}")

    # ── 6. Compute Plausibility Score ──────────────────────────────────
    plausibility_score = _compute_plausibility_score(red_flags, revenue_cagr, runway_months)

    # ── 7. LLM Explanation of Anomalies ───────────────────────────────
    anomaly_explanation = _generate_anomaly_explanation(
        red_flags=red_flags,
        revenue_cagr=revenue_cagr,
        burn_rate_monthly=burn_rate_monthly,
        runway_months=runway_months,
        cac_ltv_ratio=cac_ltv_ratio,
        gross_margin_avg=gross_margin_avg,
        plausibility_score=plausibility_score,
    )

    return {
        "revenue_cagr": round(revenue_cagr * 100, 1) if revenue_cagr is not None else None,
        "burn_rate_monthly": round(burn_rate_monthly, 0) if burn_rate_monthly else None,
        "runway_months": round(runway_months, 1) if runway_months else None,
        "cac_ltv_ratio": round(cac_ltv_ratio, 3) if cac_ltv_ratio else None,
        "gross_margin_avg": round(gross_margin_avg * 100, 1) if gross_margin_avg is not None else None,
        "gross_margin_consistent": gross_margin_consistent,
        "red_flags": red_flags,
        "unrealistic_growth_spikes": spikes,
        "financial_plausibility_score": plausibility_score,
        "anomaly_explanation": anomaly_explanation,
    }


# ── Internal Computation Functions ──────────────────────────────────────────

def _clean_series(series) -> Optional[List[float]]:
    """Filter None values and convert to floats. Return None if empty."""
    if not series:
        return None
    cleaned = [float(v) for v in series if v is not None and not (isinstance(v, float) and math.isnan(v))]
    return cleaned if len(cleaned) >= 1 else None


def _compute_cagr(revenue: List[float], years: List[int]) -> Tuple[float, List[str]]:
    """
    Compute Compound Annual Growth Rate from first to last revenue figure.
    CAGR = (end/start)^(1/n) - 1 where n = number of years
    """
    flags = []
    n_years = len(years) - 1 if len(years) >= 2 else len(revenue) - 1
    if n_years <= 0 or revenue[0] <= 0:
        return None, flags

    cagr = (revenue[-1] / revenue[0]) ** (1 / n_years) - 1

    if cagr > CAGR_RED_FLAG_THRESHOLD:
        flags.append(f"Revenue CAGR of {cagr:.0%} is extremely aggressive. Median funded SaaS CAGR is 80–150%.")
    elif cagr > CAGR_WARNING_THRESHOLD:
        flags.append(f"Revenue CAGR of {cagr:.0%} is aggressive. Requires strong justification.")

    return cagr, flags


def _detect_growth_spikes(revenue: List[float], years: List[int]) -> Tuple[List[Dict], List[str]]:
    """Detect single-year revenue jumps exceeding MAX_SINGLE_YEAR_GROWTH."""
    spikes = []
    flags = []
    for i in range(1, len(revenue)):
        if revenue[i - 1] > 0:
            growth = revenue[i] / revenue[i - 1]
            if growth > MAX_SINGLE_YEAR_GROWTH:
                year_label = years[i] if i < len(years) else f"Year {i + 1}"
                spikes.append({"year": year_label, "growth_multiple": round(growth, 1)})
                flags.append(f"Unrealistic spike: {growth:.1f}x revenue growth in {year_label} with no explanation")
    return spikes, flags


def _compute_plausibility_score(
    red_flags: List[str],
    cagr: Optional[float],
    runway: Optional[float]
) -> float:
    """
    Start at 100. Deduct points per red flag category.
    Returns 0–100 (higher = more plausible).
    """
    score = 100.0

    # Each red flag costs 10–20 points depending on severity
    for flag in red_flags:
        if "CRITICAL" in flag:
            score -= 20
        elif "inverted" in flag or "unrealistic" in flag.lower() or "spike" in flag.lower():
            score -= 15
        else:
            score -= 10

    # Extra penalty for extreme CAGR
    if cagr and cagr > 5.0:  # >500%
        score -= 15

    return max(0.0, min(100.0, round(score, 1)))


def _generate_anomaly_explanation(
    red_flags: List[str],
    revenue_cagr: Optional[float],
    burn_rate_monthly: Optional[float],
    runway_months: Optional[float],
    cac_ltv_ratio: Optional[float],
    gross_margin_avg: Optional[float],
    plausibility_score: float,
) -> str:
    """
    Use the LLM to write a 2–3 paragraph human-readable summary
    of the financial anomalies detected. The LLM explains WHY
    each anomaly matters — context a VC analyst cares about.
    """
    if not red_flags:
        return "No significant financial anomalies detected. The financial projections appear reasonable."

    flags_text = "\n".join(f"- {f}" for f in red_flags)
    metrics_summary = (
        f"Revenue CAGR: {revenue_cagr:.1f}%" if revenue_cagr is not None else ""
        + (f"\nMonthly burn: ${burn_rate_monthly:,.0f}" if burn_rate_monthly else "")
        + (f"\nRunway: {runway_months:.1f} months" if runway_months else "")
        + (f"\nCAC/LTV ratio: {cac_ltv_ratio:.2f}" if cac_ltv_ratio else "")
        + (f"\nAvg gross margin: {gross_margin_avg:.1f}%" if gross_margin_avg else "")
    )

    system = """You are a senior VC financial analyst writing a concise financial risk assessment 
for an investment memo. You are direct, analytical, and flag concerns clearly without being alarmist.
Write 2–3 short paragraphs. No lists. No headers. Plain prose."""

    prompt = f"""Based on these detected financial anomalies, write a concise risk assessment 
explaining what each issue means for an investor evaluating this startup.

Financial Plausibility Score: {plausibility_score}/100

Key Metrics:
{metrics_summary}

Detected Red Flags:
{flags_text}

Write a clear, professional 2–3 paragraph explanation of the financial risk profile."""

    try:
        return call_llm(prompt=prompt, system=system, max_tokens=600, temperature=0.2)
    except Exception as e:
        logger.error(f"LLM explanation failed: {e}")
        return f"Automated financial analysis detected {len(red_flags)} red flag(s): {'; '.join(red_flags[:3])}"