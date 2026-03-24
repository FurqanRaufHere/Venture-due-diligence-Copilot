"""
scripts/test_pipeline_manual.py
──────────────────────────────
HOW TO USE:
  This script tests the full Phase 1 pipeline WITHOUT needing to
  run the FastAPI server. Good for rapid development iteration.

  Run it with:
    cd ai_vdd
    python scripts/test_pipeline_manual.py

  It will:
    1. Parse the sample financial CSV
    2. Run financial analysis
    3. Run a mock claim extraction (or real LLM if key is set)
    4. Run risk aggregation
    5. Print a formatted report to the terminal

REQUIREMENTS:
  - Copy .env.example to .env and add your API key
  - pip install -r requirements.txt
"""

import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from utils.financial_parser import parse_financial_file
from agents.financial_analysis_agent import run_financial_analysis
from agents.risk_aggregation_engine import (
    run_risk_aggregation,
    compute_financial_risk_score,
    compute_narrative_inflation_score,
)


def print_section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def print_json(data):
    print(json.dumps(data, indent=2, default=str))


def main():
    sample_csv = os.path.join(os.path.dirname(__file__), "../sample_data/sample_financials.csv")

    # ── Step 1: Parse financials ──────────────────────────────────────
    print_section("STEP 1: Financial Parser")
    financial_data = parse_financial_file(sample_csv)
    print(f"Years detected: {financial_data['years']}")
    print(f"Revenue series: {financial_data['revenue']}")
    print(f"Detection log: {financial_data['detection_log']}")

    # ── Step 2: Financial Analysis ────────────────────────────────────
    print_section("STEP 2: Financial Analysis Engine")
    fin_result = run_financial_analysis(financial_data)
    print(f"Revenue CAGR:              {fin_result['revenue_cagr']}%")
    print(f"Burn rate (monthly):       ${fin_result.get('burn_rate_monthly', 'N/A'):,}" if fin_result.get('burn_rate_monthly') else "Burn rate: N/A")
    print(f"Runway:                    {fin_result.get('runway_months', 'N/A')} months")
    print(f"Gross margin avg:          {fin_result.get('gross_margin_avg', 'N/A')}%")
    print(f"Financial plausibility:    {fin_result['financial_plausibility_score']}/100")
    print(f"\nRed flags ({len(fin_result['red_flags'])}):")
    for flag in fin_result['red_flags']:
        print(f"  ⚠️  {flag}")
    print(f"\nAnomaly explanation:\n{fin_result.get('anomaly_explanation', 'N/A')}")

    # ── Step 3: Claim extraction (mocked for this script) ─────────────
    print_section("STEP 3: Claim Extraction (using mock data)")
    claims_result = {
        "problem_statement": "Supply chain management is fragmented and manual for mid-market companies",
        "solution_claim": "AI-powered supply chain optimization platform with real-time visibility",
        "target_market": "Mid-market manufacturers ($50M–$500M revenue)",
        "tam_claim": "$45B global supply chain software market",
        "revenue_model": "SaaS subscription, $2,500–$15,000/month per customer",
        "growth_claims": ["3x YoY revenue", "40% MoM user growth", "exponential scaling"],
        "competitive_advantage_claims": ["Proprietary AI", "10x faster than competitors"],
        "hype_indicators": ["ai-powered", "exponential growth", "revolutionize"],
        "confidence_score": 0.9,
    }
    print(f"Problem: {claims_result['problem_statement']}")
    print(f"TAM claim: {claims_result['tam_claim']}")
    print(f"Hype phrases detected: {claims_result['hype_indicators']}")
    print(f"Confidence: {claims_result['confidence_score']:.0%}")

    # ── Step 4: Risk Aggregation ──────────────────────────────────────
    print_section("STEP 4: Risk Aggregation Engine")
    financial_risk = compute_financial_risk_score(fin_result['financial_plausibility_score'])
    narrative_risk = compute_narrative_inflation_score(
        claims_result['hype_indicators'],
        claims_result['growth_claims']
    )
    print(f"Financial Risk Score:       {financial_risk}/100")
    print(f"Narrative Inflation Score:  {narrative_risk}/100")

    risk_result = run_risk_aggregation(
        financial_risk_score=financial_risk,
        narrative_inflation_score=narrative_risk,
        extracted_claims=claims_result,
        financial_metrics=fin_result,
    )

    print(f"\n{'─'*40}")
    print(f"  OVERALL RISK SCORE:  {risk_result['overall_risk_score']}/100")
    print(f"  INVESTMENT GRADE:    {risk_result['investment_grade']}")
    print(f"  CONFIDENCE:          {risk_result['confidence_level'].upper()}")
    print(f"{'─'*40}")

    memo = risk_result.get('due_diligence_memo', {})
    if memo:
        print(f"\nExecutive Summary:\n{memo.get('executive_summary', 'N/A')}")
        print(f"\nRecommendations:")
        for rec in memo.get('recommendations', []):
            print(f"  → {rec}")

    print(f"\n Phase 1 pipeline test complete.\n")


if __name__ == "__main__":
    main()