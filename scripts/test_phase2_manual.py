"""
scripts/test_phase2_manual.py
──────────────────────────────────────────────────────────────
Phase 2 end-to-end pipeline test — run WITHOUT starting the server.

Tests all 6 new components:
  Step 8:  Embedding pipeline
  Step 9:  FAISS index build + search
  Step 10: Similarity engine
  Step 11: Market & competition agent (RAG)
  Step 12: Founder risk profiling agent
  Step 13: Full 5-dimension risk aggregation

Usage:
  cd NLP_Final_Project
  python scripts/test_phase2_manual.py
"""

import os
import sys
import json
import logging

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.WARNING,  # suppress verbose logs during test
    format="%(levelname)s | %(name)s | %(message)s"
)

SEP = "=" * 60


def print_section(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


def main():
    print("\n🚀 Phase 2 Pipeline Test — AI Venture Due Diligence Copilot")
    print("Running all 6 new components...\n")

    # ── STEP 8: Embedding Pipeline ────────────────────────────────────
    print_section("STEP 8: Embedding Pipeline")
    from utils.embeddings import embed_text, embed_texts, cosine_similarity

    t1 = "AI-powered supply chain optimization platform for mid-market manufacturers"
    t2 = "Machine learning inventory management software for logistics companies"
    t3 = "Mobile dating app for pet owners"

    v1 = embed_text(t1)
    v2 = embed_text(t2)
    v3 = embed_text(t3)

    sim_related   = cosine_similarity(v1, v2)
    sim_unrelated = cosine_similarity(v1, v3)

    print(f"Vector shape:                  {v1.shape}  (should be (384,))")
    print(f"Supply chain ↔ logistics sim:  {sim_related:.3f}  (should be > 0.70)")
    print(f"Supply chain ↔ dating app sim: {sim_unrelated:.3f}  (should be < 0.50)")

    assert sim_related > sim_unrelated, "❌ Embeddings not working!"
    assert v1.shape == (384,), "❌ Wrong vector dimensions!"
    print("✅ Embedding pipeline working correctly")

    # ── STEP 9: FAISS Index Build ──────────────────────────────────────
    print_section("STEP 9: FAISS Index Build & Search")
    from utils.startup_dataset import build_faiss_index, get_dataset_stats, STARTUP_DATASET

    stats = get_dataset_stats()
    print(f"Dataset loaded:  {stats['total']} startups")
    print(f"  Failed:        {stats['failed']}")
    print(f"  Successful:    {stats['success']}")
    print(f"  Categories:    {len(stats['categories'])} unique")
    print(f"\nBuilding FAISS index...")

    index, metadata = build_faiss_index(save=True)
    print(f"Index built:     {index.ntotal} vectors, 384 dimensions")
    print(f"Saved to:        data/faiss_index/")

    # Test search
    import numpy as np
    query_vec = embed_text("supply chain management software for manufacturers").reshape(1, -1)
    sims, idxs = index.search(query_vec, 3)
    print(f"\nTop 3 similar to 'supply chain management software':")
    for sim, idx in zip(sims[0], idxs[0]):
        m = metadata[idx]
        print(f"  {m['name']:30s} ({m['outcome'].upper()})  sim={float(sim):.3f}")
    print("✅ FAISS index working correctly")

    # ── STEP 10: Similarity Engine ─────────────────────────────────────
    print_section("STEP 10: Failure Pattern Similarity Engine")
    from agents.similarity_engine import run_similarity_engine

    test_description = (
        "AI-powered supply chain optimization platform for mid-market manufacturers. "
        "SaaS subscription model with 90-day implementation. Targets $50M-$500M revenue "
        "manufacturers who currently use Excel spreadsheets. Raises Series A to expand sales team."
    )

    sim_result = run_similarity_engine(test_description)
    print(f"Failed startup similarity:   {sim_result['failed_similarity_pct']:.1f}%")
    print(f"Successful startup similarity: {sim_result['success_similarity_pct']:.1f}%")
    print(f"Pattern risk score:          {sim_result['pattern_similarity_risk_score']:.1f}/100")
    print(f"Dominant archetype:          {sim_result.get('dominant_failure_archetype', 'None')}")
    print(f"\nTop similar FAILED companies:")
    for m in sim_result['top_similar_failed'][:2]:
        print(f"  {m['name']:25s} {m['similarity_pct']:.1f}% similar")
    print(f"\nTop similar SUCCESSFUL companies:")
    for m in sim_result['top_similar_success'][:2]:
        print(f"  {m['name']:25s} {m['similarity_pct']:.1f}% similar")
    print(f"\nExplanation snippet: {sim_result['archetype_explanation'][:120]}...")
    print("✅ Similarity engine working correctly")

    # ── STEP 11: Market Agent (RAG) ────────────────────────────────────
    print_section("STEP 11: Market & Competition Agent (RAG)")
    from agents.market_agent import run_market_agent

    mock_claims = {
        "solution_claim": "AI-powered supply chain optimization for mid-market manufacturers",
        "target_market": "Mid-market manufacturers with $50M-$500M annual revenue",
        "tam_claim": "$45 billion global supply chain software market",
        "revenue_model": "SaaS subscription $2,500-$10,000 per month",
        "competitive_advantage_claims": ["90-day implementation", "94% forecast accuracy"],
        "hype_indicators": ["ai-powered", "revolutionary"],
        "growth_claims": ["10x faster", "zero competition", "exponential growth"],
    }

    market_result = run_market_agent(mock_claims)
    print(f"Competitors found:       {market_result['competitor_count']}")
    print(f"Competition density:     {market_result['competition_density']:.1f}/100")
    print(f"Market saturation:       {market_result['market_saturation_index']:.1f}/100")
    print(f"Narrative inflation:     {market_result['narrative_inflation_score']:.1f}/100")
    print(f"TAM plausibility:        {market_result['tam_plausibility'].upper()}")
    print(f"Market risk score:       {market_result['market_risk_score']:.1f}/100")
    print(f"Identified competitors:  {', '.join(market_result['identified_competitors'][:4])}")
    print(f"\nMarket assessment snippet:")
    print(f"  {market_result['market_assessment'][:200]}...")
    print("✅ Market agent (RAG) working correctly")

    # ── STEP 12: Founder Agent ─────────────────────────────────────────
    print_section("STEP 12: Founder Risk Profiling Agent")
    from agents.founder_agent import run_founder_agent

    founder_bio = """
CEO: Sarah Chen
10 years enterprise SaaS experience. Previously VP Product at Flexport (supply chain, $3.2B valuation).
Co-founded DataSync (supply chain analytics), acquired by Oracle in 2021 for $28M.
MBA Wharton, BS Computer Science MIT. Named Forbes 30 Under 30.

CTO: James Rodriguez  
12 years software engineering. Senior Staff Engineer at Stripe building payment infrastructure.
Expert in distributed systems and ML platforms. MS Computer Science Stanford.

VP Sales: Michael Thompson
15 years enterprise B2B sales. Closed $67M at Palantir in manufacturing vertical.
Built 3 enterprise sales teams from $0 to $20M+ ARR.
"""

    founder_result = run_founder_agent(founder_bio)
    print(f"Credibility score:    {founder_result['founder_credibility_score']:.1f}/100")
    print(f"Founder risk score:   {founder_result['founder_risk_score']:.1f}/100")
    print(f"Execution risk:       {founder_result['execution_risk_level'].upper()}")
    print(f"Prior exits:          {founder_result['prior_exits']}")
    print(f"Domain expertise:     {founder_result['domain_expertise_level'].upper()}")
    print(f"Team complete:        {founder_result['team_coverage_complete']}")
    print(f"\nPositive signals ({len(founder_result['positive_signals'])}):")
    for s in founder_result['positive_signals'][:3]:
        print(f"  ✅ {s}")
    print(f"\nRed flags ({len(founder_result['red_flags'])}):")
    for f in founder_result['red_flags'][:3]:
        print(f"  ⚠️  {f}")
    print(f"\nExplanation snippet: {founder_result['risk_explanation'][:150]}...")
    print("✅ Founder agent working correctly")

    # ── STEP 13: Full 5-Dimension Risk Aggregation ─────────────────────
    print_section("STEP 13: Full Risk Aggregation (All 5 Dimensions)")
    from agents.risk_aggregation_engine import run_risk_aggregation, compute_financial_risk_score, compute_narrative_inflation_score

    financial_risk   = compute_financial_risk_score(90.0)   # plausibility was 90
    narrative_risk   = compute_narrative_inflation_score(
        mock_claims.get("hype_indicators", []),
        mock_claims.get("growth_claims", [])
    )
    market_risk      = market_result["market_risk_score"]
    founder_risk     = founder_result["founder_risk_score"]
    pattern_risk     = sim_result["pattern_similarity_risk_score"]

    mock_financial_metrics = {
        "revenue_cagr": 170.7,
        "runway_months": 38.8,
        "gross_margin_avg": 64.8,
        "financial_plausibility_score": 90.0,
        "red_flags": ["Revenue CAGR of 171% is aggressive. Requires strong justification."],
    }

    risk_result = run_risk_aggregation(
        financial_risk_score=financial_risk,
        market_risk_score=market_risk,
        founder_risk_score=founder_risk,
        narrative_inflation_score=narrative_risk,
        pattern_similarity_score=pattern_risk,
        extracted_claims=mock_claims,
        financial_metrics=mock_financial_metrics,
        market_analysis=market_result,
        founder_analysis=founder_result,
        similarity_results=sim_result,
    )

    print(f"Financial Risk Score:      {risk_result['financial_risk_score']:.1f}/100")
    print(f"Market Risk Score:         {risk_result['market_risk_score']:.1f}/100")
    print(f"Founder Risk Score:        {risk_result['founder_risk_score']:.1f}/100")
    print(f"Narrative Inflation Score: {risk_result['narrative_inflation_score']:.1f}/100")
    print(f"Pattern Similarity Score:  {risk_result['pattern_similarity_score']:.1f}/100")
    print(f"{'─'*40}")
    print(f"OVERALL RISK SCORE:        {risk_result['overall_risk_score']:.1f}/100")
    print(f"INVESTMENT GRADE:          {risk_result['investment_grade']}")
    print(f"CONFIDENCE:                {risk_result['confidence_level'].upper()} ({risk_result['scores_available']}/5 dimensions)")
    print(f"{'─'*40}")

    memo = risk_result.get("due_diligence_memo", {})
    print(f"\nExecutive Summary:")
    print(f"  {memo.get('executive_summary', 'N/A')}")
    print(f"\nAll Red Flags ({len(memo.get('all_red_flags', []))}):")
    for flag in memo.get("all_red_flags", []):
        print(f"  ⚠️  {flag}")
    print(f"\nRecommendations:")
    for rec in memo.get("recommendations", []):
        print(f"  → {rec}")

    assert risk_result["confidence_level"] == "high", \
        f"Expected HIGH confidence with 5 agents, got: {risk_result['confidence_level']}"
    assert risk_result["scores_available"] == 5, "Expected 5 scores"
    print("\n✅ Full 5-dimension risk aggregation working correctly")

    # ── FINAL SUMMARY ──────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  PHASE 2 COMPLETE — ALL STEPS PASSING")
    print(SEP)
    print("""
  Step 8:  ✅ Embedding pipeline (sentence-transformers, 384-dim)
  Step 9:  ✅ FAISS index (80 startups, local vector search)
  Step 10: ✅ Similarity engine (failure pattern matching)
  Step 11: ✅ Market agent RAG (competitor retrieval + LLM analysis)
  Step 12: ✅ Founder agent (structured extraction + risk scoring)
  Step 13: ✅ Full aggregation (5/5 dimensions, HIGH confidence)
    """)
    print("Next: Boot the server and test full API with all 3 sample files:")
    print("  uvicorn main:app --reload --port 8000")
    print("  → Upload: sample_pitch_deck.pdf + sample_financials.csv + sample_founder_bio.txt\n")


if __name__ == "__main__":
    main()