"""
agents/similarity_engine.py
STEP 10 — Failure Pattern Similarity Engine

WHAT THIS DOES:
  Takes a startup description (from the pitch deck) and finds the
  most similar startups from our historical dataset.

  Then answers two key questions for the investor:
    1. How similar is this to startups that FAILED? (0-100%)
    2. How similar is this to startups that SUCCEEDED? (0-100%)

HOW THE SEARCH WORKS:
  1. Embed the new startup description → 384-dim vector
  2. Search FAISS index → find top-10 most similar historical startups
  3. Separate results into failed vs. successful groups
  4. Compute weighted average similarity for each group
  5. Classify into a named failure archetype if similarity > 0.65

THE SIMILARITY SCORE (0-100):
  Higher score = more similar to that group
  
  Example output:
    failed_similarity:  72%  ← this startup looks like past failures
    success_similarity: 45%  ← doesn't look like successful patterns
    dominant_archetype: "unit_economics"  ← main failure pattern matched

PATTERN SIMILARITY RISK SCORE:
  Combines both similarity scores into a single risk number:
    High failed_similarity + low success_similarity → HIGH RISK
    Low failed_similarity + high success_similarity → LOW RISK
  
  Formula: risk = (failed_similarity * 0.7) - (success_similarity * 0.3) + 50
  Capped at 0-100.

WHY THIS MATTERS:
  Most due diligence misses pattern recognition because analysts
  don't have time to study 500 past failures.
  This engine does it in milliseconds.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

# Minimum similarity to consider a match meaningful
SIMILARITY_THRESHOLD = 0.45

# How many results to retrieve from FAISS before filtering
TOP_K = 10

# Human-readable names for failure archetypes
ARCHETYPE_LABELS = {
    "unit_economics":       "Poor Unit Economics (CAC > LTV, unsustainable margins)",
    "premature_scaling":    "Premature Scaling (grew before product-market fit)",
    "no_product_market_fit":"No Product-Market Fit (solution looking for a problem)",
    "competition":          "Competitive Displacement (outcompeted by better-funded rival)",
    "founder_issues":       "Founder/Team Risk (culture, ethics, or execution problems)",
    "wrong_timing":         "Wrong Market Timing (too early or too late)",
    "market_conditions":    "Market Conditions (external forces disrupted the model)",
    "fraud":                "Integrity Risk (misrepresentation of product or financials)",
    "developer_tools":      "Developer Tools (strong technical foundation)",
    "b2b_saas":            "B2B SaaS (proven enterprise software model)",
    "vertical_saas":       "Vertical SaaS (deep industry focus)",
    "marketplace":         "Marketplace (network effects business model)",
    "fintech":             "Fintech (financial services innovation)",
    "ai_ml":               "AI/ML Infrastructure (AI tooling and data platform)",
}


def run_similarity_engine(startup_description: str) -> Dict[str, Any]:
    """
    Main entry point for the Failure Pattern Similarity Engine.
    
    Args:
        startup_description: 2-5 sentence description of the startup
                            (can be auto-generated from extracted claims)
    
    Returns:
        {
          "failed_similarity_pct": 0–100,
          "success_similarity_pct": 0–100,
          "pattern_similarity_risk_score": 0–100,
          "dominant_failure_archetype": str or None,
          "top_similar_failed": [{name, similarity, category, failure_reason}],
          "top_similar_success": [{name, similarity, category}],
          "archetype_explanation": str,
          "comparable_startups": [str],
        }
    """
    from utils.embeddings import embed_text
    from utils.startup_dataset import load_faiss_index
    
    logger.info("Running similarity engine...")
    
    if not startup_description or len(startup_description.strip()) < 20:
        logger.warning("Startup description too short for meaningful similarity analysis")
        return _empty_result()
    
    # Step 1: Embed the new startup
    query_vector = embed_text(startup_description)
    query_vector = query_vector.reshape(1, -1)  # FAISS expects (n, dim)
    
    # Step 2: Load index and search
    index, metadata = load_faiss_index()
    
    # Search for top-K most similar startups
    # Returns: distances (similarity scores), indices (positions in index)
    similarities, indices = index.search(query_vector, TOP_K)
    similarities = similarities[0]  # flatten from (1, k) to (k,)
    indices = indices[0]
    
    # Step 3: Organize results by outcome
    failed_matches = []
    success_matches = []
    
    for sim, idx in zip(similarities, indices):
        if idx < 0 or sim < SIMILARITY_THRESHOLD:
            continue
        
        entry = metadata[idx]
        match = {
            "name": entry["name"],
            "similarity_pct": round(float(sim) * 100, 1),
            "category": entry["category"],
            "failure_reason": entry.get("failure_reason"),
            "description_snippet": entry["description"][:120] + "...",
        }
        
        if entry["outcome"] == "failed":
            failed_matches.append(match)
        else:
            success_matches.append(match)
    
    # Sort by similarity descending
    failed_matches.sort(key=lambda x: x["similarity_pct"], reverse=True)
    success_matches.sort(key=lambda x: x["similarity_pct"], reverse=True)
    
    # Step 4: Compute aggregate similarity scores
    failed_sim_pct = _weighted_avg_similarity(failed_matches)
    success_sim_pct = _weighted_avg_similarity(success_matches)
    
    # Step 5: Identify dominant failure archetype
    dominant_archetype = _identify_dominant_archetype(failed_matches)
    
    # Step 6: Compute risk score
    risk_score = _compute_pattern_risk_score(failed_sim_pct, success_sim_pct)
    
    # Step 7: Build explanation
    archetype_explanation = _build_archetype_explanation(
        dominant_archetype, failed_matches, success_matches,
        failed_sim_pct, success_sim_pct
    )
    
    # Comparable startup names for the report
    comparable = [m["name"] for m in (failed_matches[:2] + success_matches[:2])]
    
    result = {
        "failed_similarity_pct": failed_sim_pct,
        "success_similarity_pct": success_sim_pct,
        "pattern_similarity_risk_score": risk_score,
        "dominant_failure_archetype": dominant_archetype,
        "archetype_label": ARCHETYPE_LABELS.get(dominant_archetype, dominant_archetype),
        "top_similar_failed": failed_matches[:3],
        "top_similar_success": success_matches[:3],
        "archetype_explanation": archetype_explanation,
        "comparable_startups": comparable,
        "total_matches_found": len(failed_matches) + len(success_matches),
    }
    
    logger.info(
        f"Similarity engine complete. "
        f"Failed similarity: {failed_sim_pct}%, "
        f"Success similarity: {success_sim_pct}%, "
        f"Archetype: {dominant_archetype}"
    )
    return result


def build_startup_description_from_claims(claims: Dict) -> str:
    """
    Auto-generate a startup description from extracted claims.
    Used when no manual description is provided.
    
    Takes the structured claims dict from claim_extraction_agent
    and combines them into a coherent paragraph for embedding.
    """
    parts = []
    
    if claims.get("solution_claim"):
        parts.append(claims["solution_claim"])
    if claims.get("target_market"):
        parts.append(f"Targeting {claims['target_market']}.")
    if claims.get("revenue_model"):
        parts.append(f"Revenue model: {claims['revenue_model']}.")
    if claims.get("tam_claim"):
        parts.append(f"Addressing a {claims['tam_claim']} market.")
    if claims.get("competitive_advantage_claims"):
        advantages = claims["competitive_advantage_claims"]
        if advantages:
            parts.append(f"Key advantages: {advantages[0]}.")
    
    description = " ".join(parts)
    
    # Fallback if extraction was poor
    if len(description) < 30:
        description = claims.get("problem_statement", "Early stage technology startup")
    
    return description


# ── Internal Helpers ──────────────────────────────────────────────────────────

def _weighted_avg_similarity(matches: List[Dict]) -> float:
    """
    Compute a weighted average similarity.
    Top matches get more weight than lower ones.
    Returns 0–100.
    """
    if not matches:
        return 0.0
    
    # Weights: top match gets 3x weight, second gets 2x, rest get 1x
    weights = [3, 2, 1, 1, 1]
    total_weight = 0
    weighted_sum = 0
    
    for i, match in enumerate(matches[:5]):
        w = weights[i] if i < len(weights) else 1
        weighted_sum += match["similarity_pct"] * w
        total_weight += w
    
    return round(weighted_sum / total_weight, 1)


def _identify_dominant_archetype(failed_matches: List[Dict]) -> Optional[str]:
    """
    Find the most common failure category among matched startups.
    Only returns an archetype if similarity is strong enough.
    """
    if not failed_matches:
        return None
    
    # Only consider high-similarity matches
    strong_matches = [m for m in failed_matches if m["similarity_pct"] > 55]
    if not strong_matches:
        return None
    
    # Count categories
    category_counts = {}
    for m in strong_matches:
        cat = m["category"]
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    return max(category_counts, key=category_counts.get)


def _compute_pattern_risk_score(failed_sim: float, success_sim: float) -> float:
    """
    Convert similarity scores into a 0–100 risk score.
    
    Logic:
      - High similarity to failures + low similarity to successes = HIGH RISK
      - Low similarity to failures + high similarity to successes = LOW RISK
      - Baseline: 50 (uncertain)
    """
    # failed pulls score UP (more risk), success pulls it DOWN (less risk)
    raw = 50 + (failed_sim * 0.5) - (success_sim * 0.3)
    return round(max(0.0, min(100.0, raw)), 1)


def _build_archetype_explanation(
    archetype: Optional[str],
    failed_matches: List[Dict],
    success_matches: List[Dict],
    failed_sim: float,
    success_sim: float,
) -> str:
    """Build a human-readable explanation of the pattern analysis."""
    lines = []
    
    if failed_matches:
        top_failed = failed_matches[0]
        lines.append(
            f"This startup shows {failed_sim:.0f}% pattern similarity to historically failed startups. "
            f"Closest match: {top_failed['name']} ({top_failed['similarity_pct']:.0f}% similar)"
            + (f" — {top_failed['failure_reason']}" if top_failed.get('failure_reason') else "") + "."
        )
    
    if archetype:
        label = ARCHETYPE_LABELS.get(archetype, archetype)
        lines.append(f"Dominant failure archetype: {label}.")
    
    if success_matches:
        top_success = success_matches[0]
        lines.append(
            f"Also shows {success_sim:.0f}% similarity to successful companies like "
            f"{top_success['name']} ({top_success['similarity_pct']:.0f}% similar)."
        )
    
    if not lines:
        lines.append("Insufficient similarity data to identify clear pattern archetypes.")
    
    return " ".join(lines)


def _empty_result() -> Dict:
    """Return empty result when description is insufficient."""
    return {
        "failed_similarity_pct": 0.0,
        "success_similarity_pct": 0.0,
        "pattern_similarity_risk_score": 50.0,
        "dominant_failure_archetype": None,
        "archetype_label": "Insufficient data for pattern matching",
        "top_similar_failed": [],
        "top_similar_success": [],
        "archetype_explanation": "Description too short for meaningful pattern analysis.",
        "comparable_startups": [],
        "total_matches_found": 0,
    }