"""
agents/market_agent.py
────────────────────────────────────────────────────────────────
STEP 11 — Market & Competition Agent (RAG)

WHAT IS RAG?
  RAG = Retrieval-Augmented Generation
  
  Instead of asking the LLM "what competitors exist for X?" 
  (which could hallucinate), we:
    1. RETRIEVE real competitor data from our vector database
    2. AUGMENT the LLM prompt with that retrieved data
    3. GENERATE analysis grounded in actual retrieved facts
  
  This is the difference between:
    LLM alone: "The main competitors are [hallucinated list]"
    RAG:       "Based on retrieved data, 8 similar companies exist: [actual list]"

WHAT THIS AGENT DOES:
  Step 1 — Extract keywords from the startup's claimed market
           (from ExtractedClaims: target_market, solution_claim, tam_claim)
  
  Step 2 — Search the FAISS index for similar companies
           This is our RAG retrieval step — finding real comparable companies
  
  Step 3 — Compute competition metrics:
           • competition_density: how many similar companies exist (0–100)
           • market_saturation_index: how crowded the space is (0–100)
           • narrative_inflation_score: is the TAM claim realistic?
  
  Step 4 — LLM analyzes the retrieved competitors and writes
           a structured market assessment (grounded in real data)

OUTPUT:
  {
    "competition_density": 0–100,
    "market_saturation_index": 0–100,  
    "narrative_inflation_score": 0–100,
    "competitor_count": int,
    "identified_competitors": [list of company names],
    "market_assessment": str (LLM-generated, grounded in retrieved data),
    "tam_plausibility": "low" / "medium" / "high",
    "market_risk_score": 0–100
  }
"""

import logging
import re
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

# Market size keywords that indicate potentially inflated TAM claims
INFLATED_TAM_PATTERNS = [
    r"\$\d+\s*[Tt]rillion",        # "$X trillion"
    r"\$\d{3,}\s*[Bb]illion",      # "$100+ billion" 
    r"total.*market.*\$\d+[BT]",   # "total market $XB"
]

# Competitive intensity thresholds
DENSITY_THRESHOLDS = {
    "low":    (0,  3),   # 0-3 similar companies = low competition
    "medium": (4,  7),   # 4-7 similar companies = moderate
    "high":   (8, 100),  # 8+ similar companies = saturated
}


def run_market_agent(extracted_claims: Dict, top_k: int = 10) -> Dict[str, Any]:
    """
    Main entry point for the Market & Competition Agent.
    
    Args:
        extracted_claims: Dict from claim_extraction_agent containing
                         target_market, solution_claim, tam_claim, etc.
        top_k: How many similar companies to retrieve from FAISS
    
    Returns:
        Full market analysis dict
    """
    from utils.embeddings import embed_text
    from utils.startup_dataset import load_faiss_index
    from utils.llm_client import call_llm
    
    logger.info("Running market & competition agent...")
    
    # ── Step 1: Build search query from claims ────────────────────────
    search_text = _build_market_search_query(extracted_claims)
    logger.info(f"Market search query: '{search_text[:100]}'")
    
    # ── Step 2: RAG Retrieval — find similar companies ─────────────────
    query_vec = embed_text(search_text).reshape(1, -1)
    index, metadata = load_faiss_index()
    
    similarities, indices = index.search(query_vec, top_k)
    similarities = similarities[0]
    indices = indices[0]
    
    # Collect all retrieved companies (both failed and successful)
    retrieved_companies = []
    for sim, idx in zip(similarities, indices):
        if idx < 0 or sim < 0.40:  # lower threshold to find more competitors
            continue
        entry = metadata[idx]
        retrieved_companies.append({
            "name": entry["name"],
            "similarity": round(float(sim) * 100, 1),
            "outcome": entry["outcome"],
            "category": entry["category"],
            "description_snippet": entry["description"][:100],
        })
    
    # ── Step 3: Compute competition metrics ───────────────────────────
    competitor_count = len(retrieved_companies)
    competition_density = _compute_competition_density(competitor_count, similarities)
    market_saturation = _compute_market_saturation(retrieved_companies)
    tam_plausibility = _assess_tam_plausibility(
        extracted_claims.get("tam_claim", ""),
        extracted_claims.get("target_market", ""),
    )
    narrative_inflation = _compute_narrative_inflation(
        extracted_claims,
        competition_density,
        tam_plausibility,
    )
    
    # ── Step 4: LLM generates market assessment ────────────────────────
    market_assessment = _generate_market_assessment(
        claims=extracted_claims,
        retrieved_companies=retrieved_companies,
        competition_density=competition_density,
        market_saturation=market_saturation,
        tam_plausibility=tam_plausibility,
    )
    
    # ── Step 5: Compute market risk score ─────────────────────────────
    market_risk_score = _compute_market_risk_score(
        competition_density=competition_density,
        market_saturation=market_saturation,
        narrative_inflation=narrative_inflation,
        tam_plausibility=tam_plausibility,
    )
    
    result = {
        "competition_density": competition_density,
        "market_saturation_index": market_saturation,
        "narrative_inflation_score": narrative_inflation,
        "tam_plausibility": tam_plausibility,
        "competitor_count": competitor_count,
        "identified_competitors": [c["name"] for c in retrieved_companies[:8]],
        "retrieved_companies": retrieved_companies[:8],
        "market_assessment": market_assessment,
        "market_risk_score": market_risk_score,
    }
    
    logger.info(
        f"Market agent complete. "
        f"Competitors found: {competitor_count}, "
        f"Density: {competition_density}/100, "
        f"Market risk: {market_risk_score}/100"
    )
    return result


# ── Internal Functions ────────────────────────────────────────────────────────

def _build_market_search_query(claims: Dict) -> str:
    """
    Combine the most relevant claim fields into a search query.
    This determines WHAT we search for in our vector database.
    """
    parts = []
    
    # Solution is most important for finding competitors
    if claims.get("solution_claim"):
        parts.append(claims["solution_claim"])
    if claims.get("target_market"):
        parts.append(claims["target_market"])
    if claims.get("revenue_model"):
        parts.append(claims["revenue_model"])
    
    query = " ".join(parts)
    return query if query.strip() else "technology startup software platform"


def _compute_competition_density(competitor_count: int, similarities: Any) -> float:
    """
    Convert competitor count and similarity scores into a density score (0–100).
    More competitors + higher similarities = higher density = more saturated.
    """
    if competitor_count == 0:
        return 10.0  # No matches = very niche or very novel (low density)
    
    # Base score from count
    if competitor_count >= 8:
        base = 80
    elif competitor_count >= 5:
        base = 60
    elif competitor_count >= 3:
        base = 40
    else:
        base = 20
    
    # Bonus for high-similarity matches (truly close competitors)
    high_sim = sum(1 for s in similarities if float(s) > 0.65)
    base += high_sim * 5
    
    return min(100.0, round(float(base), 1))


def _compute_market_saturation(companies: List[Dict]) -> float:
    """
    Market saturation = proportion of the retrieved companies that
    are in the same category, indicating the market is well-defined
    and populated.
    """
    if not companies:
        return 20.0
    
    # If many companies cluster in same category, market is saturated
    categories = [c["category"] for c in companies]
    most_common_cat_count = max(categories.count(c) for c in set(categories))
    saturation = (most_common_cat_count / len(companies)) * 100
    
    # Scale: 100% same category = 90 saturation, all different = 20
    scaled = 20 + (saturation * 0.7)
    return round(min(100.0, scaled), 1)


def _assess_tam_plausibility(tam_claim: str, target_market: str) -> str:
    """
    Simple rule-based TAM plausibility check.
    Returns "low", "medium", or "high" plausibility.
    """
    if not tam_claim:
        return "medium"
    
    tam_lower = tam_claim.lower()
    
    # Red flag: trillion dollar claims
    if "trillion" in tam_lower:
        return "low"
    
    # Red flag: $100B+ claims for niche markets
    match = re.search(r'\$(\d+)\s*[Bb]illion', tam_claim)
    if match:
        amount = int(match.group(1))
        if amount > 500:
            return "low"
        elif amount > 100:
            return "medium"
        else:
            return "high"
    
    # If the market description is very broad ("all businesses", "global economy")
    broad_terms = ["all businesses", "global economy", "every company", "all industries"]
    if any(term in tam_lower for term in broad_terms):
        return "low"
    
    return "medium"


def _compute_narrative_inflation(
    claims: Dict,
    competition_density: float,
    tam_plausibility: str,
) -> float:
    """
    Narrative inflation score (0–100):
    Measures how much the startup's narrative appears inflated
    relative to market realities.
    
    Factors:
    - Low TAM plausibility (unrealistic market size claim)
    - High competition density (claims to be unique in a crowded space)  
    - Hype phrases in pitch (from claim extraction)
    - Claiming to have no competition in a saturated market
    """
    score = 0.0
    
    # TAM plausibility
    if tam_plausibility == "low":
        score += 30
    elif tam_plausibility == "medium":
        score += 10
    
    # Competition gap: high density but probably claiming competitive advantage
    if competition_density > 60:
        score += 20  # Many competitors exist — advantage claims are suspect
    
    # Hype indicators already extracted in Phase 1
    hype_count = len(claims.get("hype_indicators", []))
    score += min(30, hype_count * 8)
    
    # Unsubstantiated growth claims
    growth_claims = claims.get("growth_claims", [])
    if len(growth_claims) > 4:
        score += 10  # Too many growth claims without evidence
    
    return round(min(100.0, score), 1)


def _compute_market_risk_score(
    competition_density: float,
    market_saturation: float,
    narrative_inflation: float,
    tam_plausibility: str,
) -> float:
    """
    Combine all market signals into a single 0–100 risk score.
    Higher = more market risk.
    """
    # Weighted combination
    raw = (
        competition_density * 0.35 +
        market_saturation * 0.25 +
        narrative_inflation * 0.30 +
        (30 if tam_plausibility == "low" else 10 if tam_plausibility == "medium" else 0) * 0.10
    )
    return round(min(100.0, max(0.0, raw)), 1)


def _generate_market_assessment(
    claims: Dict,
    retrieved_companies: List[Dict],
    competition_density: float,
    market_saturation: float,
    tam_plausibility: str,
) -> str:
    """
    Use the LLM to write a grounded market assessment.
    Grounds the LLM in the RETRIEVED data — this is the RAG generation step.
    """
    from utils.llm_client import call_llm
    
    # Format retrieved companies for the prompt
    competitors_text = "\n".join([
        f"- {c['name']} ({c['outcome'].upper()}, {c['similarity']}% similar): {c['description_snippet']}"
        for c in retrieved_companies[:6]
    ])
    
    system = """You are a senior venture capital analyst writing the market section of an 
investment due diligence memo. Be direct, analytical, and base your analysis on the 
provided data. Write 2-3 concise paragraphs. No headers, no lists, plain prose."""
    
    prompt = f"""Write a market competition assessment based on the following data.

STARTUP'S MARKET CLAIMS:
- Target Market: {claims.get('target_market', 'Not specified')}
- TAM Claim: {claims.get('tam_claim', 'Not specified')}
- Competitive Advantages Claimed: {claims.get('competitive_advantage_claims', [])}

RETRIEVED COMPARABLE COMPANIES (from startup database):
{competitors_text if competitors_text else "No close comparables found in database"}

COMPUTED METRICS:
- Competition Density Score: {competition_density}/100
- Market Saturation Index: {market_saturation}/100
- TAM Plausibility: {tam_plausibility.upper()}

Write a 2–3 paragraph market and competition assessment for this startup."""
    
    try:
        return call_llm(prompt=prompt, system=system, max_tokens=500, temperature=0.2)
    except Exception as e:
        logger.error(f"Market assessment LLM failed: {e}")
        count = len(retrieved_companies)
        return (
            f"Market analysis identified {count} comparable companies in this space. "
            f"Competition density is {competition_density:.0f}/100. "
            f"TAM claim plausibility rated as {tam_plausibility}."
        )