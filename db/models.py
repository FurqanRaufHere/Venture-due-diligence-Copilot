"""
db/models.py
────────────
TABLE OVERVIEW:
  Phase 1 tables:
    startups          — one row per startup evaluated
    documents         — uploaded files linked to a startup
    extracted_claims  — structured JSON from Claim Extraction Agent
    financial_metrics — computed ratios + anomaly flags
    risk_scores       — final weighted scores

  Phase 2 tables (NEW):
    similarity_results — FAISS pattern matching results
    market_analysis    — RAG market + competition analysis
    founder_profiles   — structured founder credibility scores
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, Float, Integer, Text, DateTime, JSON, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from db.database import Base


def gen_id():
    return str(uuid.uuid4())


class Startup(Base):
    __tablename__ = "startups"

    id = Column(String, primary_key=True, default=gen_id)
    name = Column(String, nullable=True)           # extracted from pitch deck
    status = Column(String, default="pending")     # pending | processing | complete | error
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    documents = relationship("Document", back_populates="startup")
    extracted_claims = relationship("ExtractedClaims", back_populates="startup", uselist=False)
    financial_metrics = relationship("FinancialMetrics", back_populates="startup", uselist=False)
    risk_scores = relationship("RiskScore", back_populates="startup", uselist=False)
    similarity_results = relationship("SimilarityResults", back_populates="startup", uselist=False)
    market_analysis = relationship("MarketAnalysis", back_populates="startup", uselist=False)
    founder_profiles = relationship("FounderProfiles", back_populates="startup", uselist=False)


class Document(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True, default=gen_id)
    startup_id = Column(String, ForeignKey("startups.id"), nullable=False)
    doc_type = Column(String, nullable=False)      # pitch_deck | financials | founder_bio
    filename = Column(String, nullable=False)
    filepath = Column(String, nullable=False)
    raw_text = Column(Text, nullable=True)         # extracted text content
    processed = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    startup = relationship("Startup", back_populates="documents")


class ExtractedClaims(Base):
    __tablename__ = "extracted_claims"

    id = Column(String, primary_key=True, default=gen_id)
    startup_id = Column(String, ForeignKey("startups.id"), nullable=False, unique=True)

    # All structured claims from the pitch deck (as JSON)
    problem_statement = Column(Text, nullable=True)
    solution_claim = Column(Text, nullable=True)
    target_market = Column(Text, nullable=True)
    tam_claim = Column(Text, nullable=True)
    revenue_model = Column(Text, nullable=True)
    growth_claims = Column(JSON, nullable=True)         # list of strings
    competitive_advantage_claims = Column(JSON, nullable=True)  # list of strings
    hype_indicators = Column(JSON, nullable=True)       # detected hype phrases
    raw_llm_output = Column(JSON, nullable=True)        # full LLM response for auditing
    confidence_score = Column(Float, nullable=True)     # 0.0 – 1.0

    created_at = Column(DateTime, default=datetime.utcnow)
    startup = relationship("Startup", back_populates="extracted_claims")


class FinancialMetrics(Base):
    __tablename__ = "financial_metrics"

    id = Column(String, primary_key=True, default=gen_id)
    startup_id = Column(String, ForeignKey("startups.id"), nullable=False, unique=True)

    # Computed metrics
    revenue_cagr = Column(Float, nullable=True)         # % annual growth rate
    burn_rate_monthly = Column(Float, nullable=True)    # monthly cash burn
    runway_months = Column(Float, nullable=True)        # months of runway
    cac_ltv_ratio = Column(Float, nullable=True)        # CAC / LTV
    gross_margin_avg = Column(Float, nullable=True)     # average gross margin %
    gross_margin_consistent = Column(Boolean, nullable=True)

    # Anomaly flags
    red_flags = Column(JSON, nullable=True)             # list of flag strings
    unrealistic_growth_spikes = Column(JSON, nullable=True)  # list of {year, growth%}
    financial_plausibility_score = Column(Float, nullable=True)  # 0–100

    # LLM explanation of anomalies
    anomaly_explanation = Column(Text, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    startup = relationship("Startup", back_populates="financial_metrics")


class RiskScore(Base):
    __tablename__ = "risk_scores"

    id = Column(String, primary_key=True, default=gen_id)
    startup_id = Column(String, ForeignKey("startups.id"), nullable=False, unique=True)

    # Sub-scores (0–100 each, higher = more risky)
    financial_risk_score = Column(Float, nullable=True)
    market_risk_score = Column(Float, nullable=True)     # Phase 2
    founder_risk_score = Column(Float, nullable=True)    # Phase 2
    narrative_inflation_score = Column(Float, nullable=True)
    pattern_similarity_score = Column(Float, nullable=True)  # Phase 2

    # Final aggregated score
    overall_risk_score = Column(Float, nullable=True)    # 0–100
    investment_grade = Column(String, nullable=True)     # A / B / C / D / F
    confidence_level = Column(String, nullable=True)     # low / medium / high

    # Full structured memo as JSON
    due_diligence_memo = Column(JSON, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    startup = relationship("Startup", back_populates="risk_scores")


# ── Phase 2 Tables ────────────────────────────────────────────────────────────

class SimilarityResults(Base):
    """Stores FAISS pattern matching results — Step 10."""
    __tablename__ = "similarity_results"

    id = Column(String, primary_key=True, default=gen_id)
    startup_id = Column(String, ForeignKey("startups.id"), nullable=False, unique=True)

    failed_similarity_pct  = Column(Float, nullable=True)   # 0–100
    success_similarity_pct = Column(Float, nullable=True)   # 0–100
    pattern_similarity_risk_score = Column(Float, nullable=True)  # 0–100
    dominant_failure_archetype = Column(String, nullable=True)
    archetype_label = Column(String, nullable=True)
    top_similar_failed  = Column(JSON, nullable=True)   # list of {name, similarity, reason}
    top_similar_success = Column(JSON, nullable=True)
    archetype_explanation = Column(Text, nullable=True)
    comparable_startups = Column(JSON, nullable=True)   # list of company names

    created_at = Column(DateTime, default=datetime.utcnow)
    startup = relationship("Startup", back_populates="similarity_results")


class MarketAnalysis(Base):
    """Stores RAG market and competition agent results — Step 11."""
    __tablename__ = "market_analysis"

    id = Column(String, primary_key=True, default=gen_id)
    startup_id = Column(String, ForeignKey("startups.id"), nullable=False, unique=True)

    competition_density      = Column(Float, nullable=True)  # 0–100
    market_saturation_index  = Column(Float, nullable=True)  # 0–100
    narrative_inflation_score = Column(Float, nullable=True) # 0–100
    tam_plausibility         = Column(String, nullable=True) # low/medium/high
    competitor_count         = Column(Integer, nullable=True)
    identified_competitors   = Column(JSON, nullable=True)   # list of names
    retrieved_companies      = Column(JSON, nullable=True)   # full detail list
    market_assessment        = Column(Text, nullable=True)   # LLM-generated prose
    market_risk_score        = Column(Float, nullable=True)  # 0–100

    created_at = Column(DateTime, default=datetime.utcnow)
    startup = relationship("Startup", back_populates="market_analysis")


class FounderProfiles(Base):
    """Stores founder risk profiling results — Step 12."""
    __tablename__ = "founder_profiles"

    id = Column(String, primary_key=True, default=gen_id)
    startup_id = Column(String, ForeignKey("startups.id"), nullable=False, unique=True)

    founder_credibility_score = Column(Float, nullable=True)  # 0–100 (higher = better)
    founder_risk_score        = Column(Float, nullable=True)  # 0–100 (higher = riskier)
    execution_risk_level      = Column(String, nullable=True) # low/medium/high
    prior_exits               = Column(Integer, nullable=True)
    domain_expertise_level    = Column(String, nullable=True) # low/medium/high
    team_coverage_complete    = Column(Boolean, nullable=True)
    missing_roles             = Column(JSON, nullable=True)
    red_flags                 = Column(JSON, nullable=True)
    positive_signals          = Column(JSON, nullable=True)
    extracted_founders        = Column(JSON, nullable=True)   # full structured data
    risk_explanation          = Column(Text, nullable=True)   # LLM prose

    created_at = Column(DateTime, default=datetime.utcnow)
    startup = relationship("Startup", back_populates="founder_profiles")