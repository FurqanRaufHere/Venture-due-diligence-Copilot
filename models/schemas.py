"""
models/schemas.py
──────────────────
WHY THIS FILE EXISTS:
  Pydantic schemas define what data looks like going IN and OUT
  of our API. They are separate from the database models because:
    - DB models = what's stored
    - Schemas   = what's sent over HTTP

  FastAPI uses these for:
    1. Automatic input validation  (400 error if wrong shape)
    2. Auto-generated docs at /docs
    3. Response serialization

NAMING CONVENTION:
  *Create  = used for POST requests (creating new data)
  *Response = used for GET responses (reading data)
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# ── Upload ────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    """Returned immediately after files are uploaded."""
    job_id: str = Field(..., description="Startup ID — use this in all follow-up requests")
    status: str
    message: str
    files_received: List[str]


class JobStatusResponse(BaseModel):
    """Returned by the /status/{job_id} polling endpoint."""
    job_id: str
    status: str          # pending | processing | complete | error
    steps_completed: List[str]
    error_message: Optional[str] = None
    created_at: datetime


# ── Extracted Claims ──────────────────────────────────────────────────

class ExtractedClaimsResponse(BaseModel):
    """Structured output from the Business Claim Extraction Agent."""
    startup_id: str
    problem_statement: Optional[str]
    solution_claim: Optional[str]
    target_market: Optional[str]
    tam_claim: Optional[str]
    revenue_model: Optional[str]
    growth_claims: Optional[List[str]]
    competitive_advantage_claims: Optional[List[str]]
    hype_indicators: Optional[List[str]]
    confidence_score: Optional[float]

    class Config:
        from_attributes = True


# ── Financial Metrics ─────────────────────────────────────────────────

class FinancialMetricsResponse(BaseModel):
    """Output from the Financial Analysis Engine."""
    startup_id: str
    revenue_cagr: Optional[float] = Field(None, description="Compound Annual Growth Rate %")
    burn_rate_monthly: Optional[float]
    runway_months: Optional[float]
    cac_ltv_ratio: Optional[float]
    gross_margin_avg: Optional[float]
    gross_margin_consistent: Optional[bool]
    red_flags: Optional[List[str]]
    unrealistic_growth_spikes: Optional[List[Dict[str, Any]]]
    financial_plausibility_score: Optional[float] = Field(None, description="0–100, higher = more plausible")
    anomaly_explanation: Optional[str]

    class Config:
        from_attributes = True


# ── Risk Score ────────────────────────────────────────────────────────

class RiskScoreResponse(BaseModel):
    """Final aggregated risk assessment output."""
    startup_id: str
    financial_risk_score: Optional[float]
    market_risk_score: Optional[float]
    founder_risk_score: Optional[float]
    narrative_inflation_score: Optional[float]
    pattern_similarity_score: Optional[float]
    overall_risk_score: Optional[float] = Field(None, description="0–100, higher = riskier")
    investment_grade: Optional[str] = Field(None, description="A=Excellent … F=Very High Risk")
    confidence_level: Optional[str]
    due_diligence_memo: Optional[Dict[str, Any]]

    class Config:
        from_attributes = True


# ── Full Analysis (combines everything) ──────────────────────────────

class FullAnalysisResponse(BaseModel):
    """Combined response returned when the full pipeline completes."""
    job_id: str
    startup_name: Optional[str]
    status: str
    extracted_claims: Optional[ExtractedClaimsResponse]
    financial_metrics: Optional[FinancialMetricsResponse]
    risk_score: Optional[RiskScoreResponse]
    processing_time_seconds: Optional[float]