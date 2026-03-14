"""
routes/report.py
────────────────────────────────────────────────────────────────
STEP 19 — PDF Report Endpoint

GET /api/report/{job_id}
  → Fetches analysis results from DB
  → Generates PDF using report_generator.py
  → Returns as file download

The frontend "Export Report" button calls this endpoint.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import Startup, ExtractedClaims, FinancialMetrics, RiskScore

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["report"])


@router.get("/report/{job_id}")
def download_report(job_id: str, db: Session = Depends(get_db)):
    """
    Generate and download a PDF investment memo for the given job.
    """
    startup = db.query(Startup).filter(Startup.id == job_id).first()
    if not startup:
        raise HTTPException(status_code=404, detail="Job not found")
    if startup.status != "complete":
        raise HTTPException(status_code=202, detail="Analysis not complete yet")

    claims   = db.query(ExtractedClaims).filter(ExtractedClaims.startup_id == job_id).first()
    fin      = db.query(FinancialMetrics).filter(FinancialMetrics.startup_id == job_id).first()
    risk     = db.query(RiskScore).filter(RiskScore.startup_id == job_id).first()

    # Build data dict matching the FullAnalysisResponse shape
    data = {
        "startup_name": startup.name,
        "risk_score": {
            "financial_risk_score":      risk.financial_risk_score if risk else None,
            "market_risk_score":         risk.market_risk_score if risk else None,
            "founder_risk_score":        risk.founder_risk_score if risk else None,
            "narrative_inflation_score": risk.narrative_inflation_score if risk else None,
            "pattern_similarity_score":  risk.pattern_similarity_score if risk else None,
            "overall_risk_score":        risk.overall_risk_score if risk else None,
            "investment_grade":          risk.investment_grade if risk else None,
            "confidence_level":          risk.confidence_level if risk else None,
            "due_diligence_memo":        risk.due_diligence_memo if risk else {},
        } if risk else {},
        "extracted_claims": {
            "problem_statement":           claims.problem_statement if claims else None,
            "solution_claim":              claims.solution_claim if claims else None,
            "target_market":               claims.target_market if claims else None,
            "tam_claim":                   claims.tam_claim if claims else None,
            "revenue_model":               claims.revenue_model if claims else None,
            "hype_indicators":             claims.hype_indicators if claims else [],
            "competitive_advantage_claims":claims.competitive_advantage_claims if claims else [],
        } if claims else {},
        "financial_metrics": {
            "revenue_cagr":                fin.revenue_cagr if fin else None,
            "burn_rate_monthly":           fin.burn_rate_monthly if fin else None,
            "runway_months":               fin.runway_months if fin else None,
            "gross_margin_avg":            fin.gross_margin_avg if fin else None,
            "financial_plausibility_score":fin.financial_plausibility_score if fin else None,
            "anomaly_explanation":         fin.anomaly_explanation if fin else None,
            "red_flags":                   fin.red_flags if fin else [],
        } if fin else {},
    }

    try:
        from utils.report_generator import generate_report
        pdf_path = generate_report(job_id, data)
    except Exception as e:
        logger.error(f"PDF generation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {e}")

    startup_slug = (startup.name or "report").replace(" ", "_").lower()
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=f"vdd_{startup_slug}_{job_id[:8]}.pdf",
    )