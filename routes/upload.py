import os
import time
import shutil
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import (Startup, Document, ExtractedClaims, FinancialMetrics, RiskScore,
                       SimilarityResults, MarketAnalysis, FounderProfiles)
from models.schemas import UploadResponse, JobStatusResponse, FullAnalysisResponse
from utils.pdf_parser import extract_text_from_pdf
from utils.financial_parser import parse_financial_file
from agents.claim_extraction_agent import run_claim_extraction
from agents.financial_analysis_agent import run_financial_analysis
from agents.risk_aggregation_engine import (
    run_risk_aggregation,
    compute_narrative_inflation_score,
    compute_financial_risk_score,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["analysis"])

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./uploads")
ALLOWED_PDF = {".pdf"}
ALLOWED_FINANCIAL = {".csv", ".xlsx", ".xls"}

# ── Preload FAISS index and embedding model at startup ────────────────────────
# This runs once when the server starts, so the first user request is fast.
_faiss_index = None
_faiss_metadata = None
_faiss_lock = threading.Lock()


def preload_faiss():
    """Load FAISS index into memory at server startup."""
    global _faiss_index, _faiss_metadata
    try:
        from utils.startup_dataset import load_faiss_index
        logger.info("Preloading FAISS index...")
        index, metadata = load_faiss_index()
        with _faiss_lock:
            _faiss_index = index
            _faiss_metadata = metadata
        logger.info(f"FAISS index preloaded: {index.ntotal} vectors ready.")
    except Exception as e:
        logger.warning(f"FAISS preload failed (will load on demand): {e}")


def get_faiss():
    """Get the preloaded FAISS index, loading it if needed."""
    global _faiss_index, _faiss_metadata
    if _faiss_index is None:
        preload_faiss()
    return _faiss_index, _faiss_metadata


# ── POST /upload ──────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse)
async def upload_startup_files(
    background_tasks: BackgroundTasks,
    pitch_deck: UploadFile = File(...),
    financials: Optional[UploadFile] = File(None),
    founder_bio: Optional[UploadFile] = File(None),
    startup_name: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    _validate_file_extension(pitch_deck.filename, ALLOWED_PDF, "pitch_deck")
    if financials:
        _validate_file_extension(financials.filename, ALLOWED_FINANCIAL, "financials")

    startup = Startup(name=startup_name, status="pending")
    db.add(startup)
    db.commit()
    db.refresh(startup)
    job_id = startup.id

    job_dir = Path(UPLOAD_DIR) / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    files_saved = []
    saved_paths = {}

    async def save_file(upload: UploadFile, doc_type: str) -> str:
        dest = job_dir / upload.filename
        with open(dest, "wb") as f:
            shutil.copyfileobj(upload.file, f)
        doc = Document(startup_id=job_id, doc_type=doc_type,
                      filename=upload.filename, filepath=str(dest))
        db.add(doc)
        files_saved.append(upload.filename)
        return str(dest)

    saved_paths["pitch_deck"] = await save_file(pitch_deck, "pitch_deck")
    if financials:
        saved_paths["financials"] = await save_file(financials, "financials")
    if founder_bio:
        saved_paths["founder_bio"] = await save_file(founder_bio, "founder_bio")
    db.commit()

    background_tasks.add_task(_run_full_pipeline, job_id=job_id, saved_paths=saved_paths)
    return UploadResponse(
        job_id=job_id, status="pending",
        message="Files received. Analysis pipeline started.",
        files_received=files_saved,
    )


# ── GET /status/{job_id} ──────────────────────────────────────────────────────

@router.get("/status/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    startup = _get_startup_or_404(job_id, db)
    steps_completed = []
    if db.query(Document).filter(Document.startup_id == job_id, Document.processed == True).first():
        steps_completed.append("document_ingestion")
    if db.query(ExtractedClaims).filter(ExtractedClaims.startup_id == job_id).first():
        steps_completed.append("claim_extraction")
    if db.query(FinancialMetrics).filter(FinancialMetrics.startup_id == job_id).first():
        steps_completed.append("financial_analysis")
    if db.query(RiskScore).filter(RiskScore.startup_id == job_id).first():
        steps_completed.append("risk_aggregation")
    return JobStatusResponse(
        job_id=job_id, status=startup.status,
        steps_completed=steps_completed,
        error_message=None, created_at=startup.created_at,
    )


# ── GET /results/{job_id} ─────────────────────────────────────────────────────

@router.get("/results/{job_id}", response_model=FullAnalysisResponse)
def get_results(job_id: str, db: Session = Depends(get_db)):
    startup = _get_startup_or_404(job_id, db)
    if startup.status in ("pending", "processing"):
        raise HTTPException(status_code=202, detail="Analysis still in progress.")
    if startup.status == "error":
        raise HTTPException(status_code=500, detail="Analysis failed. Check logs.")

    claims = db.query(ExtractedClaims).filter(ExtractedClaims.startup_id == job_id).first()
    financials = db.query(FinancialMetrics).filter(FinancialMetrics.startup_id == job_id).first()
    risk = db.query(RiskScore).filter(RiskScore.startup_id == job_id).first()

    from models.schemas import ExtractedClaimsResponse, FinancialMetricsResponse, RiskScoreResponse
    return FullAnalysisResponse(
        job_id=job_id, startup_name=startup.name, status=startup.status,
        extracted_claims=ExtractedClaimsResponse.model_validate(claims) if claims else None,
        financial_metrics=FinancialMetricsResponse.model_validate(financials) if financials else None,
        risk_score=RiskScoreResponse.model_validate(risk) if risk else None,
        processing_time_seconds=None,
    )


# ── Optimized Pipeline ────────────────────────────────────────────────────────

def _run_full_pipeline(job_id: str, saved_paths: dict):
    """
    Optimized pipeline with parallel Phase 2 agents.

    Timeline (before): ~45-60s sequential
    Timeline (after):  ~20-30s with parallelism
    
    Sequential:   parse → claims → financial → similarity → market → founder → aggregate
    Optimized:    parse → claims+financial (parallel) → similarity+market+founder (parallel) → aggregate
    """
    from db.database import SessionLocal
    db = SessionLocal()
    start_time = time.time()
    startup = db.query(Startup).filter(Startup.id == job_id).first()

    try:
        startup.status = "processing"
        db.commit()
        logger.info(f"[{job_id}] Pipeline started.")

        # ── Step 1: Parse PDF (fast with pypdf) ───────────────────────
        pitch_text = ""
        if "pitch_deck" in saved_paths:
            logger.info(f"[{job_id}] Parsing PDF...")
            t = time.time()
            pitch_text, page_count = extract_text_from_pdf(saved_paths["pitch_deck"])
            doc = db.query(Document).filter(
                Document.startup_id == job_id, Document.doc_type == "pitch_deck"
            ).first()
            if doc:
                doc.raw_text = pitch_text[:50000]  # cap stored text at 50k chars
                doc.processed = True
            db.commit()
            logger.info(f"[{job_id}] PDF parsed in {time.time()-t:.1f}s: {page_count} pages, {len(pitch_text)} chars.")

        # ── Step 2: Parse Financials ───────────────────────────────────
        financial_data = {}
        if "financials" in saved_paths:
            financial_data = parse_financial_file(saved_paths["financials"])
            doc = db.query(Document).filter(
                Document.startup_id == job_id, Document.doc_type == "financials"
            ).first()
            if doc:
                doc.processed = True
            db.commit()

        # ── Step 3+4: Claims + Financial PARALLEL ─────────────────────
        claims_result = {}
        financial_result = {}

        def run_claims():
            if not pitch_text:
                return {}
            logger.info(f"[{job_id}] Running claim extraction...")
            return run_claim_extraction(pitch_text)

        def run_financial():
            if not financial_data:
                return {}
            logger.info(f"[{job_id}] Running financial analysis...")
            return run_financial_analysis(financial_data)

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_claims = executor.submit(run_claims)
            future_financial = executor.submit(run_financial)

            try:
                claims_result = future_claims.result(timeout=60)
            except Exception as e:
                logger.error(f"[{job_id}] Claim extraction failed: {e}")
                claims_result = {}

            try:
                financial_result = future_financial.result(timeout=30)
            except Exception as e:
                logger.error(f"[{job_id}] Financial analysis failed: {e}")
                financial_result = {}

        # Save claims
        if claims_result:
            if not startup.name and claims_result.get("solution_claim"):
                startup.name = "Startup (auto)"
            claims_record = ExtractedClaims(
                startup_id=job_id,
                problem_statement=claims_result.get("problem_statement"),
                solution_claim=claims_result.get("solution_claim"),
                target_market=claims_result.get("target_market"),
                tam_claim=claims_result.get("tam_claim"),
                revenue_model=claims_result.get("revenue_model"),
                growth_claims=claims_result.get("growth_claims"),
                competitive_advantage_claims=claims_result.get("competitive_advantage_claims"),
                hype_indicators=claims_result.get("hype_indicators"),
                raw_llm_output=claims_result.get("raw_llm_output"),
                confidence_score=claims_result.get("confidence_score"),
            )
            db.add(claims_record)

        # Save financials
        if financial_result:
            fin_record = FinancialMetrics(
                startup_id=job_id,
                revenue_cagr=financial_result.get("revenue_cagr"),
                burn_rate_monthly=financial_result.get("burn_rate_monthly"),
                runway_months=financial_result.get("runway_months"),
                cac_ltv_ratio=financial_result.get("cac_ltv_ratio"),
                gross_margin_avg=financial_result.get("gross_margin_avg"),
                gross_margin_consistent=financial_result.get("gross_margin_consistent"),
                red_flags=financial_result.get("red_flags"),
                unrealistic_growth_spikes=financial_result.get("unrealistic_growth_spikes"),
                financial_plausibility_score=financial_result.get("financial_plausibility_score"),
                anomaly_explanation=financial_result.get("anomaly_explanation"),
            )
            db.add(fin_record)
        db.commit()
        logger.info(f"[{job_id}] Claims + Financial complete.")

        # ── Step 5: Phase 2 Agents PARALLEL ───────────────────────────
        similarity_result = {}
        market_result = {}
        founder_result = {}

        # Read founder bio text
        founder_bio_text = ""
        if "founder_bio" in saved_paths:
            try:
                with open(saved_paths["founder_bio"], "r", encoding="utf-8", errors="ignore") as f:
                    founder_bio_text = f.read()
            except Exception:
                pass
        if not founder_bio_text and pitch_text:
            lower = pitch_text.lower()
            for marker in ["team", "founder", "about us"]:
                idx = lower.find(marker)
                if idx > 0:
                    founder_bio_text = pitch_text[idx:idx+2000]
                    break

        def run_similarity():
            try:
                from agents.similarity_engine import run_similarity_engine, build_startup_description_from_claims
                desc = build_startup_description_from_claims(claims_result) if claims_result else pitch_text[:500]
                return run_similarity_engine(desc)
            except Exception as e:
                logger.warning(f"[{job_id}] Similarity engine failed: {e}")
                return {}

        def run_market():
            try:
                from agents.market_agent import run_market_agent
                return run_market_agent(claims_result or {})
            except Exception as e:
                logger.warning(f"[{job_id}] Market agent failed: {e}")
                return {}

        def run_founder():
            if not founder_bio_text:
                return {}
            try:
                from agents.founder_agent import run_founder_agent
                return run_founder_agent(founder_bio_text)
            except Exception as e:
                logger.warning(f"[{job_id}] Founder agent failed: {e}")
                return {}

        logger.info(f"[{job_id}] Running Phase 2 agents in parallel...")
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_sim = executor.submit(run_similarity)
            future_mkt = executor.submit(run_market)
            future_fnd = executor.submit(run_founder)

            try:
                similarity_result = future_sim.result(timeout=60)
            except Exception as e:
                logger.warning(f"[{job_id}] Similarity timeout: {e}")

            try:
                market_result = future_mkt.result(timeout=60)
            except Exception as e:
                logger.warning(f"[{job_id}] Market timeout: {e}")

            try:
                founder_result = future_fnd.result(timeout=60)
            except Exception as e:
                logger.warning(f"[{job_id}] Founder timeout: {e}")

        # Save Phase 2 results
        if similarity_result:
            db.add(SimilarityResults(
                startup_id=job_id,
                failed_similarity_pct=similarity_result.get("failed_similarity_pct"),
                success_similarity_pct=similarity_result.get("success_similarity_pct"),
                pattern_similarity_risk_score=similarity_result.get("pattern_similarity_risk_score"),
                dominant_failure_archetype=similarity_result.get("dominant_failure_archetype"),
                archetype_label=similarity_result.get("archetype_label"),
                top_similar_failed=similarity_result.get("top_similar_failed"),
                top_similar_success=similarity_result.get("top_similar_success"),
                archetype_explanation=similarity_result.get("archetype_explanation"),
                comparable_startups=similarity_result.get("comparable_startups"),
            ))

        if market_result:
            db.add(MarketAnalysis(
                startup_id=job_id,
                competition_density=market_result.get("competition_density"),
                market_saturation_index=market_result.get("market_saturation_index"),
                narrative_inflation_score=market_result.get("narrative_inflation_score"),
                tam_plausibility=market_result.get("tam_plausibility"),
                competitor_count=market_result.get("competitor_count"),
                identified_competitors=market_result.get("identified_competitors"),
                retrieved_companies=market_result.get("retrieved_companies"),
                market_assessment=market_result.get("market_assessment"),
                market_risk_score=market_result.get("market_risk_score"),
            ))

        if founder_result:
            db.add(FounderProfiles(
                startup_id=job_id,
                founder_credibility_score=founder_result.get("founder_credibility_score"),
                founder_risk_score=founder_result.get("founder_risk_score"),
                execution_risk_level=founder_result.get("execution_risk_level"),
                prior_exits=founder_result.get("prior_exits"),
                domain_expertise_level=founder_result.get("domain_expertise_level"),
                team_coverage_complete=founder_result.get("team_coverage_complete"),
                missing_roles=founder_result.get("missing_roles"),
                red_flags=founder_result.get("red_flags"),
                positive_signals=founder_result.get("positive_signals"),
                extracted_founders=founder_result.get("extracted_founders"),
                risk_explanation=founder_result.get("risk_explanation"),
            ))
        db.commit()
        logger.info(f"[{job_id}] Phase 2 agents complete.")

        # ── Step 6: Final Risk Aggregation ─────────────────────────────
        financial_risk = compute_financial_risk_score(
            financial_result["financial_plausibility_score"]
        ) if financial_result.get("financial_plausibility_score") is not None else None

        narrative_inflation = compute_narrative_inflation_score(
            hype_indicators=claims_result.get("hype_indicators", []),
            growth_claims=claims_result.get("growth_claims", []),
        ) if claims_result else None

        risk_result = run_risk_aggregation(
            financial_risk_score=financial_risk,
            market_risk_score=market_result.get("market_risk_score"),
            founder_risk_score=founder_result.get("founder_risk_score"),
            narrative_inflation_score=narrative_inflation,
            pattern_similarity_score=similarity_result.get("pattern_similarity_risk_score"),
            extracted_claims=claims_result,
            financial_metrics=financial_result,
            market_analysis=market_result,
            founder_analysis=founder_result,
            similarity_results=similarity_result,
        )

        db.add(RiskScore(
            startup_id=job_id,
            financial_risk_score=risk_result.get("financial_risk_score"),
            market_risk_score=risk_result.get("market_risk_score"),
            founder_risk_score=risk_result.get("founder_risk_score"),
            narrative_inflation_score=risk_result.get("narrative_inflation_score"),
            pattern_similarity_score=risk_result.get("pattern_similarity_score"),
            overall_risk_score=risk_result.get("overall_risk_score"),
            investment_grade=risk_result.get("investment_grade"),
            confidence_level=risk_result.get("confidence_level"),
            due_diligence_memo=risk_result.get("due_diligence_memo"),
        ))

        startup.status = "complete"
        db.commit()

        elapsed = round(time.time() - start_time, 1)
        logger.info(
            f"[{job_id}] Pipeline complete in {elapsed}s. "
            f"Grade: {risk_result.get('investment_grade')} | "
            f"Score: {risk_result.get('overall_risk_score')}/100 | "
            f"Confidence: {risk_result.get('confidence_level')}"
        )

    except Exception as e:
        logger.error(f"[{job_id}] Pipeline failed: {e}", exc_info=True)
        startup.status = "error"
        db.commit()
    finally:
        db.close()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_file_extension(filename: str, allowed: set, field_name: str):
    ext = Path(filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type for {field_name}: '{ext}'. Allowed: {allowed}"
        )


def _get_startup_or_404(job_id: str, db: Session):
    startup = db.query(Startup).filter(Startup.id == job_id).first()
    if not startup:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return startup