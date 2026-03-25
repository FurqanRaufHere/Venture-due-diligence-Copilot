import os
import logging
import tempfile
import threading
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from db.database import engine
from db import models
from routes.upload import router as upload_router, preload_faiss
from routes.report import router as report_router

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    """Parse a boolean env flag with common truthy values."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default_csv: str) -> list[str]:
    """Parse a comma-separated env var into a cleaned string list."""
    raw = os.getenv(name, default_csv)
    return [item.strip() for item in raw.split(",") if item.strip()]

app = FastAPI(
    title="AI Venture Due Diligence Copilot",
    description="""
A production-grade multi-agent AI system for startup investment risk analysis.

## Workflow
1. **POST /api/upload** — Upload pitch deck (PDF) + financials (CSV/XLSX) + founder bio (TXT)
2. **GET /api/status/{job_id}** — Poll every 2-3s until status = "complete"
3. **GET /api/results/{job_id}** — Full structured due diligence JSON
4. **GET /api/report/{job_id}** — Download PDF investment memo

## Agents
- Claim Extraction — JSON schema enforcement, structured pitch deck parsing
- Financial Analysis — Deterministic Python metrics + LLM explanation
- Market & Competition — RAG with FAISS vector search (74 startups)
- Founder Risk — Structured extraction + rule-based credibility scoring
- Pattern Similarity — Cosine similarity against historical startup database
- Risk Aggregation — Weighted 5-dimension scoring, Grade A–F

## Performance
- PDF parsing: ~200ms (pypdf, 20-page limit)
- Claims + Financial: parallel execution
- Similarity + Market + Founder: parallel execution
- Total pipeline: ~20-30 seconds

v2.0 — All 4 phases complete.
    """,
    version="2.0.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
cors_origins = _env_list(
    "FRONTEND_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5500,http://127.0.0.1:5500",
)
cors_origin_regex = os.getenv("FRONTEND_ORIGIN_REGEX", r"https://.*\\.onrender\\.com")

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_origin_regex=cors_origin_regex,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """
    Runs once when the server starts.

    IMPORTANT: Keep startup fast and non-blocking.

    What we do here:
      1. Create DB tables (instant)
      2. Create upload/report directories (instant)
      3. Start FAISS preload in a background thread (non-blocking)
                 The FAISS index is expected to be pre-built; we load it into RAM here.

    What we do NOT do here:
            - Build the FAISS index (takes 30-60s; do this before runtime)
      - Download the embedding model (same reason)
    """
    # 1. DB tables
    models.Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/verified.")

    # 2. Directories
    upload_dir = os.getenv("UPLOAD_DIR", os.path.join(tempfile.gettempdir(), "nlp_uploads"))
    Path(upload_dir).mkdir(parents=True, exist_ok=True)
    
    reports_dir = os.getenv("REPORTS_DIR", os.path.join(tempfile.gettempdir(), "nlp_reports"))
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Directories ready: {upload_dir}, {reports_dir}")

    # 3. Optional FAISS preload in background — does NOT block port binding
    preload_on_startup = _env_flag("PRELOAD_FAISS_ON_STARTUP", True)
    if preload_on_startup:
        threading.Thread(target=preload_faiss, daemon=True).start()
        logger.info("Server startup complete. FAISS preloading in background thread.")
    else:
        logger.info("Server startup complete. FAISS preload disabled by PRELOAD_FAISS_ON_STARTUP.")


# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(upload_router)
app.include_router(report_router)


# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health", tags=["system"])
def health_check():
    """Health check — also shows whether FAISS is loaded yet."""
    from routes.upload import _faiss_index
    return {
        "status": "healthy",
        "version": "2.0.0",
        "faiss_ready": _faiss_index is not None,
        "phase": "Phase 4 — Production Optimized",
        "agents": [
            "claim_extraction",
            "financial_analysis",
            "market_rag",
            "founder_risk",
            "pattern_similarity",
            "risk_aggregation",
        ],
    }


@app.get("/", tags=["system"])
def root():
    return {
        "message": "AI Venture Due Diligence Copilot API v2.0",
        "docs": "/docs",
        "health": "/health",
        "upload": "POST /api/upload",
        "report": "GET /api/report/{job_id}",
    }