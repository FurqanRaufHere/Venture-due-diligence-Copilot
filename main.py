import os
import logging
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup ───────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """
    Runs once when the server starts.

    IMPORTANT: This must be fast and non-blocking.
    Render kills the server if no port is bound within ~30 seconds.

    What we do here:
      1. Create DB tables (instant)
      2. Create upload/report directories (instant)
      3. Start FAISS preload in a background thread (non-blocking)
         The FAISS index was already BUILT during the render.yaml
         build command — we're just loading it into RAM here.

    What we do NOT do here:
      - Build the FAISS index (takes 30-60s — do this at build time)
      - Download the embedding model (same reason)
    """
    # 1. DB tables
    models.Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/verified.")

    # 2. Directories
    upload_dir = os.getenv("UPLOAD_DIR", "/tmp/uploads")
    Path(upload_dir).mkdir(parents=True, exist_ok=True)
    Path("/tmp/reports").mkdir(exist_ok=True)
    logger.info(f"Directories ready: {upload_dir}, /tmp/reports")

    # 3. Preload FAISS in background — does NOT block port binding
    threading.Thread(target=preload_faiss, daemon=True).start()
    logger.info("Server startup complete. FAISS preloading in background thread.")


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