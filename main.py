"""
main.py  (Updated — Phase 4)
FastAPI application entry point.
HOW TO RUN: uvicorn main:app --reload --port 8000
API DOCS:   http://localhost:8000/docs
"""

import os
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from db.database import engine
from db import models
from routes.upload import router as upload_router
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
2. **GET /api/status/{job_id}** — Poll until status = "complete"
3. **GET /api/results/{job_id}** — Full structured due diligence JSON
4. **GET /api/report/{job_id}** — Download PDF investment memo

## Agents
- Claim Extraction — JSON schema enforcement, structured pitch deck parsing
- Financial Analysis — Deterministic Python metrics + LLM explanation
- Market & Competition — RAG with FAISS vector search (74 startups)
- Founder Risk — Structured extraction + rule-based credibility scoring
- Pattern Similarity — Cosine similarity against historical startup database
- Risk Aggregation — Weighted 5-dimension scoring, Grade A–F

v2.0 — All 4 phases complete.
    """,
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def create_tables():
    models.Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/verified.")
    Path(os.getenv("UPLOAD_DIR", "./uploads")).mkdir(parents=True, exist_ok=True)
    Path("./reports").mkdir(exist_ok=True)
    logger.info("Directories ready.")

app.include_router(upload_router)
app.include_router(report_router)

@app.get("/health", tags=["system"])
def health_check():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "phase": "Phase 4 Complete — All 23 steps implemented",
        "agents": ["claim_extraction","financial_analysis","market_rag",
                   "founder_risk","pattern_similarity","risk_aggregation"],
    }

@app.get("/", tags=["system"])
def root():
    return {
        "message": "AI Venture Due Diligence Copilot API v2.0",
        "docs": "/docs",
        "health": "/health",
    }