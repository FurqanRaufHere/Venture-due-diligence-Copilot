import os
import logging
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
1. **POST /api/upload** — Upload pitch deck + financials + founder bio
2. **GET /api/status/{job_id}** — Poll until status = "complete"
3. **GET /api/results/{job_id}** — Full structured due diligence JSON
4. **GET /api/report/{job_id}** — Download PDF investment memo

## Performance
- PDF parsing: ~200ms (pypdf, 20-page limit)
- Claims + Financial: parallel execution
- Similarity + Market + Founder: parallel execution
- Total pipeline: ~20-30 seconds
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
async def startup_event():
    """
    Runs once when server starts.
    1. Create all DB tables
    2. Create upload/report directories
    3. Preload FAISS index + embedding model into memory
       so the first user request doesn't trigger a 30-60s load
    """
    # DB tables
    models.Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/verified.")

    # Directories
    Path(os.getenv("UPLOAD_DIR", "./uploads")).mkdir(parents=True, exist_ok=True)
    Path("./reports").mkdir(exist_ok=True)

    # Build FAISS index if it doesn't exist yet
    faiss_index_path = Path("data/faiss_index/startup_index.faiss")
    if not faiss_index_path.exists():
        logger.info("FAISS index not found — building now (one-time, ~30s)...")
        try:
            from utils.startup_dataset import build_faiss_index
            build_faiss_index(save=True)
            logger.info("FAISS index built and saved.")
        except Exception as e:
            logger.error(f"FAISS index build failed: {e}")

    # Preload FAISS + embedding model into RAM
    import threading
    threading.Thread(target=preload_faiss, daemon=True).start()
    logger.info("Server ready. Preloading FAISS index in background...")


app.include_router(upload_router)
app.include_router(report_router)


@app.get("/health", tags=["system"])
def health_check():
    from routes.upload import _faiss_index
    return {
        "status": "healthy",
        "version": "2.0.0",
        "faiss_loaded": _faiss_index is not None,
        "phase": "Phase 4 Complete — optimized for production",
    }


@app.get("/", tags=["system"])
def root():
    return {
        "message": "AI Venture Due Diligence Copilot API v2.0",
        "docs": "/docs",
        "health": "/health",
    }