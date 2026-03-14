<!-- # 🚀 AI Venture Due Diligence Copilot

> A production-grade multi-agent AI system for startup investment risk analysis.

---

## Phase 1 — Core Engine

**What's built:**
- FastAPI backend with file upload + job status polling
- PDF parsing (pitch deck → clean text)
- Financial CSV/XLSX parser (auto-detects layout)
- Business Claim Extraction Agent (LLM + strict JSON schema)
- Financial Analysis Engine (deterministic metrics + LLM explanation)
- Risk Aggregation Engine (weighted scoring → Investment Grade A–F)
- SQLite/PostgreSQL database with full schema
- Full unit test suite

---

## Project Structure

```
ai_vdd/
├── main.py                          # FastAPI entry point
├── requirements.txt
├── .env.example                     # Copy to .env and add your API key
│
├── agents/
│   ├── claim_extraction_agent.py    # Agent 1: Extract structured claims from pitch deck
│   ├── financial_analysis_agent.py  # Agent 2: Detect financial anomalies
│   └── risk_aggregation_engine.py   # Final: Compute weighted risk score
│
├── routes/
│   └── upload.py                    # POST /upload, GET /status, GET /results
│
├── db/
│   ├── database.py                  # SQLAlchemy engine + session
│   └── models.py                    # ORM table definitions
│
├── models/
│   └── schemas.py                   # Pydantic request/response schemas
│
├── utils/
│   ├── llm_client.py                # Unified LLM wrapper (Anthropic or OpenAI)
│   ├── pdf_parser.py                # Extract + clean text from PDF
│   └── financial_parser.py          # Parse CSV/XLSX financials
│
├── tests/
│   └── test_phase1.py               # Unit tests (pytest)
│
├── scripts/
│   └── test_pipeline_manual.py      # Quick end-to-end test without API server
│
└── sample_data/
    ├── sample_financials.csv
    └── sample_founder_bio.txt
```

---

## Setup

```bash
# 1. Clone and enter project
cd ai_vdd

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY or OPENAI_API_KEY

# 5. Run the server
uvicorn main:app --reload --port 8000
```

Open **http://localhost:8000/docs** for interactive API docs.

---

## Quick Test (no server needed)

```bash
python scripts/test_pipeline_manual.py
```

---

## Run Tests

```bash
pytest tests/ -v
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/upload` | Upload pitch deck + financials + founder bio |
| GET | `/api/status/{job_id}` | Poll processing status |
| GET | `/api/results/{job_id}` | Fetch full analysis results |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |

---

## Risk Scoring Weights

| Dimension | Weight | Status |
|-----------|--------|--------|
| Financial Risk | 30% | ✅ Phase 1 |
| Market Risk | 25% | 🔜 Phase 2 |
| Founder Risk | 20% | 🔜 Phase 2 |
| Narrative Inflation | 15% | ✅ Phase 1 |
| Pattern Similarity | 10% | 🔜 Phase 2 |

---

## Investment Grades

| Score | Grade | Meaning |
|-------|-------|---------|
| 0–20 | A | Low Risk — Strong candidate |
| 21–40 | B | Moderate Risk — Standard diligence |
| 41–60 | C | Elevated Risk — Deep diligence needed |
| 61–80 | D | High Risk — Major red flags |
| 81–100 | F | Very High Risk — Do not invest |
 -->


 # AI Venture Due Diligence Copilot

> A production-grade multi-agent AI system that analyzes startup pitch decks, financial models, and founder profiles to generate structured investment risk assessments in under 60 seconds.

![Phase](https://img.shields.io/badge/Phase-4_Complete-00C896?style=flat-square)
![Python](https://img.shields.io/badge/Python-3.13-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat-square)
![Groq](https://img.shields.io/badge/LLM-Groq_Llama_3.3-orange?style=flat-square)

---

## What It Does

VC due diligence is slow, subjective, and inconsistent. This system automates the first-pass screening layer:

1. **Upload** a pitch deck (PDF), financial model (CSV/XLSX), and founder bio (TXT)
2. **6 agents run** — each analyzing a different risk dimension
3. **Get a structured memo** — risk score 0–100, investment grade A–F, red flags, recommendations, and a downloadable PDF report

---

## Architecture

```
POST /api/upload
  ├── pdf_parser.py          PDF → clean text
  ├── financial_parser.py    CSV/XLSX → normalized dict
  ├── [Agent 1] claim_extraction_agent.py    LLM + JSON schema
  ├── [Agent 2] financial_analysis_agent.py  Pandas + LLM
  ├── [Agent 3] similarity_engine.py         FAISS cosine similarity
  ├── [Agent 4] market_agent.py              RAG vector search + LLM
  ├── [Agent 5] founder_agent.py             Structured extraction + rules
  └── risk_aggregation_engine.py             Weighted 5-dimension scoring

GET /api/report/{job_id}  →  PDF investment memo
```

---

## Risk Scoring

| Dimension           | Weight | Method                        |
|---------------------|--------|-------------------------------|
| Financial Risk      | 30%    | Deterministic Python + LLM    |
| Market Risk         | 25%    | RAG vector search + LLM       |
| Founder Risk        | 20%    | Structured extraction + rules |
| Narrative Inflation | 15%    | Hype detection                |
| Pattern Similarity  | 10%    | FAISS cosine similarity       |

**Grades:** A (0–20) · B (21–40) · C (41–60) · D (61–80) · F (81–100)

---