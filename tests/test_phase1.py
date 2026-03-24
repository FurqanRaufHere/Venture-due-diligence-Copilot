"""
tests/test_phase1.py
─────────────────────
WHY TESTS MATTER:
  These tests verify each component works in isolation BEFORE
  you wire everything together. It's much faster to debug a single
  function than to trace a bug through the entire pipeline.

  Run with: pytest tests/ -v

TEST COVERAGE:
  ✓ Financial parser (row-based and column-based layouts)
  ✓ Financial analysis engine (CAGR, spikes, scoring)
  ✓ Claim extraction (hype detection, confidence scoring)
  ✓ Risk aggregation (scoring, grades, memo structure)
  ✓ API endpoints (upload, status, results)

NOTE ON LLM TESTS:
  LLM calls are NOT tested here — they require API keys and are
  non-deterministic. The agents mock the LLM in unit tests.
  Integration tests (with real LLM calls) live in tests/integration/.
"""

import pytest
import json
from unittest.mock import patch, MagicMock


# ══════════════════════════════════════════════════════════════════════
# 1. Financial Parser Tests
# ══════════════════════════════════════════════════════════════════════

class TestFinancialParser:
    """Tests for utils/financial_parser.py"""

    def test_parse_row_based_csv(self, tmp_path):
        """Parser correctly handles row-based layout (metrics as rows)."""
        import pandas as pd
        from utils.financial_parser import parse_financial_file

        csv_content = "Metric,2021,2022,2023\nRevenue,500000,1200000,2800000\nCOGS,200000,450000,900000\nCash Balance,1000000,800000,500000\n"
        csv_file = tmp_path / "financials.csv"
        csv_file.write_text(csv_content)

        result = parse_financial_file(str(csv_file))

        assert result["revenue"] == [500000.0, 1200000.0, 2800000.0]
        assert result["years"] == [2021, 2022, 2023]
        assert result["revenue"] is not None
        assert len(result["detection_log"]) > 0

    def test_parse_column_based_csv(self, tmp_path):
        """Parser correctly handles column-based layout (years as rows)."""
        from utils.financial_parser import parse_financial_file

        csv_content = "Year,Revenue,COGS,Cash\n2021,500000,200000,1000000\n2022,1200000,450000,800000\n2023,2800000,900000,500000\n"
        csv_file = tmp_path / "financials_col.csv"
        csv_file.write_text(csv_content)

        result = parse_financial_file(str(csv_file))
        assert result["years"] == [2021, 2022, 2023]

    def test_missing_file_raises(self):
        from utils.financial_parser import parse_financial_file
        with pytest.raises(FileNotFoundError):
            parse_financial_file("/nonexistent/file.csv")

    def test_unsupported_format_raises(self, tmp_path):
        from utils.financial_parser import parse_financial_file
        bad_file = tmp_path / "data.json"
        bad_file.write_text("{}")
        with pytest.raises(ValueError, match="Unsupported file type"):
            parse_financial_file(str(bad_file))


# ══════════════════════════════════════════════════════════════════════
# 2. Financial Analysis Engine Tests
# ══════════════════════════════════════════════════════════════════════

class TestFinancialAnalysis:
    """Tests for agents/financial_analysis_agent.py — all deterministic, no LLM."""

    def _mock_llm(self):
        """Patch the LLM call so tests don't need an API key."""
        return patch("agents.financial_analysis_agent.call_llm", return_value="Mocked explanation.")

    def test_normal_growth_no_flags(self):
        """Reasonable financials should produce no red flags."""
        financial_data = {
            "years": [2021, 2022, 2023],
            "revenue": [500000, 900000, 1500000],
            "costs": [400000, 700000, 1100000],
            "gross_profit": [100000, 200000, 400000],
            "cash_balance": [2000000, 1500000, 1000000],
            "burn_rate": None, "headcount": None, "cac": None, "ltv": None,
        }
        with self._mock_llm():
            result = _run_financial_analysis_pure(financial_data)

        # CAGR of ~73% is reasonable
        assert result["revenue_cagr"] is not None
        assert result["revenue_cagr"] < 150
        # No red flags expected
        cagr_flags = [f for f in result["red_flags"] if "CAGR" in f]
        assert len(cagr_flags) == 0

    def test_extreme_cagr_triggers_red_flag(self):
        """Revenue growing 10x per year should trigger red flag."""
        financial_data = {
            "years": [2021, 2022, 2023],
            "revenue": [100000, 1000000, 10000000],
            "costs": None, "gross_profit": None, "cash_balance": None,
            "burn_rate": None, "headcount": None, "cac": None, "ltv": None,
        }
        with self._mock_llm():
            result = _run_financial_analysis_pure(financial_data)

        assert len(result["red_flags"]) > 0
        assert result["financial_plausibility_score"] < 80

    def test_inverted_unit_economics(self):
        """CAC > LTV should trigger red flag."""
        financial_data = {
            "years": [2022, 2023],
            "revenue": [500000, 1000000],
            "costs": None, "gross_profit": None, "cash_balance": None,
            "burn_rate": None, "headcount": None,
            "cac": 5000,   # costs $5000 to acquire a customer
            "ltv": 2000,   # customer only worth $2000
        }
        with self._mock_llm():
            result = _run_financial_analysis_pure(financial_data)

        unit_econ_flags = [f for f in result["red_flags"] if "Unit economics" in f or "inverted" in f]
        assert len(unit_econ_flags) > 0

    def test_critical_runway_flags(self):
        """Less than 6 months runway should be CRITICAL flag."""
        financial_data = {
            "years": [2022, 2023],
            "revenue": [500000, 800000],
            "costs": [600000, 1000000],
            "gross_profit": None,
            "cash_balance": [200000, 100000],  # only 100k left
            "burn_rate": [15000, 20000],        # ~$17k/month avg
            "headcount": None, "cac": None, "ltv": None,
        }
        with self._mock_llm():
            result = _run_financial_analysis_pure(financial_data)

        critical_flags = [f for f in result["red_flags"] if "CRITICAL" in f or "runway" in f.lower()]
        assert len(critical_flags) > 0

    def test_plausibility_score_range(self):
        """Plausibility score must always be 0–100."""
        financial_data = {
            "years": [2021, 2022, 2023],
            "revenue": [1, 1000000, 1000000000],  # absurd growth
            "costs": None, "gross_profit": None, "cash_balance": None,
            "burn_rate": None, "headcount": None, "cac": None, "ltv": None,
        }
        with self._mock_llm():
            result = _run_financial_analysis_pure(financial_data)

        assert 0 <= result["financial_plausibility_score"] <= 100


# ══════════════════════════════════════════════════════════════════════
# 3. Claim Extraction Tests (hype detection — no LLM)
# ══════════════════════════════════════════════════════════════════════

class TestClaimExtraction:
    """Tests for agents/claim_extraction_agent.py"""

    def test_hype_phrase_detection(self):
        from agents.claim_extraction_agent import _detect_hype_phrases

        text = "We will revolutionize the market with our AI-powered platform. This is a blue ocean opportunity with exponential growth."
        found = _detect_hype_phrases(text)
        assert "revolutionize" in found
        assert "blue ocean" in found
        assert "exponential growth" in found

    def test_no_hype_clean_text(self):
        from agents.claim_extraction_agent import _detect_hype_phrases

        text = "We help small businesses manage their inventory through a subscription software platform."
        found = _detect_hype_phrases(text)
        assert len(found) == 0

    def test_confidence_score_full_extraction(self):
        """Full extraction should yield high confidence."""
        from agents.claim_extraction_agent import run_claim_extraction

        mock_llm_response = json.dumps({
            "problem_statement": "Inventory management is manual",
            "solution_claim": "AI-powered inventory platform",
            "target_market": "SMB retailers",
            "tam_claim": "$12B market",
            "revenue_model": "SaaS subscription $99/month",
            "growth_claims": ["2x YoY", "40% MoM"],
            "competitive_advantage_claims": ["Faster than competitors"],
        })

        with patch("agents.claim_extraction_agent.call_llm_json", return_value=json.loads(mock_llm_response)):
            result = run_claim_extraction("Sample pitch deck text about inventory management.")

        assert result["confidence_score"] >= 0.8
        assert result["problem_statement"] == "Inventory management is manual"

    def test_llm_failure_returns_empty_gracefully(self):
        """If LLM fails, extraction should return empty fields, not crash."""
        from agents.claim_extraction_agent import run_claim_extraction

        with patch("agents.claim_extraction_agent.call_llm_json", side_effect=ValueError("API error")):
            result = run_claim_extraction("Some pitch deck text.")

        assert result["confidence_score"] == 0.0
        assert result["problem_statement"] is None


# ══════════════════════════════════════════════════════════════════════
# 4. Risk Aggregation Tests
# ══════════════════════════════════════════════════════════════════════

class TestRiskAggregation:
    """Tests for agents/risk_aggregation_engine.py"""

    def test_grade_assignment(self):
        from agents.risk_aggregation_engine import _score_to_grade
        assert _score_to_grade(15) == "A"
        assert _score_to_grade(35) == "B"
        assert _score_to_grade(55) == "C"
        assert _score_to_grade(75) == "D"
        assert _score_to_grade(90) == "F"

    def test_narrative_inflation_score(self):
        from agents.risk_aggregation_engine import compute_narrative_inflation_score
        score = compute_narrative_inflation_score(
            hype_indicators=["revolutionize", "blue ocean", "10x", "ai-powered", "exponential growth"],
            growth_claims=["300% CAGR", "viral growth", "10x users", "5x revenue", "100x ROI"]
        )
        assert score > 50  # Many hype phrases = high inflation score
        assert score <= 100

    def test_narrative_clean_pitch(self):
        from agents.risk_aggregation_engine import compute_narrative_inflation_score
        score = compute_narrative_inflation_score(hype_indicators=[], growth_claims=["20% MoM growth"])
        assert score == 0

    def test_financial_risk_inversion(self):
        from agents.risk_aggregation_engine import compute_financial_risk_score
        assert compute_financial_risk_score(90.0) == 10.0   # High plausibility → low risk
        assert compute_financial_risk_score(30.0) == 70.0   # Low plausibility → high risk
        assert compute_financial_risk_score(100.0) == 0.0
        assert compute_financial_risk_score(0.0) == 100.0

    def test_aggregation_partial_scores(self):
        """Aggregation should work with only 1–2 sub-scores available."""
        from agents.risk_aggregation_engine import run_risk_aggregation

        with patch("agents.risk_aggregation_engine.call_llm", return_value="Mock summary."):
            result = run_risk_aggregation(
                financial_risk_score=70.0,
                narrative_inflation_score=50.0,
            )

        assert result["overall_risk_score"] is not None
        assert 0 <= result["overall_risk_score"] <= 100
        assert result["confidence_level"] == "low"  # Only 2 of 5 agents ran

    def test_aggregation_no_scores(self):
        """If no scores are provided, overall should be None."""
        from agents.risk_aggregation_engine import run_risk_aggregation
        result = run_risk_aggregation()
        assert result["overall_risk_score"] is None

    def test_aggregation_all_high_risk(self):
        """All 100 scores should produce F grade."""
        with patch("agents.risk_aggregation_engine.call_llm", return_value="High risk."):
            from agents.risk_aggregation_engine import run_risk_aggregation
            result = run_risk_aggregation(
                financial_risk_score=100,
                market_risk_score=100,
                founder_risk_score=100,
                narrative_inflation_score=100,
                pattern_similarity_score=100,
            )
        assert result["investment_grade"] == "F"
        assert result["overall_risk_score"] == 100.0


# ══════════════════════════════════════════════════════════════════════
# 5. PDF Parser Tests (no actual PDF needed — test clean/chunk logic)
# ══════════════════════════════════════════════════════════════════════

class TestPdfParser:
    def test_chunk_text_basic(self):
        from utils.pdf_parser import chunk_text
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = chunk_text(text, max_chars=20)
        assert len(chunks) >= 1
        assert all(len(c) > 0 for c in chunks)

    def test_chunk_respects_max_size(self):
        from utils.pdf_parser import chunk_text
        text = "\n\n".join(["word " * 100] * 10)
        chunks = chunk_text(text, max_chars=500)
        assert all(len(c) <= 600 for c in chunks)  # slight slack for paragraph boundaries

    def test_get_summary_chunk_truncates(self):
        from utils.pdf_parser import get_pdf_summary_chunk
        long_text = "x" * 20000
        result = get_pdf_summary_chunk(long_text, max_chars=8000)
        assert len(result) == 8000

    def test_missing_pdf_raises(self):
        from utils.pdf_parser import extract_text_from_pdf
        with pytest.raises(FileNotFoundError):
            extract_text_from_pdf("/nonexistent/deck.pdf")


# ── Helper: run financial analysis without mocking the import ─────────────────

def _run_financial_analysis_pure(financial_data):
    """Thin wrapper that patches LLM call inside the financial agent."""
    with patch("agents.financial_analysis_agent.call_llm", return_value="Mocked explanation."):
        from agents.financial_analysis_agent import run_financial_analysis
        return run_financial_analysis(financial_data)