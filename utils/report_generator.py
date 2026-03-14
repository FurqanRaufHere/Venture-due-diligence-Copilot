"""
utils/report_generator.py
────────────────────────────────────────────────────────────────
STEP 19 — PDF Report Generation

Generates a professional investment memo PDF from analysis results.
Uses ReportLab — already installed (used in sample data generation).

WHAT THE PDF CONTAINS:
  Page 1: Cover — startup name, grade, score, date
  Page 2: Executive Summary + Risk Breakdown table
  Page 3: Financial Analysis — metrics + red flags
  Page 4: Market & Competition Analysis
  Page 5: Founder Risk Profile
  Page 6: Pattern Similarity + Recommendations

TRIGGERED FROM:
  GET /api/report/{job_id}  → returns PDF file download
  Frontend "Export Report" button calls this endpoint
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("./reports")
REPORTS_DIR.mkdir(exist_ok=True)

# ── Colors ────────────────────────────────────────────────────
NAVY       = (0.039, 0.086, 0.157)   # #0A1628
TEAL       = (0.0,   0.784, 0.588)   # #00C896
GOLD       = (0.961, 0.784, 0.259)   # #F5C842
CORAL      = (1.0,   0.361, 0.361)   # #FF5C5C
WHITE      = (0.973, 0.976, 0.980)   # #F8F9FA
GRAY_DARK  = (0.290, 0.333, 0.408)   # #4A5568
GRAY_MID   = (0.533, 0.573, 0.643)   # #8892A4
SURFACE    = (0.067, 0.118, 0.208)   # #111E35
SURFACE2   = (0.094, 0.153, 0.267)   # #182744


def generate_report(job_id: str, data: Dict[str, Any]) -> str:
    """
    Generate a PDF investment memo for the given analysis results.

    Args:
        job_id: The analysis job ID (used for filename)
        data:   Full results dict from /api/results/{job_id}

    Returns:
        Path to the generated PDF file
    """
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
        HRFlowable, PageBreak
    )
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    from reportlab.lib import colors

    output_path = REPORTS_DIR / f"vdd_report_{job_id[:8]}.pdf"

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=0.75*inch, leftMargin=0.75*inch,
        topMargin=0.75*inch, bottomMargin=0.75*inch,
        title="VDD Investment Memo",
        author="AI Venture Due Diligence Copilot",
    )

    # ── Extract data ──────────────────────────────────────────
    risk    = data.get("risk_score") or {}
    claims  = data.get("extracted_claims") or {}
    fin     = data.get("financial_metrics") or {}
    memo    = risk.get("due_diligence_memo") or {}
    name    = data.get("startup_name") or "Unknown Startup"
    grade   = risk.get("investment_grade") or "N/A"
    score   = risk.get("overall_risk_score")
    conf    = risk.get("confidence_level") or "low"

    # ── Style helpers ─────────────────────────────────────────
    def rl_color(rgb): return colors.Color(*rgb)

    def style(name, **kw):
        kw.setdefault('fontName', 'Helvetica')
        kw.setdefault('fontSize', 10)
        kw.setdefault('textColor', rl_color(WHITE))
        kw.setdefault('leading', 14)
        return ParagraphStyle(name, **kw)

    S = {
        'cover_title':  style('ct', fontName='Helvetica-Bold', fontSize=32,
                               textColor=rl_color(WHITE), alignment=TA_CENTER, spaceAfter=8),
        'cover_sub':    style('cs', fontSize=14, textColor=rl_color(GRAY_MID),
                               alignment=TA_CENTER, spaceAfter=6),
        'cover_grade':  style('cg', fontName='Helvetica-Bold', fontSize=52,
                               textColor=rl_color(TEAL), alignment=TA_CENTER),
        'h1':           style('h1', fontName='Helvetica-Bold', fontSize=18,
                               textColor=rl_color(TEAL), spaceBefore=16, spaceAfter=8),
        'h2':           style('h2', fontName='Helvetica-Bold', fontSize=13,
                               textColor=rl_color(WHITE), spaceBefore=12, spaceAfter=6),
        'body':         style('body', fontSize=10, textColor=rl_color(GRAY_MID),
                               leading=15, spaceAfter=6),
        'body_white':   style('bw', fontSize=10, textColor=rl_color(WHITE),
                               leading=15, spaceAfter=4),
        'mono':         style('mono', fontName='Courier', fontSize=9,
                               textColor=rl_color(TEAL)),
        'label':        style('lbl', fontName='Helvetica-Bold', fontSize=9,
                               textColor=rl_color(GRAY_MID), spaceAfter=2),
        'flag':         style('fl', fontSize=10, textColor=rl_color(CORAL), leading=14),
        'rec':          style('rec', fontSize=10, textColor=rl_color(WHITE), leading=15),
        'footer':       style('ft', fontSize=8, textColor=rl_color(GRAY_DARK),
                               alignment=TA_CENTER),
    }

    def sp(h=12): return Spacer(1, h)
    def divider(): return HRFlowable(width="100%", thickness=0.5,
                                      color=rl_color(SURFACE2), spaceAfter=10, spaceBefore=4)

    def section_header(text):
        return [
            HRFlowable(width="100%", thickness=2, color=rl_color(TEAL),
                       spaceAfter=6, spaceBefore=16),
            Paragraph(text, S['h1']),
        ]

    def metric_table(rows):
        """2-column key-value table for metrics."""
        t = Table(rows, colWidths=[2.8*inch, 4.5*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (0,-1), rl_color(SURFACE2)),
            ('BACKGROUND', (1,0), (1,-1), rl_color(SURFACE)),
            ('TEXTCOLOR',  (0,0), (0,-1), rl_color(GRAY_MID)),
            ('TEXTCOLOR',  (1,0), (1,-1), rl_color(WHITE)),
            ('FONTNAME',   (0,0), (0,-1), 'Helvetica'),
            ('FONTNAME',   (1,0), (1,-1), 'Helvetica'),
            ('FONTSIZE',   (0,0), (-1,-1), 10),
            ('ROWBACKGROUNDS', (0,0), (-1,-1), [rl_color(SURFACE), rl_color(SURFACE2)]),
            ('GRID',       (0,0), (-1,-1), 0.3, rl_color(SURFACE2)),
            ('LEFTPADDING',(0,0), (-1,-1), 10),
            ('RIGHTPADDING',(0,0),(-1,-1), 10),
            ('TOPPADDING', (0,0), (-1,-1), 7),
            ('BOTTOMPADDING',(0,0),(-1,-1), 7),
        ]))
        return t

    def score_table(dims):
        """Risk score breakdown table."""
        rows = [['Risk Dimension', 'Weight', 'Score', 'Status']]
        for d in dims:
            v = d['value']
            status = 'N/A' if v is None else ('LOW' if v <= 30 else 'MEDIUM' if v <= 60 else 'HIGH')
            rows.append([d['label'], d['weight'], f"{v:.1f}/100" if v is not None else 'N/A', status])

        t = Table(rows, colWidths=[2.8*inch, 1*inch, 1.2*inch, 1.3*inch])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), rl_color(NAVY)),
            ('TEXTCOLOR',  (0,0), (-1,0), rl_color(TEAL)),
            ('FONTNAME',   (0,0), (-1,0), 'Helvetica-Bold'),
            ('FONTSIZE',   (0,0), (-1,-1), 10),
            ('ROWBACKGROUNDS', (0,1), (-1,-1), [rl_color(SURFACE), rl_color(SURFACE2)]),
            ('TEXTCOLOR',  (0,1), (-1,-1), rl_color(WHITE)),
            ('GRID',       (0,0), (-1,-1), 0.3, rl_color(SURFACE2)),
            ('LEFTPADDING',(0,0), (-1,-1), 10),
            ('TOPPADDING', (0,0), (-1,-1), 8),
            ('BOTTOMPADDING',(0,0),(-1,-1), 8),
        ]))
        return t

    # ── Build story ───────────────────────────────────────────
    story = []

    # ── PAGE 1: COVER ─────────────────────────────────────────
    story += [
        sp(60),
        Paragraph("AI VENTURE DUE DILIGENCE COPILOT", S['cover_sub']),
        sp(20),
        Paragraph(name, S['cover_title']),
        sp(8),
        Paragraph("Investment Risk Assessment", S['cover_sub']),
        sp(40),
        Paragraph(grade, S['cover_grade']),
        Paragraph(f"Investment Grade", S['cover_sub']),
        sp(8),
        Paragraph(f"Overall Risk Score: {score:.1f}/100" if score else "Score: N/A",
                  style('sc', fontSize=16, textColor=rl_color(WHITE), alignment=TA_CENTER)),
        sp(6),
        Paragraph(f"Confidence: {conf.upper()}", S['cover_sub']),
        sp(60),
        HRFlowable(width="100%", thickness=0.5, color=rl_color(SURFACE2)),
        sp(12),
        Paragraph(
            f"Generated: {datetime.now().strftime('%B %d, %Y at %H:%M')}  ·  "
            f"AI Venture Due Diligence Copilot v2.0  ·  Powered by Groq LLM",
            S['footer']
        ),
        PageBreak(),
    ]

    # ── PAGE 2: EXECUTIVE SUMMARY + RISK BREAKDOWN ────────────
    story += section_header("Executive Summary")
    exec_summary = memo.get('executive_summary', 'No summary available.')
    story += [Paragraph(exec_summary, S['body_white']), sp(16)]

    story += section_header("Risk Score Breakdown")
    dims = [
        {'label': 'Financial Risk',     'weight': '30%', 'value': risk.get('financial_risk_score')},
        {'label': 'Market Risk',        'weight': '25%', 'value': risk.get('market_risk_score')},
        {'label': 'Founder Risk',       'weight': '20%', 'value': risk.get('founder_risk_score')},
        {'label': 'Narrative Inflation','weight': '15%', 'value': risk.get('narrative_inflation_score')},
        {'label': 'Pattern Similarity', 'weight': '10%', 'value': risk.get('pattern_similarity_score')},
    ]
    story += [score_table(dims), sp(16)]

    all_flags = memo.get('all_red_flags', [])
    if all_flags:
        story += section_header(f"Red Flags ({len(all_flags)})")
        for flag in all_flags:
            story.append(Paragraph(f"⚠  {flag}", S['flag']))
            story.append(sp(4))

    story.append(PageBreak())

    # ── PAGE 3: EXTRACTED CLAIMS + FINANCIAL ──────────────────
    story += section_header("Extracted Business Claims")
    claim_rows = [
        ['Problem Statement', claims.get('problem_statement') or 'N/A'],
        ['Solution',          claims.get('solution_claim') or 'N/A'],
        ['Target Market',     claims.get('target_market') or 'N/A'],
        ['TAM Claim',         claims.get('tam_claim') or 'N/A'],
        ['Revenue Model',     claims.get('revenue_model') or 'N/A'],
        ['Hype Phrases',      ', '.join(claims.get('hype_indicators') or []) or 'None detected'],
    ]
    story += [metric_table(claim_rows), sp(16)]

    story += section_header("Financial Analysis")
    fin_rows = [
        ['Revenue CAGR',        f"{fin.get('revenue_cagr'):.1f}%" if fin.get('revenue_cagr') is not None else 'N/A'],
        ['Monthly Burn Rate',   f"${fin.get('burn_rate_monthly'):,.0f}" if fin.get('burn_rate_monthly') is not None else 'N/A'],
        ['Runway',              f"{fin.get('runway_months'):.1f} months" if fin.get('runway_months') is not None else 'N/A'],
        ['Gross Margin (avg)',  f"{fin.get('gross_margin_avg'):.1f}%" if fin.get('gross_margin_avg') is not None else 'N/A'],
        ['Plausibility Score',  f"{fin.get('financial_plausibility_score')}/100" if fin.get('financial_plausibility_score') is not None else 'N/A'],
    ]
    story += [metric_table(fin_rows), sp(12)]

    anomaly = fin.get('anomaly_explanation', '')
    if anomaly:
        story += [Paragraph("Anomaly Analysis", S['h2']),
                  Paragraph(anomaly, S['body']), sp(8)]

    story.append(PageBreak())

    # ── PAGE 4: RECOMMENDATIONS ───────────────────────────────
    story += section_header("Investment Recommendations")
    recs = memo.get('recommendations', [])
    for i, rec in enumerate(recs, 1):
        story.append(Paragraph(f"{i}.  {rec}", S['rec']))
        story.append(sp(8))

    sp_comps = memo.get('comparable_startups', [])
    if sp_comps:
        story += section_header("Comparable Companies")
        story.append(Paragraph(', '.join(sp_comps), S['body_white']))
        story.append(sp(12))

    archetype = memo.get('dominant_failure_archetype')
    if archetype:
        story += section_header("Failure Pattern Archetype")
        story.append(Paragraph(
            archetype.replace('_', ' ').title(),
            style('at', fontName='Helvetica-Bold', fontSize=14,
                  textColor=rl_color(GOLD))
        ))
        story.append(sp(8))

    # ── FOOTER PAGE ───────────────────────────────────────────
    story.append(PageBreak())
    story += [
        sp(100),
        HRFlowable(width="100%", thickness=0.5, color=rl_color(SURFACE2)),
        sp(16),
        Paragraph("DISCLAIMER", style('disc', fontName='Helvetica-Bold', fontSize=9,
                                       textColor=rl_color(GRAY_MID), alignment=TA_CENTER)),
        sp(8),
        Paragraph(
            "This report was generated by the AI Venture Due Diligence Copilot, an automated "
            "analysis system. It is intended as a decision-support tool only and does not "
            "constitute financial, legal, or investment advice. All scores and assessments "
            "should be reviewed by qualified investment professionals before making any "
            "investment decisions. Past startup patterns do not guarantee future outcomes.",
            style('dbt', fontSize=8, textColor=rl_color(GRAY_DARK),
                  alignment=TA_CENTER, leading=13)
        ),
        sp(16),
        Paragraph(
            f"AI Venture Due Diligence Copilot  ·  Report ID: {job_id[:8]}  ·  "
            f"{datetime.now().strftime('%Y-%m-%d')}",
            S['footer']
        ),
    ]

    # Build with dark background on every page
    def dark_background(canvas, doc):
        canvas.saveState()
        canvas.setFillColorRGB(*NAVY)
        canvas.rect(0, 0, letter[0], letter[1], fill=1, stroke=0)
        canvas.restoreState()

    doc.build(story, onFirstPage=dark_background, onLaterPages=dark_background)
    logger.info(f"PDF report generated: {output_path}")
    return str(output_path)
