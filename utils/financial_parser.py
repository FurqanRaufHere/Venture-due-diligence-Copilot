"""
utils/financial_parser.py
──────────────────────────
WHY THIS FILE EXISTS:
  Financial spreadsheets come in many formats. This module:
    1. Reads CSV or XLSX using Pandas
    2. Tries to auto-detect which columns are revenue, costs, etc.
       by matching common header names (fuzzy matching)
    3. Returns a clean, normalized dictionary that the Financial
       Analysis Engine can work with directly

WHY AUTO-DETECTION MATTERS:
  Every startup formats their spreadsheet differently.
  One says "Revenue", another says "Total Sales", another says "ARR".
  Rather than requiring a strict template, we detect the most likely
  column for each financial concept.

THE NORMALIZED SCHEMA:
  {
    "years": [2022, 2023, 2024, ...],
    "revenue": [500000, 1200000, 2800000, ...],
    "costs": [...],
    "gross_profit": [...],
    "cash_balance": [...],
    "headcount": [...],   # optional
    "raw_columns": {...}  # all original columns for reference
  }
"""

import re
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ── Column name matchers (regex patterns) ────────────────────────────────────
# We try each pattern against lowercased column names.
# First match wins.

COLUMN_PATTERNS = {
    "revenue":       [r"revenue", r"total.?sales", r"arr", r"mrr", r"income", r"turnover"],
    "costs":         [r"cost", r"cogs", r"expense", r"opex", r"total.?cost"],
    "gross_profit":  [r"gross.?profit", r"gross.?margin"],
    "net_income":    [r"net.?income", r"net.?profit", r"ebitda", r"net.?loss"],
    "cash_balance":  [r"cash", r"balance", r"runway"],
    "burn_rate":     [r"burn", r"monthly.?spend", r"cash.?out"],
    "headcount":     [r"headcount", r"employees", r"team.?size", r"fte"],
    "cac":           [r"\bcac\b", r"customer.?acquisition.?cost"],
    "ltv":           [r"\bltv\b", r"lifetime.?value", r"clv"],
}


def parse_financial_file(filepath: str) -> Dict[str, Any]:
    """
    Main entry point. Reads a CSV or XLSX and returns a normalized dict.

    Returns:
        {
          "years": [...],
          "revenue": [...],
          "costs": [...],
          "gross_profit": [...],
          "cash_balance": [...],
          "burn_rate": [...],
          "headcount": [...],
          "cac": float or None,
          "ltv": float or None,
          "raw_columns": {col_name: [values...]},
          "detection_log": ["revenue → 'Total Revenue' column", ...],
          "warnings": [...]
        }

    Raises:
        FileNotFoundError, ValueError on bad input
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Financial file not found: {filepath}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        df = pd.read_csv(str(path), thousands=',')
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(str(path), thousands=',')
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Expected .csv or .xlsx")

    logger.info(f"Loaded financial file: {path.name} — {df.shape[0]} rows × {df.shape[1]} cols")

    # Try row-based layout first (each row = metric, each col = year)
    # then column-based (each col = metric, each row = year)
    result = _try_parse_row_based(df)
    if not result:
        result = _try_parse_column_based(df)
    if not result:
        result = _parse_fallback(df)

    return result


def _try_parse_row_based(df: pd.DataFrame) -> Optional[Dict]:
    """
    Layout: First column = metric names, remaining columns = years
    Example:
        Metric       | 2022    | 2023    | 2024
        Revenue      | 500000  | 1200000 | 2800000
        COGS         | 200000  | 450000  | 900000
    """
    first_col = df.iloc[:, 0].astype(str).str.lower().tolist()
    other_cols = df.columns[1:]

    # Check if other column headers look like years
    years = _extract_years_from_headers(other_cols)
    if not years or len(years) < 2:
        return None

    detection_log = []
    warnings = []
    extracted = {"years": years, "detection_log": detection_log, "warnings": warnings}

    for concept, patterns in COLUMN_PATTERNS.items():
        matched_row = _find_row_by_patterns(first_col, patterns)
        if matched_row is not None:
            values = _safe_numeric_row(df, matched_row, other_cols)
            extracted[concept] = values
            detection_log.append(f"{concept} → row '{df.iloc[matched_row, 0]}' matched")
        else:
            extracted[concept] = None

    # Store all rows for reference
    extracted["raw_columns"] = {
        str(df.iloc[i, 0]): _safe_numeric_row(df, i, other_cols)
        for i in range(len(df))
    }

    return extracted


def _try_parse_column_based(df: pd.DataFrame) -> Optional[Dict]:
    """
    Layout: First column = years, remaining columns = metrics
    Example:
        Year | Revenue  | COGS    | Gross Profit
        2022 | 500000   | 200000  | 300000
        2023 | 1200000  | 450000  | 750000
    """
    first_col_values = pd.to_numeric(df.iloc[:, 0], errors='coerce')
    if first_col_values.isna().sum() > len(df) * 0.5:
        return None  # First column isn't mostly numeric years

    years = [int(y) for y in first_col_values.dropna() if 1990 <= y <= 2050]
    if len(years) < 2:
        return None

    col_names = [str(c).lower() for c in df.columns[1:]]
    detection_log = []
    warnings = []
    extracted = {"years": years, "detection_log": detection_log, "warnings": warnings}

    for concept, patterns in COLUMN_PATTERNS.items():
        col_idx = _find_col_by_patterns(col_names, patterns)
        if col_idx is not None:
            actual_col = df.columns[col_idx + 1]
            values = pd.to_numeric(df[actual_col], errors='coerce').tolist()
            extracted[concept] = [v if not np.isnan(v) else None for v in values]
            detection_log.append(f"{concept} → column '{actual_col}' matched")
        else:
            extracted[concept] = None

    extracted["raw_columns"] = {
        str(c): pd.to_numeric(df[c], errors='coerce').tolist()
        for c in df.columns[1:]
    }
    return extracted


def _parse_fallback(df: pd.DataFrame) -> Dict:
    """Last resort: just return all columns as raw data with a warning."""
    logger.warning("Could not detect financial layout. Returning raw data.")
    return {
        "years": [],
        "revenue": None,
        "costs": None,
        "gross_profit": None,
        "cash_balance": None,
        "burn_rate": None,
        "headcount": None,
        "cac": None,
        "ltv": None,
        "raw_columns": {str(c): df[c].tolist() for c in df.columns},
        "detection_log": [],
        "warnings": ["Could not auto-detect financial layout. Manual review needed."]
    }


# ── Helpers ────────────────────────────────────────────────────────────────

def _extract_years_from_headers(headers) -> List[int]:
    years = []
    for h in headers:
        match = re.search(r'(20\d{2}|19\d{2})', str(h))
        if match:
            years.append(int(match.group(1)))
    return sorted(set(years))


def _find_row_by_patterns(row_names: List[str], patterns: List[str]) -> Optional[int]:
    for i, name in enumerate(row_names):
        for pattern in patterns:
            if re.search(pattern, name):
                return i
    return None


def _find_col_by_patterns(col_names: List[str], patterns: List[str]) -> Optional[int]:
    for i, name in enumerate(col_names):
        for pattern in patterns:
            if re.search(pattern, name):
                return i
    return None


def _safe_numeric_row(df: pd.DataFrame, row_idx: int, cols) -> List[Optional[float]]:
    values = []
    for col in cols:
        try:
            val = pd.to_numeric(df.loc[df.index[row_idx], col], errors='coerce')
            values.append(None if np.isnan(val) else float(val))
        except Exception:
            values.append(None)
    return values