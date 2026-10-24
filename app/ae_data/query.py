"""Read-only access to the A&E database.

Everything the app or the question assistant runs goes through
``run_select``, which only allows a single SELECT (or WITH ... SELECT)
statement on a read-only connection, with a row cap and a time limit.
"""

from __future__ import annotations

import gzip
import re
import shutil
import sqlite3
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "ae.sqlite"
DB_GZ = ROOT / "data" / "ae.sqlite.gz"

MAX_ROWS = 5000
TIME_LIMIT_S = 10


class QueryError(ValueError):
    """Raised for a query that is not allowed or fails."""


def ensure_database() -> Path:
    """Unpack the compressed database on first run (it is shipped as .gz)."""
    if not DB_PATH.exists():
        if not DB_GZ.exists():
            raise FileNotFoundError("No database found. Run `python prepare_data.py` first.")
        with gzip.open(DB_GZ, "rb") as src, open(DB_PATH, "wb") as dst:
            shutil.copyfileobj(src, dst)
    return DB_PATH


def connect() -> sqlite3.Connection:
    path = ensure_database()
    con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)
    allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION}

    def authorizer(action, *_):
        return sqlite3.SQLITE_OK if action in allowed else sqlite3.SQLITE_DENY

    con.set_authorizer(authorizer)
    return con


def _strip_comments(sql: str) -> str:
    sql = re.sub(r"/\*.*?\*/", " ", sql, flags=re.S)
    return re.sub(r"--[^\n]*", " ", sql)


def validate(sql: str) -> str:
    body = _strip_comments(sql).strip().rstrip(";").strip()
    if not body:
        raise QueryError("The query is empty.")
    if ";" in body:
        raise QueryError("Only one statement can be run at a time.")
    if not re.match(r"(?is)^(select|with)\b", body):
        raise QueryError("Only SELECT queries are allowed.")
    return body


def run_select(sql: str, con: sqlite3.Connection | None = None, max_rows: int = MAX_ROWS,
               params: tuple | list = ()) -> pd.DataFrame:
    body = validate(sql)
    own = con is None
    con = con or connect()
    deadline = time.monotonic() + TIME_LIMIT_S
    con.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cur = con.execute(body, params)
        cols = [d[0] for d in cur.description]
        rows = cur.fetchmany(max_rows + 1)
    except sqlite3.DatabaseError as e:
        msg = "The query took too long and was stopped." if "interrupt" in str(e).lower() else str(e)
        raise QueryError(msg) from e
    finally:
        con.set_progress_handler(None, 0)
        if own:
            con.close()
    df = pd.DataFrame(rows[:max_rows], columns=cols)
    df.attrs["truncated"] = len(rows) > max_rows
    return df


# --------------------------------------------------------------------------
# A description of the data, for people and for the assistant
# --------------------------------------------------------------------------

SCHEMA = """
Tables (SQLite):

national(publication, period_type, period, financial_year, dimension, category, measure, value)
  England-level ECDS open data. period_type is 'month' (period 'YYYY-MM', April 2020 to March 2026),
  'year' (period is the financial year 'YYYY-YY', 2020-21 to 2025-26) or 'weekday' (Time of day only).
  Diagnoses (First/Secondary diagnosis) are stored by year only.
  category is NULL for dimensions Attendances, Waiting times and Reattendances; 'Total' rows hold dimension totals.

provider(financial_year, org_code, dimension, category, measure, value)
  Provider level analysis for 2023-24 and 2024-25: trusts, the 7 NHS England regions and England.
  measure is 'attendances' except for dimension 'Waiting times'. Suppressed small counts (1-7) are not stored.

organisations(org_code, org_name, geography_level)
  geography_level is 'Provider', 'Commissioning Region' or 'National' (org_code 'ENG').

provider_waits(financial_year, geography_level, org_code, org_name, attendances, admitted,
               within_4h, over_4h, over_12h, pct_within_4h, pct_over_12h, pct_admitted)
  One row per organisation and year. Percentages are 0-100.

monthly_attendances(month, financial_year, attendances, admitted, pct_admitted)   -- view
annual_waits(financial_year, attendances, pct_within_4h, pct_over_12h, over_12h)   -- view
glossary(kind, name, description)

All percentages are on a 0-100 scale. Category text matching should be case-insensitive
(use LIKE or lower()). Financial years run April to March.
"""


def describe_values(con: sqlite3.Connection | None = None) -> str:
    """List the dimensions and measures that exist, so queries use real labels."""
    own = con is None
    con = con or connect()
    try:
        nat = pd.read_sql("SELECT dimension, GROUP_CONCAT(DISTINCT measure) AS measures, "
                          "COUNT(DISTINCT category) AS categories FROM national GROUP BY dimension", con)
        prov = pd.read_sql("SELECT dimension, COUNT(DISTINCT category) AS categories FROM provider "
                           "GROUP BY dimension", con)
        small = pd.read_sql("SELECT dimension, GROUP_CONCAT(DISTINCT category) AS cats FROM national "
                            "WHERE period_type = 'year' AND dimension IN ('Age','Gender','IMD decile','Ethnicity',"
                            "'Department type','Arrival mode','Attendance category') GROUP BY dimension", con)
        acu = pd.read_sql("SELECT GROUP_CONCAT(DISTINCT category) AS cats FROM provider WHERE dimension='Acuity'", con)
    finally:
        if own:
            con.close()
    lines = ["National dimensions (measures; number of categories):"]
    lines += [f"- {r.dimension}: {r.measures}; {r.categories} categories" for r in nat.itertuples()]
    lines.append("Provider dimensions (number of categories):")
    lines += [f"- {r.dimension}: {r.categories}" for r in prov.itertuples()]
    lines.append("Category labels for small dimensions:")
    lines += [f"- {r.dimension}: {r.cats}" for r in small.itertuples()]
    if len(acu) and acu["cats"].iloc[0]:
        lines.append(f"- Acuity (provider): {acu['cats'].iloc[0]}")
    return "\n".join(lines)
