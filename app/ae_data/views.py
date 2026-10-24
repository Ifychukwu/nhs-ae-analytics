"""The queries behind each page of the app, kept out of the Streamlit file so
they can be tested without a browser. Values are always passed as SQL
parameters, never pasted into the query text."""

from __future__ import annotations

import pandas as pd

from .query import run_select


def headline(con) -> pd.DataFrame:
    """Annual totals with admission rate, oldest year first."""
    return run_select("""
        SELECT w.financial_year, w.attendances, w.pct_within_4h, w.pct_over_12h, w.over_12h,
               a.value AS pct_admitted
        FROM annual_waits w
        LEFT JOIN national a ON a.dimension = 'Attendances' AND a.measure = 'pct_admitted'
             AND a.period_type = 'year' AND a.period = w.financial_year
        ORDER BY w.financial_year""", con)


def monthly(con) -> pd.DataFrame:
    df = run_select("SELECT * FROM monthly_attendances ORDER BY month", con)
    df["date"] = pd.to_datetime(df["month"] + "-01")
    df["per_day"] = df["attendances"] / df["date"].dt.days_in_month
    return df


def dimensions(con) -> list[str]:
    df = run_select("SELECT DISTINCT dimension FROM national WHERE category IS NOT NULL "
                    "AND period_type = 'year' ORDER BY dimension", con)
    return df["dimension"].tolist()


def measures_for(con, dimension: str) -> list[str]:
    df = run_select("SELECT DISTINCT measure FROM national WHERE dimension = ? AND category IS NOT NULL",
                    con, params=(dimension,))
    order = ["attendances", "rate_per_100k", "pct_of_total", "population"]
    return sorted(df["measure"], key=lambda m: order.index(m) if m in order else 99)


def years(con) -> list[str]:
    return run_select("SELECT DISTINCT period FROM national WHERE period_type = 'year' ORDER BY period",
                      con)["period"].tolist()


def has_months(con, dimension: str) -> bool:
    df = run_select("SELECT COUNT(*) AS n FROM national WHERE dimension = ? AND period_type = 'month'",
                    con, params=(dimension,))
    return int(df["n"].iloc[0]) > 0


def breakdown(con, dimension: str, measure: str, year: str, top: int = 20, search: str = "") -> pd.DataFrame:
    return run_select("""
        SELECT category, value FROM national
        WHERE dimension = ? AND measure = ? AND period_type = 'year' AND period = ?
          AND category IS NOT NULL AND category <> 'Total' AND lower(category) LIKE ?
        ORDER BY value DESC LIMIT ?""", con,
                      params=(dimension, measure, year, f"%{search.lower()}%", int(top)))


def trend(con, dimension: str, measure: str, categories: list[str], by: str = "year") -> pd.DataFrame:
    if not categories:
        return pd.DataFrame(columns=["period", "category", "value"])
    marks = ",".join("?" * len(categories))
    return run_select(f"""
        SELECT period, category, value FROM national
        WHERE dimension = ? AND measure = ? AND period_type = ? AND category IN ({marks})
        ORDER BY period""", con, params=(dimension, measure, "month" if by == "month" else "year", *categories))


def trusts(con) -> pd.DataFrame:
    """Organisations with waiting-time figures (trusts and independent providers)."""
    return run_select("SELECT DISTINCT org_code, org_name FROM provider_waits "
                      "WHERE geography_level = 'Provider' ORDER BY org_name", con)


def trust_waits(con, org_code: str) -> pd.DataFrame:
    return run_select("""
        SELECT w.financial_year, w.org_name, w.attendances, w.pct_within_4h, w.pct_over_12h, w.pct_admitted,
               e.pct_within_4h AS england_pct_within_4h, e.pct_over_12h AS england_pct_over_12h
        FROM provider_waits w
        LEFT JOIN provider_waits e ON e.org_code = 'ENG' AND e.financial_year = w.financial_year
        WHERE w.org_code = ? ORDER BY w.financial_year""", con, params=(org_code,))


def trust_mix(con, org_code: str, dimension: str, year: str) -> pd.DataFrame:
    """Share of the trust's attendances in each category, beside England's."""
    return run_select("""
        WITH t AS (SELECT category, value FROM provider
                   WHERE org_code = ? AND dimension = ? AND financial_year = ?),
             e AS (SELECT category, value FROM provider
                   WHERE org_code = 'ENG' AND dimension = ? AND financial_year = ?)
        SELECT t.category,
               100.0 * t.value / (SELECT SUM(value) FROM t) AS trust_pct,
               100.0 * e.value / (SELECT SUM(value) FROM e) AS england_pct
        FROM t LEFT JOIN e ON e.category = t.category
        ORDER BY t.category""", con, params=(org_code, dimension, year, dimension, year))


def provider_years(con) -> list[str]:
    return run_select("SELECT DISTINCT financial_year FROM provider_waits ORDER BY 1", con)["financial_year"].tolist()


MIX_DIMENSIONS = ["Acuity", "Age", "Arrival mode", "Attendance source", "Day of arrival", "Department type",
                  "Discharge destination", "Duration in department", "Ethnicity", "Gender", "Hour of arrival",
                  "IMD decile"]


def league(con, year: str, min_major: int = 10000) -> pd.DataFrame:
    """Trusts with a major (Type 1) A&E of at least ``min_major`` attendances."""
    return run_select("""
        SELECT w.org_code, w.org_name, w.attendances, w.pct_within_4h, w.pct_over_12h, w.pct_admitted
        FROM provider_waits w
        JOIN provider d ON d.org_code = w.org_code AND d.financial_year = w.financial_year
             AND d.dimension = 'Department type' AND d.category LIKE '01%'
        WHERE w.geography_level = 'Provider' AND w.financial_year = ? AND d.value >= ?
        ORDER BY w.pct_over_12h DESC""", con, params=(year, min_major))
