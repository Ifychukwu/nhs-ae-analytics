"""Data checks run before the database is written. Each returns failure
messages; an empty list means the check passed."""

from __future__ import annotations

import pandas as pd


def check_no_duplicate_keys(national: pd.DataFrame) -> list[str]:
    key = ["dimension", "period_type", "period", "category", "measure"]
    dup = national.duplicated(key).sum()
    return [f"{dup} duplicate national rows"] if dup else []


def check_months_add_up_to_years(national: pd.DataFrame, tolerance: float = 0.005) -> list[str]:
    """Monthly attendances must add up to the published yearly total (within 0.5%)."""
    att = national[(national["dimension"] == "Attendances") & (national["measure"] == "attendances")]
    months = att[att["period_type"] == "month"].groupby("financial_year")
    years = att[att["period_type"] == "year"].set_index("period")["value"]
    fails = []
    for fy, g in months:
        if len(g) == 12 and fy in years:
            diff = abs(g["value"].sum() / years[fy] - 1)
            if diff > tolerance:
                fails.append(f"{fy}: months sum to {g['value'].sum():,.0f}, year total {years[fy]:,.0f}")
    return fails


def check_percentages_in_range(national: pd.DataFrame, waits: pd.DataFrame) -> list[str]:
    fails = []
    pct = national[national["measure"].str.startswith("pct_")]["value"]
    if not pct.between(0, 100).all():
        fails.append(f"{(~pct.between(0, 100)).sum()} national percentages outside 0-100")
    for c in ["pct_within_4h", "pct_over_12h"]:
        s = waits[c].dropna()
        if not s.between(0, 100).all():
            fails.append(f"provider {c} outside 0-100")
    return fails


def check_wait_counts_consistent(national: pd.DataFrame) -> list[str]:
    """Published counts and percentages for waiting times must agree."""
    w = national[(national["dimension"] == "Waiting times") & (national["period_type"] == "year")]
    p = w.pivot_table(index="period", columns="measure", values="value")
    fails = []
    if {"over_12h", "attendances", "pct_over_12h"} <= set(p.columns):
        calc = 100 * p["over_12h"] / p["attendances"]
        bad = (calc - p["pct_over_12h"]).abs() > 0.5
        fails += [f"{y}: 12 hour % does not match counts" for y in p.index[bad]]
    return fails


def check_providers_sum_to_england(provider: pd.DataFrame, tolerance: float = 0.01) -> list[str]:
    """Trust totals must add up to the England total in the same provider file
    (counts are rounded to the nearest 5, so a small gap is allowed)."""
    tot = provider[provider["dimension"] == "Total"]
    fails = []
    for fy, g in tot.groupby("financial_year"):
        eng = g.loc[g["org_code"] == "ENG", "value"]
        prov = g.loc[g["geography_level"] == "Provider", "value"].sum()
        if len(eng) and abs(prov / eng.iloc[0] - 1) > tolerance:
            fails.append(f"{fy}: trust totals sum to {prov:,.0f} vs England {eng.iloc[0]:,.0f}")
    return fails


def waiting_time_coverage(waits: pd.DataFrame, national: pd.DataFrame) -> list[str]:
    """Not a failure: report how much of each year's published England total
    the provider waiting-time counts cover."""
    eng = national[(national["dimension"] == "Waiting times") & (national["measure"] == "attendances")
                   & (national["period_type"] == "year")].set_index("period")["value"]
    notes = []
    for fy, g in waits[waits["geography_level"] == "Provider"].groupby("financial_year"):
        if fy in eng:
            notes.append(f"{fy}: provider waiting times cover {g['attendances'].sum() / eng[fy]:.1%} "
                         f"of published England attendances")
    return notes


def run_checks(national, provider, waits) -> list[str]:
    out = check_no_duplicate_keys(national) + check_months_add_up_to_years(national) \
        + check_wait_counts_consistent(national)
    if len(waits):
        out += check_percentages_in_range(national, waits) + check_providers_sum_to_england(provider)
    return out
