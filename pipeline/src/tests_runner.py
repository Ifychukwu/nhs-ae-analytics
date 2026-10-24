"""Data quality tests run before any analysis.

Each test returns a list of failure messages (empty list = pass), so the
pipeline can report every problem at once rather than stopping at the first.
"""

from __future__ import annotations

import pandas as pd

from load_data import COUNT_COLUMNS


def test_provider_sums_match_published_totals(msitae, published_totals) -> list[str]:
    """Summing providers must reproduce NHS England's own England total row."""
    fails = []
    sums = msitae.groupby("month")[COUNT_COLUMNS].sum()
    for month, row in published_totals.iterrows():
        diff = (sums.loc[month] - row[COUNT_COLUMNS]).abs()
        bad = diff[diff > 0]
        if len(bad):
            fails.append(f"{month:%b %Y}: provider sum != published total for {list(bad.index)}")
    return fails


def test_no_negative_counts(msitae) -> list[str]:
    neg = (msitae[COUNT_COLUMNS] < 0).sum()
    return [f"negative values in {c}" for c, n in neg.items() if n]


def test_one_row_per_provider_month(msitae) -> list[str]:
    dup = msitae.duplicated(["month", "org_code"]).sum()
    return [f"{dup} duplicate provider-month rows"] if dup else []


def test_breaches_do_not_exceed_attendances(msitae) -> list[str]:
    fails = []
    for kind in ["t1", "t2", "other", "booked_t1", "booked_t2", "booked_other"]:
        bad = (msitae[f"over4_{kind}"] > msitae[f"att_{kind}"]).sum()
        if bad:
            fails.append(f"{bad} rows where over-4-hour {kind} exceeds attendances")
    return fails


def test_ecds_rates_in_range(ecds4, ecds12) -> list[str]:
    fails = []
    for col in ["ecds_pct4_total", "ecds_pct4_admitted"]:
        s = ecds4[col].dropna()
        if not s.between(0, 1).all():
            fails.append(f"{col} outside 0-1")
    if not ecds12["ecds_pct12_t12"].dropna().between(0, 1).all():
        fails.append("ecds_pct12_t12 outside 0-1")
    return fails


def test_history_complete(history, annual) -> list[str]:
    fails = []
    if len(history) != 60:
        fails.append(f"expected 60 months of history, got {len(history)}")
    # Monthly figures must add up to the published annual totals.
    fy = history.assign(fy=history["month"].apply(
        lambda m: f"{m.year if m.month >= 4 else m.year - 1}-{str((m.year + 1 if m.month >= 4 else m.year))[2:]}"))
    by_fy = fy.groupby("fy")["attendances"].sum()
    ann = annual.set_index("financial_year")["attendances"]
    for y, v in by_fy.items():
        if y in ann and abs(v - ann[y]) > 1:
            fails.append(f"{y}: monthly sum {v:,.0f} != annual total {ann[y]:,.0f}")
    return fails


def run_all_tests(msitae, published_totals, ecds4, ecds12, history, annual) -> list[str]:
    return (test_provider_sums_match_published_totals(msitae, published_totals)
            + test_no_negative_counts(msitae)
            + test_one_row_per_provider_month(msitae)
            + test_breaches_do_not_exceed_attendances(msitae)
            + test_ecds_rates_in_range(ecds4, ecds12)
            + test_history_complete(history, annual))
