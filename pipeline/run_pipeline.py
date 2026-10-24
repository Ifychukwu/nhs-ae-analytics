"""Run the full pipeline: load -> test -> analyse -> write outputs.

Usage:
    python run_pipeline.py            # reads data/, writes outputs/

Every published number used downstream passes the data tests in
tests/test_data_quality.py first; the run stops if any test fails.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import analysis as A  # noqa: E402
import load_data as L  # noqa: E402
from tests_runner import run_all_tests  # noqa: E402

DATA, OUT = ROOT / "data", ROOT / "outputs"


def find(pattern: str) -> Path:
    hits = sorted(DATA.glob(pattern))
    if not hits:
        raise FileNotFoundError(f"No file in data/ matches {pattern!r}")
    return hits[-1]


def main() -> None:
    OUT.mkdir(exist_ok=True)

    # ---- Load ----------------------------------------------------------
    msitae, published_totals = L.load_all_msitae(sorted(DATA.glob("*-2026-CSV-*.csv")))
    ecds_files = sorted(DATA.glob("*ECDS_Performance_*.xlsx"))
    ecds4 = pd.concat([L.load_ecds_4hr(f) for f in ecds_files], ignore_index=True)
    ecds12 = pd.concat([L.load_ecds_12hr(f) for f in ecds_files], ignore_index=True)
    icb12 = pd.concat([L.load_ecds_icb_12hr(f) for f in ecds_files], ignore_index=True)
    tables = find("*AE2223_ECDS_MSitAE_Tables.xlsx")
    history = L.load_monthly_history(tables)
    annual = L.load_annual_history(tables)
    open_data = L.load_open_data(find("*AE2223_Open_Data*.csv"))

    # ---- Test ----------------------------------------------------------
    failures = run_all_tests(msitae, published_totals, ecds4, ecds12, history, annual)
    if failures:
        print("Data tests failed:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("All data tests passed.")

    # ---- Analyse -------------------------------------------------------
    metrics = A.provider_metrics(msitae)
    nat = A.national_month(metrics)
    t1 = A.type1_providers(metrics)

    model = A.fit_seasonal_model(history)
    fc = A.backtest_and_forecast(model, nat[["month", "att_unplanned"]])

    latest, first = metrics["month"].max(), metrics["month"].min()
    seg_tbl = A.build_segment_table(t1, ecds4, ecds12, latest, first)
    segments, seg_info = A.segment_providers(seg_tbl)

    gm = metrics[metrics["org_code"].isin(A.GREATER_MANCHESTER)].copy()
    gm["short_name"] = gm["org_code"].map(A.GREATER_MANCHESTER)
    gm = gm.merge(ecds12[["month", "org_code", "ecds_pct12_t12"]], on=["month", "org_code"], how="left")

    imd = A.deprivation_gradient(open_data)
    age = A.age_profile(open_data)
    eth = A.ethnicity_recording(open_data)
    waits = A.ecds_waiting_trend(open_data)

    # ---- Write ---------------------------------------------------------
    metrics.to_csv(OUT / "provider_month_metrics.csv", index=False)
    nat.to_csv(OUT / "national_month.csv", index=False)
    fc["backtest"].to_csv(OUT / "forecast_backtest.csv", index=False)
    fc["forecast"].to_csv(OUT / "forecast_winter_2026_27.csv", index=False)
    model["fitted"].to_csv(OUT / "forecast_model_fit.csv", index=False)
    segments.reset_index().to_csv(OUT / "provider_segments.csv", index=False)
    gm.to_csv(OUT / "greater_manchester.csv", index=False)
    icb12.to_csv(OUT / "icb_12hr.csv", index=False)
    imd.to_csv(OUT / "deprivation_gradient.csv", index=False)
    age.to_csv(OUT / "age_profile.csv", index=False)
    waits.to_csv(OUT / "ecds_waits_by_year.csv", index=False)

    n_latest = nat[nat["month"] == latest].iloc[0]
    n_first = nat[nat["month"] == first].iloc[0]
    summary = {
        "latest_month": latest.strftime("%B %Y"),
        "pct4_all_latest": round(100 * n_latest["pct4_all"], 1),
        "pct4_all_first": round(100 * n_first["pct4_all"], 1),
        "pct4_t1_latest": round(100 * n_latest["pct4_t1"], 1),
        "att_all_latest": int(n_latest["att_all"]),
        "dta12_latest": int(n_latest["dta_12h_plus"]),
        "dta12_first": int(n_first["dta_12h_plus"]),
        "model_trend_pct_per_year": round(model["trend_pct_per_year"], 2),
        "model_in_sample_mape": round(model["in_sample_mape"], 2),
        "backtest_mape": round(fc["backtest_mape"], 2),
        "level_shift_pct": round(fc["level_shift_pct"], 1),
        "winter_forecast_total": int(fc["forecast"]["forecast"].sum()),
        "segments_k": seg_info["k"],
        "silhouette_by_k": {str(k): round(v, 3) for k, v in seg_info["silhouette_by_k"].items()},
        "n_segmented": int(len(segments)),
        "imd_ratio_most_to_least": round(imd["rate_per_100k"].iloc[0] / imd["rate_per_100k"].iloc[-1], 2),
        "ethnicity_pct_not_recorded": round(eth["pct_not_recorded"], 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
