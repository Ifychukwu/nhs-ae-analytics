"""Analysis steps: provider metrics, demand forecast, segmentation, inequalities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Greater Manchester acute trusts with a Type 1 emergency department, plus
# The Christie (specialist cancer centre: emergency admissions, no A&E).
GREATER_MANCHESTER = {
    "R0A": "Manchester University NHS FT",
    "RM3": "Northern Care Alliance NHS FT",
    "RWJ": "Stockport NHS FT",
    "RMP": "Tameside and Glossop ICFT",
    "RMC": "Bolton NHS FT",
    "RRF": "Wrightington, Wigan and Leigh FT",
    "RBV": "The Christie NHS FT",
}


# --------------------------------------------------------------------------
# 1. Provider metrics from MSitAE
# --------------------------------------------------------------------------

def provider_metrics(msitae: pd.DataFrame) -> pd.DataFrame:
    """Derive the standard operational measures for each provider and month.

    4 hour performance follows NHS England's published definition, which
    counts booked appointments alongside unplanned attendances.
    """
    d = msitae.copy()
    d["att_unplanned"] = d["att_t1"] + d["att_t2"] + d["att_other"]
    d["att_booked"] = d["att_booked_t1"] + d["att_booked_t2"] + d["att_booked_other"]
    d["att_all"] = d["att_unplanned"] + d["att_booked"]
    d["over4_all"] = d[[c for c in d.columns if c.startswith("over4_")]].sum(axis=1)
    d["att_t1_all"] = d["att_t1"] + d["att_booked_t1"]
    d["over4_t1_all"] = d["over4_t1"] + d["over4_booked_t1"]
    d["adm_via_ae"] = d["adm_t1"] + d["adm_t2"] + d["adm_other"]

    with np.errstate(divide="ignore", invalid="ignore"):
        d["pct4_all"] = 1 - d["over4_all"] / d["att_all"]
        d["pct4_t1"] = 1 - d["over4_t1_all"] / d["att_t1_all"]
        d["conversion_t1"] = d["adm_t1"] / d["att_t1"]
        d["dta12_per_100_adm"] = 100 * d["dta_12h_plus"] / d["adm_via_ae"]
    return d.replace([np.inf, -np.inf], np.nan)


def national_month(metrics: pd.DataFrame) -> pd.DataFrame:
    cols = ["att_unplanned", "att_booked", "att_all", "over4_all", "att_t1_all",
            "over4_t1_all", "adm_via_ae", "dta_12h_plus", "dta_4to12h"]
    n = metrics.groupby("month")[cols].sum()
    n["pct4_all"] = 1 - n["over4_all"] / n["att_all"]
    n["pct4_t1"] = 1 - n["over4_t1_all"] / n["att_t1_all"]
    n["dta12_per_100_adm"] = 100 * n["dta_12h_plus"] / n["adm_via_ae"]
    return n.reset_index()


def type1_providers(metrics: pd.DataFrame, min_t1: int = 2000) -> pd.DataFrame:
    """Acute providers running a major (Type 1) emergency department."""
    keep = metrics.groupby("org_code")["att_t1"].min() >= min_t1
    return metrics[metrics["org_code"].isin(keep[keep].index)].copy()


# --------------------------------------------------------------------------
# 2. National demand forecast
# --------------------------------------------------------------------------

# March 2020 to May 2021 is excluded from model fitting: lockdowns suppressed
# attendances far below any normal seasonal pattern.
COVID_START, COVID_END = pd.Timestamp("2020-03-01"), pd.Timestamp("2021-05-01")


def _design(months: pd.Series, origin: pd.Timestamp) -> np.ndarray:
    t = ((months.dt.year - origin.year) * 12 + months.dt.month - origin.month) / 12.0
    dummies = np.zeros((len(months), 11))
    for i, m in enumerate(months.dt.month):
        if m != 4:                       # April is the reference month
            dummies[i, (m - 5) % 12] = 1  # May..March -> columns 0..10
    return np.column_stack([np.ones(len(months)), t, dummies])


def fit_seasonal_model(history: pd.DataFrame) -> dict:
    """OLS on log(attendances per day) with a linear trend and month effects."""
    h = history.copy()
    h["per_day"] = h["attendances"] / h["month"].dt.days_in_month
    train = h[(h["month"] < COVID_START) | (h["month"] > COVID_END)]
    origin = h["month"].min()
    X, y = _design(train["month"], origin), np.log(train["per_day"].to_numpy())
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    sigma = float(np.sqrt(resid @ resid / (len(y) - X.shape[1])))
    fitted = train[["month", "per_day"]].assign(fitted=np.exp(X @ beta))
    return {"beta": beta, "sigma": sigma, "origin": origin, "n_train": len(y),
            "trend_pct_per_year": float(np.expm1(beta[1]) * 100),
            "in_sample_mape": float(np.mean(np.abs(fitted["per_day"] / fitted["fitted"] - 1)) * 100),
            "fitted": fitted}


def predict(model: dict, months: pd.Series) -> np.ndarray:
    return np.exp(_design(months, model["origin"]) @ model["beta"])


def backtest_and_forecast(model: dict, actual_2026: pd.DataFrame, horizon: int = 6) -> dict:
    """Score the model on the 2026 months it never saw, then forecast ahead.

    The historical file ends in March 2023, so the model is projected across a
    three-year gap. The backtest measures how far that projection drifts; the
    forward forecast keeps the seasonal shape but re-anchors the level to the
    latest observed months (the mean log error across them).
    """
    a = actual_2026.copy()
    a["per_day"] = a["att_unplanned"] / a["month"].dt.days_in_month
    a["predicted_per_day"] = predict(model, a["month"])
    a["error_pct"] = 100 * (a["per_day"] / a["predicted_per_day"] - 1)
    level = float(np.mean(np.log(a["per_day"] / a["predicted_per_day"])))

    last = a["month"].max()
    future = pd.Series(pd.date_range(last + pd.offsets.MonthBegin(1), periods=horizon, freq="MS"))
    base = predict(model, future) * np.exp(level)
    s = model["sigma"] * np.sqrt(1 + 1 / len(a))  # month noise + level uncertainty
    days = future.dt.days_in_month.to_numpy()
    fc = pd.DataFrame({
        "month": future,
        "forecast": base * days,
        "lower_95": base * np.exp(-1.96 * s) * days,
        "upper_95": base * np.exp(1.96 * s) * days,
    })
    return {"backtest": a[["month", "att_unplanned", "per_day", "predicted_per_day", "error_pct"]],
            "backtest_mape": float(a["error_pct"].abs().mean()),
            "level_shift_pct": float(np.expm1(level) * 100),
            "forecast": fc}


# --------------------------------------------------------------------------
# 3. Provider segmentation (k-means)
# --------------------------------------------------------------------------

SEGMENT_FEATURES = {
    "pct4_t1": "Type 1 seen within 4 hours",
    "ecds_pct4_admitted": "Admitted patients within 4 hours (ECDS)",
    "ecds_pct12_t12": "Share spending 12+ hours in A&E (ECDS)",
    "dta12_per_100_adm": "12+ hour trolley waits per 100 admissions",
    "pct4_t1_change_pp": "Change in Type 1 4 hour % since July",
}

# Admission rate is kept as context but left out of clustering: it reflects
# case mix and local counting practice more than operational pressure.

# Direction in which each feature signals *more* pressure.
PRESSURE_SIGN = {"pct4_t1": -1, "ecds_pct4_admitted": -1, "ecds_pct12_t12": 1,
                 "dta12_per_100_adm": 1, "pct4_t1_change_pp": -1}


def build_segment_table(t1_metrics: pd.DataFrame, ecds4: pd.DataFrame,
                        ecds12: pd.DataFrame, month: pd.Timestamp,
                        baseline: pd.Timestamp) -> pd.DataFrame:
    cur = t1_metrics[t1_metrics["month"] == month].set_index("org_code")
    base = t1_metrics[t1_metrics["month"] == baseline].set_index("org_code")
    tbl = cur[["org_name", "region", "att_t1_all", "pct4_t1", "conversion_t1", "dta12_per_100_adm"]].copy()
    tbl["pct4_t1_change_pp"] = 100 * (cur["pct4_t1"] - base["pct4_t1"])
    e4 = ecds4[ecds4["month"] == month].set_index("org_code")[["ecds_pct4_admitted", "ecds_pct4_non_admitted"]]
    e12 = ecds12[ecds12["month"] == month].set_index("org_code")[["ecds_pct12_t12"]]
    return tbl.join(e4, how="inner").join(e12, how="inner").dropna(subset=list(SEGMENT_FEATURES))


def segment_providers(tbl: pd.DataFrame, k_range=range(3, 7), seed: int = 42) -> tuple[pd.DataFrame, dict]:
    feats = list(SEGMENT_FEATURES)
    X = StandardScaler().fit_transform(tbl[feats])
    scores = {}
    for k in k_range:
        labels = KMeans(n_clusters=k, n_init=25, random_state=seed).fit_predict(X)
        scores[k] = float(silhouette_score(X, labels))
    best_k = max(scores, key=scores.get)
    km = KMeans(n_clusters=best_k, n_init=25, random_state=seed).fit(X)

    out = tbl.copy()
    out["cluster"] = km.labels_
    z = pd.DataFrame(X, columns=feats, index=tbl.index)
    out["pressure_score"] = sum(z[f] * s for f, s in PRESSURE_SIGN.items() if s) / sum(
        1 for s in PRESSURE_SIGN.values() if s)
    # Order clusters from least to most pressured so labels are reproducible.
    order = out.groupby("cluster")["pressure_score"].mean().sort_values().index.tolist()
    names = {3: ["Lower pressure", "Moderate pressure", "High pressure"],
             4: ["Lower pressure", "Moderate pressure", "High pressure", "Severe pressure"],
             5: ["Lower pressure", "Moderate pressure", "Elevated pressure", "High pressure", "Severe pressure"],
             6: ["Lowest pressure", "Lower pressure", "Moderate pressure", "Elevated pressure", "High pressure",
                 "Severe pressure"]}[best_k]
    out["segment"] = out["cluster"].map({c: names[i] for i, c in enumerate(order)})
    out["segment_rank"] = out["cluster"].map({c: i for i, c in enumerate(order)})
    return out.drop(columns="cluster"), {"silhouette_by_k": scores, "k": best_k}


# --------------------------------------------------------------------------
# 4. Inequalities from the ECDS open data
# --------------------------------------------------------------------------

IMD_ORDER = ["Most deprived 10%", "More deprived 10-20%", "More deprived 20-30%",
             "More deprived 30-40%", "More deprived 40-50%", "Less deprived 40-50%",
             "Less deprived 30-40%", "Less deprived 20-30%", "Less deprived 10-20%",
             "Least deprived 10%"]

AGE_ORDER = ["0", "1-4", "5-9", "10-14", "15", "16", "17", "18", "19", "20-24", "25-29",
             "30-34", "35-39", "40-44", "45-49", "50-54", "55-59", "60-64", "65-69",
             "70-74", "75-79", "80-84", "85-89", "90+"]


def _rates(open_data: pd.DataFrame, group: str, period: str) -> pd.DataFrame:
    s = open_data[(open_data["measure_group"] == group) & (open_data["reporting_period"] == period)]
    return s.pivot_table(index="measure_type", columns="measure", values="measure_value")


def deprivation_gradient(open_data: pd.DataFrame, period: str = "2223") -> pd.DataFrame:
    p = _rates(open_data, "IMD_DECILE", period).reindex(IMD_ORDER)
    p["decile"] = range(1, 11)
    return p.rename(columns={"Rate per 100,000 Population": "rate_per_100k",
                             "Attendances": "attendances", "Population": "population"}).reset_index()


def age_profile(open_data: pd.DataFrame, period: str = "2223") -> pd.DataFrame:
    p = _rates(open_data, "AGE", period).reindex(AGE_ORDER)
    return p.rename(columns={"Rate per 100,000 Population": "rate_per_100k",
                             "Attendances": "attendances", "Population": "population"}).reset_index()


def ethnicity_recording(open_data: pd.DataFrame, period: str = "2223") -> dict:
    p = _rates(open_data, "ETHNICITY", period)
    total = p.loc["ETHNICITY_TOTAL", "Attendances"]
    missing = p.loc[["Not stated", "Unknown"], "Attendances"].sum()
    return {"total": float(total), "not_recorded": float(missing),
            "pct_not_recorded": float(100 * missing / total)}


def ecds_waiting_trend(open_data: pd.DataFrame) -> pd.DataFrame:
    s = open_data[open_data["measure_group"] == "WAITING_TIMES"]
    p = s.pivot_table(index="reporting_period", columns="measure", values="measure_value")
    p.index = p.index.map({"2021": "2020-21", "2122": "2021-22", "2223": "2022-23"})
    return p[["%_LESS_THAN_4_HOURS_FROM_ARRIVAL_TO_DEPARTURE",
              "%_MORE_THAN_12_HOURS_FROM_ARRIVAL_TO_DEPARTURE",
              "TOTAL_ATTENDANCES"]].rename(columns={
        "%_LESS_THAN_4_HOURS_FROM_ARRIVAL_TO_DEPARTURE": "pct_within_4h",
        "%_MORE_THAN_12_HOURS_FROM_ARRIVAL_TO_DEPARTURE": "pct_over_12h",
        "TOTAL_ATTENDANCES": "attendances"}).reset_index()
