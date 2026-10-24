"""Turn NHS England's yearly ECDS open-data files into one consistent table.

Each yearly publication uses slightly different labels for the same thing
(``IMD_DECILE`` / ``Imd_Decile`` / ``IMD Decile``; ``2122`` / ``2021/22``;
``2025-4`` / ``2025-04``). The functions here map every variant onto one
vocabulary so that years can be compared and queried together. No published
value is changed except that suppressed cells ("*") become missing.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------

def read_csv_any_encoding(path: Path, **kw) -> pd.DataFrame:
    """The 2024-25 and 2025-26 files are Windows-1252, earlier ones UTF-8."""
    for enc in ("utf-8", "cp1252"):
        try:
            return pd.read_csv(path, dtype=str, encoding=enc, **kw)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode {path}")


def to_number(s: pd.Series) -> pd.Series:
    cleaned = s.astype(str).str.replace(",", "", regex=False).str.strip()
    cleaned = cleaned.replace({"*": np.nan, "": np.nan, "nan": np.nan, "-": np.nan})
    return pd.to_numeric(cleaned, errors="coerce")


def tidy_text(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.replace(r"\s+", " ", regex=True).replace({"nan": np.nan})


# --------------------------------------------------------------------------
# Periods
# --------------------------------------------------------------------------

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def financial_year_of_month(month: str) -> str:
    y, m = int(month[:4]), int(month[5:7])
    start = y if m >= 4 else y - 1
    return f"{start}-{str(start + 1)[2:]}"


def parse_period(raw: str) -> tuple[str, str, str | None]:
    """Return (period_type, period, financial_year).

    Examples: '2025-4' -> ('month', '2025-04', '2025-26');
    '2021/22' and '2122' -> ('year', '2021-22', '2021-22');
    "('Friday',)" -> ('weekday', 'Friday', None).
    """
    p = re.sub(r"[()',\s]", "", str(raw))
    if p in WEEKDAYS:
        return "weekday", p, None
    m = re.fullmatch(r"(\d{4})-(\d{1,2})", p)
    if m:
        month = f"{m.group(1)}-{int(m.group(2)):02d}"
        return "month", month, financial_year_of_month(month)
    m = re.fullmatch(r"(\d{4})/(\d{2})", p)
    if m:
        fy = f"{m.group(1)}-{m.group(2)}"
        return "year", fy, fy
    m = re.fullmatch(r"(\d{2})(\d{2})", p)          # '2122', '2223', and '2021' meaning 2020-21
    if m and int(m.group(2)) == int(m.group(1)) + 1:
        fy = f"20{m.group(1)}-{m.group(2)}"
        return "year", fy, fy
    return "other", p, None


# --------------------------------------------------------------------------
# Dimensions, categories and measures
# --------------------------------------------------------------------------

DIMENSIONS = {
    "department type": "Department type",
    "attendance category": "Attendance category",
    "age": "Age",
    "arrival mode": "Arrival mode",
    "attendances": "Attendances",
    "attendance source": "Attendance source",
    "chief complaint": "Chief complaint",
    "discharge destination": "Discharge destination",
    "ethnicity": "Ethnicity",
    "first diagnosis code": "First diagnosis",
    "gender": "Gender",
    "imd decile": "IMD decile",
    "investigation": "Investigation",
    "reattendances": "Reattendances",
    "secondary diagnosis code": "Secondary diagnosis",
    "time of day distribution": "Time of day",
    "treatment code": "Treatment",
    "waiting times": "Waiting times",
    "acuity": "Acuity",
    "duration in department": "Duration in department",
    "hour of arrival": "Hour of arrival",
    "day of arrival": "Day of arrival",
    "total": "Total",
}


def dimension_name(raw: str) -> str:
    key = re.sub(r"^\d+\.", "", str(raw)).replace("_", " ").strip().lower()
    key = re.sub(r"\s+", " ", key)
    if key not in DIMENSIONS:
        raise ValueError(f"Unknown measure group {raw!r}")
    return DIMENSIONS[key]


def measure_name(raw: str) -> str:
    """Map every published measure label onto a short snake_case name."""
    m = str(raw).replace("_", " ").strip().lower()
    m = re.sub(r"\s+", " ", m)
    rules = [
        (r"^% (4 hours or less|less than 4 hours)", "pct_within_4h"),
        (r"^% more than 4 hours", "pct_over_4h"),
        (r"^% more than 12 hours", "pct_over_12h"),
        (r"^(4 hours or less|less than 4 hours)", "within_4h"),
        (r"^more than 4 hours", "over_4h"),
        (r"^more than 12 hours", "over_12h"),
        (r"^total attendances ex crs$", "attendances_ex_crs"),
        (r"^total admissions$", "admitted"),
        (r"^total unplanned attendances$", "unplanned_attendances"),
        (r"^(total attendances|attendances|number of a&e records)$", "attendances"),
        (r"^% of total attendances$", "pct_of_total"),
        (r"^% admitted$", "pct_admitted"),
        (r"^admitted$", "admitted"),
        (r"^% of attendances that are reattendances$", "pct_reattendances"),
        (r"^reattendances$", "reattendances"),
        (r"^population$", "population"),
        (r"^rate per 100,000 population$", "rate_per_100k"),
    ]
    for pattern, name in rules:
        if re.search(pattern, m):
            return name
    return re.sub(r"[^a-z0-9]+", "_", m).strip("_")


def category_name(raw, dimension: str):
    """Collapse case and spacing variants; give every 'total' row one label."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    c = re.sub(r"\s+", " ", str(raw)).strip()
    if c in ("", "nan"):
        return None
    low = c.lower().replace("_", " ")
    if low.endswith(" total") or low == "total" or low.startswith(dimension.lower() + " total") \
            or low.startswith("attendance category total"):
        return "Total"
    if dimension in ("IMD decile",):
        return c[0].upper() + c[1:].lower()
    if dimension == "Age":
        # '01-04' and '1-4' are the same band in different publications.
        return re.sub(r"\b0(\d)\b", r"\1", c)
    return c


# --------------------------------------------------------------------------
# National open data
# --------------------------------------------------------------------------

def normalise_open_data(path: Path, publication: str) -> pd.DataFrame:
    raw = read_csv_any_encoding(path, usecols=range(5))
    raw.columns = [c.strip().upper() for c in raw.columns]
    df = pd.DataFrame({
        "publication": publication,
        "raw_period": raw["REPORTING_PERIOD"],
        "dimension": raw["MEASURE_GROUP"].map(dimension_name),
        "raw_category": tidy_text(raw["MEASURE_TYPE"]),
        "raw_measure": tidy_text(raw["MEASURE"]),
        "value": to_number(raw["MEASURE_VALUE"]),
    })
    parsed = df["raw_period"].map(parse_period)
    df["period_type"] = parsed.str[0]
    df["period"] = parsed.str[1]
    df["financial_year"] = parsed.str[2]

    # Time-of-day rows carry the hour inside the measure label.
    tod = df["dimension"] == "Time of day"
    hour = df.loc[tod, "raw_measure"].str.extract(r"(\d{1,2}):(\d{2})\s*$")
    df.loc[tod, "category"] = hour[0].astype(float).astype("Int64").astype(str).str.zfill(2) + ":00"
    measure_text = df.loc[tod, "raw_measure"].str.replace(r":\s*\d{1,2}:\d{2}\s*$", "", regex=True)
    kind = df.loc[tod, "raw_category"].fillna("")
    label = measure_text.where(~kind.str.lower().str.startswith("number"), kind).str.lower()
    df.loc[tod, "measure"] = np.select(
        [label.str.startswith("number"), label.str.contains("ambulance"), label.str.startswith("%")],
        ["ambulance_arrivals", "pct_of_ambulance_arrivals", "pct_of_attendances"], default="total_arrivals")

    rest = ~tod
    df.loc[rest, "category"] = [category_name(c, d) for c, d in zip(df.loc[rest, "raw_category"], df.loc[rest, "dimension"])]
    df.loc[rest, "measure"] = df.loc[rest, "raw_measure"].map(measure_name)
    return df[["publication", "period_type", "period", "financial_year", "dimension",
               "category", "measure", "value"]]


NO_CATEGORY = {"Attendances", "Reattendances", "Waiting times"}


def harmonise_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Publications spell the same category in different cases
    ('Brought In By Ambulance' / 'Brought in by ambulance'). Give each
    case-insensitive variant one label: the one with the fewest capitals."""
    key = df["dimension"] + "\x00" + df["category"].fillna("").str.lower()
    variants = pd.DataFrame({"key": key, "category": df["category"]}).dropna().drop_duplicates()
    variants["caps"] = variants["category"].str[1:].str.count(r"[A-Z]")
    best = variants.sort_values(["key", "caps"]).drop_duplicates("key").set_index("key")["category"]
    out = df.copy()
    out["category"] = key.map(best).where(df["category"].notna(), None)
    return out


def _inconsistent_rate_groups(df: pd.DataFrame, tolerance: float = 0.01) -> set[tuple]:
    """(publication, dimension, period_type, period) groups whose published
    rate per 100,000 disagrees with attendances / population by more than 1%."""
    keys = ["publication", "dimension", "period_type", "period", "category"]
    sub = df[df["measure"].isin(["attendances", "population", "rate_per_100k"])]
    if sub.empty:
        return set()
    w = sub.pivot_table(index=keys, columns="measure", values="value", aggfunc="first").dropna()
    if not {"attendances", "population", "rate_per_100k"} <= set(w.columns):
        return set()
    calc = 1e5 * w["attendances"] / w["population"]
    bad = (w["rate_per_100k"] / calc - 1).abs() > tolerance
    return {k[:4] for k in w.index[bad]}


def combine_publications(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Keep, for each dimension and period, only the newest publication's rows.

    Later publications revise earlier figures and sometimes relabel categories,
    so mixing two publications inside one period would double count.

    Exception: a publication whose rates per 100,000 do not match its own
    attendances and population (the 2025-26 file's Age rows) is passed over
    for an older, consistent one. Where no consistent publication exists, the
    rate is recalculated from attendances and population. What was done is
    listed in ``result.attrs["notes"]``.
    """
    all_rows = pd.concat(frames, ignore_index=True)
    all_rows.loc[all_rows["dimension"].isin(NO_CATEGORY), "category"] = None
    all_rows = harmonise_categories(all_rows)
    order = {p: i for i, p in enumerate(sorted(all_rows["publication"].unique()))}
    all_rows["_rank"] = all_rows["publication"].map(order)

    group = ["dimension", "period_type", "period"]
    bad = _inconsistent_rate_groups(all_rows)
    is_bad = pd.Series([tuple(k) in bad for k in all_rows[["publication"] + group].itertuples(index=False)],
                       index=all_rows.index)
    good_rank = all_rows["_rank"].where(~is_bad)
    has_good = good_rank.groupby([all_rows[g] for g in group]).transform("max")
    newest = all_rows.groupby(group)["_rank"].transform("max")
    chosen = has_good.fillna(newest)
    out = all_rows[all_rows["_rank"] == chosen].copy()

    notes = []
    replaced = sorted({(p, d, per) for (p, d, pt, per) in bad}
                      - {(p, d, per) for p, d, per in out[["publication", "dimension", "period"]]
                         .drop_duplicates().itertuples(index=False)})
    if replaced:
        notes.append(f"Used an older publication for {len(replaced)} dimension-periods whose newest rates "
                     f"did not match their counts (e.g. {replaced[0][1]} {replaced[0][2]} in {replaced[0][0]}).")

    # Recalculate rates that are still inconsistent (no consistent publication).
    still = _inconsistent_rate_groups(out.drop(columns="_rank"))
    if still:
        keys = ["dimension", "period_type", "period", "category"]
        w = out.pivot_table(index=keys, columns="measure", values="value", aggfunc="first")
        rate = (1e5 * w["attendances"] / w["population"]).round(0).rename("new_rate").reset_index()
        flag = pd.Series([(p, d, pt, per) in still for p, d, pt, per in
                          out[["publication", "dimension", "period_type", "period"]].itertuples(index=False)],
                         index=out.index)
        target = flag & (out["measure"] == "rate_per_100k")
        merged = out.loc[target, keys].merge(rate, on=keys, how="left")
        out.loc[target, "value"] = merged["new_rate"].to_numpy()
        periods = sorted({(d, per) for _, d, _, per in still})
        notes.append(f"Recalculated rate per 100,000 from attendances and population for {len(periods)} "
                     f"dimension-periods where the published rate did not match ({periods[0][0]}, "
                     f"{periods[0][1]} to {periods[-1][1]}).")

    out = out.drop(columns="_rank").dropna(subset=["value"])
    out = out[out["period_type"] != "other"]
    # Diagnoses are kept by year only: monthly diagnosis rows are 75% of the
    # data but rarely needed, and dropping them keeps the database small.
    diag = out["dimension"].isin(["First diagnosis", "Secondary diagnosis"]) & (out["period_type"] == "month")
    out = out[~diag]
    out = out.drop_duplicates(["dimension", "period_type", "period", "category", "measure"], keep="last") \
             .reset_index(drop=True)
    out.attrs["notes"] = notes
    return out


# --------------------------------------------------------------------------
# Provider-level analysis (PLA) files
# --------------------------------------------------------------------------

def normalise_provider_file(path: Path) -> pd.DataFrame:
    raw = read_csv_any_encoding(path)
    raw.columns = [c.strip().upper() for c in raw.columns]
    mtype = tidy_text(raw["MEASURE_TYPE"])
    has_colon = mtype.str.contains(":", regex=False)
    dim_raw = mtype.where(~has_colon, mtype.str.split(":", n=1).str[0])
    cat_raw = tidy_text(raw["MEASURE"]).where(~has_colon, mtype.str.split(":", n=1).str[1].str.strip())

    period_type, period, fy = zip(*raw["REPORTING_PERIOD"].map(parse_period))
    df = pd.DataFrame({
        "financial_year": fy,
        "geography_level": tidy_text(raw["GEOGRAPHY_LEVEL"]),
        "org_code": tidy_text(raw["ORG_CODE"]),
        "org_name": tidy_text(raw["ORG_DESCRIPTION"]).str.title()
                    .str.replace("Nhs", "NHS", regex=False).str.replace(" And ", " and ", regex=False)
                    .str.replace(" Of ", " of ", regex=False).str.replace("'S", "'s", regex=False),
        "dimension": dim_raw.map(dimension_name),
        "raw_category": cat_raw,
        "value": to_number(raw["MEASURE_VALUE"]),
    })
    waits = df["dimension"] == "Waiting times"
    df["category"] = [category_name(c, d) for c, d in zip(df["raw_category"], df["dimension"])]
    df["measure"] = "attendances"
    df.loc[waits, "measure"] = df.loc[waits, "raw_category"].map(measure_name)
    df.loc[waits, "category"] = None
    return harmonise_categories(df.drop(columns="raw_category"))


def provider_waits(provider: pd.DataFrame) -> pd.DataFrame:
    """One row per organisation and year with 4 and 12 hour measures."""
    w = provider[provider["dimension"] == "Waiting times"]
    p = w.pivot_table(index=["financial_year", "geography_level", "org_code", "org_name"],
                      columns="measure", values="value", aggfunc="first").reset_index()
    p.columns.name = None
    if "within_4h" not in p:
        p["within_4h"] = np.nan
    p["within_4h"] = p["within_4h"].fillna(p["attendances"] - p["over_4h"])
    # NHS England's 4 hour % leaves out sites piloting new standards (CRS),
    # whose attendances have no 4 hour outcome: the base is within + over.
    p["pct_within_4h"] = 100 * p["within_4h"] / (p["within_4h"] + p["over_4h"])
    p["pct_over_12h"] = 100 * p["over_12h"] / p["attendances"]
    p["pct_admitted"] = 100 * p["admitted"] / p["attendances"]
    keep = ["financial_year", "geography_level", "org_code", "org_name", "attendances",
            "admitted", "within_4h", "over_4h", "over_12h", "pct_within_4h", "pct_over_12h", "pct_admitted"]
    p = p[keep]
    # The provider file has no England waiting-times row, and its waiting-time
    # base can differ from the published national total (2023-24 covers 23.3m
    # of 24.7m attendances). England is therefore summed from the same trust
    # rows, so trusts are always compared with a benchmark on the same basis.
    counts = ["attendances", "admitted", "within_4h", "over_4h", "over_12h"]
    eng = p[p["geography_level"] == "Provider"].groupby("financial_year")[counts].sum().reset_index()
    eng["geography_level"], eng["org_code"], eng["org_name"] = "National", "ENG", "England (sum of providers)"
    eng["pct_within_4h"] = 100 * eng["within_4h"] / (eng["within_4h"] + eng["over_4h"])
    eng["pct_over_12h"] = 100 * eng["over_12h"] / eng["attendances"]
    eng["pct_admitted"] = 100 * eng["admitted"] / eng["attendances"]
    return pd.concat([p, eng[keep]], ignore_index=True)
