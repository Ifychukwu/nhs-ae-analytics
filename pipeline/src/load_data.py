"""Loaders for the public NHS England A&E files used in this project.

Each function reads one published file and returns a tidy pandas DataFrame.
Nothing here changes the published numbers; cleaning is limited to renaming
columns, parsing dates and turning suppressed cells ("*") into missing values.

Sources (all published by NHS England, Open Government Licence):
  * Monthly A&E Attendances and Emergency Admissions (MSitAE), provider CSVs
  * ECDS Performance workbooks (4 hour and 12 hour in department, by provider)
  * Hospital A&E Activity 2022-23: summary tables and ECDS open data CSV
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

# --------------------------------------------------------------------------
# MSitAE monthly provider CSVs
# --------------------------------------------------------------------------

MSITAE_COLUMNS = {
    "Period": "period",
    "Org Code": "org_code",
    "Parent Org": "region",
    "Org name": "org_name",
    "A&E attendances Type 1": "att_t1",
    "A&E attendances Type 2": "att_t2",
    "A&E attendances Other A&E Department": "att_other",
    "A&E attendances Booked Appointments Type 1": "att_booked_t1",
    "A&E attendances Booked Appointments Type 2": "att_booked_t2",
    "A&E attendances Booked Appointments Other Department": "att_booked_other",
    "Attendances over 4hrs Type 1": "over4_t1",
    "Attendances over 4hrs Type 2": "over4_t2",
    "Attendances over 4hrs Other Department": "over4_other",
    "Attendances over 4hrs Booked Appointments Type 1": "over4_booked_t1",
    "Attendances over 4hrs Booked Appointments Type 2": "over4_booked_t2",
    "Attendances over 4hrs Booked Appointments Other Department": "over4_booked_other",
    "Patients who have waited 4-12 hs from DTA to admission": "dta_4to12h",
    "Patients who have waited 12+ hrs from DTA to admission": "dta_12h_plus",
    "Emergency admissions via A&E - Type 1": "adm_t1",
    "Emergency admissions via A&E - Type 2": "adm_t2",
    "Emergency admissions via A&E - Other A&E department": "adm_other",
    "Other emergency admissions": "adm_non_ae",
}

COUNT_COLUMNS = [c for c in MSITAE_COLUMNS.values()
                 if c not in ("period", "org_code", "region", "org_name")]


def _period_to_date(period: str) -> pd.Timestamp:
    """'MSitAE-JULY-2026' -> Timestamp('2026-07-01')."""
    match = re.search(r"([A-Z]+)-(\d{4})$", period.upper())
    if not match:
        raise ValueError(f"Unrecognised period label: {period!r}")
    return pd.to_datetime(f"1 {match.group(1)} {match.group(2)}", format="%d %B %Y")


def load_msitae_csv(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Read one MSitAE monthly CSV.

    Returns (providers, published_total) so that the published England total
    row can be used later as a data test against the sum of providers.
    """
    raw = pd.read_csv(path)
    missing = set(MSITAE_COLUMNS) - set(raw.columns)
    if missing:
        raise ValueError(f"{path.name}: missing expected columns {sorted(missing)}")

    df = raw.rename(columns=MSITAE_COLUMNS)[list(MSITAE_COLUMNS.values())]
    is_total = df["org_code"].astype(str).str.upper() == "TOTAL"
    published_total = df.loc[is_total, COUNT_COLUMNS].iloc[0].astype(int)

    df = df.loc[~is_total].copy()
    df["month"] = _period_to_date(df["period"].iloc[0])
    df["region"] = df["region"].str.strip().str.replace("NHS ENGLAND ", "", regex=False).str.title()
    df["org_name"] = df["org_name"].str.strip()
    df[COUNT_COLUMNS] = df[COUNT_COLUMNS].astype(int)
    published_total.name = df["month"].iloc[0]
    return df.drop(columns="period"), published_total


def load_all_msitae(paths: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    frames, totals = [], []
    for p in sorted(paths):
        df, tot = load_msitae_csv(p)
        frames.append(df)
        totals.append(tot)
    providers = pd.concat(frames, ignore_index=True).sort_values(["month", "org_code"])
    published_totals = pd.DataFrame(totals)
    published_totals.index.name = "month"
    return providers.reset_index(drop=True), published_totals.sort_index()


# --------------------------------------------------------------------------
# ECDS Performance workbooks
# --------------------------------------------------------------------------

def _num(v):
    """Published cells can be numbers, '*' (suppressed 1-7) or blank."""
    if v is None or (isinstance(v, str) and v.strip() in {"", "*", "-"}):
        return np.nan
    return float(v)


def _sheet_rows(path: Path, sheet: str) -> list[tuple]:
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = list(wb[sheet].iter_rows(values_only=True))
    wb.close()
    return rows


def _period_from_sheet(rows: list[tuple]) -> pd.Timestamp:
    for r in rows[:12]:
        text = " ".join(str(v) for v in r if v is not None)
        m = re.search(r"Period:\s*([A-Za-z]+ \d{4})", text)
        if m:
            return pd.to_datetime("1 " + m.group(1), format="%d %B %Y")
    raise ValueError("Could not find 'Period:' line in sheet header")


def load_ecds_4hr(path: Path) -> pd.DataFrame:
    """Provider-level Type 1 & 2 four-hour performance by discharge group.

    Provider rows are the ones whose 'Organisation' cell reads 'Total'
    (the trust total above its individual sites).
    """
    rows = _sheet_rows(path, "4 Hour Performance")
    month = _period_from_sheet(rows)
    groups = ["admitted", "non_admitted", "other", "unknown", "total"]
    out = []
    for r in rows[15:]:
        if r[1] is None or r[2] != "Total":
            continue
        vals = [_num(v) for v in r[4:19]]
        rec = {"month": month, "trust_name": str(r[1]).strip(), "org_code": str(r[3]).strip()}
        for i, g in enumerate(groups):
            rec[f"ecds_att_{g}"] = vals[i]
            rec[f"ecds_over4_{g}"] = vals[5 + i]
        out.append(rec)
    df = pd.DataFrame(out)
    for g in groups:
        df[f"ecds_pct4_{g}"] = 1 - df[f"ecds_over4_{g}"] / df[f"ecds_att_{g}"]
    return df


def load_ecds_12hr(path: Path) -> pd.DataFrame:
    """Provider-level share of Type 1 & 2 attendances spending 12+ hours in A&E
    (arrival to departure). This measure exists only in ECDS, not in MSitAE."""
    rows = _sheet_rows(path, "12 Hour Performance")
    month = _period_from_sheet(rows)
    out = []
    for r in rows[15:]:
        if r[1] is None or r[2] != "Total":
            continue
        out.append({
            "month": month,
            "org_code": str(r[3]).strip(),
            "ecds_att_t12": _num(r[4]),
            "ecds_over12_t12": _num(r[5]),
        })
    df = pd.DataFrame(out)
    df["ecds_pct12_t12"] = df["ecds_over12_t12"] / df["ecds_att_t12"]
    return df


def load_ecds_icb_12hr(path: Path) -> pd.DataFrame:
    """ICB-level (system) 12 hour share, used for the Greater Manchester view."""
    rows = _sheet_rows(path, "12 Hour Performance")
    month = _period_from_sheet(rows)
    out = []
    for r in rows[15:]:
        name = str(r[2]) if r[2] else ""
        if name.startswith("NHS ") and name.endswith("ICB"):
            out.append({"month": month, "icb": name, "region": r[1],
                        "att": _num(r[4]), "over12": _num(r[5])})
    df = pd.DataFrame(out)
    df["pct12"] = df["over12"] / df["att"]
    return df


# --------------------------------------------------------------------------
# Hospital A&E Activity 2022-23 publication
# --------------------------------------------------------------------------

MONTHS = ["April", "May", "June", "July", "August", "September", "October",
          "November", "December", "January", "February", "March"]


def load_monthly_history(path: Path) -> pd.DataFrame:
    """National monthly unplanned A&E attendances, April 2018 to March 2023
    (Table 9 of the 2022-23 publication; planned attendances excluded)."""
    rows = _sheet_rows(path, "Table 9")
    header = next(r for r in rows if r[0] is None and r[1] and str(r[1]).startswith("2018"))
    years = [str(v) for v in header[1:] if v]
    out = []
    for r in rows:
        if r[0] in MONTHS:
            for k, fy in enumerate(years):
                start_year = int(fy[:4])
                m_idx = MONTHS.index(r[0])
                cal_year = start_year if m_idx < 9 else start_year + 1
                cal_month = (m_idx + 3) % 12 + 1
                out.append({
                    "month": pd.Timestamp(cal_year, cal_month, 1),
                    "emergency_admissions": float(r[1 + 3 * k]),
                    "attendances": float(r[2 + 3 * k]),
                })
    return pd.DataFrame(out).sort_values("month").reset_index(drop=True)


def load_annual_history(path: Path) -> pd.DataFrame:
    """Annual MSitAE attendances 2011-12 to 2022-23 (Table 1)."""
    rows = _sheet_rows(path, "Table 1")
    years_row = next(r for r in rows if r[1] == "2011-12")
    vals_row = next(r for r in rows if r[0] == "MSitAE")
    years = [v for v in years_row[1:] if v]
    vals = [float(v) for v in vals_row[1:1 + len(years)]]
    return pd.DataFrame({"financial_year": years, "attendances": vals})


def load_open_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["MEASURE_TYPE"] = df["MEASURE_TYPE"].astype(str).str.strip()
    df.columns = [c.lower() for c in df.columns]
    return df
