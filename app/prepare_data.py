"""Build the app's database from NHS England's Hospital A&E Activity files.

Usage:
    python prepare_data.py            # reads raw/, writes data/ae.sqlite

Source: https://digital.nhs.uk/data-and-information/publications/statistical/hospital-accident--emergency-activity
Download, for each year, the "ECDS open data" CSV and the "provider level
analysis" CSV into raw/. File names are matched by year code (2223, 2324 ...).
"""

from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ae_data import normalise as N  # noqa: E402
from ae_data.checks import run_checks, waiting_time_coverage  # noqa: E402
from ae_data.glossary import DIMENSION_NOTES, MEASURE_NOTES  # noqa: E402

RAW, OUT = ROOT / "raw", ROOT / "data" / "ae.sqlite"


def publication_of(path: Path) -> str | None:
    m = re.search(r"(?:AE_?)(\d{2})(\d{2})", path.name)
    return f"20{m.group(1)}-{m.group(2)}" if m else None


def find_files() -> tuple[dict[str, Path], dict[str, Path]]:
    open_data, provider = {}, {}
    for p in sorted(RAW.glob("*.csv")):
        pub = publication_of(p)
        if not pub:
            continue
        name = p.name.lower()
        if "pla" in name:
            provider.setdefault(pub, p)
        elif "open" in name:
            open_data.setdefault(pub, p)        # duplicate downloads are ignored
    return open_data, provider


def main() -> None:
    open_files, provider_files = find_files()
    if not open_files:
        sys.exit("No ECDS open data files found in raw/")
    print("Open data:", {k: v.name for k, v in open_files.items()})
    print("Provider level:", {k: v.name for k, v in provider_files.items()})

    national = N.combine_publications([N.normalise_open_data(p, pub) for pub, p in open_files.items()])
    provider = pd.concat([N.normalise_provider_file(p) for p in provider_files.values()], ignore_index=True) \
        if provider_files else pd.DataFrame()
    waits = N.provider_waits(provider) if len(provider) else pd.DataFrame()

    failures = run_checks(national, provider, waits)
    if failures:
        print("Data checks failed:\n  " + "\n  ".join(failures))
        sys.exit(1)
    for note in national.attrs.get("notes", []):
        print("Fixed:", note)
    print("All data checks passed.")
    for note in waiting_time_coverage(waits, national):
        print("Note:", note)

    glossary = pd.DataFrame(
        [{"kind": "dimension", "name": k, "description": v} for k, v in DIMENSION_NOTES.items()]
        + [{"kind": "measure", "name": k, "description": v} for k, v in MEASURE_NOTES.items()])

    OUT.parent.mkdir(exist_ok=True)
    if OUT.exists():
        OUT.unlink()
    with sqlite3.connect(OUT) as con:
        national.to_sql("national", con, index=False)
        orgs = provider[["org_code", "org_name", "geography_level"]].drop_duplicates("org_code")
        orgs.to_sql("organisations", con, index=False)
        # Counts of 1 to 7 are suppressed ("*") by NHS England; they carry no
        # value, so they are not stored.
        provider.dropna(subset=["value"]).drop(columns=["org_name", "geography_level"]) \
            .to_sql("provider", con, index=False)
        waits.to_sql("provider_waits", con, index=False)
        glossary.to_sql("glossary", con, index=False)
        con.executescript("""
            CREATE INDEX ix_nat ON national(dimension, measure, period_type, period);
            CREATE INDEX ix_prov ON provider(org_code, financial_year, dimension);
            CREATE VIEW monthly_attendances AS
                SELECT period AS month, financial_year,
                       MAX(CASE WHEN measure = 'attendances' THEN value END) AS attendances,
                       MAX(CASE WHEN measure = 'admitted' THEN value END) AS admitted,
                       MAX(CASE WHEN measure = 'pct_admitted' THEN value END) AS pct_admitted
                FROM national
                WHERE dimension = 'Attendances' AND period_type = 'month'
                GROUP BY period, financial_year;
            CREATE VIEW annual_waits AS
                SELECT period AS financial_year,
                       MAX(CASE WHEN measure = 'attendances' THEN value END) AS attendances,
                       MAX(CASE WHEN measure = 'pct_within_4h' THEN value END) AS pct_within_4h,
                       MAX(CASE WHEN measure = 'pct_over_12h' THEN value END) AS pct_over_12h,
                       MAX(CASE WHEN measure = 'over_12h' THEN value END) AS over_12h
                FROM national
                WHERE dimension = 'Waiting times' AND period_type = 'year'
                GROUP BY period;
        """)
        con.execute("VACUUM")
    size = OUT.stat().st_size / 1e6
    print(f"Wrote {OUT.relative_to(ROOT)} ({size:.1f} MB): {len(national):,} national rows, "
          f"{provider['value'].notna().sum():,} provider rows, {len(waits):,} provider-year waits")


if __name__ == "__main__":
    main()
