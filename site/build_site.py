"""Build the static dashboard for GitHub Pages.

Usage (from the repository root):
    python site/build_site.py

Reads the app database (app/data/ae.sqlite.gz) and the pipeline outputs
(pipeline/outputs/), and writes docs/index.html: one self-contained page with
the data embedded and no external scripts, so it works on GitHub Pages.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "app"))

from ae_data import views  # noqa: E402
from ae_data.query import connect, run_select  # noqa: E402

OUT = ROOT / "docs" / "index.html"
TEMPLATE = Path(__file__).resolve().parent / "template.html"


def records(df: pd.DataFrame, digits: int = 2) -> list[dict]:
    return json.loads(df.round(digits).to_json(orient="records"))


def main() -> None:
    con = connect()
    head = views.headline(con)
    monthly = views.monthly(con)[["month", "attendances", "pct_admitted"]]

    imd = run_select("""
        SELECT period AS year, category, value FROM national
        WHERE dimension = 'IMD decile' AND measure = 'rate_per_100k' AND period_type = 'year'
          AND category IN ('Most deprived 10%', 'Least deprived 10%') ORDER BY period""", con)
    imd = imd.pivot(index="year", columns="category", values="value").reset_index()
    imd.columns = ["year", "least", "most"]

    age = run_select("""
        SELECT category AS age, value AS rate FROM national
        WHERE dimension = 'Age' AND measure = 'rate_per_100k' AND period_type = 'year'
          AND period = (SELECT MAX(period) FROM national WHERE period_type = 'year')""", con)
    order = ["0", "1-4", "5-9", "10-14", "15", "16", "17", "18", "19", "20-24", "25-29", "30-34", "35-39",
             "40-44", "45-49", "50-54", "55-59", "60-64", "65-69", "70-74", "75-79", "80-84", "85-89", "90+"]
    age = age[age["age"].isin(order)].set_index("age").reindex(order).reset_index()

    year = views.provider_years(con)[-1]
    trusts = views.league(con, year)[["org_code", "org_name", "attendances", "pct_within_4h", "pct_over_12h"]]
    missing = trusts[trusts[["pct_within_4h", "pct_over_12h"]].isna().any(axis=1)]
    if len(missing):
        print("Left out (no waiting-time figures):", ", ".join(missing["org_name"]))
    trusts = trusts.dropna(subset=["pct_within_4h", "pct_over_12h"])
    eng = run_select("SELECT pct_within_4h, pct_over_12h FROM provider_waits WHERE org_code = 'ENG' "
                     "AND financial_year = ?", con, params=(year,)).iloc[0]

    out_dir = ROOT / "pipeline" / "outputs"
    summary = json.loads((out_dir / "summary.json").read_text())
    segs = pd.read_csv(out_dir / "provider_segments.csv")
    seg_counts = segs["segment"].value_counts().to_dict()

    data = {
        "headline": records(head),
        "monthly": records(monthly),
        "imd": records(imd, 0),
        "age": records(age, 0),
        "trust_year": year,
        "trusts": records(trusts, 1),
        "england": {"pct_within_4h": round(float(eng.pct_within_4h), 1),
                    "pct_over_12h": round(float(eng.pct_over_12h), 1)},
        "autumn": {**summary, "segments": seg_counts},
    }
    html = TEMPLATE.read_text().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html)
    print(f"Wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
