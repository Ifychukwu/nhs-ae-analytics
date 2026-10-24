# A&E Data Explorer

A Streamlit app for exploring six years of NHS England emergency department data, with a
question box that answers plain-English questions by writing and running SQL against the data.

**Data source:** NHS England, [Hospital Accident & Emergency Activity](https://digital.nhs.uk/data-and-information/publications/statistical/hospital-accident--emergency-activity)
(Emergency Care Data Set, ECDS). Open Government Licence.

## What it does

| Tab | What you can do |
| --- | --- |
| Overview | Latest-year headline figures, monthly attendances April 2020 to March 2026, 4 hour and 12 hour trends |
| Explore | Break attendances down by any recorded characteristic (chief complaint, diagnosis, age, deprivation, ethnicity, arrival mode and more), filter, see trends by year or month, download CSV |
| Trusts | Compare any trust's 4 and 12 hour performance and case mix with England; league table of all trusts with a major A&E |
| Ask the data | Ask a question in plain English. Claude writes SQL, the app runs it read-only, and the answer is written from the results with every query shown. Without an API key, run ready-made questions or your own SQL |
| About the data | Sources, how the files were combined, checks, caveats and a glossary |

## Data

| Files | Coverage |
| --- | --- |
| ECDS open data, 2022-23 to 2025-26 publications | England, monthly April 2020 to March 2026, by 18 characteristics |
| Provider level analysis, 2023-24 and 2024-25 | 201 trusts and other providers, plus regions and England |

`prepare_data.py` combines them into one SQLite database (`data/ae.sqlite`, shipped compressed as
`data/ae.sqlite.gz` and unpacked on first run).

**How the files are combined.** Each yearly file repeats earlier years, sometimes revised and with
different labels (`IMD_DECILE` / `Imd_Decile` / `IMD Decile`; `2122` / `2021/22`; `2025-4`). Every label
is mapped to one vocabulary, and for each characteristic and period the newest publication wins, so
figures are never double counted.

**Checks run before the database is written** (`ae_data/checks.py`): no duplicate rows; monthly
attendances add up to each year's published total; published 12 hour counts and percentages agree;
percentages lie between 0 and 100; trust totals add up to England's. The build stops if any check fails.

## Run it locally

```bash
pip install -r requirements.txt
python prepare_data.py          # only if you change the raw files; needs the CSVs in raw/
streamlit run app.py
```

To enable plain-English questions, add an Anthropic API key:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

or create `.streamlit/secrets.toml` from `.streamlit/secrets.toml.example`. On Streamlit Community
Cloud, paste the key into the app's Secrets settings.

## How the question box works

`ae_data/assistant.py` gives Claude the database schema and the real category labels, plus one tool,
`run_sql`. Claude writes SQL; the app runs it on a read-only connection that accepts only a single
SELECT, caps results at 5,000 rows and stops after 10 seconds (`ae_data/query.py`). Claude must take
every number from a query result, and each query is shown under the answer so it can be checked.

## Tests

```bash
python tests/test_app.py
```

Covers the SQL safety rules, published totals, every view the app uses, the assistant loop (with a
fake Claude client), and a full run of `app.py` with and without an API key.

## Things to know

- ECDS figures are published as experimental and come from provider submissions.
- The 4 hour percentage excludes sites piloting new standards; the 12 hour percentage uses all attendances.
- Trust waiting times for 2023-24 cover 94.3% of England's attendances (99.7% in 2024-25). England's
  trust-level benchmark is built from the same trust rows, so compare each trust with England in the
  same year rather than across years.
- Counts of 1 to 7 are suppressed in the trust files and are not stored.
- Diagnoses are stored by year only, to keep the database small.
- Ethnicity is not stated or unknown for a large share of attendances.
