# NHS A&E Pressure Monitor

A reproducible Python pipeline that turns NHS England's public A&E data into
operational and planning insight: four hour performance, 12 hour waits,
a seasonal demand forecast, a segmentation of trusts by pressure profile,
and an inequalities view by deprivation.

## Questions it answers

1. How is A&E performing nationally, and how did it change between July and September 2026?
2. How much demand should the NHS expect this winter, and how well does a model trained on pre-2023 data hold up?
3. Which trusts share a similar pressure profile, and which are deteriorating fastest?
4. How are the Greater Manchester trusts doing?
5. Who uses A&E most, by deprivation?

## Data (all public, Open Government Licence)

Source: NHS England, [Hospital Accident & Emergency Activity](https://digital.nhs.uk/data-and-information/publications/statistical/hospital-accident--emergency-activity) and [A&E Attendances and Emergency Admissions](https://www.england.nhs.uk/statistics/statistical-work-areas/ae-waiting-times-and-activity/).


| File | Publisher | Used for |
| --- | --- | --- |
| Monthly A&E Attendances and Emergency Admissions (MSitAE), provider CSVs, Jul to Sep 2026 | NHS England | Provider and national performance |
| ECDS Performance workbooks (July finalised, September provisional) | NHS England | Admitted vs non-admitted 4 hour, 12 hours in A&E |
| Hospital A&E Activity 2022-23: summary tables | NHS England | Monthly history Apr 2018 to Mar 2023, annual totals |
| Hospital A&E Activity 2022-23: ECDS open data CSV | NHS England | Deprivation, age, ethnicity recording |

Put the files in `data/` with their published names.

## Run it

```bash
pip install pandas numpy scikit-learn openpyxl
python run_pipeline.py
```

Outputs are written to `outputs/` as CSVs plus `summary.json`.

## How it works

- `src/load_data.py` parses each published file into a tidy table without changing any published value.
- `src/tests_runner.py` runs data tests before any analysis: provider rows must sum exactly to NHS England's published England total, counts must be non-negative and unique per provider and month, 4 hour breaches must not exceed attendances, and monthly history must add up to the published annual totals. The pipeline stops if any test fails.
- `src/analysis.py`
  - **Metrics:** 4 hour performance on NHS England's definition (booked appointments included), Type 1 performance, 12+ hour decision-to-admit waits per 100 admissions.
  - **Forecast:** OLS on log attendances per day with a linear trend and month effects, fitted on April 2018 to March 2023 with lockdown months (March 2020 to May 2021) excluded. Backtested on July to September 2026, then re-anchored to those months to forecast October 2026 to March 2027 with an approximate 95% interval.
  - **Segmentation:** k-means on five standardised pressure measures for trusts with a Type 1 department; k chosen by silhouette score; segments ordered by a composite pressure score so labels are reproducible.
  - **Inequalities:** attendance rates per 100,000 by IMD decile, and the share of attendances with ethnicity not recorded.

## Key results (September 2026)

- 74.4% of patients were seen within 4 hours (75.4% in July); 60.6% in major (Type 1) departments.
- 50,582 patients waited over 12 hours from decision to admit, up from 47,438 in July.
- A model trained on 2018-23 under-forecast July to September 2026 by 11.8% on average: demand has stepped up since the last year in this file.
- Three pressure profiles emerge (silhouette 0.22, so a screening tool rather than natural groups).
- People in the most deprived tenth of areas attended A&E 1.93 times as often as those in the least deprived; ethnicity was not recorded for 17% of attendances.

## Limitations and next steps

- The history file ends in March 2023. Adding the monthly MSitAE files for April 2023 to June 2026 would remove the gap and let the model learn the recent level and trend directly.
- ECDS measures are experimental and come straight from provider submissions.
- Next: add the missing months, compare against a SARIMA or gradient-boosted model, and forecast at trust level.
