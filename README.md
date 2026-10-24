# NHS A&E Analytics

Two projects built on NHS England's public A&E data.

| Folder | What it is |
| --- | --- |
| [`app/`](app/) | **A&E Data Explorer**: a Streamlit app to explore six years of ECDS data (April 2020 to March 2026, England and trusts), with plain-English questions answered by Claude through read-only SQL. |
| [`pipeline/`](pipeline/) | **A&E Pressure Monitor**: a tested Python pipeline for July to September 2026: 4 and 12 hour performance, a seasonal demand forecast, k-means segmentation of trusts and a deprivation analysis. |

Data source: NHS England, [Hospital Accident & Emergency Activity](https://digital.nhs.uk/data-and-information/publications/statistical/hospital-accident--emergency-activity)
and [A&E Attendances and Emergency Admissions](https://www.england.nhs.uk/statistics/statistical-work-areas/ae-waiting-times-and-activity/).
Open Government Licence.
