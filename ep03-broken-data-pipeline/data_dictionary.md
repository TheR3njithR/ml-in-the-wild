# Data dictionary (synthetic)

All timestamps are calendar dates in the QuickBasket fiction (1 Dec 2024–31 Mar 2026). Nothing in these files is a real company's production log.

## `pipeline_daily.csv` — one row per pipeline per day (the model-facing table)

| Column | Meaning |
|---|---|
| date | Calendar date of the scheduled daily job. |
| pipeline | One of eight pipelines: orders_facts, inventory_snapshot, delivery_events, customer_signups, payments_ledger, marketing_spend, warehouse_picks, support_tickets. |
| pipeline_id | Integer code for the pipeline, 0–7. |
| dow | Monday = 0 … Sunday = 6. |
| week_of_month | 1–5. |
| month | 1–12. |
| is_weekend | 1 if Saturday or Sunday. |
| is_month_end | 1 on the last two calendar days of the month. |
| is_holiday | 1 on the generated holiday calendar (see `calendar.csv`). |
| is_promo | 1 on a generated promotion window. |
| status | SUCCESS or FAILED. Silent breaks are always SUCCESS. |
| rows | Raw row count written by the job. |
| usable_rows | Rows that still have the grain filled in. This is the volume we monitor. |
| null_rate | Fraction of rows that do not count as usable. |
| sources_expected | How many source files or feeds this pipeline usually waits for. |
| sources_arrived | How many actually arrived before the job ran. |
| source_fill | arrived / expected. |
| runtime_s | Job runtime in seconds (not used as a leak of the break type). |
| lag_7, lag_14 | This pipeline's usable_rows 7 and 14 days earlier. |
| same_dow_median_8w | Median usable_rows of the last 8 same weekdays (d-7 … d-56). |
| raw_med7, raw_same_dow, usable_med7 | Helpers for the naive baseline rule only. |

## `calendar.csv`

| Column | Meaning |
|---|---|
| date | Calendar date. |
| is_holiday, is_promo | The same flags as in `pipeline_daily.csv`. |

## `hook_timeline.csv`

Illustrative minute-by-minute cumulative rows for Orders on 16 Mar 2026, used in the hook chart. Label it SYNTHETIC / ILLUSTRATIVE if you show it.

## `incident_log.csv` and `incident_labels.csv` — answer key, not a model input

| Column | Meaning |
|---|---|
| date, pipeline | The job that broke. |
| incident_type | SOURCE_SILENT, FILTER_REGRESSION, LATE_ARRIVAL, DUP_LOAD, NULL_SURGE, TRUNCATED_EXTRACT, JOIN_COLLAPSE, or HARD_FAIL. |
| is_silent, is_hard_fail | Labels. Silent breaks finished with status SUCCESS. |
| healthy_rows | Hidden generator value: what usable volume would have been without the break. **Do not use this as a feature.** |

The detector in `pipeline_guard.py` never reads these two files at prediction time.
