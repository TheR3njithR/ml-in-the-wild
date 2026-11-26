# Data dictionary (SYNTHETIC data, fictional company "QuickBasket")

## data/kpi_daily.csv: one row per city per day, 2024-09-01 to 2026-10-31 (3,955 rows)
| Column | Type | Meaning |
|---|---|---|
| date | date (YYYY-MM-DD) | Business day (IST) |
| city | text | Bengaluru, Delhi, Hyderabad, Mumbai, Pune |
| orders | integer | Completed orders that day |
| sessions | integer | App + web visits, counted by the analytics tracking tag |
| revenue_inr | integer | Revenue in rupees, GST-inclusive item prices |
| refunds | integer | Orders refunded that day |
| delivery_min | decimal | Average delivery time in minutes |

Built in: +22% a year growth, a weekly rhythm (weekends higher), mild seasonality, random day-to-day noise.

## data/event_calendar.csv: real business events (the numbers really moved)
| Column | Meaning |
|---|---|
| date | Day of the event |
| event | Diwali, New Year's Eve, Republic Day Sale, Holi, Big Summer Sale, Independence Day Sale, Ganesh Chaturthi, Dussehra, Heavy rain |
| cities | "All" or a comma-separated list |
| type | sale/festival or weather |

## data/incident_log.csv: ground truth for grading ONLY (the model never sees it)
| Column | Meaning |
|---|---|
| date, city | The broken city-day |
| incident | DUPLICATE_LOAD, TRACKING_TAG_BROKEN, TRACKING_DOUBLE_FIRE, PARTIAL_LOAD, STALE_COPY, TAX_DOUBLE_COUNTED, REFUND_FEED_MISSING |

| Incident | What broke | What it does to the dashboard |
|---|---|---|
| DUPLICATE_LOAD | Orders file loaded twice (or one batch twice) | Orders, revenue, refunds x1.5 to x2; sessions unchanged |
| TRACKING_TAG_BROKEN | Analytics tag stopped firing | Sessions x0.38 to x0.45; orders unchanged |
| TRACKING_DOUBLE_FIRE | Tag fires twice on some pages | Sessions x1.55 to x1.65 |
| PARTIAL_LOAD | Nightly job ran before all orders landed | Orders, revenue, refunds x0.62 to x0.83 |
| STALE_COPY | Pipeline failed silently, yesterday copied forward | Every number identical to the day before |
| TAX_DOUBLE_COUNTED | GST added on top of GST-inclusive prices | Revenue x1.18 |
| REFUND_FEED_MISSING | Refunds table did not arrive, join filled zeros | Refunds = 0 |

## Features built by kpi_guard.py (each against the city's same weekday over the previous 8 weeks)
| Feature | Formula | Catches |
|---|---|---|
| conv_dev | log(conversion / usual conversion), conversion = orders / sessions | duplicate loads, tracking problems, partial loads |
| aov_dev | log(order value / usual), order value = revenue / orders | tax and currency errors |
| refund_dev | log of refund rate vs usual | missing refunds feed |
| frozen | share of orders, sessions, revenue identical to yesterday | stale copies |
