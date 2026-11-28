# Student challenge

The packed code already catches the April 2026 beverages shift. Now try this, without looking at `incident_log.csv` while you work.

1. Add a **concept-drift** check that does not wait for a week of labels: for example, a rolling comparison of predicted-versus-a-proxy (promo-adjusted lag_7) rather than actuals. Does it fire earlier or later than the feature PSI on this dataset?
2. Rebuild the monitor with a **28-day** window and five PSI bins. How many extra days of delay do you buy, and do the six pre-shift amber looks disappear?
3. The hook over-order is almost thirty thousand units in ten days. Write a one-page note that a planning manager could read at 06:30, with a single recommended action and the top two features that moved.

Paste your numbers into a short write-up. Every figure should come from a script, not from memory.
