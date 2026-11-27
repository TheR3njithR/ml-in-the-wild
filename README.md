# ML in the Wild

Code and synthetic datasets for **ML in the Wild: 100 Production-Ready Case Studies**, a YouTube series on [TheDataScientist](https://www.youtube.com/@TheDataScientist). Each episode solves one real business decision with machine learning — Data → Model → Decision → Action → Feedback — and ships a runnable Python project.

## Episodes

- [Episode 1 — Your KPI Dashboard Is Lying: Catch It With Isolation Forest](ep01-kpi-dashboard-lying/)
- [Episode 2 — Predict a Power BI Refresh Failure Before It Happens](ep02-power-bi-refresh-failure/)
- [Episode 3 — Detect a Broken Data Pipeline Without Reading the Logs](ep03-broken-data-pipeline/)

## How to run an episode

In the episode folder:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
./run_all.sh
```

Chart PNGs and generated output files are not committed to this repository. `./run_all.sh` rebuilds them, including every number and chart shown in the video.

All datasets are synthetic. See each episode README for the story, the files, and the expected results.
