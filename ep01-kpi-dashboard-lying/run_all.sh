#!/bin/bash
# Rebuild everything for Episode 1 from scratch: synthetic data -> baseline -> model -> decision -> feedback -> charts
set -e; cd "$(dirname "$0")"; PY=${PY:-python}
$PY make_data.py
$PY 01_baseline_rules.py
$PY 02_isolation_forest.py
$PY 03_decide_and_act.py 2026-01-21
$PY 04_feedback_loop.py
$PY 05_charts.py
