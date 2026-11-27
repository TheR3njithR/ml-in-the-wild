#!/bin/bash
# Rebuild everything for Episode 3 from scratch: synthetic data -> baseline -> model -> decision -> feedback -> charts
set -e; cd "$(dirname "$0")"; PY=${PY:-python}
$PY make_data.py
$PY 01_baseline_rules.py
$PY 02_detect.py
$PY 03_decide_and_act.py 2026-03-16
$PY 04_feedback_loop.py
$PY 05_charts.py
