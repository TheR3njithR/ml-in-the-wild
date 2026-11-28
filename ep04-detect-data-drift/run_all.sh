#!/bin/bash
# Rebuild everything for Episode 4 from scratch.
set -e; cd "$(dirname "$0")"; PY=${PY:-python}
$PY make_data.py
$PY 02_model.py
$PY 01_baseline.py
$PY 03_detect.py
$PY 04_decide_and_act.py 2026-04-20
$PY 05_feedback_loop.py
$PY 06_charts.py
