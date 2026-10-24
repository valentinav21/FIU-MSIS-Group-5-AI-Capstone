#!/usr/bin/env bash
# FORTPRO AI Planner - start on macOS or Linux:  bash start_mac_linux.sh
set -e
cd "$(dirname "$0")"
PY=$(command -v python3 || command -v python || true)
if [ -z "$PY" ]; then echo "Python 3.10 or newer is needed: https://www.python.org/downloads/"; exit 1; fi
"$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)' || { echo "Python 3.10 or newer is needed."; exit 1; }
if [ ! -x ".venv/bin/python" ]; then
  echo "First run: setting up a private Python environment (a few minutes, once)."
  "$PY" -m venv .venv
  .venv/bin/python -m pip install --upgrade pip
  .venv/bin/python -m pip install -r requirements.txt
fi
if [ ! -f "data/sample/INV1.csv" ]; then
  echo "Building the synthetic sample data..."
  .venv/bin/python tools/generate_synthetic.py
fi
echo "Starting FORTPRO AI Planner at http://localhost:8501 (Ctrl+C to stop)"
.venv/bin/python -m streamlit run app/Home.py --server.headless false --browser.gatherUsageStats false
