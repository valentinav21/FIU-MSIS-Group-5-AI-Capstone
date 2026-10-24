@echo off
REM FORTPRO AI Planner - double-click to start (Windows)
cd /d "%~dp0"
set PY=
where py >nul 2>nul && set PY=py -3
if "%PY%"=="" where python >nul 2>nul && set PY=python
if "%PY%"=="" (
  echo Python 3.10 or newer is needed. Install it from https://www.python.org/downloads/ and tick "Add python.exe to PATH".
  pause
  exit /b 1
)
if not exist ".venv\Scripts\python.exe" (
  echo First run: setting up a private Python environment. This takes a few minutes, once.
  %PY% -m venv .venv || (echo Could not create the environment. & pause & exit /b 1)
  ".venv\Scripts\python.exe" -m pip install --upgrade pip
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt || (echo Package install failed. Check the internet connection and try again. & pause & exit /b 1)
)
if not exist "data\sample\INV1.csv" (
  echo Building the synthetic sample data...
  ".venv\Scripts\python.exe" tools\generate_synthetic.py
)
echo Starting FORTPRO AI Planner. Your browser will open at http://localhost:8501
echo Keep this window open while you use the app. Close it to stop.
".venv\Scripts\python.exe" -m streamlit run app\Home.py --server.headless false --browser.gatherUsageStats false
pause
