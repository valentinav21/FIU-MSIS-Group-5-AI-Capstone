@echo off
REM Backtest for the Fabric scorecard: writes exports\ai_forecast.csv and exports\ai_forecast_forward.csv
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run start_windows.bat once first, to set up the environment.
  pause
  exit /b 1
)
set /p DATA=Folder with the cleaned CSV files (paste the path, then Enter): 
set DATA=%DATA:"=%
".venv\Scripts\python.exe" tools\backtest_export.py --data "%DATA%"
echo.
echo Files are in the exports folder. Send ai_forecast.csv and ai_forecast_forward.csv to the data engineer.
pause
