@echo off
rem Runs the automated tests (about 3-5 minutes) with the small offline model.
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist .venv\Scripts\python.exe (
  echo Run setup_windows.bat first.
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m pytest -q
echo.
echo If the last line says "passed" with no "failed", everything works.
pause
