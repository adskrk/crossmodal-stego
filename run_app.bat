@echo off
rem Starts the browser demo (hide / extract). Close this window to stop it.
cd /d "%~dp0"
set PYTHONUTF8=1
if not exist .venv\Scripts\python.exe (
  echo Run setup_windows.bat first.
  pause
  exit /b 1
)
echo Starting the demo with GPT-2. Your browser opens when it is ready.
echo Keep this window open during the demo; close it to stop.
.venv\Scripts\python.exe scripts\app.py --model gpt2 --open
pause
