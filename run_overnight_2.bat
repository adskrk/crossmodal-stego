@echo off
rem Final text channel on GPT-2 (line-break fix, 16 texts per prompt and budget) and
rem the selection experiments on it. About 2.5 hours. Keep the laptop plugged in and awake.
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PY=.venv\Scripts\python.exe
if not exist %PY% (
  echo Run setup_windows.bat first.
  pause
  exit /b 1
)
if not exist results_gpt2_nlfix mkdir results_gpt2_nlfix
copy /y results_gpt2\image_measurements* results_gpt2_nlfix\ >nul
echo Running GPT-2 experiments (about 2.5 hours). Progress is written to results_gpt2_nlfix\overnight3.log
echo Started %date% %time% > results_gpt2_nlfix\overnight3.log

echo [1/4] Final text channel: 384 messages with the line-break fix (about 2 hours)
echo ==== [1/4] Final text channel: 384 messages with the line-break fix (about 2 hours) ==== >> results_gpt2_nlfix\overnight3.log
%PY% scripts\exp_text.py --model gpt2 --variant nlfix --cs 0,0.1 --per-cell 16 --out results_gpt2_nlfix >> results_gpt2_nlfix\overnight3.log 2>&1 || goto :fail

echo [2/4] Style detector on the new texts (1 min)
echo ==== [2/4] Style detector on the new texts (1 min) ==== >> results_gpt2_nlfix\overnight3.log
%PY% scripts\add_style_detector.py --results results_gpt2_nlfix --variant nlfix >> results_gpt2_nlfix\overnight3.log 2>&1 || goto :fail

echo [3/4] Selection experiments, one warden and several (about 20 min)
echo ==== [3/4] Selection experiments, one warden and several (about 20 min) ==== >> results_gpt2_nlfix\overnight3.log
%PY% scripts\exp_selection.py --results results_gpt2_nlfix --text-tag text_nlfix_ --ensemble >> results_gpt2_nlfix\overnight3.log 2>&1 || goto :fail

echo [4/4] Rebuilding docs\RESULTS.md and the figures
echo ==== [4/4] Rebuilding docs\RESULTS.md and the figures ==== >> results_gpt2_nlfix\overnight3.log
%PY% scripts\make_report.py >> results_gpt2_nlfix\overnight3.log 2>&1 || goto :fail

echo Finished %date% %time% >> results_gpt2_nlfix\overnight3.log
echo.
echo FINISHED. Commit and push, then say "pushed".
pause
exit /b 0

:fail
echo.
echo A step failed. The details are at the end of results_gpt2_nlfix\overnight3.log
pause
exit /b 1
