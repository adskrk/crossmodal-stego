@echo off
rem GPT-2 measurements for the line-break fix, the fingerprint check and the split experiment.
rem About 3 hours on a laptop CPU. Keep the laptop plugged in and awake.
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set PY=.venv\Scripts\python.exe
if not exist %PY% (
  echo Run setup_windows.bat first.
  pause
  exit /b 1
)
echo Running GPT-2 experiments (about 3 hours). Progress is written to results_gpt2\overnight2.log
echo Started %date% %time% > results_gpt2\overnight2.log

echo [1/6] Line-break fix on GPT-2 (about 25 min)
echo ==== [1/6] Line-break fix on GPT-2 (about 25 min) ==== >> results_gpt2\overnight2.log
%PY% scripts\exp_text.py --model gpt2 --variant nlfix --cs 0 --per-cell 8 --out results_gpt2 >> results_gpt2\overnight2.log 2>&1 || goto :fail
%PY% scripts\add_style_detector.py --results results_gpt2 --variant nlfix >> results_gpt2\overnight2.log 2>&1 || goto :fail

echo [2/6] Fingerprint check on GPT-2 (about 10 min)
echo ==== [2/6] Fingerprint check on GPT-2 (about 10 min) ==== >> results_gpt2\overnight2.log
%PY% scripts\diag_text_fingerprint.py --model gpt2 --results results_gpt2 --variant filter >> results_gpt2\overnight2.log 2>&1 || goto :fail
%PY% scripts\diag_text_fingerprint.py --model gpt2 --results results_gpt2 --variant nlfix >> results_gpt2\overnight2.log 2>&1 || goto :fail

echo [3/6] Longer texts: 64-byte fragments (about 30 min)
echo ==== [3/6] Longer texts: 64-byte fragments (about 30 min) ==== >> results_gpt2\overnight2.log
%PY% scripts\exp_text.py --model gpt2 --variant filter --cs 0 --per-cell 4 --payload 64 --out results_gpt2\len64 >> results_gpt2\overnight2.log 2>&1 || goto :fail
%PY% scripts\add_style_detector.py --results results_gpt2\len64 >> results_gpt2\overnight2.log 2>&1 || goto :fail

echo [4/6] Longer texts: 160-byte fragments (about 70 min)
echo ==== [4/6] Longer texts: 160-byte fragments (about 70 min) ==== >> results_gpt2\overnight2.log
%PY% scripts\exp_text.py --model gpt2 --variant filter --cs 0 --per-cell 4 --payload 160 --out results_gpt2\len160 >> results_gpt2\overnight2.log 2>&1 || goto :fail
%PY% scripts\add_style_detector.py --results results_gpt2\len160 >> results_gpt2\overnight2.log 2>&1 || goto :fail

echo [5/6] Split experiment: RL choosing the text share (about 20 min)
echo ==== [5/6] Split experiment: RL choosing the text share (about 20 min) ==== >> results_gpt2\overnight2.log
%PY% scripts\exp_split.py --results results_gpt2 --sizes 24:results_gpt2,64:results_gpt2\len64,160:results_gpt2\len160 >> results_gpt2\overnight2.log 2>&1 || goto :fail

echo [6/6] Rebuilding docs\RESULTS.md and the figures
echo ==== [6/6] Rebuilding docs\RESULTS.md and the figures ==== >> results_gpt2\overnight2.log
%PY% scripts\make_report.py >> results_gpt2\overnight2.log 2>&1 || goto :fail

echo Finished %date% %time% >> results_gpt2\overnight2.log
echo.
echo FINISHED. Send results_gpt2\overnight2.log, or just say it finished.
pause
exit /b 0

:fail
echo.
echo A step failed. The details are at the end of results_gpt2\overnight2.log.
pause
exit /b 1
