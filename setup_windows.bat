@echo off
rem One-time setup on Windows. Double-click this file, or run it from a terminal.
rem Needs: Python 3.11 (python.org installer) and Git for Windows. Downloads about 1.5 GB.
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
echo ============================================================
echo  Cross-modal steganography: Windows setup (20-40 minutes)
echo ============================================================

py -3.11 --version >nul 2>nul
if errorlevel 1 (
  echo.
  echo Python 3.11 was not found.
  echo Install it from https://www.python.org/downloads/release/python-3119/
  echo   ^(scroll down, "Windows installer 64-bit"^), then run this file again.
  goto :fail
)
git --version >nul 2>nul
if errorlevel 1 (
  echo Git was not found. Install Git for Windows from https://git-scm.com/download/win
  goto :fail
)

echo [1/6] Creating the Python environment (.venv)
if not exist .venv\Scripts\python.exe (
  py -3.11 -m venv .venv || goto :fail
)
set PY=.venv\Scripts\python.exe
%PY% -m pip install --upgrade pip || goto :fail

echo [2/6] Installing PyTorch (CPU)
%PY% -m pip install torch || goto :fail

echo [3/6] Installing the other libraries
%PY% -m pip install -r requirements.txt || goto :fail

if not exist vendor mkdir vendor
if not exist data mkdir data
echo [4/6] Downloading the pretrained image detectors (Aletheia, about 250 MB)
if not exist vendor\aletheia\aletheia-models (
  if exist vendor\aletheia rmdir /s /q vendor\aletheia
  git clone --depth 1 --filter=blob:none --sparse https://github.com/daniellerch/aletheia.git vendor\aletheia || goto :fail
  git -C vendor\aletheia sparse-checkout set aletheia-models || goto :fail
)

echo [5/6] Downloading the Kodak cover photos and cutting them into crops
if not exist data\kodak (
  git clone --depth 1 https://github.com/MohamedBakrAli/Kodak-Lossless-True-Color-Image-Suite.git data\kodak || goto :fail
)
%PY% scripts\prepare_covers.py --kodak data\kodak\PhotoCD_PCD0992 --out data\covers || goto :fail

echo [6/6] Downloading GPT-2 (about 550 MB)
%PY% -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('gpt2'); AutoModelForCausalLM.from_pretrained('gpt2'); print('gpt2 ready')" || goto :fail

echo.
echo ============================================================
echo  Setup finished.
echo  Next: double-click run_tests.bat, then run_app.bat
echo ============================================================
pause
exit /b 0

:fail
echo.
echo Setup stopped because of the error above. Copy the last lines and ask for help.
pause
exit /b 1
