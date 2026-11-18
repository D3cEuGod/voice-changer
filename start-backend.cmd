@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
set OPENBLAS_NUM_THREADS=1
echo VOICE_STARTUP env Checking the runtime
rem The portable package ships its own Python with every dependency in runtime\.
if exist "runtime\python.exe" (
    set "PY=runtime\python.exe"
    goto :models
)
set "PY=.venv\Scripts\python.exe"
if not exist ".venv\Scripts\python.exe" (
    py -3.13 -m venv .venv
    if errorlevel 1 goto :failed
)
".venv\Scripts\python.exe" -c "from importlib.metadata import distribution; distribution('onnxruntime')" >nul 2>&1
if not errorlevel 1 (
    ".venv\Scripts\python.exe" -m pip uninstall -y onnxruntime onnxruntime-directml
    if errorlevel 1 goto :failed
)
".venv\Scripts\python.exe" -c "from importlib.metadata import version; from pathlib import Path; assert all(version(p)==v for p,v in (l.split('==') for l in Path('requirements.txt').read_text().splitlines() if l))" >nul 2>&1
if errorlevel 1 (
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto :failed
)
:models
echo VOICE_STARTUP files Checking model files
"%PY%" setup_models.py
if errorlevel 1 goto :failed
if exist "setup_reference_engine.py" (
    "%PY%" setup_reference_engine.py
    if errorlevel 1 echo Seed-VC setup failed. Starting available legacy voices.
)
echo VOICE_STARTUP server Starting the backend service
"%PY%" -m uvicorn backend.app:app --host 127.0.0.1 --port 8765 --ws-max-size 16384 --ws-max-queue 2
if errorlevel 1 goto :failed
exit /b 0
:failed
echo Backend failed to start. See the error above.
pause
exit /b 1
