@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
rem The portable package ships its own Python with every dependency in runtime\.
if exist "runtime\pythonw.exe" (
    start "" "runtime\pythonw.exe" -m gui %*
    exit /b 0
)
if not exist ".venv-gui\Scripts\python.exe" (
    echo Creating the GUI environment in .venv-gui ...
    py -3.13 -m venv .venv-gui
    if errorlevel 1 goto :failed
)
".venv-gui\Scripts\python.exe" -c "from importlib.metadata import version; from pathlib import Path; assert all(version(p)==v for p,v in (l.split('==') for l in Path('gui/requirements.txt').read_text().splitlines() if l))" >nul 2>&1
if errorlevel 1 (
    echo Installing GUI dependencies ...
    ".venv-gui\Scripts\python.exe" -m pip install -r gui\requirements.txt
    if errorlevel 1 goto :failed
)
start "" ".venv-gui\Scripts\pythonw.exe" -m gui %*
exit /b 0
:failed
echo GUI failed to start. See the error above.
pause
exit /b 1
