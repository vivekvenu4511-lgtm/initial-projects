@echo off
title Aria Personal Agent v7
cd /d "%~dp0"

echo.
echo  Aria Personal Agent v7 - HuggingFace + llama-cpp
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo  [ERROR] Python not found. Install from https://python.org
    pause & exit /b 1
)

:: Enable Windows long paths (silent - may need admin for registry but pip config works)
python -c "import subprocess; subprocess.run(['reg','add','HKLM\\SYSTEM\\CurrentControlSet\\Control\\FileSystem','/v','LongPathsEnabled','/t','REG_DWORD','/d','1','/f'], capture_output=True)" >nul 2>&1

:: Check if llama-cpp-python is installed
python -c "import llama_cpp" >nul 2>&1
if errorlevel 1 (
    echo  [Setup] Running first-time setup...
    echo  [Setup] This will download pre-built wheels - no compiler needed.
    echo.
    python setup_v7.py
    echo.
) else (
    pip install -r requirements.txt --quiet --no-warn-script-location >nul 2>&1
)

echo  [Launch] Dashboard  ->  http://localhost:8000
echo  [Launch] HF Manager ->  http://localhost:8000/hf
echo.

:: Open browser after 4 seconds in background
start "" /min cmd /c "timeout /t 4 /nobreak >nul & start http://localhost:8000"

python main.py %*

if errorlevel 1 (
    echo.
    echo  [ERROR] Aria exited with an error. Check output above.
    echo  [Help]  Try:  python setup_v7.py
    pause
)
