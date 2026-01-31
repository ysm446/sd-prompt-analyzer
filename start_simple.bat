@echo off
REM Simple startup script - Run from Anaconda Prompt or Miniconda Prompt

echo Starting SD Prompt Analyzer...
echo.

REM Change to project directory
cd /d "%~dp0"

REM Activate environment and run
call conda activate sd-prompt-analyzer && python app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ERROR: Please run this file from Anaconda/Miniconda Prompt
    pause
)
