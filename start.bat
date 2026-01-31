@echo off
chcp 65001 >nul
REM SD Prompt Analyzer Startup Script

echo ========================================
echo   SD Prompt Analyzer
echo ========================================
echo.

REM Initialize Conda
if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\miniconda3\Scripts\activate.bat"
) else if exist "D:\miniconda3\Scripts\activate.bat" (
    call "D:\miniconda3\Scripts\activate.bat"
) else if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\anaconda3\Scripts\activate.bat"
) else (
    echo [ERROR] Conda not found. Please check your Conda installation.
    echo.
    echo Please run this command in Conda prompt:
    echo   conda activate sd-prompt-analyzer
    echo   python app.py
    echo.
    pause
    exit /b 1
)

REM Activate Conda environment
echo Activating conda environment: sd-prompt-analyzer
call conda activate sd-prompt-analyzer

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Environment 'sd-prompt-analyzer' not found
    echo.
    echo Please setup the environment first:
    echo   1. conda create -n sd-prompt-analyzer python=3.10 -y
    echo   2. conda activate sd-prompt-analyzer
    echo   3. pip install -r requirements.txt
    echo   4. python scripts/setup.py
    echo.
    pause
    exit /b 1
)

echo Environment activated successfully
echo.

REM Start application
echo Starting SD Prompt Analyzer...
echo Browser will open automatically
echo.
echo Press Ctrl+C to stop
echo ========================================
echo.

python app.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ========================================
    echo [ERROR] Failed to start application
    echo ========================================
    pause
)
