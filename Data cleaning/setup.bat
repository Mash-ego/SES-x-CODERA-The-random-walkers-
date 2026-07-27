@echo off
REM Sets up a self-contained virtual environment for this project.
REM Run once by double-clicking this file, or: setup.bat

cd /d "%~dp0"

echo Creating virtual environment (.venv)...
python -m venv .venv

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing required libraries...
pip install -r requirements.txt

echo.
echo Setup complete.
echo To run the analysis:
echo   .venv\Scripts\activate.bat
echo   python multi_predictor_analysis.py
pause
