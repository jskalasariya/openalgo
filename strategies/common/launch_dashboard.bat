@echo off
REM Strategy Dashboard Launcher (Windows CMD)
REM =========================================
REM Multi-strategy dashboard - all strategies in one view

echo ============================================================
echo 📊 Strategy Dashboard Launcher
echo ============================================================

REM Navigate to common folder
cd /d "%~dp0"

echo.
echo 🚀 Launching Multi-Strategy Dashboard...
echo All active strategies will be displayed in separate tabs
echo Dashboard will open at: http://localhost:8501
echo.
echo ============================================================
echo.

REM Launch dashboard
streamlit run strategy_dashboard.py
