# Strategy Dashboard Launcher (PowerShell)
# ========================================
# Multi-strategy dashboard - all strategies in one view

Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host "📊 Strategy Dashboard Launcher" -ForegroundColor Green
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 59) -ForegroundColor Cyan

# Navigate to common folder
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

Write-Host "`n🚀 Launching Multi-Strategy Dashboard..." -ForegroundColor Green
Write-Host "All active strategies will be displayed in separate tabs" -ForegroundColor Gray
Write-Host "Dashboard will open at: http://localhost:8501`n" -ForegroundColor Gray
Write-Host "=" -ForegroundColor Cyan -NoNewline
Write-Host ("=" * 59) -ForegroundColor Cyan
Write-Host ""

# Launch dashboard
streamlit run strategy_dashboard.py
