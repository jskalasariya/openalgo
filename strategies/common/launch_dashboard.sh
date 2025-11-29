#!/bin/bash
# Strategy Dashboard Launcher (Bash)
# ===================================
# Multi-strategy dashboard - all strategies in one view

echo "============================================================"
echo "📊 Strategy Dashboard Launcher"
echo "============================================================"

# Navigate to common folder
cd "$(dirname "$0")"

echo ""
echo "🚀 Launching Multi-Strategy Dashboard..."
echo "All active strategies will be displayed in separate tabs"
echo "Dashboard will open at: http://localhost:8501"
echo ""
echo "============================================================"
echo ""

# Launch dashboard
streamlit run strategy_dashboard.py
