# Common Strategies Utilities

This folder contains shared utilities and tools that can be used across multiple strategies.

## Contents

### 📊 Strategy Dashboard
Generic Streamlit dashboard for visualizing strategy performance, positions, and events.

**Files:**
- `strategy_dashboard.py` - Main dashboard application
- `*_dashboard_config.yaml` - Strategy-specific configurations
- `requirements.txt` - Python dependencies
- `README.md` - Comprehensive documentation
- `MIGRATION_GUIDE.md` - Migration guide from old dashboard

**Quick Start:**
```bash
# Install dependencies
pip install -r requirements.txt

# Launch dashboard (shows all strategies)
cd strategies/common
.\launch_dashboard.ps1              # Windows
./launch_dashboard.sh               # Linux/Mac

# All strategies appear in separate tabs automatically!
```

### 🎯 Supported Strategies

Currently supports:
- **Expiry Blast** - Multi-leg options expiry day trading
- **Nifty Sehwag** - Multi-leg options NIFTY expiry trading

### 🔧 Adding Support for New Strategies

1. Add your strategy config to `dashboard_config.yaml` under `strategies` section
2. Use `PositionPersistenceManager` in your strategy
3. Launch dashboard - it will auto-detect your strategy

See `README.md` for detailed instructions.

## Architecture

The dashboard is designed to work with any strategy that:
1. Uses the `position_persistence_db` system
2. Saves sessions, positions, and events to the database
3. Has a config file (optional, defaults provided)

## Dependencies

- Streamlit 1.51.0+
- Plotly 6.3.0+
- Pandas 2.2.3+
- SQLAlchemy 2.0.31+
- PyYAML 6.0.1+

## Database Schema

Works with tables:
- `strategy_sessions` - Strategy execution sessions
- `strategy_positions` - Position tracking
- `position_events` - Event audit trail

## Features

- ✅ Real-time session monitoring
- ✅ Position tracking and visualization
- ✅ P&L analysis and statistics
- ✅ Event timeline and audit trail
- ✅ Interactive price charts
- ✅ Configurable metrics and features
- ✅ Auto-refresh capability
- ✅ Multi-strategy support

## Documentation

See `README.md` for complete documentation.
