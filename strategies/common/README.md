# Generic Strategy Dashboard

Real-time visualization dashboard for strategy positions, orders, and performance tracking. This dashboard works with any strategy that uses the `position_persistence_db` system.

## Features

✅ **Live Session Monitoring**
- View all active strategy sessions
- Session status and performance metrics
- Real-time position updates

✅ **Position Tracking**
- Current positions table with entry/exit prices
- Stop loss and profit levels
- Position status and exit reasons

✅ **Price Action Charts**
- Interactive candlestick-like charts
- Entry/exit markers
- Stop loss and breakout level lines
- Multiple leg selection (CE/PE)

✅ **P&L Analysis**
- P&L distribution histogram
- Cumulative P&L chart
- Win rate and profit factor
- Individual trade details

✅ **Event Timeline**
- Order execution events
- Trailing stop updates
- ATM changes
- Audit trail of all position events

✅ **Performance Statistics**
- Total trades and win rate
- Average profit/loss
- Profit factor
- Open positions count

## Installation

```bash
# Install Streamlit and dependencies
cd strategies/common
pip install -r requirements.txt
```

## Running the Dashboard

Navigate to the common folder first:
```bash
cd strategies/common
```

### Launch Dashboard (All Strategies)

The dashboard now automatically displays **all active strategies** in separate tabs!

**Windows PowerShell:**
```powershell
.\launch_dashboard.ps1
```

**Linux/Mac/WSL:**
```bash
chmod +x launch_dashboard.sh
./launch_dashboard.sh
```

**Windows CMD:**
```cmd
launch_dashboard.bat
```

### Manual Launch
```bash
streamlit run strategy_dashboard.py
```

The dashboard will open at `http://localhost:8501` and display all strategies that have sessions in the database.

### What You'll See

- **Strategy Overview:** Summary cards showing status of all strategies
- **Strategy Tabs:** Each active strategy gets its own tab with full details
- **Auto-Detection:** Automatically discovers strategies from the database
- **No Configuration Needed:** Just launch and go!

## Configuration

All strategies are configured in a single unified file: **`dashboard_config.yaml`**

### Adding a New Strategy

Edit `strategies/common/dashboard_config.yaml` and add your strategy under the `strategies` section:

```yaml
strategies:
  your_strategy:
    display_name: "Your Strategy Name"
    page_icon: "🎯"
    description: "Your strategy description"
    show_price_chart: true
    show_pnl_analysis: true
    show_event_timeline: true
    show_statistics: true
    metrics:
      total_trades: true
      win_rate: true
      total_pnl: true
      avg_profit: true
      profit_factor: true
      open_positions: true
```

### Configuration Structure

**Global Settings:**
- `global.page_title`: Overall dashboard title
- `global.refresh_interval`: Auto-refresh interval in seconds
- `global.colors`: Color scheme for all strategies

**Strategy-Specific Settings:**
- `display_name`: Strategy name shown in UI
- `page_icon`: Emoji icon for the strategy tab
- `description`: Brief description of the strategy
- `show_price_chart`: Toggle price chart tab
- `show_pnl_analysis`: Toggle P&L analysis tab
- `show_event_timeline`: Toggle event timeline tab
- `show_statistics`: Toggle statistics section
- `metrics`: Individual metric toggles for overview section

### Benefits of Unified Config

✅ **Single Source of Truth**: All strategy configs in one place  
✅ **Easy Comparison**: See all strategy settings side-by-side  
✅ **Global Settings**: Share common settings across strategies  
✅ **Less Clutter**: One file instead of many  
✅ **Version Control**: Easier to track changes

## Database Connection

The dashboard automatically connects to your SQLite database:
- Default location: `broker_strategies.db` in project root
- Can be overridden via `DATABASE_URL` environment variable

The dashboard uses the following tables:
- `strategy_sessions`: Strategy execution sessions
- `strategy_positions`: Position state and tracking
- `position_events`: Event audit trail

## Dashboard Sections

### 1. Session Selector (Sidebar)
- Lists all sessions for the selected strategy
- Shows underlying, timestamp, and status
- Select a session to view details

### 2. Session Overview
- Key performance metrics
- Total trades, win rate, P&L
- Average profit per trade

### 3. Positions Tab
- Table of all positions in the session
- Entry/exit prices and times
- Current status and profit/loss

### 4. Price Chart Tab
- Interactive price visualization
- Entry and exit markers
- Stop loss and breakout levels
- Per-leg chart selection

### 5. P&L Analysis Tab
- P&L distribution histogram
- Cumulative P&L line chart
- Detailed statistics
- Profit factor calculation

### 6. Events Tab
- Chronological event timeline
- Event types and summaries
- Detailed event data

## Creating Strategy-Specific Launchers

You can create convenience launcher scripts in your strategy folders:

### Windows PowerShell (launch_dashboard.ps1)
```powershell
# Expiry Blast Dashboard Launcher
Set-Location "$PSScriptRoot\..\..\common"
streamlit run strategy_dashboard.py -- --strategy expiry_blast
```

### Linux/Mac (launch_dashboard.sh)
```bash
#!/bin/bash
cd "$(dirname "$0")/../../common"
streamlit run strategy_dashboard.py -- --strategy expiry_blast
```

## Customization

### Adding Custom Visualizations

To add strategy-specific visualizations, you can:

1. Create a custom config file
2. Extend the `DashboardConfig` class
3. Add custom rendering functions
4. Use the config to conditionally show custom sections

### Color Schemes

Customize colors in the config file:
```yaml
colors:
  positive: "#2ecc71"  # Green for profits
  negative: "#e74c3c"  # Red for losses
  neutral: "#1f77b4"   # Blue for neutral
```

## Integration with Strategies

Your strategy should use the `PositionPersistenceManager` from `position_persistence.py`:

```python
from strategies.expiries.position_persistence import PositionPersistenceManager

# Initialize
persistence = PositionPersistenceManager(
    strategy_name='your_strategy',
    api_client=client
)

# Start session
persistence.start_session(session_id, underlying)

# Save positions
persistence.save_position_entry(...)
persistence.update_position_state(...)
persistence.save_position_exit(...)
```

## Notes

- Dashboard reads live data from the persistence database
- No data is modified by the dashboard (read-only)
- Auto-refresh updates data without page reload
- Can be deployed on Streamlit Cloud for remote access
- Works with SQLite, PostgreSQL, MySQL databases

## Troubleshooting

### No Sessions Found
- Ensure your strategy is running and creating sessions
- Check that the `strategy_name` matches exactly
- Verify database connection

### Dashboard Won't Start
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Check Python version (3.8+)
- Verify Streamlit is installed correctly

### Data Not Updating
- Click "Refresh Data" button
- Enable auto-refresh
- Check that strategy is actively writing to database

## Support

For issues or questions:
1. Check the configuration file
2. Verify database connection
3. Review strategy integration
4. Check logs for errors
