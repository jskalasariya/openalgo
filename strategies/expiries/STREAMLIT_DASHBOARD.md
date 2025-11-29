# Expiry Blast Streamlit Dashboard

Real-time visualization dashboard for the Expiry Blast strategy with position tracking, P&L analysis, and event monitoring.

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
pip install -r streamlit_requirements.txt
```

## Running the Dashboard

```bash
# From the strategies/examples directory
streamlit run streamlit_dashboard.py
```

The dashboard will open at `http://localhost:8501`

## Database Connection

The dashboard automatically connects to your SQLite database:
- Location: `database/position_persistence.db` (default)
- Fetches all sessions and positions
- Real-time data updates

## Usage

1. **Select a Session**: Use the sidebar dropdown to select a strategy session
2. **View Overview**: Check total trades, win rate, and P&L at the top
3. **Analyze Positions**: 
   - **Positions tab**: See all open and closed positions
   - **Price Chart tab**: View price action with entry/exit markers
   - **P&L Analysis tab**: Visualize P&L distribution and cumulative returns
   - **Events tab**: Review detailed event timeline

## Dashboard Sections

### Session Overview
- Total trades executed
- Win rate percentage
- Total P&L (Green for profit, Red for loss)
- Average profit per trade

### Positions Table
- Symbol and leg type (CE/PE)
- Entry price and current price
- Stop loss level
- Current profit %
- Entry/exit times
- Exit reason (PROFIT_TARGET, STOP_LOSS, ATM_CHANGE, etc.)

### Price Chart
- Interactive chart with Plotly
- Entry point marker (green circle)
- Exit point marker (red X)
- Stop loss line (dashed red)
- Breakout level line (dotted blue)

### P&L Analysis
- **P&L Distribution**: Histogram showing win/loss distribution
- **Cumulative P&L**: Line chart showing running total
- Statistics: Avg profit, avg loss, profit factor

### Event Timeline
- Real-time events (last 50)
- Order placements
- Trailing stop updates
- ATM changes
- Exit reasons

## Auto-Refresh

Toggle "Auto-refresh (30s)" in the sidebar to automatically refresh data every 30 seconds for live monitoring.

## Customization

Edit `streamlit_dashboard.py` to:
- Change refresh intervals
- Add more metrics
- Customize chart styles
- Add filters or additional analysis

## Troubleshooting

**"No sessions found"**
- Make sure the strategy has recorded at least one session
- Check that `position_persistence.db` exists
- Run the strategy first to create data

**Database not connecting**
- Verify the database path in `position_persistence_db.py`
- Check file permissions

**Charts not rendering**
- Ensure Plotly is installed: `pip install plotly`
- Try refreshing the page

## Notes

- Dashboard reads live data from the persistence database
- No data is modified by the dashboard (read-only)
- Perfect for backtesting analysis and live monitoring
- Can be deployed on Streamlit Cloud for remote access

## Performance Tips

- For large datasets, filter by date range in code
- Limit event timeline to recent events (current: 50)
- Use auto-refresh sparingly in production
