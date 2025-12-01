"""
Generic Strategy Dashboard - Streamlit Visualization
====================================================
Real-time visualization dashboard for strategy positions, orders, and performance.
Works with any strategy that uses the position_persistence_db system.

Run with: streamlit run strategy_dashboard.py

The dashboard automatically detects all strategies and displays them in tabs.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pytz
from sqlalchemy import create_engine, desc, func
from sqlalchemy.orm import sessionmaker
import os
import sys
import argparse
import yaml
from pathlib import Path
from typing import List, Dict, Optional

# Add project to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from database.expiry_blast_db import (
    StrategySession,
    StrategyPosition,
    PositionEvent,
    SessionStatus,
    PositionStatus,
    EventType,
    init_position_persistence_db,
    db_session
)

# ==================== CONFIG MANAGEMENT ====================

class DashboardConfig:
    """Configuration for strategy-specific dashboard settings"""
    
    _config_cache = None  # Cache the config file
    _global_config = None
    
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self._load_unified_config()
        self.config = self._get_strategy_config()
    
    def _load_unified_config(self):
        """Load unified config file once"""
        if DashboardConfig._config_cache is None:
            config_path = Path(__file__).parent / "dashboard_config.yaml"
            
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    full_config = yaml.safe_load(f)
                    DashboardConfig._config_cache = full_config.get('strategies', {})
                    DashboardConfig._global_config = full_config.get('global', {})
            else:
                DashboardConfig._config_cache = {}
                DashboardConfig._global_config = {}
    
    def _get_strategy_config(self) -> dict:
        """Get configuration for specific strategy"""
        # Try to get strategy-specific config
        strategy_config = DashboardConfig._config_cache.get(self.strategy_name, {})
        
        # Merge with default configuration
        default_config = {
            'display_name': self.strategy_name.replace('_', ' ').title(),
            'page_icon': "📊",
            'description': f"{self.strategy_name.replace('_', ' ').title()} strategy",
            'show_price_chart': True,
            'show_pnl_analysis': True,
            'show_event_timeline': True,
            'show_statistics': True,
            'metrics': {
                'total_trades': True,
                'win_rate': True,
                'total_pnl': True,
                'avg_profit': True,
                'profit_factor': True,
                'open_positions': True
            }
        }
        
        # Merge strategy config over defaults
        merged_config = {**default_config, **strategy_config}
        
        # Merge metrics if both exist
        if 'metrics' in default_config and 'metrics' in strategy_config:
            merged_config['metrics'] = {**default_config['metrics'], **strategy_config['metrics']}
        
        return merged_config
    
    def get(self, key: str, default=None):
        """Get config value"""
        return self.config.get(key, default)
    
    @classmethod
    def get_global(cls, key: str, default=None):
        """Get global configuration value"""
        if cls._global_config is None:
            temp = cls('temp')  # Force load
        return cls._global_config.get(key, default)

# ==================== STREAMLIT CONFIG ====================

def setup_page_config():
    """Setup Streamlit page configuration"""
    st.set_page_config(
        page_title="Strategy Dashboard - All Strategies",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .metric-card {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #1f77b4;
        }
        .metric-positive {
            border-left-color: #2ecc71;
        }
        .metric-negative {
            border-left-color: #e74c3c;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding-left: 20px;
            padding-right: 20px;
        }
        </style>
    """, unsafe_allow_html=True)

# ==================== UTILITY FUNCTIONS ====================

def get_database_session():
    """Get database session"""
    init_position_persistence_db()
    return db_session

def get_all_strategies() -> List[str]:
    """Get list of all strategies that have sessions in the database"""
    session = get_database_session()
    try:
        strategies = session.query(StrategySession.strategy_name).distinct().all()
        return [s[0] for s in strategies if s[0]]
    finally:
        session.close()

def get_active_sessions(strategy_name: str = None):
    """Fetch active strategy sessions"""
    session = get_database_session()
    try:
        query = session.query(StrategySession)
        if strategy_name:
            query = query.filter(StrategySession.strategy_name == strategy_name)
        sessions = query.order_by(
            desc(StrategySession.start_time)
        ).limit(50).all()
        return sessions
    finally:
        session.close()

def get_strategy_summary(strategy_name: str) -> Dict:
    """Get summary statistics for a strategy"""
    session = get_database_session()
    try:
        # Get latest session
        latest_session = session.query(StrategySession).filter(
            StrategySession.strategy_name == strategy_name
        ).order_by(desc(StrategySession.start_time)).first()
        
        if not latest_session:
            return {
                'has_data': False,
                'total_sessions': 0,
                'active_sessions': 0,
                'latest_status': 'N/A'
            }
        
        # Count sessions
        total_sessions = session.query(func.count(StrategySession.id)).filter(
            StrategySession.strategy_name == strategy_name
        ).scalar()
        
        active_sessions = session.query(func.count(StrategySession.id)).filter(
            StrategySession.strategy_name == strategy_name,
            StrategySession.status == SessionStatus.RUNNING.value
        ).scalar()
        
        return {
            'has_data': True,
            'total_sessions': total_sessions,
            'active_sessions': active_sessions,
            'latest_status': latest_session.status,
            'latest_time': latest_session.start_time
        }
    finally:
        session.close()

def get_session_positions(session_id: int):
    """Fetch positions for a session"""
    session = get_database_session()
    try:
        positions = session.query(StrategyPosition).filter(
            StrategyPosition.session_id == session_id
        ).all()
        return positions
    finally:
        session.close()

def get_position_events(position_id: int):
    """Fetch events for a position"""
    session = get_database_session()
    try:
        events = session.query(PositionEvent).filter(
            PositionEvent.position_id == position_id
        ).order_by(PositionEvent.created_at).all()
        return events
    finally:
        session.close()

def calculate_session_stats(positions):
    """Calculate performance statistics for a session"""
    if not positions:
        return {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'win_rate': 0,
            'avg_profit': 0,
            'avg_loss': 0,
        }
    
    closed_positions = [p for p in positions if p.exit_time is not None]
    
    if not closed_positions:
        return {
            'total_trades': len(positions),
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0,
            'win_rate': 0,
            'avg_profit': 0,
            'avg_loss': 0,
        }
    
    pnls = []
    for p in closed_positions:
        pnl = (p.exit_price - p.entry_price) * 1  # Assuming 1 lot for simplicity
        pnls.append(pnl)
    
    winning = [pnl for pnl in pnls if pnl > 0]
    losing = [pnl for pnl in pnls if pnl < 0]
    
    return {
        'total_trades': len(closed_positions),
        'winning_trades': len(winning),
        'losing_trades': len(losing),
        'total_pnl': sum(pnls),
        'win_rate': len(winning) / len(closed_positions) * 100 if closed_positions else 0,
        'avg_profit': sum(winning) / len(winning) if winning else 0,
        'avg_loss': sum(losing) / len(losing) if losing else 0,
    }

def position_to_dict(pos):
    """Convert position object to dict for DataFrame"""
    return {
        'Symbol': pos.symbol,
        'Leg': pos.leg_type,
        'Entry Price': f"₹{pos.entry_price:.2f}",
        'Current Price': f"₹{pos.current_price:.2f}" if pos.current_price else "N/A",
        'Stop Loss': f"₹{pos.stop_price:.2f}",
        'Profit %': f"{pos.profit_percent*100:.2f}%",
        'Status': pos.status,
        'Exit Reason': pos.exit_reason or "Open",
        'Entry Time': pos.entry_time.strftime('%H:%M:%S') if pos.entry_time else "N/A",
        'Exit Time': pos.exit_time.strftime('%H:%M:%S') if pos.exit_time else "Open",
    }

# ==================== DASHBOARD SECTIONS ====================

def render_header():
    """Render dashboard header"""
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("📊 Strategy Dashboard - Multi-Strategy View")
    with col2:
        if st.button("🔄 Refresh All", use_container_width=True):
            st.rerun()
    st.markdown("Real-time monitoring for all your trading strategies")
    st.markdown("---")

def render_strategy_overview():
    """Render overview of all strategies"""
    strategies = get_all_strategies()
    
    if not strategies:
        st.warning("⚠️ No strategies found in database. Start a strategy to see data.")
        return None
    
    st.subheader("Strategy Overview")
    
    cols = st.columns(min(len(strategies), 4))
    
    for idx, strategy_name in enumerate(strategies):
        config = DashboardConfig(strategy_name)
        summary = get_strategy_summary(strategy_name)
        
        with cols[idx % len(cols)]:
            with st.container():
                st.markdown(f"### {config.get('page_icon', '📊')} {config.get('display_name')}")
                
                if summary['has_data']:
                    status_color = "🟢" if summary['active_sessions'] > 0 else "🔴"
                    st.markdown(f"{status_color} **Status:** {summary['latest_status']}")
                    st.metric("Total Sessions", summary['total_sessions'])
                    st.metric("Active Now", summary['active_sessions'])
                    if summary.get('latest_time'):
                        st.caption(f"Last: {summary['latest_time'].strftime('%Y-%m-%d %H:%M')}")
                else:
                    st.info("No data yet")
    
    st.markdown("---")
    return strategies

def render_session_selector(strategy_name: str):
    """Render session selection for a specific strategy"""
    sessions = get_active_sessions(strategy_name)
    
    if not sessions:
        st.info(f"No sessions found for {strategy_name}")
        return None
    
    session_options = {
        f"{s.underlying} - {s.start_time.strftime('%Y-%m-%d %H:%M:%S')} ({s.status})": s
        for s in sessions
    }
    
    selected = st.selectbox(
        "Select Session:",
        options=list(session_options.keys()),
        key=f"session_selector_{strategy_name}"
    )
    
    return session_options[selected] if selected else None
    selected = st.sidebar.selectbox(
        "Select Session:",
        options=list(session_options.keys())
    )
    
    return session_options[selected] if selected else None

def render_session_overview(session, config: DashboardConfig):
    """Render session overview metrics"""
    st.subheader(f"Session Overview - {session.underlying}")
    
    positions = get_session_positions(session.id)
    stats = calculate_session_stats(positions)
    
    metrics_config = config.get('metrics', {})
    
    # Determine which metrics to show
    metrics_to_show = []
    if metrics_config.get('total_trades', True):
        metrics_to_show.append(('Total Trades', stats['total_trades'], None))
    if metrics_config.get('win_rate', True):
        metrics_to_show.append(('Win Rate', f"{stats['win_rate']:.1f}%", f"{stats['winning_trades']} wins"))
    if metrics_config.get('total_pnl', True):
        color = "🟢" if stats['total_pnl'] >= 0 else "🔴"
        metrics_to_show.append((f"{color} Total P&L", f"₹{stats['total_pnl']:.2f}", None))
    if metrics_config.get('avg_profit', True):
        avg_profit = stats['avg_profit'] if stats['avg_profit'] else 0
        metrics_to_show.append(('Avg Profit/Trade', f"₹{avg_profit:.2f}", None))
    
    # Create columns dynamically based on metrics
    cols = st.columns(len(metrics_to_show))
    for col, (label, value, delta) in zip(cols, metrics_to_show):
        with col:
            st.metric(label, value, delta=delta)
    
    st.markdown("---")

def render_positions_table(session):
    """Render positions table"""
    st.subheader("Positions")
    
    positions = get_session_positions(session.id)
    
    if not positions:
        st.info("No positions in this session")
        return
    
    position_data = [position_to_dict(p) for p in positions]
    df = pd.DataFrame(position_data)
    
    st.dataframe(df, use_container_width=True, hide_index=True)

def render_price_chart(session):
    """Render interactive price chart"""
    st.subheader("Price Action & Positions")
    
    positions = get_session_positions(session.id)
    
    if not positions:
        st.info("No positions to chart")
        return
    
    # Create chart for first position (or selected position)
    leg_options = {f"{p.leg_type} - {p.symbol}": p for p in positions}
    selected_leg = st.selectbox("Select leg to view:", options=list(leg_options.keys()))
    
    position = leg_options[selected_leg]
    
    # Create candlestick-like chart
    fig = go.Figure()
    
    # Add entry point
    fig.add_trace(go.Scatter(
        x=[position.entry_time],
        y=[position.entry_price],
        mode='markers+text',
        name='Entry',
        marker=dict(size=12, color='green', symbol='circle'),
        text=['ENTRY'],
        textposition='top center'
    ))
    
    # Add stop loss line
    fig.add_hline(
        y=position.stop_price,
        line_dash="dash",
        line_color="red",
        annotation_text=f"SL: ₹{position.stop_price:.2f}",
        annotation_position="right"
    )
    
    # Add breakout level line
    fig.add_hline(
        y=position.highest_high_breakout,
        line_dash="dot",
        line_color="blue",
        annotation_text=f"Breakout: ₹{position.highest_high_breakout:.2f}",
        annotation_position="right"
    )
    
    # Add exit point if exists
    if position.exit_time:
        fig.add_trace(go.Scatter(
            x=[position.exit_time],
            y=[position.exit_price],
            mode='markers+text',
            name='Exit',
            marker=dict(
                size=12,
                color='red' if position.exit_reason == 'STOP_LOSS' else 'green',
                symbol='x'
            ),
            text=[position.exit_reason or 'EXIT'],
            textposition='top center'
        ))
    
    fig.update_layout(
        title=f"{position.symbol} - {position.leg_type}",
        xaxis_title="Time",
        yaxis_title="Price (₹)",
        height=400,
        hovermode='x unified'
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_pnl_analysis(session):
    """Render P&L analysis"""
    st.subheader("P&L Analysis")
    
    positions = get_session_positions(session.id)
    closed_positions = [p for p in positions if p.exit_time is not None]
    
    if not closed_positions:
        st.info("No closed positions to analyze")
        return
    
    col1, col2 = st.columns(2)
    
    # P&L distribution
    with col1:
        pnls = [(p.exit_price - p.entry_price) for p in closed_positions]
        
        fig = px.histogram(
            x=pnls,
            nbins=20,
            title="P&L Distribution",
            labels={"x": "P&L (₹)", "y": "Count"}
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    # Cumulative P&L
    with col2:
        cumulative_pnl = []
        cumsum = 0
        for p in sorted(closed_positions, key=lambda x: x.exit_time):
            pnl = p.exit_price - p.entry_price
            cumsum += pnl
            cumulative_pnl.append({
                'time': p.exit_time,
                'cumsum': cumsum,
                'symbol': p.symbol
            })
        
        df_cumsum = pd.DataFrame(cumulative_pnl)
        
        fig = px.line(
            df_cumsum,
            x='time',
            y='cumsum',
            title="Cumulative P&L",
            labels={"time": "Exit Time", "cumsum": "Cumulative P&L (₹)"},
            markers=True
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

def render_event_timeline(session):
    """Render event timeline"""
    st.subheader("Event Timeline")
    
    session_obj = get_database_session()
    try:
        events = session_obj.query(PositionEvent).filter(
            PositionEvent.session_id == session.id
        ).order_by(desc(PositionEvent.created_at)).limit(50).all()
    finally:
        session_obj.close()
    
    if not events:
        st.info("No events recorded")
        return
# ==================== STRATEGY DASHBOARD ====================

def render_strategy_dashboard(strategy_name: str):
    """Render complete dashboard for a single strategy"""
    config = DashboardConfig(strategy_name)
    
    st.header(f"{config.get('page_icon', '📊')} {config.get('display_name')}")
    
    # Session selector
    session = render_session_selector(strategy_name)
    
    if session is None:
        return
    
    # Main content
    render_session_overview(session, config)
    
    # Tabs for different views
    tabs_list = ["Positions"]
    if config.get('show_price_chart', True):
        tabs_list.append("Price Chart")
    if config.get('show_pnl_analysis', True):
        tabs_list.append("P&L Analysis")
    if config.get('show_event_timeline', True):
        tabs_list.append("Events")
    
    tabs = st.tabs(tabs_list)
    
    tab_index = 0
    with tabs[tab_index]:
        render_positions_table(session)
    tab_index += 1
    
    if config.get('show_price_chart', True):
        with tabs[tab_index]:
            render_price_chart(session)
        tab_index += 1
    
    if config.get('show_pnl_analysis', True):
        with tabs[tab_index]:
            render_pnl_analysis(session)
        tab_index += 1
    
    if config.get('show_event_timeline', True):
        with tabs[tab_index]:
            render_event_timeline(session)

# ==================== MAIN APP ====================

def main():
    # Setup page
    setup_page_config()
    render_header()
    
    # Sidebar controls
    st.sidebar.header("⚙️ Dashboard Settings")
    
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
    if auto_refresh:
        import time
        time.sleep(30)
        st.rerun()
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Active Strategies")
    
    # Get all strategies
    strategies = render_strategy_overview()
    
    if not strategies:
        return
    
    # Create tabs for each strategy
    strategy_configs = [DashboardConfig(s) for s in strategies]
    tab_labels = [f"{c.get('page_icon', '📊')} {c.get('display_name')}" for c in strategy_configs]
    
    strategy_tabs = st.tabs(tab_labels)
    
    for idx, (tab, strategy_name) in enumerate(zip(strategy_tabs, strategies)):
        with tab:
            render_strategy_dashboard(strategy_name)
    
    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        st.markdown("**Multi-Strategy Dashboard** | Real-time Monitoring")

if __name__ == "__main__":
    main()
