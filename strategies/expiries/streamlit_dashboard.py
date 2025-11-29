"""
Expiry Blast Strategy - Streamlit Dashboard
===========================================
Real-time visualization of Expiry Blast strategy positions, orders, and performance.

Run with: streamlit run streamlit_dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import pytz
from sqlalchemy import create_engine, desc
from sqlalchemy.orm import sessionmaker
import os
import sys

# Add project to path
sys.path.insert(0, os.path.dirname(__file__))

from database.position_persistence_db import (
    StrategySession,
    StrategyPosition,
    PositionEvent,
    SessionStatus,
    PositionStatus,
    EventType,
    init_position_persistence_db,
    db_session
)

# ==================== STREAMLIT CONFIG ====================

st.set_page_config(
    page_title="Expiry Blast Dashboard",
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
    </style>
""", unsafe_allow_html=True)

# ==================== UTILITY FUNCTIONS ====================

def get_database_session():
    """Get database session"""
    init_position_persistence_db()
    return db_session

def get_active_sessions():
    """Fetch active strategy sessions"""
    session = get_database_session()
    try:
        sessions = session.query(StrategySession).order_by(
            desc(StrategySession.start_time)
        ).limit(50).all()
        return sessions
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
    st.title("📊 Expiry Blast Strategy Dashboard")
    st.markdown("---")

def render_session_selector():
    """Render session selection sidebar"""
    st.sidebar.header("📋 Sessions")
    sessions = get_active_sessions()
    
    if not sessions:
        st.sidebar.warning("No sessions found in database")
        return None
    
    session_options = {
        f"{s.underlying} - {s.start_time.strftime('%Y-%m-%d %H:%M:%S')} ({s.status})": s
        for s in sessions
    }
    
    selected = st.sidebar.selectbox(
        "Select Session:",
        options=list(session_options.keys())
    )
    
    return session_options[selected] if selected else None

def render_session_overview(session):
    """Render session overview metrics"""
    st.subheader(f"Session Overview - {session.underlying}")
    
    positions = get_session_positions(session.id)
    stats = calculate_session_stats(positions)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Trades", stats['total_trades'], delta=None)
    
    with col2:
        st.metric("Win Rate", f"{stats['win_rate']:.1f}%", 
                 delta=f"{stats['winning_trades']} wins")
    
    with col3:
        color = "🟢" if stats['total_pnl'] >= 0 else "🔴"
        st.metric(f"{color} Total P&L", f"₹{stats['total_pnl']:.2f}")
    
    with col4:
        avg_profit = stats['avg_profit'] if stats['avg_profit'] else 0
        st.metric("Avg Profit/Trade", f"₹{avg_profit:.2f}")
    
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
    
    event_data = []
    for event in events:
        event_data.append({
            'Time': event.created_at.strftime('%H:%M:%S'),
            'Type': event.event_type,
            'Summary': event.summary,
            'Details': event.event_data[:100] if event.event_data else "N/A"
        })
    
    df_events = pd.DataFrame(event_data)
    st.dataframe(df_events, use_container_width=True, hide_index=True)

def render_statistics(session):
    """Render detailed statistics"""
    st.subheader("Detailed Statistics")
    
    positions = get_session_positions(session.id)
    stats = calculate_session_stats(positions)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Winning Trades", stats['winning_trades'])
        st.metric("Losing Trades", stats['losing_trades'])
    
    with col2:
        st.metric("Avg Win", f"₹{stats['avg_profit']:.2f}")
        st.metric("Avg Loss", f"₹{stats['avg_loss']:.2f}")
    
    with col3:
        if stats['winning_trades'] + stats['losing_trades'] > 0:
            profit_factor = abs(stats['avg_profit'] * stats['winning_trades']) / abs(stats['avg_loss'] * stats['losing_trades']) if stats['avg_loss'] != 0 else 0
            st.metric("Profit Factor", f"{profit_factor:.2f}")
        
        if positions:
            open_positions = [p for p in positions if p.exit_time is None]
            st.metric("Open Positions", len(open_positions))

# ==================== MAIN APP ====================

def main():
    render_header()
    
    # Refresh data periodically
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
    if auto_refresh:
        st.write("Auto-refresh enabled")
        import time
        time.sleep(30)
        st.rerun()
    
    # Session selector
    session = render_session_selector()
    
    if session is None:
        st.warning("Please select or create a session first")
        return
    
    # Main content
    render_session_overview(session)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["Positions", "Price Chart", "P&L Analysis", "Events"])
    
    with tab1:
        render_positions_table(session)
    
    with tab2:
        render_price_chart(session)
    
    with tab3:
        render_pnl_analysis(session)
        render_statistics(session)
    
    with tab4:
        render_event_timeline(session)
    
    # Footer
    st.markdown("---")
    st.markdown("**Expiry Blast Strategy Dashboard** | Real-time position and performance tracking")

if __name__ == "__main__":
    main()
