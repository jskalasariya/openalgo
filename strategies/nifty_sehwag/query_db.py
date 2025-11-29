#!/usr/bin/env python3
"""
Nifty Sehwag Strategy Database Query Tool
==========================================
Utility script to query strategy execution data from database.

Usage:
  python query_db.py summary <session_id>     - Get session summary
  python query_db.py daily <YYYY-MM-DD>       - Get daily performance
  python query_db.py positions <session_id>   - List positions
  python query_db.py orders <session_id>      - List orders
  python query_db.py events <session_id>      - List events
  python query_db.py recent <N>               - Show N most recent sessions
"""

import sys
import json
from datetime import datetime
from tabulate import tabulate

try:
    from database.nifty_sehwag_db import (
        db_session, NiftySehwagSession, NiftySehwagPosition,
        NiftySehwagOrder, NiftySehwagEvent,
        get_session_summary, get_daily_performance, get_session
    )
except ImportError:
    print("❌ Database module not available. Install: pip install sqlalchemy")
    sys.exit(1)


def format_datetime(dt):
    """Format datetime for display"""
    if dt is None:
        return "N/A"
    return dt.strftime('%Y-%m-%d %H:%M:%S')


def cmd_summary(session_id):
    """Display session summary"""
    print(f"\n📊 Session Summary: {session_id}\n")
    
    summary = get_session_summary(session_id)
    if not summary:
        print(f"❌ Session not found: {session_id}")
        return
    
    # Session info
    print(f"📅 Expiry Date: {summary['expiry_date']}")
    print(f"📍 Status: {summary['status']}")
    print(f"⏱️  Start: {summary['start_time']}")
    print(f"⏱️  End: {summary['end_time']}")
    print(f"📊 Total Orders: {summary['total_orders']}")
    print(f"💰 Net PnL: {summary['net_pnl']:.2f}")
    print(f"📈 Positions: {summary['total_positions']} ({summary['closed_positions']} closed, {summary['active_positions']} active)")
    
    # Position breakdown
    if summary['positions']:
        print(f"\n📋 Position Details:\n")
        positions_data = []
        for pos in summary['positions']:
            positions_data.append([
                f"Leg {pos['leg']}",
                pos['symbol'],
                f"₹{pos['entry_price']:.2f}" if pos['entry_price'] else "N/A",
                f"₹{pos['exit_price']:.2f}" if pos['exit_price'] else "N/A",
                f"₹{pos['realized_pnl']:.2f}",
                f"{pos['pnl_percentage']:.2f}%",
                pos['status']
            ])
        
        headers = ["Leg", "Symbol", "Entry", "Exit", "PnL", "PnL %", "Status"]
        print(tabulate(positions_data, headers=headers, tablefmt="grid"))


def cmd_daily(expiry_date):
    """Display daily performance"""
    print(f"\n📊 Daily Performance: {expiry_date}\n")
    
    daily = get_daily_performance(expiry_date)
    if not daily:
        print(f"❌ No sessions found for: {expiry_date}")
        return
    
    print(f"📅 Date: {daily['expiry_date']}")
    print(f"🔢 Sessions: {daily['num_sessions']}")
    print(f"📊 Total Positions: {daily['total_positions']}")
    print(f"📈 Total Orders: {daily['total_orders']}")
    print(f"💰 Net PnL: {daily['net_pnl']:.2f}")
    print(f"\n📝 Sessions: {', '.join(daily['sessions'])}")


def cmd_positions(session_id):
    """List all positions for a session"""
    print(f"\n📊 Positions: {session_id}\n")
    
    session = get_session(session_id)
    if not session:
        print(f"❌ Session not found: {session_id}")
        return
    
    from database.nifty_sehwag_db import NiftySehwagPosition
    positions = db_session.query(NiftySehwagPosition).filter_by(session_id=session.id).all()
    
    if not positions:
        print("No positions found")
        return
    
    pos_data = []
    for p in positions:
        pos_data.append([
            p.leg_number,
            p.symbol,
            format_datetime(p.entry_time),
            f"₹{p.entry_price:.2f}" if p.entry_price else "N/A",
            format_datetime(p.exit_time),
            f"₹{p.exit_price:.2f}" if p.exit_price else "N/A",
            f"₹{p.realized_pnl:.2f}" if p.realized_pnl else "N/A",
            f"{p.pnl_percentage:.2f}%" if p.pnl_percentage else "N/A",
            p.status
        ])
    
    headers = ["Leg", "Symbol", "Entry Time", "Entry Price", "Exit Time", "Exit Price", "PnL", "PnL %", "Status"]
    print(tabulate(pos_data, headers=headers, tablefmt="grid"))


def cmd_orders(session_id):
    """List all orders for a session"""
    print(f"\n📋 Orders: {session_id}\n")
    
    session = get_session(session_id)
    if not session:
        print(f"❌ Session not found: {session_id}")
        return
    
    orders = db_session.query(NiftySehwagOrder).filter_by(session_id=session.id).all()
    
    if not orders:
        print("No orders found")
        return
    
    orders_data = []
    for o in orders:
        orders_data.append([
            o.leg_number or "-",
            o.symbol,
            o.order_type,
            o.side,
            o.quantity,
            f"₹{o.price:.2f}" if o.price else "Market",
            o.status,
            f"₹{o.execution_price:.2f}" if o.execution_price else "-",
            format_datetime(o.execution_time) if o.execution_time else "-"
        ])
    
    headers = ["Leg", "Symbol", "Type", "Side", "Qty", "Price", "Status", "Exec Price", "Exec Time"]
    print(tabulate(orders_data, headers=headers, tablefmt="grid"))


def cmd_events(session_id):
    """List all events for a session"""
    print(f"\n📝 Events: {session_id}\n")
    
    session = get_session(session_id)
    if not session:
        print(f"❌ Session not found: {session_id}")
        return
    
    events = db_session.query(NiftySehwagEvent).filter_by(session_id=session.id).order_by(NiftySehwagEvent.event_time).all()
    
    if not events:
        print("No events found")
        return
    
    events_data = []
    for e in events:
        data_str = ""
        if e.data:
            try:
                data_obj = json.loads(e.data)
                data_str = json.dumps(data_obj, indent=0)[:30] + "..." if len(json.dumps(data_obj)) > 30 else json.dumps(data_obj)
            except:
                data_str = e.data[:30]
        
        events_data.append([
            format_datetime(e.event_time),
            e.event_type,
            e.leg_number or "-",
            e.symbol or "-",
            e.description or "-",
            data_str
        ])
    
    headers = ["Time", "Event Type", "Leg", "Symbol", "Description", "Data"]
    print(tabulate(events_data, headers=headers, tablefmt="grid", maxcolwidths=[19, 20, 5, 10, 20, 25]))


def cmd_recent(n=10):
    """Show N most recent sessions"""
    print(f"\n📊 Most Recent {n} Sessions\n")
    
    sessions = db_session.query(NiftySehwagSession).order_by(
        NiftySehwagSession.created_at.desc()
    ).limit(n).all()
    
    if not sessions:
        print("No sessions found")
        return
    
    sessions_data = []
    for s in sessions:
        # Calculate PnL from positions
        positions = db_session.query(NiftySehwagPosition).filter_by(session_id=s.id).all()
        total_pnl = sum(p.realized_pnl or 0.0 for p in positions)
        
        sessions_data.append([
            s.session_id[-12:],  # Last 12 chars of session ID
            s.expiry_date,
            s.status,
            len(positions),
            f"₹{total_pnl:.2f}",
            format_datetime(s.start_time),
            format_datetime(s.end_time)
        ])
    
    headers = ["Session ID", "Expiry", "Status", "Positions", "PnL", "Start", "End"]
    print(tabulate(sessions_data, headers=headers, tablefmt="grid"))


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    cmd = sys.argv[1]
    
    try:
        if cmd == "summary" and len(sys.argv) > 2:
            cmd_summary(sys.argv[2])
        elif cmd == "daily" and len(sys.argv) > 2:
            cmd_daily(sys.argv[2])
        elif cmd == "positions" and len(sys.argv) > 2:
            cmd_positions(sys.argv[2])
        elif cmd == "orders" and len(sys.argv) > 2:
            cmd_orders(sys.argv[2])
        elif cmd == "events" and len(sys.argv) > 2:
            cmd_events(sys.argv[2])
        elif cmd == "recent":
            n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
            cmd_recent(n)
        else:
            print(__doc__)
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
