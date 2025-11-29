#!/usr/bin/env python3
"""
Nifty Sehwag Database Setup
============================
Initialize and manage Nifty Sehwag strategy database.

Usage:
  python setup_db.py init      - Initialize database tables
  python setup_db.py status    - Check database status
  python setup_db.py clean     - Delete old sessions (>90 days)
  python setup_db.py reset     - Reset entire database (WARNING: deletes all data)
"""

import sys
from datetime import datetime, timedelta
import os

try:
    from database.nifty_sehwag_db import (
        init_db, db_session, engine, 
        NiftySehwagSession, NiftySehwagPosition, NiftySehwagOrder, NiftySehwagEvent
    )
except ImportError as e:
    print(f"❌ Database module not available: {e}")
    print("Make sure you're in the project root directory")
    sys.exit(1)


def cmd_init():
    """Initialize database"""
    print("\n🔧 Initializing Nifty Sehwag Database...\n")
    
    try:
        init_db()
        print("✅ Database initialized successfully!")
        
        # Check tables
        from sqlalchemy import inspect
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        print(f"\n📊 Created tables:")
        for table in sorted(tables):
            if 'nifty_sehwag' in table:
                count = db_session.query(table).count() if hasattr(db_session.query(table), 'count') else 0
                print(f"   • {table}")
        
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cmd_status():
    """Check database status"""
    print("\n📊 Database Status\n")
    
    try:
        # Get database URL
        from database.nifty_sehwag_db import DATABASE_URL
        print(f"📍 Database: {DATABASE_URL}")
        
        # Get file size if SQLite
        if 'sqlite' in DATABASE_URL:
            db_path = DATABASE_URL.replace('sqlite:///', '').replace('sqlite:///', '')
            if os.path.exists(db_path):
                size_mb = os.path.getsize(db_path) / (1024 * 1024)
                print(f"💾 File size: {size_mb:.2f} MB")
            else:
                print(f"💾 Database file not found: {db_path}")
        
        # Count records
        sessions = db_session.query(NiftySehwagSession).count()
        positions = db_session.query(NiftySehwagPosition).count()
        orders = db_session.query(NiftySehwagOrder).count()
        events = db_session.query(NiftySehwagEvent).count()
        
        print(f"\n📊 Records:")
        print(f"   • Sessions: {sessions}")
        print(f"   • Positions: {positions}")
        print(f"   • Orders: {orders}")
        print(f"   • Events: {events}")
        
        # Get latest session
        latest_session = db_session.query(NiftySehwagSession).order_by(
            NiftySehwagSession.created_at.desc()
        ).first()
        
        if latest_session:
            print(f"\n📅 Latest Session:")
            print(f"   • ID: {latest_session.session_id}")
            print(f"   • Date: {latest_session.session_date}")
            print(f"   • Status: {latest_session.status}")
            print(f"   • PnL: ₹{latest_session.net_pnl:.2f}")
        
        # Get total PnL
        total_pnl = db_session.query(NiftySehwagSession).all()
        total_pnl_sum = sum(s.net_pnl or 0.0 for s in total_pnl)
        print(f"\n💰 Total PnL (all sessions): ₹{total_pnl_sum:.2f}")
        
        print("\n✅ Database is healthy!")
        
    except Exception as e:
        print(f"❌ Error checking status: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cmd_clean():
    """Clean old sessions"""
    print("\n🧹 Cleaning old sessions...\n")
    
    try:
        # Delete sessions older than 90 days
        old_date = datetime.now() - timedelta(days=90)
        
        old_sessions = db_session.query(NiftySehwagSession).filter(
            NiftySehwagSession.session_date < old_date
        ).all()
        
        if not old_sessions:
            print("✅ No old sessions to clean")
            return
        
        print(f"⚠️  Found {len(old_sessions)} sessions older than 90 days")
        print("   These will be deleted along with their positions, orders, and events\n")
        
        response = input("Continue with cleanup? (yes/no): ").strip().lower()
        if response != 'yes':
            print("❌ Cleanup cancelled")
            return
        
        # Count related records
        total_positions = 0
        total_orders = 0
        total_events = 0
        
        for session in old_sessions:
            total_positions += db_session.query(NiftySehwagPosition).filter_by(session_id=session.id).count()
            total_orders += db_session.query(NiftySehwagOrder).filter_by(session_id=session.id).count()
            total_events += db_session.query(NiftySehwagEvent).filter_by(session_id=session.id).count()
        
        # Delete sessions (cascades to related records)
        for session in old_sessions:
            db_session.delete(session)
        
        db_session.commit()
        
        print(f"\n✅ Cleanup completed!")
        print(f"   • Deleted {len(old_sessions)} sessions")
        print(f"   • Deleted {total_positions} positions")
        print(f"   • Deleted {total_orders} orders")
        print(f"   • Deleted {total_events} events")
        
    except Exception as e:
        print(f"❌ Error during cleanup: {e}")
        db_session.rollback()
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cmd_reset():
    """Reset entire database"""
    print("\n🚨 WARNING: This will delete ALL data!\n")
    
    response = input("Type 'DELETE ALL DATA' to confirm: ").strip()
    if response != 'DELETE ALL DATA':
        print("❌ Reset cancelled")
        return
    
    try:
        from sqlalchemy import MetaData, inspect
        
        print("\n🔄 Resetting database...\n")
        
        # Drop all tables
        from database.nifty_sehwag_db import Base
        Base.metadata.drop_all(bind=engine)
        print("✅ Dropped all tables")
        
        # Recreate tables
        init_db()
        print("✅ Recreated database schema")
        
        print("\n✅ Database has been reset!")
        
    except Exception as e:
        print(f"❌ Error during reset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print(__doc__)
        return
    
    cmd = sys.argv[1]
    
    if cmd == "init":
        cmd_init()
    elif cmd == "status":
        cmd_status()
    elif cmd == "clean":
        cmd_clean()
    elif cmd == "reset":
        cmd_reset()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
