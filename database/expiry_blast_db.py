"""
Position Persistence Database Module
====================================
Handles storage and recovery of position state for crash recovery.
Designed to work alongside existing strategy execution without modification.

Key Features:
- Persistent position state storage
- Crash detection and identification
- Recovery of crashed trades
- Audit trail of all position events
- Session management
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, Index
from sqlalchemy.orm import scoped_session, sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from sqlalchemy.pool import NullPool
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# Get DATABASE_URL from environment or use default SQLite database
DATABASE_URL = os.getenv('DATABASE_URL')

if not DATABASE_URL:
    # Default to SQLite database in db folder
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'db', 'expiry_blast.db')
    DATABASE_URL = f'sqlite:///{db_path}'
    logger.info(f"📁 Using default SQLite database: {db_path}")

# Conditionally create engine based on DB type
if 'sqlite' in DATABASE_URL:
    engine = create_engine(
        DATABASE_URL,
        poolclass=NullPool,
        connect_args={'check_same_thread': False}
    )
else:
    engine = create_engine(
        DATABASE_URL,
        pool_size=50,
        max_overflow=100,
        pool_timeout=10
    )

db_session = scoped_session(sessionmaker(autocommit=False, autoflush=False, bind=engine))
Base = declarative_base()
Base.query = db_session.query_property()


# ==================== ENUMS ====================

class PositionStatus(str, Enum):
    """Status of a position"""
    ACTIVE = "ACTIVE"           # Position is open and being monitored
    CLOSED = "CLOSED"           # Position was closed normally (profit/stop)
    CRASHED = "CRASHED"         # Session crashed while position was open
    RECOVERED = "RECOVERED"     # Position recovered from crash and resumed
    ORPHANED = "ORPHANED"       # Position exists on broker but not in DB
    MANUAL_EXIT = "MANUAL_EXIT" # Manually exited
    EXPIRED = "EXPIRED"         # Position expired (end of day)


class EventType(str, Enum):
    """Types of events that can occur to a position"""
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    TRAIL_UPDATE = "TRAIL_UPDATE"
    ATM_CHANGE = "ATM_CHANGE"
    STOP_HIT = "STOP_HIT"
    PROFIT_TARGET_HIT = "PROFIT_TARGET_HIT"
    CRASH = "CRASH"
    RECOVERY = "RECOVERY"
    MANUAL_INTERVENTION = "MANUAL_INTERVENTION"


class SessionStatus(str, Enum):
    """Status of a strategy session"""
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    CRASHED = "CRASHED"
    RECOVERED = "RECOVERED"
    MANUAL_STOP = "MANUAL_STOP"


# ==================== DATABASE MODELS ====================

class StrategySession(Base):
    """Track strategy execution sessions for crash detection"""
    __tablename__ = 'strategy_sessions'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(String(36), unique=True, nullable=False, index=True)  # UUID
    strategy_name = Column(String(100), nullable=False, index=True)  # e.g., 'expiry_blast'
    underlying = Column(String(50), nullable=False)  # e.g., 'NIFTY', 'BANKNIFTY'
    
    start_time = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    
    status = Column(String(20), default=SessionStatus.RUNNING.value, index=True)  # RUNNING, CRASHED, COMPLETED, etc.
    
    # Crash details
    crash_timestamp = Column(DateTime(timezone=True), nullable=True)
    crash_error = Column(Text, nullable=True)  # Exception message/traceback
    crash_reason = Column(String(200), nullable=True)  # Human-readable reason
    
    # Recovery tracking
    recovery_attempted = Column(Boolean, default=False)
    recovery_timestamp = Column(DateTime(timezone=True), nullable=True)
    recovery_successful = Column(Boolean, default=False)
    recovery_notes = Column(Text, nullable=True)
    
    # Metadata
    positions_open_at_crash = Column(Integer, default=0)
    user_api_key = Column(String(100), nullable=True, index=True)  # For recovery
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    positions = relationship("StrategyPosition", back_populates="session", cascade="all, delete-orphan")
    events = relationship("PositionEvent", back_populates="session", cascade="all, delete-orphan")


class StrategyPosition(Base):
    """Store position state for persistence and recovery"""
    __tablename__ = 'strategy_positions'
    
    id = Column(Integer, primary_key=True)
    position_id = Column(String(36), unique=True, nullable=False, index=True)  # UUID
    session_id = Column(Integer, ForeignKey('strategy_sessions.id'), nullable=False, index=True)
    
    # Instrument details
    symbol = Column(String(100), nullable=False)  # Full symbol (e.g., 'NIFTY25NOV13900CE')
    leg_type = Column(String(10), nullable=False)  # CE or PE
    strike = Column(Float, nullable=False)  # Strike price
    underlying = Column(String(50), nullable=False)
    
    # Position state
    status = Column(String(20), default=PositionStatus.ACTIVE.value, index=True)
    
    # Entry details
    entry_price = Column(Float, nullable=False)
    entry_time = Column(DateTime(timezone=True), nullable=False)
    entry_order_id = Column(String(100), nullable=True)
    
    # Current state
    current_price = Column(Float, nullable=True)
    stop_price = Column(Float, nullable=False)
    highest_price = Column(Float, nullable=False)
    highest_high_breakout = Column(Float, nullable=False)  # Breakout level from 5 candles
    
    # Trailing stop state
    last_trail_level = Column(Integer, default=0)  # Number of trail steps taken
    profit_percent = Column(Float, default=0.0)  # Current profit %
    
    # Exit details
    exit_price = Column(Float, nullable=True)
    exit_time = Column(DateTime(timezone=True), nullable=True)
    exit_order_id = Column(String(100), nullable=True)
    exit_reason = Column(String(50), nullable=True)  # PROFIT_TARGET, STOP_LOSS, ATM_CHANGE, etc.
    
    # Crash tracking
    was_crashed = Column(Boolean, default=False)
    crash_recovery_count = Column(Integer, default=0)  # How many times this was recovered
    
    # Metadata
    candle_count_at_entry = Column(Integer, nullable=True)
    atm_checks_since_entry = Column(Integer, default=0)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    session = relationship("StrategySession", back_populates="positions")
    events = relationship("PositionEvent", back_populates="position", cascade="all, delete-orphan")
    
    # Indexes for fast queries
    __table_args__ = (
        Index('idx_session_status_active', 'session_id', 'status'),
        Index('idx_session_symbol', 'session_id', 'symbol'),
        Index('idx_crash_recovery', 'was_crashed', 'status'),
    )


class PositionEvent(Base):
    """Audit trail of position events"""
    __tablename__ = 'position_events'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('strategy_sessions.id'), nullable=False, index=True)
    position_id = Column(Integer, ForeignKey('strategy_positions.id'), nullable=True, index=True)
    
    event_type = Column(String(30), nullable=False, index=True)  # ENTRY, EXIT, TRAIL_UPDATE, CRASH, etc.
    event_timestamp = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    
    # Event details as JSON for flexibility
    event_data = Column(Text, nullable=True)  # JSON string with event-specific data
    
    # Event summary for quick viewing
    summary = Column(String(500), nullable=True)
    
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    session = relationship("StrategySession", back_populates="events")
    position = relationship("StrategyPosition", back_populates="events")


# ==================== DATABASE INITIALIZATION ====================

def init_position_persistence_db():
    """Initialize the position persistence database"""
    try:
        Base.metadata.create_all(bind=engine, checkfirst=True)
        logger.info("✓ Position persistence database initialized successfully")
    except Exception as e:
        logger.error(f"✗ Error initializing position persistence database: {e}")
        raise


# ==================== HELPER FUNCTIONS ====================

def get_active_crashed_positions(strategy_name: str, underlying: str) -> List[Dict[str, Any]]:
    """
    Get list of positions that were crashed and need recovery
    
    Returns:
        List of crashed positions with full state for recovery
    """
    try:
        # Find crashed sessions
        crashed_sessions = db_session.query(StrategySession).filter(
            StrategySession.strategy_name == strategy_name,
            StrategySession.underlying == underlying,
            StrategySession.status == SessionStatus.CRASHED.value,
            StrategySession.recovery_attempted == False
        ).all()
        
        crashed_positions = []
        for session in crashed_sessions:
            # Get all ACTIVE positions from crashed session
            positions = db_session.query(StrategyPosition).filter(
                StrategyPosition.session_id == session.id,
                StrategyPosition.status == PositionStatus.ACTIVE.value
            ).all()
            
            for pos in positions:
                crashed_positions.append({
                    'session_id': session.session_id,
                    'position_id': pos.position_id,
                    'symbol': pos.symbol,
                    'leg_type': pos.leg_type,
                    'strike': pos.strike,
                    'entry_price': pos.entry_price,
                    'entry_time': pos.entry_time.isoformat() if pos.entry_time else None,
                    'stop_price': pos.stop_price,
                    'highest_price': pos.highest_price,
                    'highest_high_breakout': pos.highest_high_breakout,
                    'last_trail_level': pos.last_trail_level,
                    'was_crashed': pos.was_crashed,
                    'crash_recovery_count': pos.crash_recovery_count,
                    'crash_timestamp': session.crash_timestamp.isoformat() if session.crash_timestamp else None,
                    'crash_reason': session.crash_reason,
                })
        
        return crashed_positions
    
    except Exception as e:
        logger.error(f"✗ Error fetching crashed positions: {e}")
        return []


def get_active_sessions() -> List[Dict[str, Any]]:
    """Get all currently active sessions"""
    try:
        sessions = db_session.query(StrategySession).filter(
            StrategySession.status == SessionStatus.RUNNING.value
        ).all()
        
        return [
            {
                'session_id': s.session_id,
                'strategy_name': s.strategy_name,
                'underlying': s.underlying,
                'start_time': s.start_time.isoformat() if s.start_time else None,
                'positions_open': len(s.positions),
            }
            for s in sessions
        ]
    
    except Exception as e:
        logger.error(f"✗ Error fetching active sessions: {e}")
        return []


def mark_session_crashed(session_id: str, error: str, reason: str = None):
    """Mark a session as crashed"""
    try:
        session = db_session.query(StrategySession).filter(
            StrategySession.session_id == session_id
        ).first()
        
        if session:
            session.status = SessionStatus.CRASHED.value
            session.crash_timestamp = datetime.now()
            session.crash_error = error
            session.crash_reason = reason or "Unknown error"
            
            # Count open positions
            open_positions = db_session.query(StrategyPosition).filter(
                StrategyPosition.session_id == session.id,
                StrategyPosition.status == PositionStatus.ACTIVE.value
            ).all()
            
            session.positions_open_at_crash = len(open_positions)
            
            # Mark positions as crashed
            for pos in open_positions:
                pos.was_crashed = True
                pos.status = PositionStatus.CRASHED.value
            
            db_session.commit()
            logger.warning(f"⚠️  Session {session_id} marked as CRASHED with {len(open_positions)} open positions")
            return True
        
        return False
    
    except Exception as e:
        logger.error(f"✗ Error marking session as crashed: {e}")
        db_session.rollback()
        return False


def get_session_stats(session_id: str) -> Dict[str, Any]:
    """Get statistics for a session"""
    try:
        session = db_session.query(StrategySession).filter(
            StrategySession.session_id == session_id
        ).first()
        
        if not session:
            return {}
        
        positions = db_session.query(StrategyPosition).filter(
            StrategyPosition.session_id == session.id
        ).all()
        
        stats = {
            'session_id': session_id,
            'strategy': session.strategy_name,
            'underlying': session.underlying,
            'status': session.status,
            'start_time': session.start_time.isoformat() if session.start_time else None,
            'end_time': session.end_time.isoformat() if session.end_time else None,
            'total_positions': len(positions),
            'active_positions': len([p for p in positions if p.status == PositionStatus.ACTIVE.value]),
            'closed_positions': len([p for p in positions if p.status == PositionStatus.CLOSED.value]),
            'crashed_positions': len([p for p in positions if p.status == PositionStatus.CRASHED.value]),
            'recovered_positions': len([p for p in positions if p.status == PositionStatus.RECOVERED.value]),
        }
        
        return stats
    
    except Exception as e:
        logger.error(f"✗ Error getting session stats: {e}")
        return {}
