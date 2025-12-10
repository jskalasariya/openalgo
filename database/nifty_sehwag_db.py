"""
Nifty Sehwag Strategy Database Module
=====================================
Handles persistent storage of trade history, positions, and strategy execution logs
for the Nifty Sehwag multi-leg options expiry strategy.

Key Features:
- Trade history tracking (all orders placed)
- Position snapshots (entry, management, exit events)
- Real-time position state recovery
- Strategy execution audit trail
- Leg-specific performance metrics
- Daily performance summary
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum

from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, Text, ForeignKey, Index, event
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
    db_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'db', 'nifty_sehwag.db')
    DATABASE_URL = f'sqlite:///{db_path}'
    logger.info(f"ðŸ“ Using default SQLite database: {db_path}")

# Conditionally create engine based on DB type
if 'sqlite' in DATABASE_URL:
    engine = create_engine(
        DATABASE_URL,
        poolclass=NullPool,
        connect_args={
            'check_same_thread': False,
            'timeout': 30,  # 30 second timeout for locks
            'isolation_level': None  # Autocommit mode
        }
    )

    # Enable WAL mode for better concurrent access
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")  # 30 seconds in milliseconds
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.close()
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

class LegStatus(str, Enum):
    """Status of a leg in the strategy"""
    WAITING = "WAITING"           # Waiting for entry conditions
    ENTRY_TRIGGERED = "ENTRY_TRIGGERED"  # Entry signal received, waiting for Wait & Trade
    PENDING_ENTRY = "PENDING_ENTRY"      # Waiting for order confirmation
    ACTIVE = "ACTIVE"             # Position is open and being managed
    SL_HIT = "SL_HIT"             # Stop loss was hit
    PROFIT_TARGET_HIT = "PROFIT_TARGET_HIT"  # Profit target reached
    MANUALLY_EXITED = "MANUALLY_EXITED"  # Manually closed
    EXPIRED = "EXPIRED"           # Position expired (end of day)
    RECOVERED = "RECOVERED"       # Position recovered from crash


class OrderStatus(str, Enum):
    """Status of an order"""
    PENDING = "PENDING"
    ACCEPTED = "ACCEPTED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"
    EXECUTED = "EXECUTED"
    PARTIAL = "PARTIAL"


class StrategyEventType(str, Enum):
    """Types of strategy events"""
    STRATEGY_START = "STRATEGY_START"
    STRATEGY_STOP = "STRATEGY_STOP"
    ENTRY_CONDITION_MET = "ENTRY_CONDITION_MET"
    WAIT_TRADE_CONFIRMED = "WAIT_TRADE_CONFIRMED"
    WAIT_TRADE_FAILED = "WAIT_TRADE_FAILED"
    ORDER_PLACED = "ORDER_PLACED"
    ORDER_EXECUTED = "ORDER_EXECUTED"
    POSITION_ACTIVE = "POSITION_ACTIVE"
    SL_UPDATED = "SL_UPDATED"
    PROFIT_LOCK_UPDATED = "PROFIT_LOCK_UPDATED"
    EXIT_EXECUTED = "EXIT_EXECUTED"
    CRASH_DETECTED = "CRASH_DETECTED"
    ERROR = "ERROR"


# ==================== DATABASE MODELS ====================

class NiftySehwagSession(Base):
    """Represents a strategy execution session"""
    __tablename__ = 'nifty_sehwag_sessions'
    
    id = Column(Integer, primary_key=True)
    session_date = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    session_id = Column(String(50), unique=True, nullable=False)  # Unique session identifier
    expiry_date = Column(String(10), nullable=False)  # YYYY-MM-DD format
    status = Column(String(20), default='RUNNING')  # RUNNING, COMPLETED, ERROR, INTERRUPTED
    
    # Performance metrics
    total_orders_placed = Column(Integer, default=0)
    total_orders_executed = Column(Integer, default=0)
    total_orders_rejected = Column(Integer, default=0)
    net_pnl = Column(Float, default=0.0)
    total_legs_opened = Column(Integer, default=0)
    total_legs_closed = Column(Integer, default=0)
    
    # Session metadata
    notes = Column(Text)
    start_time = Column(DateTime(timezone=True))
    end_time = Column(DateTime(timezone=True))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    positions = relationship("NiftySehwagPosition", back_populates="session", cascade="all, delete-orphan")
    orders = relationship("NiftySehwagOrder", back_populates="session", cascade="all, delete-orphan")
    events = relationship("NiftySehwagEvent", back_populates="session", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_session_date', 'session_date'),
        Index('idx_expiry_date', 'expiry_date'),
    )


class NiftySehwagPosition(Base):
    """Represents a leg position in the strategy"""
    __tablename__ = 'nifty_sehwag_positions'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('nifty_sehwag_sessions.id'), nullable=False)
    leg_number = Column(Integer, nullable=False)  # 1, 2, or 3
    symbol = Column(String(50), nullable=False)  # e.g., NIFTY23N3624700CE
    entry_time = Column(DateTime(timezone=True))
    exit_time = Column(DateTime(timezone=True))
    
    # Entry details
    entry_price = Column(Float)
    entry_quantity = Column(Integer)
    entry_order_id = Column(String(50))
    atm_strike = Column(Integer)  # ATM strike used
    itm_level = Column(Integer)  # ITM level applied (ITM3, ITM4)
    
    # Exit details
    exit_price = Column(Float)
    exit_quantity = Column(Integer)
    exit_order_id = Column(String(50))
    exit_reason = Column(String(50))  # SL_HIT, PROFIT_TARGET, MANUAL, EXPIRED
    
    # Position management
    status = Column(String(20), default='WAITING')  # WAITING, ACTIVE, CLOSED, etc.
    current_sl = Column(Float)  # Current stop loss
    lock_profit = Column(Float, default=0.0)  # Lock profit level
    current_price = Column(Float)  # Last known price
    
    # PnL tracking
    entry_pnl = Column(Float, default=0.0)
    exit_pnl = Column(Float, default=0.0)
    realized_pnl = Column(Float, default=0.0)
    unrealized_pnl = Column(Float, default=0.0)
    pnl_percentage = Column(Float, default=0.0)
    
    # Configuration used
    sl_percentage = Column(Float)  # SL % used
    profit_target = Column(Float)  # Profit target used
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    session = relationship("NiftySehwagSession", back_populates="positions")
    snapshots = relationship("NiftySehwagPositionSnapshot", back_populates="position", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_session_id', 'session_id'),
        Index('idx_leg_number', 'leg_number'),
        Index('idx_symbol', 'symbol'),
    )


class NiftySehwagPositionSnapshot(Base):
    """Snapshot of position state at key moments"""
    __tablename__ = 'nifty_sehwag_position_snapshots'
    
    id = Column(Integer, primary_key=True)
    position_id = Column(Integer, ForeignKey('nifty_sehwag_positions.id'), nullable=False)
    timestamp = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    event_type = Column(String(50))  # ENTRY, SL_UPDATE, PROFIT_UPDATE, EXIT, etc.
    
    # Position state at this moment
    current_price = Column(Float)
    current_sl = Column(Float)
    lock_profit = Column(Float)
    unrealized_pnl = Column(Float)
    pnl_percentage = Column(Float)
    
    # Additional details
    notes = Column(Text)
    
    # Relationships
    position = relationship("NiftySehwagPosition", back_populates="snapshots")
    
    __table_args__ = (
        Index('idx_position_id', 'position_id'),
        Index('idx_timestamp', 'timestamp'),
    )


class NiftySehwagOrder(Base):
    """Order placed for the strategy"""
    __tablename__ = 'nifty_sehwag_orders'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('nifty_sehwag_sessions.id'), nullable=False)
    order_id = Column(String(50), unique=True)  # Broker order ID
    leg_number = Column(Integer)  # 1, 2, or 3
    order_type = Column(String(20))  # ENTRY, EXIT, SL_ORDER
    
    # Order details
    symbol = Column(String(50), nullable=False)
    exchange = Column(String(10), default='NFO')
    side = Column(String(10))  # BUY, SELL
    quantity = Column(Integer)
    price = Column(Float)
    order_time = Column(DateTime(timezone=True))
    
    # Order status
    status = Column(String(20), default='PENDING')
    execution_price = Column(Float)
    executed_quantity = Column(Integer)
    execution_time = Column(DateTime(timezone=True))
    
    # Error tracking
    error_message = Column(Text)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    session = relationship("NiftySehwagSession", back_populates="orders")
    
    __table_args__ = (
        Index('idx_session_id', 'session_id'),
        Index('idx_order_id', 'order_id'),
        Index('idx_symbol', 'symbol'),
    )


class NiftySehwagEvent(Base):
    """Strategy execution event log"""
    __tablename__ = 'nifty_sehwag_events'
    
    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey('nifty_sehwag_sessions.id'), nullable=False)
    event_time = Column(DateTime(timezone=True), default=func.now(), nullable=False)
    event_type = Column(String(50), nullable=False)  # ENTRY_CONDITION_MET, WAIT_TRADE_CONFIRMED, etc.
    
    # Event details
    leg_number = Column(Integer)
    symbol = Column(String(50))
    description = Column(Text)
    data = Column(Text)  # JSON data
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    session = relationship("NiftySehwagSession", back_populates="events")
    
    __table_args__ = (
        Index('idx_session_id', 'session_id'),
        Index('idx_event_time', 'event_time'),
        Index('idx_event_type', 'event_type'),
    )


# ==================== DATABASE OPERATIONS ====================

def init_db():
    """Initialize the database"""
    try:
        Base.metadata.create_all(bind=engine, checkfirst=True)
        logger.info(f"âœ… Nifty Sehwag database initialized successfully")
    except Exception as e:
        logger.error(f"âŒ Failed to initialize Nifty Sehwag database: {e}")
        raise


# ==================== SESSION OPERATIONS ====================

def create_session(session_id: str, expiry_date: str, notes: str = None) -> Optional[NiftySehwagSession]:
    """Create a new strategy session with retry logic for database locks"""
    import time

    # Ensure database is initialized
    ensure_db_initialized()

    max_retries = 3
    retry_delay = 1  # seconds

    for attempt in range(max_retries):
        try:
            session = NiftySehwagSession(
                session_id=session_id,
                expiry_date=expiry_date,
                status='RUNNING',
                start_time=datetime.now(),
                notes=notes
            )
            db_session.add(session)
            db_session.commit()
            logger.info(f"✅ Created session: {session_id}")
            return session
        except Exception as e:
            db_session.rollback()

            # Check if it's a database lock error
            if 'database is locked' in str(e).lower() and attempt < max_retries - 1:
                logger.warning(f"⚠️  Database locked, retrying in {retry_delay}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
                continue

            logger.error(f"❌ Error creating session: {e}")
            return None

    return None


def get_session(session_id: str) -> Optional[NiftySehwagSession]:
    """Get session by ID"""
    try:
        return db_session.query(NiftySehwagSession).filter_by(session_id=session_id).first()
    except Exception as e:
        logger.error(f"âŒ Error fetching session: {e}")
        return None


def update_session_status(session_id: str, status: str, notes: str = None) -> bool:
    """Update session status"""
    try:
        session = get_session(session_id)
        if session:
            session.status = status
            if notes:
                session.notes = (session.notes or '') + f"\n{notes}"
            if status == 'COMPLETED':
                session.end_time = datetime.now()
            db_session.commit()
            logger.info(f"âœ… Updated session {session_id} status to {status}")
            return True
        return False
    except Exception as e:
        logger.error(f"âŒ Error updating session: {e}")
        db_session.rollback()
        return False


# ==================== POSITION OPERATIONS ====================

def create_position(session_id: str, leg_number: int, symbol: str, atm_strike: int,
                   strike: int, option_type: str, entry_price: float = None,
                   quantity: int = None, initial_sl: float = None) -> Optional[int]:
    """Create a new position record and return position ID"""
    try:
        session = get_session(session_id)
        if not session:
            logger.error(f"❌ Session not found: {session_id}")
            return None
        
        # Calculate itm_level from strike and atm_strike difference
        strike_diff = 50  # NIFTY default
        itm_level = abs(strike - atm_strike) // strike_diff

        position = NiftySehwagPosition(
            session_id=session.id,
            leg_number=leg_number,
            symbol=symbol,
            status='ENTERED' if entry_price else 'WAITING',
            atm_strike=atm_strike,
            itm_level=itm_level,
            entry_price=entry_price,
            entry_quantity=quantity,  # Database column is entry_quantity, not quantity
            sl_percentage=None,  # Can be updated later
            profit_target=None   # Can be updated later
        )
        db_session.add(position)
        db_session.commit()
        logger.info(f"✅ Created position: Leg {leg_number} - {symbol} (ID: {position.id})")
        return position.id
    except Exception as e:
        logger.error(f"❌ Error creating position: {e}")
        db_session.rollback()
        return None


def update_position_entry(position_id: int, entry_time: datetime, entry_price: float, 
                         entry_quantity: int, entry_order_id: str, current_sl: float,
                         lock_profit: float = 0.0) -> bool:
    """Update position with entry details"""
    try:
        position = db_session.query(NiftySehwagPosition).get(position_id)
        if position:
            position.entry_time = entry_time
            position.entry_price = entry_price
            position.entry_quantity = entry_quantity
            position.entry_order_id = entry_order_id
            position.status = 'ACTIVE'
            position.current_sl = current_sl
            position.lock_profit = lock_profit
            position.current_price = entry_price
            db_session.commit()
            logger.info(f"âœ… Updated position {position_id} with entry details")
            return True
        return False
    except Exception as e:
        logger.error(f"âŒ Error updating position entry: {e}")
        db_session.rollback()
        return False


def update_position_sl_and_profit(position_id: int, current_sl: float, lock_profit: float) -> bool:
    """Update position SL and lock profit"""
    try:
        position = db_session.query(NiftySehwagPosition).get(position_id)
        if position:
            position.current_sl = current_sl
            position.lock_profit = lock_profit
            db_session.commit()
            
            # Create snapshot
            create_position_snapshot(
                position_id=position_id,
                event_type='SL_UPDATED' if current_sl != position.current_sl else 'PROFIT_UPDATED',
                current_price=position.current_price,
                current_sl=current_sl,
                lock_profit=lock_profit,
                unrealized_pnl=position.unrealized_pnl,
                pnl_percentage=position.pnl_percentage
            )
            return True
        return False
    except Exception as e:
        logger.error(f"âŒ Error updating position SL/Profit: {e}")
        db_session.rollback()
        return False


def update_position_exit(position_id: int, exit_time: datetime, exit_price: float, 
                        exit_quantity: int, exit_order_id: str, exit_reason: str,
                        realized_pnl: float, pnl_percentage: float) -> bool:
    """Update position with exit details"""
    try:
        position = db_session.query(NiftySehwagPosition).get(position_id)
        if position:
            position.exit_time = exit_time
            position.exit_price = exit_price
            position.exit_quantity = exit_quantity
            position.exit_order_id = exit_order_id
            position.exit_reason = exit_reason
            position.status = 'CLOSED'
            position.realized_pnl = realized_pnl
            position.pnl_percentage = pnl_percentage
            position.unrealized_pnl = 0.0
            db_session.commit()
            logger.info(f"âœ… Updated position {position_id} with exit details (PnL: {realized_pnl:.2f})")
            return True
        return False
    except Exception as e:
        logger.error(f"âŒ Error updating position exit: {e}")
        db_session.rollback()
        return False


def update_position_price(position_id: int, current_price: float, unrealized_pnl: float, 
                         pnl_percentage: float) -> bool:
    """Update position with current price and PnL"""
    try:
        position = db_session.query(NiftySehwagPosition).get(position_id)
        if position:
            position.current_price = current_price
            position.unrealized_pnl = unrealized_pnl
            position.pnl_percentage = pnl_percentage
            db_session.commit()
            return True
        return False
    except Exception as e:
        logger.error(f"âŒ Error updating position price: {e}")
        db_session.rollback()
        return False


def create_position_snapshot(position_id: int, event_type: str, current_price: float,
                            current_sl: float, lock_profit: float, unrealized_pnl: float,
                            pnl_percentage: float, notes: str = None) -> Optional[NiftySehwagPositionSnapshot]:
    """Create a position snapshot"""
    try:
        snapshot = NiftySehwagPositionSnapshot(
            position_id=position_id,
            event_type=event_type,
            current_price=current_price,
            current_sl=current_sl,
            lock_profit=lock_profit,
            unrealized_pnl=unrealized_pnl,
            pnl_percentage=pnl_percentage,
            notes=notes
        )
        db_session.add(snapshot)
        db_session.commit()
        return snapshot
    except Exception as e:
        logger.error(f"âŒ Error creating position snapshot: {e}")
        db_session.rollback()
        return None


# ==================== ORDER OPERATIONS ====================

def create_order(session_id: str, order_type: str, symbol: str, side: str, quantity: int,
                price: float, leg_number: int = None, exchange: str = 'NFO') -> Optional[NiftySehwagOrder]:
    """Create an order record"""
    try:
        session = get_session(session_id)
        if not session:
            logger.error(f"âŒ Session not found: {session_id}")
            return None
        
        order = NiftySehwagOrder(
            session_id=session.id,
            order_type=order_type,
            symbol=symbol,
            exchange=exchange,
            side=side,
            quantity=quantity,
            price=price,
            leg_number=leg_number,
            status='PENDING',
            order_time=datetime.now()
        )
        db_session.add(order)
        db_session.commit()
        logger.info(f"âœ… Created order: {symbol} {side} {quantity} @ {price}")
        return order
    except Exception as e:
        logger.error(f"âŒ Error creating order: {e}")
        db_session.rollback()
        return None


def update_order_execution(order_id: int, broker_order_id: str, status: str,
                          execution_price: float, executed_quantity: int) -> bool:
    """Update order with execution details"""
    try:
        order = db_session.query(NiftySehwagOrder).get(order_id)
        if order:
            order.order_id = broker_order_id
            order.status = status
            order.execution_price = execution_price
            order.executed_quantity = executed_quantity
            order.execution_time = datetime.now()
            db_session.commit()
            logger.info(f"âœ… Updated order {order_id}: {status} @ {execution_price}")
            return True
        return False
    except Exception as e:
        logger.error(f"âŒ Error updating order: {e}")
        db_session.rollback()
        return False


def update_order_error(order_id: int, error_message: str) -> bool:
    """Update order with error details"""
    try:
        order = db_session.query(NiftySehwagOrder).get(order_id)
        if order:
            order.status = 'REJECTED'
            order.error_message = error_message
            db_session.commit()
            logger.error(f"âŒ Order {order_id} rejected: {error_message}")
            return True
        return False
    except Exception as e:
        logger.error(f"âŒ Error updating order error: {e}")
        db_session.rollback()
        return False


# ==================== EVENT OPERATIONS ====================

def create_event(session_id: str, event_type: str, description: str = None, 
                leg_number: int = None, symbol: str = None, data: Dict = None) -> Optional[NiftySehwagEvent]:
    """Create a strategy event"""
    try:
        session = get_session(session_id)
        if not session:
            logger.error(f"âŒ Session not found: {session_id}")
            return None
        
        event = NiftySehwagEvent(
            session_id=session.id,
            event_type=event_type,
            leg_number=leg_number,
            symbol=symbol,
            description=description,
            data=json.dumps(data) if data else None
        )
        db_session.add(event)
        db_session.commit()
        return event
    except Exception as e:
        logger.error(f"âŒ Error creating event: {e}")
        db_session.rollback()
        return None


# ==================== REPORTING OPERATIONS ====================

def get_session_summary(session_id: str) -> Optional[Dict]:
    """Get summary of a session"""
    try:
        session = get_session(session_id)
        if not session:
            return None
        
        positions = db_session.query(NiftySehwagPosition).filter_by(session_id=session.id).all()
        total_pnl = sum(p.realized_pnl or 0.0 for p in positions)
        
        summary = {
            'session_id': session.session_id,
            'expiry_date': session.expiry_date,
            'status': session.status,
            'start_time': session.start_time.isoformat() if session.start_time else None,
            'end_time': session.end_time.isoformat() if session.end_time else None,
            'total_positions': len(positions),
            'closed_positions': sum(1 for p in positions if p.status == 'CLOSED'),
            'active_positions': sum(1 for p in positions if p.status == 'ACTIVE'),
            'total_orders': session.total_orders_executed,
            'net_pnl': total_pnl,
            'positions': [
                {
                    'leg': p.leg_number,
                    'symbol': p.symbol,
                    'entry_price': p.entry_price,
                    'exit_price': p.exit_price,
                    'realized_pnl': p.realized_pnl,
                    'pnl_percentage': p.pnl_percentage,
                    'status': p.status
                }
                for p in positions
            ]
        }
        return summary
    except Exception as e:
        logger.error(f"âŒ Error generating session summary: {e}")
        return None


def get_daily_performance(expiry_date: str) -> Optional[Dict]:
    """Get daily performance for an expiry date"""
    try:
        sessions = db_session.query(NiftySehwagSession).filter_by(expiry_date=expiry_date).all()
        
        if not sessions:
            return None
        
        total_pnl = 0.0
        total_positions = 0
        total_orders = 0
        
        for session in sessions:
            positions = db_session.query(NiftySehwagPosition).filter_by(session_id=session.id).all()
            total_pnl += sum(p.realized_pnl or 0.0 for p in positions)
            total_positions += len(positions)
            total_orders += session.total_orders_executed or 0
        
        return {
            'expiry_date': expiry_date,
            'num_sessions': len(sessions),
            'total_positions': total_positions,
            'total_orders': total_orders,
            'net_pnl': total_pnl,
            'sessions': [s.session_id for s in sessions]
        }
    except Exception as e:
        logger.error(f"âŒ Error generating daily performance: {e}")
        return None




# ==================== FUNCTION ALIASES ====================
# For backward compatibility with persistence manager imports

def log_position(session_id: str, leg_number: int, symbol: str, atm_strike: int,
                 strike: int, option_type: str, entry_price: float = None,
                 quantity: int = None, initial_sl: float = None) -> Optional[int]:
    """Alias for create_position - logs a new position"""
    return create_position(session_id, leg_number, symbol, atm_strike,
                          strike, option_type, entry_price, quantity, initial_sl)


def update_position_status(position_id: int, status: str, exit_price: float = None,
                           exit_time: datetime = None, realized_pnl: float = None,
                           pnl_percentage: float = None) -> bool:
    """Alias for update_position_exit - updates position status and exit details"""
    if exit_time is None:
        exit_time = datetime.now()
    # update_position_exit(position_id, exit_time, exit_price, exit_quantity, exit_order_id, exit_reason, realized_pnl, pnl_percentage)
    return update_position_exit(
        position_id=position_id,
        exit_time=exit_time,
        exit_price=exit_price or 0.0,
        exit_quantity=0,  # Not tracked in this alias
        exit_order_id="",  # Not tracked in this alias
        exit_reason=status,
        realized_pnl=realized_pnl or 0.0,
        pnl_percentage=pnl_percentage or 0.0
    )


def log_order(session_id: str, order_type: str, symbol: str, side: str,
              quantity: int, price: float = None, leg_number: int = None) -> Optional[int]:
    """Alias for create_order - logs a new order and returns order ID"""
    # create_order(session_id, order_type, symbol, side, quantity, price, leg_number, exchange)
    order = create_order(
        session_id=session_id,
        order_type=order_type,
        symbol=symbol,
        side=side,
        quantity=quantity,
        price=price or 0.0,
        leg_number=leg_number
    )
    return order.id if order else None


def update_order_status(order_id: int, status: str, broker_order_id: str = None,
                       executed_price: float = None, executed_quantity: int = None,
                       executed_time: datetime = None) -> bool:
    """Alias for update_order_execution - updates order status (executed_time is ignored, function sets it automatically)"""
    # update_order_execution(order_id, broker_order_id, status, execution_price, executed_quantity)
    return update_order_execution(
        order_id=order_id,
        broker_order_id=broker_order_id or "",
        status=status,
        execution_price=executed_price or 0.0,
        executed_quantity=executed_quantity or 0
    )


def log_event(session_id: str, event_type: str, description: str = None,
              metadata: Dict = None) -> Optional[int]:
    """Alias for create_event - logs a strategy event"""
    return create_event(session_id, event_type, description, data=metadata)

# Don't initialize tables on import - do it lazily when first needed
# This prevents database lock errors during module import
_db_initialized = False

def ensure_db_initialized():
    """Ensure database is initialized (lazy initialization)"""
    global _db_initialized
    if not _db_initialized:
        try:
            init_db()
            _db_initialized = True
        except Exception as e:
            logger.warning(f"⚠️  Could not initialize DB tables: {e}")
            # Don't fail - allow strategy to continue without persistence
