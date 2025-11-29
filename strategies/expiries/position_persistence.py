"""
Position Persistence Manager
============================
Manages position state persistence with batch writes and crash recovery.

Usage:
    persistence = PositionPersistenceManager(
        strategy_name='expiry_blast',
        api_client=client,
        batch_interval=10  # seconds
    )
    
    # On strategy start
    persistence.start_session(session_id, underlying)
    
    # On position entry
    persistence.save_position_entry(position_id, symbol, entry_price, ...)
    
    # Periodically during monitoring
    persistence.update_position_state(position_id, current_price, stop_price, ...)
    
    # On position exit
    persistence.save_position_exit(position_id, exit_price, exit_reason, ...)
    
    # On crash
    persistence.handle_crash(session_id, error, reason)
    
    # On recovery/next startup
    crashed_trades = persistence.get_crashed_positions()
"""

import uuid
import json
import logging
import queue
import threading
from datetime import datetime
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, asdict
import time

from database.position_persistence_db import (
    db_session,
    init_position_persistence_db,
    StrategySession,
    StrategyPosition,
    PositionEvent,
    SessionStatus,
    PositionStatus,
    EventType,
    get_active_crashed_positions,
    mark_session_crashed,
    get_session_stats,
)

logger = logging.getLogger(__name__)


@dataclass
class PositionSnapshot:
    """Current snapshot of position state"""
    position_id: str
    symbol: str
    leg_type: str
    strike: float
    entry_price: float
    entry_time: datetime
    current_price: float
    stop_price: float
    highest_price: float
    highest_high_breakout: float
    last_trail_level: int
    profit_percent: float
    was_crashed: bool = False


@dataclass
class CrashedPosition:
    """Position that needs recovery"""
    session_id: str
    position_id: str
    symbol: str
    leg_type: str
    entry_price: float
    entry_time: datetime
    stop_price: float
    highest_price: float
    highest_high_breakout: float
    last_trail_level: int
    crash_reason: str
    crash_timestamp: datetime


class PositionPersistenceManager:
    """
    Manages position persistence with batch writes and crash recovery.
    
    Features:
    - Batched DB writes to minimize performance overhead
    - Crash detection and recovery
    - Position state snapshots
    - Audit trail of all events
    - Background batch writer thread
    """
    
    def __init__(
        self,
        strategy_name: str,
        api_client: Optional[Any] = None,
        batch_interval: int = 10,
        enable_async: bool = False,
        logger_instance: Optional[logging.Logger] = None
    ):
        """
        Initialize the persistence manager.
        
        Args:
            strategy_name: Name of the strategy (e.g., 'expiry_blast')
            api_client: OpenAlgo API client for position verification
            batch_interval: Seconds between batch writes (default 10)
            enable_async: Enable async batch writer thread
            logger_instance: Optional custom logger
        """
        self.strategy_name = strategy_name
        self.api_client = api_client
        self.batch_interval = batch_interval
        self.enable_async = enable_async
        
        if logger_instance:
            self.logger = logger_instance
        else:
            self.logger = logger
        
        # Initialize database
        init_position_persistence_db()
        
        # Session management
        self.current_session_id: Optional[str] = None
        self.current_session_db_id: Optional[int] = None
        
        # Batch write queue and buffer
        self.update_queue: queue.Queue = queue.Queue()
        self.last_batch_time: Dict[str, float] = {}  # Track last write time per position
        
        # Background writer thread
        self.batch_writer_thread: Optional[threading.Thread] = None
        self.is_running = False
        
        # Position cache for quick access
        self.position_cache: Dict[str, PositionSnapshot] = {}
        
        if enable_async:
            self._start_async_batch_writer()
    
    def _start_async_batch_writer(self):
        """Start background batch writer thread"""
        self.is_running = True
        self.batch_writer_thread = threading.Thread(
            target=self._batch_writer_loop,
            daemon=True,
            name=f"{self.strategy_name}_batch_writer"
        )
        self.batch_writer_thread.start()
        self.logger.info(f"✓ Async batch writer started for {self.strategy_name}")
    
    def _batch_writer_loop(self):
        """Background thread that processes batched updates"""
        while self.is_running:
            try:
                # Process all queued updates
                updates = []
                try:
                    while True:
                        updates.append(self.update_queue.get_nowait())
                except queue.Empty:
                    pass
                
                if updates:
                    self._flush_updates(updates)
                
                time.sleep(1)  # Check queue every second
            
            except Exception as e:
                self.logger.error(f"✗ Error in batch writer: {e}")
                time.sleep(2)
    
    def stop(self):
        """Stop the persistence manager"""
        self.is_running = False
        if self.batch_writer_thread:
            self.batch_writer_thread.join(timeout=5)
            self.logger.info(f"✓ Batch writer stopped")
    
    # ==================== SESSION MANAGEMENT ====================
    
    def start_session(self, underlying: str) -> str:
        """
        Start a new strategy session.
        
        Returns:
            session_id: UUID of the new session
        """
        try:
            self.current_session_id = str(uuid.uuid4())
            
            session = StrategySession(
                session_id=self.current_session_id,
                strategy_name=self.strategy_name,
                underlying=underlying,
                status=SessionStatus.RUNNING.value
            )
            
            db_session.add(session)
            db_session.commit()
            
            self.current_session_db_id = session.id
            
            self.logger.info(f"✓ Session started: {self.current_session_id} ({self.strategy_name} - {underlying})")
            return self.current_session_id
        
        except Exception as e:
            self.logger.error(f"✗ Error starting session: {e}")
            db_session.rollback()
            raise
    
    def end_session(self, status: str = SessionStatus.COMPLETED.value):
        """End the current session"""
        try:
            if not self.current_session_id:
                self.logger.warning("⚠️  No active session to end")
                return
            
            session = db_session.query(StrategySession).filter(
                StrategySession.session_id == self.current_session_id
            ).first()
            
            if session:
                session.end_time = datetime.now()
                session.status = status
                db_session.commit()
                
                self.logger.info(f"✓ Session ended: {self.current_session_id} (Status: {status})")
                stats = get_session_stats(self.current_session_id)
                self.logger.info(f"📊 Session Stats: {stats}")
        
        except Exception as e:
            self.logger.error(f"✗ Error ending session: {e}")
            db_session.rollback()
    
    # ==================== POSITION ENTRY ====================
    
    def save_position_entry(
        self,
        symbol: str,
        leg_type: str,
        strike: float,
        entry_price: float,
        highest_high_breakout: float,
        entry_quantity: int = 1,
        entry_order_id: Optional[str] = None,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Save a new position entry.
        
        Returns:
            position_id: UUID of the position
        """
        try:
            if not self.current_session_db_id:
                raise RuntimeError("No active session. Call start_session() first.")
            
            position_id = str(uuid.uuid4())
            
            position = StrategyPosition(
                position_id=position_id,
                session_id=self.current_session_db_id,
                symbol=symbol,
                leg_type=leg_type,
                strike=strike,
                underlying=self.strategy_name.replace('_', '-'),
                entry_price=entry_price,
                entry_time=datetime.now(),
                entry_order_id=entry_order_id,
                current_price=entry_price,
                stop_price=entry_price * 0.95,  # 5% initial stop
                highest_price=entry_price,
                highest_high_breakout=highest_high_breakout,
                last_trail_level=0,
                profit_percent=0.0,
                status=PositionStatus.ACTIVE.value,
                was_crashed=False
            )
            
            db_session.add(position)
            db_session.commit()
            
            # Create entry event
            self._log_event(
                EventType.ENTRY,
                position_id=position_id,
                summary=f"Entry: {symbol} @ {entry_price:.2f}",
                data={
                    'symbol': symbol,
                    'leg_type': leg_type,
                    'strike': strike,
                    'entry_price': entry_price,
                    'entry_order_id': entry_order_id,
                }
            )
            
            # Cache position
            self.position_cache[position_id] = PositionSnapshot(
                position_id=position_id,
                symbol=symbol,
                leg_type=leg_type,
                strike=strike,
                entry_price=entry_price,
                entry_time=position.entry_time,
                current_price=entry_price,
                stop_price=position.stop_price,
                highest_price=entry_price,
                highest_high_breakout=highest_high_breakout,
                last_trail_level=0,
                profit_percent=0.0,
            )
            
            self.logger.info(f"✓ Position entry saved: {position_id} ({symbol})")
            return position_id
        
        except Exception as e:
            self.logger.error(f"✗ Error saving position entry: {e}")
            db_session.rollback()
            raise
    
    # ==================== POSITION UPDATES (BATCHED) ====================
    
    def queue_position_update(
        self,
        position_id: str,
        current_price: float,
        stop_price: float,
        highest_price: float,
        last_trail_level: int,
        profit_percent: float,
        force_immediate: bool = False
    ):
        """
        Queue a position update for batched writing.
        
        By default, updates are batched to reduce DB writes.
        Set force_immediate=True for critical updates like trailing stops.
        """
        try:
            update = {
                'position_id': position_id,
                'current_price': current_price,
                'stop_price': stop_price,
                'highest_price': highest_price,
                'last_trail_level': last_trail_level,
                'profit_percent': profit_percent,
                'timestamp': time.time(),
            }
            
            if self.enable_async:
                self.update_queue.put(update)
            else:
                # For sync mode, check if we should flush
                current_time = time.time()
                pos_last_time = self.last_batch_time.get(position_id, 0)
                
                if force_immediate or (current_time - pos_last_time) >= self.batch_interval:
                    self._flush_position_update(position_id, update)
                    self.last_batch_time[position_id] = current_time
        
        except Exception as e:
            self.logger.error(f"✗ Error queuing position update: {e}")
    
    def _flush_position_update(self, position_id: str, update: Dict):
        """Flush a single position update to database"""
        try:
            position = db_session.query(StrategyPosition).filter(
                StrategyPosition.position_id == position_id
            ).first()
            
            if not position:
                self.logger.warning(f"⚠️  Position not found: {position_id}")
                return
            
            position.current_price = update['current_price']
            position.stop_price = update['stop_price']
            position.highest_price = update['highest_price']
            position.last_trail_level = update['last_trail_level']
            position.profit_percent = update['profit_percent']
            position.updated_at = datetime.now()
            
            db_session.commit()
            
            # Update cache
            if position_id in self.position_cache:
                snap = self.position_cache[position_id]
                snap.current_price = update['current_price']
                snap.stop_price = update['stop_price']
                snap.highest_price = update['highest_price']
                snap.last_trail_level = update['last_trail_level']
                snap.profit_percent = update['profit_percent']
        
        except Exception as e:
            self.logger.error(f"✗ Error flushing position update: {e}")
            db_session.rollback()
    
    def _flush_updates(self, updates: List[Dict]):
        """Flush multiple batched updates to database"""
        if not updates:
            return
        
        try:
            # Group updates by position
            updates_by_pos = {}
            for update in updates:
                pos_id = update['position_id']
                if pos_id not in updates_by_pos:
                    updates_by_pos[pos_id] = []
                updates_by_pos[pos_id].append(update)
            
            # Write the latest update for each position
            for pos_id, pos_updates in updates_by_pos.items():
                latest = max(pos_updates, key=lambda x: x['timestamp'])
                self._flush_position_update(pos_id, latest)
        
        except Exception as e:
            self.logger.error(f"✗ Error flushing batch updates: {e}")
    
    # ==================== POSITION EXIT ====================
    
    def save_position_exit(
        self,
        position_id: str,
        exit_price: float,
        exit_reason: str,
        exit_order_id: Optional[str] = None,
        profit_loss: Optional[float] = None
    ):
        """
        Save position exit (close).
        
        Args:
            position_id: UUID of position to close
            exit_price: Price at which position was exited
            exit_reason: Reason for exit (PROFIT_TARGET, STOP_LOSS, ATM_CHANGE, etc.)
            exit_order_id: Optional broker order ID
            profit_loss: Optional profit/loss amount
        """
        try:
            position = db_session.query(StrategyPosition).filter(
                StrategyPosition.position_id == position_id
            ).first()
            
            if not position:
                self.logger.warning(f"⚠️  Position not found for exit: {position_id}")
                return
            
            position.exit_price = exit_price
            position.exit_time = datetime.now()
            position.exit_reason = exit_reason
            position.exit_order_id = exit_order_id
            position.status = PositionStatus.CLOSED.value
            position.updated_at = datetime.now()
            
            db_session.commit()
            
            # Create exit event
            self._log_event(
                EventType.EXIT,
                position_id=position_id,
                summary=f"Exit: {position.symbol} @ {exit_price:.2f} ({exit_reason})",
                data={
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'profit_loss': profit_loss,
                    'exit_order_id': exit_order_id,
                }
            )
            
            self.logger.info(f"✓ Position exit saved: {position_id} (Reason: {exit_reason})")
        
        except Exception as e:
            self.logger.error(f"✗ Error saving position exit: {e}")
            db_session.rollback()
    
    # ==================== CRASH HANDLING ====================
    
    def handle_crash(
        self,
        error: str,
        reason: str = None,
        traceback_str: str = None
    ):
        """
        Handle strategy crash.
        
        Args:
            error: Error message
            reason: Human-readable reason
            traceback_str: Optional full traceback
        """
        try:
            if not self.current_session_id:
                self.logger.warning("⚠️  No active session to mark as crashed")
                return
            
            full_error = error
            if traceback_str:
                full_error = f"{error}\n\nTraceback:\n{traceback_str}"
            
            mark_session_crashed(
                self.current_session_id,
                full_error,
                reason or "Unexpected crash"
            )
            
            self.logger.error(f"🔴 CRASH HANDLED: {reason or error}")
        
        except Exception as e:
            self.logger.error(f"✗ Error handling crash: {e}")
    
    # ==================== CRASH RECOVERY ====================
    
    def get_crashed_positions(self, underlying: str = None) -> List[CrashedPosition]:
        """
        Get all positions that crashed and need recovery.
        
        Returns:
            List of CrashedPosition objects ready for recovery
        """
        try:
            underlying = underlying or self.strategy_name
            crashed = get_active_crashed_positions(self.strategy_name, underlying)
            
            result = []
            for pos_data in crashed:
                crashed_pos = CrashedPosition(
                    session_id=pos_data['session_id'],
                    position_id=pos_data['position_id'],
                    symbol=pos_data['symbol'],
                    leg_type=pos_data['leg_type'],
                    entry_price=pos_data['entry_price'],
                    entry_time=datetime.fromisoformat(pos_data['entry_time']) if pos_data['entry_time'] else None,
                    stop_price=pos_data['stop_price'],
                    highest_price=pos_data['highest_price'],
                    highest_high_breakout=pos_data['highest_high_breakout'],
                    last_trail_level=pos_data['last_trail_level'],
                    crash_reason=pos_data['crash_reason'],
                    crash_timestamp=datetime.fromisoformat(pos_data['crash_timestamp']) if pos_data['crash_timestamp'] else None,
                )
                result.append(crashed_pos)
            
            if result:
                self.logger.warning(f"⚠️  Found {len(result)} crashed positions to recover")
                for pos in result:
                    self.logger.warning(f"   - {pos.symbol} @ {pos.entry_price:.2f} (crashed: {pos.crash_timestamp})")
            
            return result
        
        except Exception as e:
            self.logger.error(f"✗ Error getting crashed positions: {e}")
            return []
    
    def mark_position_recovered(self, position_id: str, recovery_notes: str = None):
        """Mark a crashed position as recovered and resumed"""
        try:
            position = db_session.query(StrategyPosition).filter(
                StrategyPosition.position_id == position_id
            ).first()
            
            if not position:
                self.logger.warning(f"⚠️  Position not found: {position_id}")
                return
            
            position.status = PositionStatus.RECOVERED.value
            position.crash_recovery_count += 1
            position.updated_at = datetime.now()
            db_session.commit()
            
            self._log_event(
                EventType.RECOVERY,
                position_id=position_id,
                summary=f"Position recovered from crash",
                data={'recovery_count': position.crash_recovery_count}
            )
            
            self.logger.info(f"✓ Position marked as recovered: {position_id}")
        
        except Exception as e:
            self.logger.error(f"✗ Error marking position as recovered: {e}")
            db_session.rollback()
    
    def verify_position_with_broker(self, position_id: str) -> Dict[str, Any]:
        """
        Verify if a position still exists on the broker.
        
        Returns:
            Dict with verification status and any mismatches
        """
        try:
            position = db_session.query(StrategyPosition).filter(
                StrategyPosition.position_id == position_id
            ).first()
            
            if not position:
                return {'verified': False, 'reason': 'Position not found in DB'}
            
            if not self.api_client:
                return {'verified': True, 'reason': 'No API client for verification', 'db_state': 'ACTIVE'}
            
            # Use broker API to check holdings
            try:
                holdings = self.api_client.get_holdings()
                position_exists = any(h['tradingsymbol'] == position.symbol for h in holdings.get('data', []))
                
                return {
                    'verified': True,
                    'db_state': position.status,
                    'broker_state': 'ACTIVE' if position_exists else 'CLOSED',
                    'mismatch': position_exists and position.status != PositionStatus.ACTIVE.value
                }
            
            except Exception as api_error:
                self.logger.warning(f"⚠️  Could not verify with broker: {api_error}")
                return {'verified': False, 'reason': f'Broker API error: {api_error}'}
        
        except Exception as e:
            self.logger.error(f"✗ Error verifying position: {e}")
            return {'verified': False, 'reason': str(e)}
    
    # ==================== EVENT LOGGING ====================
    
    def _log_event(
        self,
        event_type: EventType,
        position_id: str = None,
        summary: str = None,
        data: Dict = None
    ):
        """Log an event to the audit trail"""
        try:
            if not self.current_session_db_id:
                return
            
            position_db_id = None
            if position_id:
                position = db_session.query(StrategyPosition).filter(
                    StrategyPosition.position_id == position_id
                ).first()
                if position:
                    position_db_id = position.id
            
            event = PositionEvent(
                session_id=self.current_session_db_id,
                position_id=position_db_id,
                event_type=event_type.value,
                summary=summary,
                event_data=json.dumps(data) if data else None,
            )
            
            db_session.add(event)
            db_session.commit()
        
        except Exception as e:
            self.logger.error(f"✗ Error logging event: {e}")
            db_session.rollback()
    
    # ==================== UTILITIES ====================
    
    def get_session_stats(self) -> Dict[str, Any]:
        """Get current session statistics"""
        if not self.current_session_id:
            return {}
        return get_session_stats(self.current_session_id)
    
    def get_position_snapshot(self, position_id: str) -> Optional[PositionSnapshot]:
        """Get cached position snapshot"""
        return self.position_cache.get(position_id)
    
    def flush_all_pending(self):
        """Flush all pending updates to database"""
        try:
            if self.enable_async:
                # For async mode, drain the queue
                updates = []
                try:
                    while True:
                        updates.append(self.update_queue.get_nowait())
                except queue.Empty:
                    pass
                
                if updates:
                    self._flush_updates(updates)
            
            # Also ensure all in-memory changes are committed
            db_session.commit()
            self.logger.info(f"✓ Flushed all pending updates")
        
        except Exception as e:
            self.logger.error(f"✗ Error flushing pending updates: {e}")
            db_session.rollback()
