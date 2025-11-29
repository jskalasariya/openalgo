"""
Nifty Sehwag Strategy - Main Orchestrator
==========================================
Coordinates all strategy components and executes the trading logic.
"""

import logging
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import pytz

from .models import StrategyState, LegPosition, LegSchedule
from .market_data import MarketDataManager
from .order_manager import OrderManager
from .position_manager import PositionManager
from .persistence_manager import NiftySehwagPersistence
from .utils import (
    is_market_open, get_current_atm_strike,
    get_option_symbol, get_next_expiry_date, calculate_itm_strike
)

logger = logging.getLogger(__name__)


class NiftySehwagStrategy:
    """Main strategy orchestrator"""
    
    def __init__(self, client, config: Dict, websocket_client=None):
        """
        Initialize strategy
        
        Args:
            client: OpenAlgo API client
            config: Full configuration dictionary
            websocket_client: Optional WebSocket client
        """
        self.client = client
        self.config = config
        self.tz = pytz.timezone(config.get('strategy', {}).get('timezone', 'Asia/Kolkata'))
        
        # Initialize components
        self.market_data = MarketDataManager(
            client, 
            self._build_market_data_config(), 
            websocket_client
        )
        self.order_manager = OrderManager(client, self._build_order_config())
        self.position_manager = PositionManager(self.order_manager, self._build_position_config())
        
        # Strategy parameters
        self.underlying = config.get('strategy', {}).get('underlying', 'NIFTY')
        # Note: Strategy runs every day, not just expiry day
        self.wait_trade_threshold = config.get('strategy', {}).get('wait_trade_threshold_pct', 3.0)
        
        # Schedule
        self.start_hour = config.get('schedule', {}).get('strategy_start_hour', 9)
        self.start_minute = config.get('schedule', {}).get('strategy_start_minute', 15)
        self.start_second = config.get('schedule', {}).get('strategy_start_second', 10)
        self.end_hour = config.get('schedule', {}).get('end_hour', 15)
        self.end_minute = config.get('schedule', {}).get('end_minute', 30)
        
        # Legs configuration
        self.legs_config = self._load_legs_config()
        
        # Cache previous day's candle data at initialization (one-time activity)
        logger.info("📊 Fetching previous trading day's candle data (one-time)...")
        self.cached_prev_day_candles = self.market_data.get_previous_day_candles(self.tz)
        
        # Pre-calculate and cache highest_high and lowest_low (optimization)
        self.cached_highest_high = 0.0
        self.cached_lowest_low = 0.0
        
        if self.cached_prev_day_candles is not None:
            self.cached_highest_high = float(self.cached_prev_day_candles['high'].max())
            self.cached_lowest_low = float(self.cached_prev_day_candles['low'].min())
            logger.info(f"✓ Cached {len(self.cached_prev_day_candles)} previous day candles")
            logger.info(f"✓ Pre-calculated: High={self.cached_highest_high:.2f}, Low={self.cached_lowest_low:.2f}")
        else:
            logger.warning("⚠️  Could not fetch previous day candles at initialization")
        
        # Initialize persistence (will be set up when strategy runs)
        self.persistence = None
        
        logger.info(f"🚀 Nifty Sehwag Strategy initialized with {len(self.legs_config)} legs")
    
    def _build_market_data_config(self) -> Dict:
        """Build market data manager config"""
        return {
            'underlying': self.config.get('strategy', {}).get('underlying', 'NIFTY'),
            'underlying_exchange': self.config.get('strategy', {}).get('underlying_exchange', 'NSE_INDEX'),
            'option_exchange': self.config.get('strategy', {}).get('option_exchange', 'NFO'),
            'candle_interval': self.config.get('strategy', {}).get('candle_interval', '3m'),
            'lookback_candles': self.config.get('strategy', {}).get('lookback_candles_minutes', 3),
            'use_websocket': self.config.get('websocket', {}).get('enabled', False)
        }
    
    def _build_order_config(self) -> Dict:
        """Build order manager config"""
        return {
            'test_mode': self.config.get('strategy', {}).get('test_mode', False),
            'auto_place_orders': self.config.get('orders', {}).get('auto_place_orders', False),
            'option_exchange': self.config.get('strategy', {}).get('option_exchange', 'NFO'),
            'price_type': self.config.get('orders', {}).get('price_type', 'MARKET'),
            'product': self.config.get('orders', {}).get('product', 'NRML'),
            'instrument_type': self.config.get('orders', {}).get('instrument_type', 'options'),
            'entry_action': self.config.get('orders', {}).get('entry_action', 'BUY'),
            'exit_action': self.config.get('orders', {}).get('exit_action', 'SELL')
        }
    
    def _build_position_config(self) -> Dict:
        """Build position manager config"""
        return {
            'entry_action': self.config.get('orders', {}).get('entry_action', 'BUY'),
            'exit_action': self.config.get('orders', {}).get('exit_action', 'SELL')
        }
    
    def _load_legs_config(self) -> List[Dict]:
        """Load and validate legs configuration"""
        legs_raw = self.config.get('legs', [])
        
        # Support both list and dict formats
        if isinstance(legs_raw, list):
            legs = [leg for leg in legs_raw if leg.get('enabled', True)]
        elif isinstance(legs_raw, dict):
            # Convert old format to new
            legs = self._convert_old_format(legs_raw)
        else:
            logger.error("Invalid legs configuration format")
            legs = []
        
        if not legs:
            logger.warning("⚠️  No legs configured!")
        else:
            logger.info(f"📊 Loaded {len(legs)} active legs")
            for i, leg in enumerate(legs, 1):
                logger.info(f"   {leg.get('name', f'Leg {i}')}: "
                           f"ITM{leg.get('itm_level', 0)} @ +{leg.get('entry_delay_seconds', 0)}s "
                           f"({leg.get('management_type', 'fixed')})")
        
        return legs
    
    def _convert_old_format(self, legs_dict: Dict) -> List[Dict]:
        """Convert old dict format to new list format"""
        legs = []
        for i in range(1, 10):
            leg_key = f'leg{i}'
            if leg_key in legs_dict:
                leg = legs_dict[leg_key].copy()
                leg['name'] = f'Leg {i}'
                leg['enabled'] = leg.get('enabled', True)
                
                # Set default delays
                if i == 1:
                    leg['entry_delay_seconds'] = 0
                elif i == 2:
                    leg['entry_delay_seconds'] = 35
                elif i == 3:
                    leg['entry_delay_seconds'] = 50
                else:
                    leg['entry_delay_seconds'] = 60 * i
                
                # Determine management type
                if leg.get('dynamic_sl_decrease'):
                    leg['management_type'] = 'dynamic_sl'
                else:
                    leg['management_type'] = 'escalating_profit'
                
                if leg['enabled']:
                    legs.append(leg)
        
        return legs
    
    def run(self):
        """Main strategy execution"""
        logger.info("\n" + "="*70)
        logger.info("🚀 NIFTY SEHWAG STRATEGY STARTED")
        logger.info("="*70)
        
        # Pre-flight checks
        if not is_market_open(self.tz):
            logger.warning("⚠️  Market is closed. Strategy will not run.")
            return
        
        if not self.legs_config:
            logger.error("❌ No legs configured. Cannot run strategy.")
            return
        
        # Check for crashed sessions and handle recovery
        logger.info("🔍 Checking for crashed sessions...")
        crashed_sessions = NiftySehwagPersistence.detect_crashed_sessions()
        
        if crashed_sessions:
            logger.warning(f"⚠️  Found {len(crashed_sessions)} crashed session(s) with active positions")
            
            # Mark crashed sessions
            session_ids = [s['session_id'] for s in crashed_sessions]
            NiftySehwagPersistence.mark_sessions_as_crashed(
                session_ids,
                crash_reason="Strategy restarted - previous session did not complete"
            )
            
            # Log recovered positions
            for crashed in crashed_sessions:
                logger.warning(f"  📋 Crashed session: {crashed['session_id']}")
                for pos in crashed['active_positions']:
                    logger.warning(f"    - {pos['leg_name']}: {pos['symbol']} (Entry: ₹{pos['entry_price']:.2f})")
            
            logger.info("💡 Tip: Check crashed positions in database and manually square off if needed")
            logger.info("    Use: python strategies/nifty_sehwag/query_db.py")
        else:
            logger.info("✓ No crashed sessions detected")
        
        # Initialize persistence
        try:
            expiry_date = get_next_expiry_date(self.underlying, self.tz)
            self.persistence = NiftySehwagPersistence(expiry_date=expiry_date)
        except Exception as e:
            logger.warning(f"⚠️  Could not initialize persistence: {e}")
            logger.warning("Strategy will continue without database tracking")
            self.persistence = None
        
        # Initialize state
        state = StrategyState()
        
        # Build leg schedule
        strategy_start_time = datetime.now(self.tz).replace(
            hour=self.start_hour,
            minute=self.start_minute,
            second=self.start_second,
            microsecond=0
        )
        
        leg_schedule = self._build_leg_schedule(strategy_start_time)
        self._log_schedule(strategy_start_time, leg_schedule)
        
        # Wait for strategy start time
        self._wait_for_start_time(strategy_start_time)
        
        # Step 1: Execute legs (each leg will independently check entry condition)
        # No upfront entry check - legs will detect breakout at their scheduled times
        logger.info("📍 Strategy active - monitoring for breakout at each leg's entry time")
        self._execute_legs(leg_schedule, state)
        
        # Step 4: Monitor positions
        self._monitor_positions(leg_schedule, state)
        
        # Step 5: Close remaining positions
        self._close_all_positions(state)
        
        # Step 6: Summary
        self._print_summary(state)
        
        # Step 7: Close persistence session
        if self.persistence:
            total_pnl, total_pnl_pct = state.get_total_pnl()
            self.persistence.close_session(total_pnl, total_pnl_pct)
    
    def _build_leg_schedule(self, start_time: datetime) -> List[LegSchedule]:
        """Build leg entry schedule"""
        schedule = []
        
        for i, leg_config in enumerate(self.legs_config, 1):
            entry_delay = leg_config.get('entry_delay_seconds', 0)
            entry_time = start_time + timedelta(seconds=entry_delay)
            
            leg_schedule = LegSchedule(
                leg_num=i,
                config=leg_config,
                entry_time=entry_time
            )
            schedule.append(leg_schedule)
            
            # Initialize state
            self.config['strategy']['leg_configs'] = self.config.get('strategy', {}).get('leg_configs', {})
        
        return schedule
    
    def _log_schedule(self, start_time: datetime, schedule: List[LegSchedule]):
        """Log entry schedule"""
        logger.info(f"\n📅 Leg Entry Schedule:")
        logger.info(f"Strategy Start: {start_time.strftime('%H:%M:%S')}")
        for leg_info in schedule:
            logger.info(f"   {leg_info.name} @ {leg_info.entry_time.strftime('%H:%M:%S')}")
    
    def _wait_for_start_time(self, start_time: datetime):
        """Wait until strategy start time"""
        while datetime.now(self.tz) < start_time:
            sleep_time = (start_time - datetime.now(self.tz)).total_seconds()
            if sleep_time > 60:
                logger.info(f"⏳ Waiting for strategy start... {sleep_time:.0f} seconds remaining")
                time.sleep(60)
            elif sleep_time > 0:
                time.sleep(min(sleep_time, 5))
            else:
                break
        
        logger.info(f"\n✓ Strategy start time reached: {datetime.now(self.tz).strftime('%H:%M:%S')}")
    
    def _wait_for_trade_confirmation(self, state: StrategyState, timeout: int = 300) -> bool:
        """Wait for Wait & Trade confirmation"""
        logger.info(f"\n📍 Waiting for {self.wait_trade_threshold:.1f}% price movement...")
        
        wait_start = time.time()
        
        while not self.market_data.check_wait_trade(
            state.reference_price, 
            state.entry_direction, 
            self.wait_trade_threshold
        ):
            if time.time() - wait_start > timeout:
                return False
            time.sleep(2)
        
        return True
    
    def _execute_legs(self, schedule: List[LegSchedule], state: StrategyState):
        """Execute leg entries"""
        logger.info("\n📍 Executing legs...")
        
        for leg_info in schedule:
            # Wait for entry time
            while datetime.now(self.tz) < leg_info.entry_time:
                sleep_time = (leg_info.entry_time - datetime.now(self.tz)).total_seconds()
                if sleep_time > 10:
                    time.sleep(5)
                elif sleep_time > 0:
                    time.sleep(min(sleep_time, 1))
                else:
                    break
            
            logger.info(f"\n✓ {leg_info.name} entry time reached: {datetime.now(self.tz).strftime('%H:%M:%S')}")
            
            # Execute entry
            position = self._execute_leg_entry(leg_info, state)
            if position:
                state.set_position(leg_info.leg_num, position)
                state.leg_configs[leg_info.leg_num] = leg_info.config
                leg_info.entered = True
                logger.info(f"✅ {leg_info.name} position entered successfully")
            else:
                logger.error(f"✗ {leg_info.name} entry failed")
    
    def _execute_leg_entry(self, leg_info: LegSchedule, state: StrategyState) -> Optional[LegPosition]:
        """Execute a single leg entry"""
        try:
            config = leg_info.config
            
            # Check entry condition at this leg's entry time
            logger.info(f"{leg_info.name}: Checking entry condition at {datetime.now(self.tz).strftime('%H:%M:%S')}")
            should_enter, current_direction, _, _ = self.market_data.analyze_entry_condition(
                self.tz,
                cached_candles=None,
                cached_high=self.cached_highest_high,
                cached_low=self.cached_lowest_low
            )
            
            if not should_enter:
                logger.warning(f"{leg_info.name}: Entry condition NOT met - Skipping this leg")
                if self.persistence and current_direction:
                    self.persistence.log_entry_condition(
                        current_direction, self.cached_highest_high, self.cached_lowest_low,
                        self.market_data.get_underlying_price() or 0, False
                    )
                return None
            
            # First leg sets the direction for all subsequent legs
            if state.entry_direction is None:
                state.entry_direction = current_direction
                state.reference_price = self.market_data.get_underlying_price()
                state.reference_time = datetime.now(self.tz).strftime('%H:%M:%S')
                logger.info(f"✓ First entry detected: Direction={current_direction}, Reference Price={state.reference_price:.2f}")
                
                # Log entry condition met
                if self.persistence:
                    self.persistence.log_entry_condition(
                        current_direction, self.cached_highest_high, self.cached_lowest_low,
                        state.reference_price, True
                    )
                
                # Wait & Trade confirmation for first leg
                logger.info("⏳ Waiting for Wait & Trade confirmation...")
                wait_trade_success = self._wait_for_trade_confirmation(state)
                
                if self.persistence:
                    self.persistence.log_wait_trade(
                        wait_trade_success, state.reference_price, self.wait_trade_threshold
                    )
                
                if not wait_trade_success:
                    logger.warning(f"⚠️  Wait & Trade timeout - {leg_info.name} entry cancelled")
                    return None
                logger.info("✓ Wait & Trade confirmed")
            else:
                # Subsequent legs must match the established direction
                if current_direction != state.entry_direction:
                    logger.warning(f"{leg_info.name}: Direction mismatch - Expected {state.entry_direction}, got {current_direction} - Skipping")
                    return None
                logger.info(f"{leg_info.name}: Direction consistent - {current_direction}")
            
            # Get ATM and calculate ITM strike
            spot_price = self.market_data.get_underlying_price()
            if not spot_price:
                logger.error("Could not get spot price")
                return None
            
            atm_strike = get_current_atm_strike(spot_price, self.underlying)
            itm_level = config.get('itm_level', 3)
            option_type = state.entry_direction
            
            strike = calculate_itm_strike(atm_strike, itm_level, option_type, self.underlying)
            expiry_date = get_next_expiry_date(self.underlying, self.tz)
            symbol = get_option_symbol(strike, expiry_date, option_type, self.underlying)
            
            logger.info(f"{leg_info.name}: {symbol} (ATM: {atm_strike}, ITM: {itm_level}, Strike: {strike})")
            
            # Get quote
            quote = self.market_data.get_quote(symbol)
            if not quote:
                logger.error(f"Could not get quote for {symbol}")
                return None
            
            entry_price = float(quote.get('ltp', 0))
            quantity = config.get('quantity', 1)
            
            logger.info(f"Entry Price: {entry_price:.2f}, Quantity: {quantity}")
            
            # Place order
            order_id = self.order_manager.place_order(symbol, quantity, self.order_manager.entry_action)
            
            # Log order to database
            if self.persistence and order_id:
                self.persistence.log_order(
                    leg_num=leg_info.leg_num,
                    order_type='ENTRY',
                    symbol=symbol,
                    quantity=quantity,
                    price=entry_price,
                    order_id=order_id
                )
            
            if not order_id:
                logger.error(f"Failed to place order for {leg_info.name}")
                return None
            
            # Create position tracking
            initial_sl = entry_price * (1 - config.get('initial_sl_pct', 7.0) / 100)
            lock_profit = config.get('lock_profit_pct', 2.0)
            
            position = LegPosition(
                leg_id=leg_info.leg_num,
                symbol=symbol,
                entry_price=entry_price,
                entry_time=datetime.now(self.tz).strftime('%Y-%m-%d %H:%M:%S'),
                quantity=quantity,
                option_type=option_type,
                itm_level=itm_level,
                current_sl=initial_sl,
                lock_profit_pct=lock_profit,
                profit_level_for_lock_increase=2.0
            )
            
            logger.info(f"✅ {leg_info.name} entered - SL: {initial_sl:.2f}")
            
            # Log leg entry to database
            if self.persistence:
                self.persistence.log_leg_entry(
                    leg_num=leg_info.leg_num,
                    leg_name=leg_info.name,
                    symbol=symbol,
                    entry_price=entry_price,
                    quantity=quantity,
                    option_type=option_type,
                    strike=strike,
                    initial_sl=initial_sl,
                    lock_profit_pct=lock_profit
                )
            
            return position
            
        except Exception as e:
            logger.error(f"Error executing {leg_info.name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _monitor_positions(self, schedule: List[LegSchedule], state: StrategyState):
        """Monitor active positions"""
        logger.info("\n📍 Monitoring positions until end time...")
        
        end_time = datetime.now(self.tz).replace(
            hour=self.end_hour,
            minute=self.end_minute,
            second=0,
            microsecond=0
        )
        
        while datetime.now(self.tz) < end_time:
            try:
                active_positions = state.get_all_active_positions()
                
                if not active_positions:
                    logger.info("No active positions remaining")
                    break
                
                for leg_num, position in active_positions:
                    leg_config = state.leg_configs.get(leg_num)
                    if not leg_config:
                        continue
                    
                    quote = self.market_data.get_quote(position.symbol)
                    if quote:
                        current_price = float(quote.get('ltp', 0))
                        self.position_manager.manage_position(leg_num, leg_config, state, current_price)
                
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in position monitoring: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(5)
    
    def _close_all_positions(self, state: StrategyState):
        """Force close all remaining positions"""
        logger.info(f"\n✓ End time reached: {datetime.now(self.tz).strftime('%H:%M:%S')}")
        logger.info("📍 Closing all remaining positions...")
        
        active_positions = state.get_all_active_positions()
        if active_positions:
            for leg_num, position in active_positions:
                leg_config = state.leg_configs.get(leg_num)
                leg_name = leg_config.get('name', f'Leg {leg_num}') if leg_config else f'Leg {leg_num}'
                
                quote = self.market_data.get_quote(position.symbol)
                if quote:
                    final_price = float(quote.get('ltp', 0))
                    logger.info(f"Closing {leg_name}: {position.symbol} @ {final_price:.2f}")
                    self.position_manager.exit_position(position, final_price, "END_OF_DAY")
        else:
            logger.info("No active positions to close")
    
    def _print_summary(self, state: StrategyState):
        """Print strategy summary"""
        logger.info(f"\n{'='*70}")
        logger.info(f"📊 STRATEGY SUMMARY")
        logger.info(f"{'='*70}")
        
        total_pnl, total_pnl_pct = state.get_total_pnl()
        
        for leg_num in sorted(state.leg_positions.keys()):
            position = state.leg_positions[leg_num]
            leg_config = state.leg_configs.get(leg_num)
            leg_name = leg_config.get('name', f'Leg {leg_num}') if leg_config else f'Leg {leg_num}'
            
            if position and not position.is_active:
                logger.info(f"{leg_name}: PnL {position.pnl_pct:.2f}% (₹{position.pnl:.2f})")
            elif position and position.is_active:
                logger.info(f"{leg_name}: Still ACTIVE (not closed properly)")
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Total PnL: ₹{total_pnl:.2f} ({total_pnl_pct:.2f}%)")
        logger.info(f"{'='*70}\n")
