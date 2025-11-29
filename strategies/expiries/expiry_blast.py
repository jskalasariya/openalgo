"""
Expiry Blast Strategy - OpenAlgo
=========================================
A comprehensive weekly options expiry trading strategy that:

✓ Only runs on expiry day after specified time (default 10 AM IST)
✓ Monitors ATM CE and PE for breakout entry
✓ Auto-detects ATM changes during the day
✓ Uses 5 candle breakout on 3m interval
✓ 50% initial stop loss with 1% trailing on every 1% profit move
✓ Exits at 100% profit target
✓ Fully configurable for NIFTY, SENSEX, BANKNIFTY, etc.
✓ Configuration-driven (no hardcoded values in code)

Configuration: expiry_blast_config.yaml
Author: OpenAlgo Team
"""

import os
import time
import json
import yaml
from datetime import datetime, timedelta
import pytz
from typing import Dict, Tuple, Optional, List, Callable
from dataclasses import dataclass
from threading import Thread, Event, Lock
import logging
from logging.handlers import RotatingFileHandler
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
from openalgo import api
import pandas as pd
import sys
import traceback
from position_persistence import PositionPersistenceManager
from database.position_persistence_db import init_position_persistence_db, EventType
from utils.websocket_ltp_client import WebSocketLTPClient, create_websocket_client

# ==================== LOGGING SETUP ====================

def setup_logging():
    """Configure logging to both file and console"""
    # Create logs directory if it doesn't exist
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate log filename with timestamp
    log_filename = os.path.join(log_dir, f"expiry_blast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    # Create logger
    logger = logging.getLogger('ExpiryBlast')
    logger.setLevel(logging.DEBUG)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # File handler with rotation - UTF-8 encoding for emoji support
    file_handler = RotatingFileHandler(
        log_filename,
        maxBytes=10*1024*1024,  # 10MB per file
        backupCount=5,  # Keep 5 backup files
        encoding='utf-8'  # Support emoji in log files
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    # Force UTF-8 encoding on Windows console
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_filename

# Initialize logger
logger, log_file = setup_logging()
logger.info(f"{'='*70}")
logger.info(f"Expiry Blast Strategy Started")
logger.info(f"Log file: {log_file}")
logger.info(f"{'='*70}")

# ==================== CONFIGURATION LOADER ====================

class ConfigLoader:
    """Load and validate configuration from YAML file"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            # Default to config file in same directory as this script
            config_path = os.path.join(
                os.path.dirname(__file__),
                "expiry_blast_config.yaml"
            )
        
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict:
        """Load YAML configuration file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"✓ Configuration loaded from {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"✗ Configuration file not found: {self.config_path}")
            sys.exit(1)
        except yaml.YAMLError as e:
            logger.error(f"✗ Invalid YAML configuration: {e}")
            sys.exit(1)
    
    def _validate_config(self) -> None:
        """Validate required configuration keys"""
        required_keys = ['strategy', 'position', 'schedule', 'monitoring']
        for key in required_keys:
            if key not in self.config:
                logger.error(f"✗ Missing required configuration section: {key}")
                sys.exit(1)
    
    def get(self, section: str, key: str, default=None):
        """Get configuration value with fallback to default"""
        return self.config.get(section, {}).get(key, default)
    
    def get_section(self, section: str) -> Dict:
        """Get entire configuration section"""
        return self.config.get(section, {})


# ==================== CONFIG INITIALIZATION ====================

config_loader = ConfigLoader()

# Extract configuration
API_KEY = os.getenv(
    "OPENALGO_APIKEY",
    config_loader.get('api', 'key', 'your_openalgo_api_key')
)
API_HOST = os.getenv(
    "OPENALGO_API_HOST",
    config_loader.get('api', 'host', 'http://127.0.0.1:5000')
)

# Validate and fix API_HOST - ensure it has http:// or https://
if API_HOST and not API_HOST.startswith(('http://', 'https://')):
    API_HOST = f'http://{API_HOST}'
    logger.warning(f"⚠️  API_HOST didn't have protocol, added http:// → {API_HOST}")

if not API_HOST:
    API_HOST = 'http://127.0.0.1:5000'
    logger.warning(f"⚠️  API_HOST not set, using default → {API_HOST}")

UNDERLYING = config_loader.get('strategy', 'underlying', 'NIFTY')
UNDERLYING_EXCHANGE = config_loader.get('strategy', 'underlying_exchange', 'NSE_INDEX')
OPTION_EXCHANGE = config_loader.get('strategy', 'option_exchange', 'NFO')
RUN_ONLY_ON_EXPIRY_DAY = config_loader.get('strategy', 'run_only_on_expiry_day', True)

START_HOUR = config_loader.get('schedule', 'start_hour', 10)
START_MINUTE = config_loader.get('schedule', 'start_minute', 0)
END_HOUR = config_loader.get('schedule', 'end_hour', 15)
END_MINUTE = config_loader.get('schedule', 'end_minute', 30)
CHECK_SLEEP = config_loader.get('schedule', 'check_interval_seconds', 5)

MONITOR_INTERVAL = config_loader.get('monitoring', 'candle_interval', '3m')
LOOKBACK_CANDLES = config_loader.get('monitoring', 'lookback_candles', 5)
BREAKOUT_THRESHOLD_PCT = config_loader.get('monitoring', 'breakout_threshold_pct', 0.0)

ENTRY_QUANTITY = config_loader.get('position', 'entry_quantity', 75)
LOT_MULTIPLIER = config_loader.get('position', 'lot_multiplier', 1)
FINAL_QUANTITY = int(ENTRY_QUANTITY * LOT_MULTIPLIER)
INITIAL_STOP_PCT = config_loader.get('position', 'initial_stop_pct', 0.50)
TRAIL_PERCENT_STEP = config_loader.get('position', 'trail_step_pct', 0.01)
PROFIT_TARGET_PCT = config_loader.get('position', 'profit_target_pct', 1.00)
ATM_RECHECK_INTERVAL = config_loader.get('position', 'atm_recheck_interval', 5)

AUTO_PLACE_ORDERS = config_loader.get('orders', 'auto_place_orders', False)
PRICE_TYPE = config_loader.get('orders', 'price_type', 'MARKET')
PRODUCT = config_loader.get('orders', 'product', 'NRML')
INSTRUMENT_TYPE = config_loader.get('orders', 'instrument_type', 'options')
ENTRY_ACTION = config_loader.get('orders', 'entry_action', 'BUY')
EXIT_ACTION = config_loader.get('orders', 'exit_action', 'SELL')

PRINT_QUOTES_IMMEDIATELY = config_loader.get('logging', 'print_quotes_immediately', True)
PRINT_CANDLE_DATA = config_loader.get('logging', 'print_candle_data', True)
PRINT_POSITION_UPDATES = config_loader.get('logging', 'print_position_updates', True)

MONITOR_ATM_CHANGES = config_loader.get('atm_monitoring', 'monitor_atm_changes', True)
ATM_CHECK_INTERVAL_MINUTES = config_loader.get('atm_monitoring', 'atm_check_interval_minutes', 5)
CLOSE_ON_ATM_CHANGE = config_loader.get('atm_monitoring', 'close_on_atm_change', True)

MONITOR_CE = config_loader.get('legs', 'monitor_ce', True)
MONITOR_PE = config_loader.get('legs', 'monitor_pe', True)

# ==================== WEBSOCKET CONFIGURATION ====================
USE_WEBSOCKET = config_loader.get('websocket', 'enabled', True)
WEBSOCKET_URL = os.getenv('WEBSOCKET_URL', config_loader.get('websocket', 'url', 'ws://127.0.0.1:8765'))
WEBSOCKET_TIMEOUT = config_loader.get('websocket', 'timeout_seconds', 10)
WEBSOCKET_RECONNECT_INTERVAL = config_loader.get('websocket', 'reconnect_interval_seconds', 5)

# ==================== TEST MODE CONFIGURATION ====================
# Set to True to simulate all strategy logic without placing actual orders
# All positions will be tracked in DB as if they were real trades
TEST_MODE = config_loader.get('strategy', 'test_mode', False)

TZ = pytz.timezone(config_loader.get('strategy', 'timezone', 'Asia/Kolkata'))

logger.info(f"\n{'='*60}")
logger.info(f"🔁 Expiry Blast Strategy - OpenAlgo")
logger.info(f"{'='*60}")
logger.info(f"Underlying: {UNDERLYING} | Exchange: {UNDERLYING_EXCHANGE}")
logger.info(f"Start Time: {START_HOUR:02d}:{START_MINUTE:02d} IST | End Time: {END_HOUR:02d}:{END_MINUTE:02d} IST")
logger.info(f"Expiry Day Only: {RUN_ONLY_ON_EXPIRY_DAY}")
logger.info(f"Auto Place Orders: {AUTO_PLACE_ORDERS}")
logger.info(f"Monitor ATM Changes: {MONITOR_ATM_CHANGES}")
logger.info(f"WebSocket: {'📡 ENABLED' if USE_WEBSOCKET else '🔄 DISABLED (REST API polling)'} @ {WEBSOCKET_URL if USE_WEBSOCKET else 'N/A'}")
logger.info(f"TEST MODE: {'🟢 ENABLED (Orders simulated, not placed)' if TEST_MODE else '🔴 DISABLED (Live trading)'}")
logger.info(f"{'='*60}\n")

# ==================== API CLIENT ====================

client = api(api_key=API_KEY, host=API_HOST)


# ==================== WEBSOCKET LTP CLIENT ====================

# Initialize WebSocket client (if enabled)
ws_client: Optional[WebSocketLTPClient] = None

def initialize_websocket():
    """Initialize WebSocket connection using reusable client from utils"""
    global ws_client
    
    if not USE_WEBSOCKET:
        logger.info("⏭️  WebSocket disabled in configuration")
        return False
    
    try:
        ws_client = create_websocket_client(
            ws_url=WEBSOCKET_URL,
            api_key=API_KEY,
            logger_instance=logger,
            timeout_seconds=WEBSOCKET_TIMEOUT,
            reconnect_interval_seconds=WEBSOCKET_RECONNECT_INTERVAL
        )
        ws_client.start_background()
        logger.info("✓ WebSocket client initialized (background thread)")
        return True
    
    except Exception as e:
        logger.error(f"✗ Error initializing WebSocket: {e}")
        return False


# ==================== UTILITY FUNCTIONS ====================

def is_market_open() -> bool:
    # # Jaysukh
    # return True
    """Check if current time is within market hours"""
    now = datetime.now(TZ)
    current_time = now.time()
    start_time = datetime.strptime(f"{START_HOUR:02d}:{START_MINUTE:02d}", "%H:%M").time()
    end_time = datetime.strptime(f"{END_HOUR:02d}:{END_MINUTE:02d}", "%H:%M").time()
    return start_time <= current_time <= end_time


def is_expiry_day() -> bool:
    """Check if today is expiry day by comparing with actual broker expiry data"""
    if not RUN_ONLY_ON_EXPIRY_DAY:
        return True
    
    try:
        # Fetch expiry dates from broker
        resp = client.expiry(
            symbol=UNDERLYING,
            exchange=OPTION_EXCHANGE,
            instrumenttype=INSTRUMENT_TYPE
        )
        
        if resp.get('status') != 'success' or not isinstance(resp.get('data'), list) or len(resp['data']) == 0:
            logger.error(f"✗ Could not fetch expiry data: {resp}")
            return False
        
        # Get today's date in DD-MMM-YY format to match broker response
        now = datetime.now(TZ)
        today_str = now.strftime('%d-%b-%y').upper()  # e.g., '28-NOV-25'
        
        # Get nearest expiry from broker
        nearest_expiry_str = resp['data'][0]  # e.g., '02-DEC-25'
        
        # Compare today's date with nearest expiry
        if today_str == nearest_expiry_str:
            logger.info(f"✓ Expiry day detected: {nearest_expiry_str} (Today is expiry)")
            return True
        else:
            logger.info(f"ℹ️  Not expiry day. Today: {today_str}, Nearest expiry: {nearest_expiry_str}")
            return False
    
    except Exception as e:
        logger.error(f"✗ Error checking expiry day: {e}")
        return False


def get_nearest_expiry() -> str:
    """Fetch nearest weekly expiry and format it"""
    try:
        resp = client.expiry(
            symbol=UNDERLYING,
            exchange=OPTION_EXCHANGE,
            instrumenttype=INSTRUMENT_TYPE
        )
        
        if resp.get('status') != 'success' or not isinstance(resp.get('data'), list) or len(resp['data']) == 0:
            raise RuntimeError(f"Invalid expiry response: {resp}")
        
        raw_expiry = resp['data'][0]  # Nearest expiry
        logger.info(f"📅 Nearest Expiry from API: {raw_expiry}")
        
        # Convert format: '02-DEC-25' → '02DEC25'
        day, mon, year = raw_expiry.split('-')
        expiry_fmt = f"{day}{mon.upper()}{year[-2:]}"
        
        logger.info(f"📅 Formatted Expiry: {expiry_fmt}")
        return expiry_fmt
    except Exception as e:
        logger.error(f"✗ Error fetching expiry: {e}")
        raise


def resolve_option_symbol(
    option_type: str,
    expiry: Optional[str] = None,
    underlying: Optional[str] = None
) -> Tuple[str, str, float]:
    """
    Resolve ATM CE/PE symbol using optionchain API (READ-ONLY, no order placement)
    
    Returns:
        Tuple of (symbol, expiry, strike_price)
    """
    if underlying is None:
        underlying = UNDERLYING
    if expiry is None:
        expiry = get_nearest_expiry()
    
    try:
        # Fetch option chain using read-only API (no order placement)
        resp = client.optionchain(
            underlying=underlying,
            exchange=UNDERLYING_EXCHANGE,
            expiry_date=expiry,
            strike_count=1  # Only need ATM, so fetch 1 strike around ATM
        )
        
        if resp.get('status') != 'success':
            raise RuntimeError(f"Option chain fetch failed: {resp}")
        
        atm_strike = float(resp.get('atm_strike', 0))
        chain = resp.get('chain', [])
        
        if not chain:
            raise RuntimeError(f"No chain data in response: {resp}")
        
        # Find ATM item in chain
        atm_item = next((item for item in chain if item['strike'] == atm_strike), None)
        if not atm_item:
            raise RuntimeError(f"ATM strike {atm_strike} not found in chain")
        
        # Extract symbol for the requested option type
        option_data = atm_item.get(option_type.lower())
        if not option_data or not option_data.get('symbol'):
            raise RuntimeError(f"No {option_type} data for ATM strike {atm_strike}")
        
        symbol = option_data['symbol']
        resolved_expiry = resp.get('expiry_date', expiry)
        strike = atm_strike
        
        logger.info(f"✓ Resolved {option_type:2s} @ Strike {strike:7.0f}: {symbol}")
        return symbol, resolved_expiry, strike
    
    except Exception as e:
        logger.error(f"✗ Error resolving {option_type} symbol: {e}")
        raise


def fetch_last_candles(symbol: str) -> pd.DataFrame:
    """Fetch last N candles for a symbol"""
    try:
        end_date = datetime.now(TZ)
        start_date = end_date - timedelta(days=5)  # Get more data for safety
        
        df = client.history(
            symbol=symbol,
            exchange=OPTION_EXCHANGE,
            interval=MONITOR_INTERVAL,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )
        
        if not isinstance(df, pd.DataFrame) or df.empty:
            raise RuntimeError(f"Invalid history response for {symbol}")
        
        df.index = pd.to_datetime(df.index)
        df = df.tail(LOOKBACK_CANDLES)
        
        if PRINT_CANDLE_DATA:
            logger.info(f"\n📊 Last {LOOKBACK_CANDLES} {MONITOR_INTERVAL} candles for {symbol}:")
            logger.info(df[['open', 'high', 'low', 'close']].to_string())
        
        return df
    
    except Exception as e:
        logger.error(f"✗ Error fetching candles for {symbol}: {e}")
        raise


def get_ltp(symbol: str) -> Optional[float]:
    """Get last traded price - try WebSocket first, fallback to REST API"""
    # Try WebSocket first if available
    if ws_client and ws_client.is_connected():
        cached_price = ws_client.get_last_price(symbol)
        if cached_price is not None:
            if PRINT_QUOTES_IMMEDIATELY:
                logger.info(f"💹 {symbol:15s} LTP: {cached_price:8.2f} (WebSocket)")
            return cached_price
    
    # Fallback to REST API
    try:
        q = client.quotes(symbol=symbol, exchange=OPTION_EXCHANGE)
        ltp = q['data']['ltp']
        
        if PRINT_QUOTES_IMMEDIATELY:
            logger.info(f"💹 {symbol:15s} LTP: {ltp:8.2f} (REST)")
        
        return ltp
    
    except Exception as e:
        logger.error(f"✗ Error fetching LTP for {symbol}: {e}")
        return None


# ==================== PERSISTENCE INITIALIZATION ====================

try:
    init_position_persistence_db()
    persistence = PositionPersistenceManager(
        strategy_name='expiry_blast',
        api_client=client,
        batch_interval=10,
        enable_async=False,
        logger_instance=logger
    )
    logger.info("✓ Position persistence initialized")
except Exception as e:
    logger.error(f"✗ Error initializing persistence: {e}")
    persistence = None

# ==================== LEG MONITOR CLASS ====================

@dataclass
class PositionState:
    """Track position state"""
    symbol: str
    leg_type: str  # "CE" or "PE"
    entry_price: float
    entry_time: datetime
    stop_price: float
    highest_price: float
    position_open: bool
    last_trail_level: int
    last_atm_check: int  # Candle count since last ATM check
    current_strike: float
    position_id: Optional[str] = None  # Persistence tracking


class LegMonitor:
    """Monitor and trade a single leg (CE or PE)"""
    
    def __init__(self, symbol: str, leg_type: str, strike: float):
        self.symbol = symbol
        self.leg_type = leg_type
        self.strike = strike
        self.state = PositionState(
            symbol=symbol,
            leg_type=leg_type,
            entry_price=0,
            entry_time=None,
            stop_price=0,
            highest_price=0,
            position_open=False,
            last_trail_level=0,
            last_atm_check=0,
            current_strike=strike,
            position_id=None
        )
        self.candle_count = 0
        self.highest_high = 0
    
    def initialize(self):
        """Initialize by fetching initial candles and setting breakout level"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 Initializing {self.leg_type} Monitor: {self.symbol}")
        logger.info(f"{'='*60}")
        
        df = fetch_last_candles(self.symbol)
        self.highest_high = df['high'].max()
        logger.info(f"🎯 Breakout Level ({self.leg_type}): {self.highest_high:.2f}")
        self.candle_count = len(df)
    
    def check_atm_changed(self) -> bool:
        """Check if ATM has changed during the day"""
        if not MONITOR_ATM_CHANGES:
            return False
        
        self.state.last_atm_check += 1
        
        if self.state.last_atm_check >= ATM_RECHECK_INTERVAL:
            self.state.last_atm_check = 0
            
            try:
                new_symbol, _, new_strike = resolve_option_symbol(
                    self.leg_type,
                    underlying=UNDERLYING
                )
                
                if new_strike != self.state.current_strike:
                    logger.warning(f"\n⚠️  ATM CHANGED for {self.leg_type}!")
                    logger.warning(f"   Old Strike: {self.state.current_strike} ({self.symbol})")
                    logger.warning(f"   New Strike: {new_strike} ({new_symbol})")
                    return True
            
            except Exception as e:
                logger.error(f"⚠️  Error checking ATM change: {e}")
        
        return False
    
    def monitor(self):
        """Main monitoring loop for this leg - supports WebSocket or polling"""
        logger.info(f"▶️  Starting {self.leg_type} monitoring...")
        self.initialize()
        
        # Register WebSocket callback if available
        if ws_client and ws_client.is_connected():
            logger.info(f"📡 {self.leg_type} using WebSocket for real-time LTP updates")
            
            price_event = Event()
            last_price_holder = {'price': None}
            
            def on_price_update(ltp: float):
                """Callback when price updates via WebSocket"""
                last_price_holder['price'] = ltp
                price_event.set()
            
            ws_client.on_price_update(self.symbol, on_price_update)
        else:
            logger.info(f"🔄 {self.leg_type} using polling (REST API) for LTP updates")
            price_event = None
        
        iteration = 0
        while True:
            # Check if market hours have ended
            if not is_market_open():
                logger.info(f"⏹️  Market hours ended. Stopping {self.leg_type} monitor.")
                if self.state.position_open:
                    logger.warning(f"⚠️  Position still open for {self.leg_type}. Would exit at market price.")
                break
            
            # Check if ATM has changed
            if self.check_atm_changed() and CLOSE_ON_ATM_CHANGE:
                logger.info(f"🔄 ATM changed. Exiting {self.leg_type} position and resetting.")
                if self.state.position_open:
                    logger.info(f"📊 Exit Summary {self.leg_type}: Entry={self.state.entry_price:.2f}, Current=N/A")
                # Reset for new ATM
                self.initialize()
                self.state.position_open = False
                self.state.last_trail_level = 0
                continue
            
            # Get current LTP
            if ws_client and ws_client.is_connected() and price_event:
                # WebSocket mode: wait for price update or timeout
                price_event.clear()
                if price_event.wait(timeout=CHECK_SLEEP):
                    ltp = last_price_holder['price']
                else:
                    # Timeout - continue to next iteration
                    continue
            else:
                # Polling mode: fetch via REST API
                ltp = get_ltp(self.symbol)
                if ltp is None:
                    time.sleep(CHECK_SLEEP)
                    continue
                
                time.sleep(CHECK_SLEEP)
            
            iteration += 1
            
            # ========== ENTRY LOGIC ==========
            if not self.state.position_open and ltp > self.highest_high * (1 + BREAKOUT_THRESHOLD_PCT):
                self.state.entry_price = ltp
                self.state.entry_time = datetime.now(TZ)
                self.state.position_open = True
                self.state.stop_price = ltp * (1 - INITIAL_STOP_PCT)
                self.state.highest_price = ltp
                self.state.last_trail_level = 0
                
                # PERSISTENCE: Save entry
                if persistence:
                    try:
                        self.state.position_id = persistence.save_position_entry(
                            symbol=self.symbol,
                            leg_type=self.leg_type,
                            strike=self.strike,
                            entry_price=ltp,
                            highest_high_breakout=self.highest_high
                        )
                    except Exception as e:
                        logger.error(f"✗ Error saving position entry to persistence: {e}")
                
                entry_msg = (
                    f"\n{'='*60}\n"
                    f"✅ ENTRY SIGNAL: {self.leg_type}\n"
                    f"   Symbol: {self.symbol}\n"
                    f"   Entry Price: {self.state.entry_price:.2f}\n"
                    f"   Stop Loss: {self.state.stop_price:.2f}\n"
                    f"   Quantity: {ENTRY_QUANTITY}\n"
                    f"   Time: {self.state.entry_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"{'='*60}\n"
                )
                logger.info(entry_msg)
                
                if AUTO_PLACE_ORDERS:
                    self._place_order(ENTRY_ACTION, FINAL_QUANTITY)
            
            # ========== POSITION MANAGEMENT ==========
            if self.state.position_open:
                profit_pct = (ltp / self.state.entry_price) - 1
                gain_points = ltp - self.state.entry_price
                
                # PERSISTENCE: Queue update
                if persistence and self.state.position_id:
                    try:
                        persistence.queue_position_update(
                            position_id=self.state.position_id,
                            current_price=ltp,
                            stop_price=self.state.stop_price,
                            highest_price=self.state.highest_price,
                            last_trail_level=self.state.last_trail_level,
                            profit_percent=profit_pct
                        )
                    except Exception as e:
                        logger.error(f"✗ Error queuing position update to persistence: {e}")
                
                # Check for 100% profit exit
                if profit_pct >= PROFIT_TARGET_PCT:
                    exit_msg = (
                        f"\n{'='*60}\n"
                        f"🎯 PROFIT TARGET HIT: {self.leg_type}\n"
                        f"   Symbol: {self.symbol}\n"
                        f"   Entry Price: {self.state.entry_price:.2f}\n"
                        f"   Exit Price: {ltp:.2f}\n"
                        f"   Profit: {gain_points:.2f} ({profit_pct*100:.1f}%)\n"
                        f"   Exit Time: {datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"{'='*60}\n"
                    )
                    logger.info(exit_msg)
                    
                    # PERSISTENCE: Save exit
                    if persistence and self.state.position_id:
                        try:
                            persistence.save_position_exit(
                                position_id=self.state.position_id,
                                exit_price=ltp,
                                exit_reason='PROFIT_TARGET',
                                profit_loss=gain_points * ENTRY_QUANTITY
                            )
                        except Exception as e:
                            logger.error(f"✗ Error saving position exit to persistence: {e}")
                    
                    if AUTO_PLACE_ORDERS:
                        self._place_order(EXIT_ACTION, FINAL_QUANTITY)
                    
                    self.state.position_open = False
                    break
                
                # Trailing stop logic - Move stop up by exactly the profit amount
                if profit_pct > 0:
                    # Calculate new stop: current_price - (entry_price * initial_stop_pct)
                    # This maintains a constant % buffer below price
                    new_stop = ltp - (self.state.entry_price * INITIAL_STOP_PCT)
                    
                    # Only update if stop has moved up
                    if new_stop > self.state.stop_price:
                        old_stop = self.state.stop_price
                        self.state.stop_price = new_stop
                        
                        if PRINT_POSITION_UPDATES:
                            logger.info(f"📈 {self.leg_type} Trail Update: Stop {old_stop:.2f} → {new_stop:.2f} | Profit: {profit_pct*100:.1f}%")
                
                # Check for stop loss
                if ltp <= self.state.stop_price:
                    exit_msg = (
                        f"\n{'='*60}\n"
                        f"🛑 STOP LOSS HIT: {self.leg_type}\n"
                        f"   Symbol: {self.symbol}\n"
                        f"   Entry Price: {self.state.entry_price:.2f}\n"
                        f"   Exit Price: {ltp:.2f}\n"
                        f"   Loss: {gain_points:.2f} ({profit_pct*100:.1f}%)\n"
                        f"   Stop Price: {self.state.stop_price:.2f}\n"
                        f"   Exit Time: {datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"{'='*60}\n"
                    )
                    logger.info(exit_msg)
                    
                    # PERSISTENCE: Save exit
                    if persistence and self.state.position_id:
                        try:
                            persistence.save_position_exit(
                                position_id=self.state.position_id,
                                exit_price=ltp,
                                exit_reason='STOP_LOSS',
                                profit_loss=gain_points * ENTRY_QUANTITY
                            )
                        except Exception as e:
                            logger.error(f"✗ Error saving position exit to persistence: {e}")
                    
                    if AUTO_PLACE_ORDERS:
                        self._place_order(EXIT_ACTION, FINAL_QUANTITY)
                    
                    self.state.position_open = False
                    break
            
            time.sleep(CHECK_SLEEP)
    
    def _place_order(self, action: str, quantity: int):
        """Place an order (or simulate in test mode)"""
        try:
            if TEST_MODE:
                # Simulate order in TEST_MODE - persist to DB
                logger.info(f"🧪 TEST MODE - SIMULATED ORDER ({action}): {self.symbol} x {quantity}")
                
                # Log simulated order to persistence
                if persistence and self.state.position_id:
                    try:
                        order_data = {
                            'symbol': self.symbol,
                            'action': action,
                            'quantity': quantity,
                            'price_type': PRICE_TYPE,
                            'product': PRODUCT,
                            'simulated': True
                        }
                        persistence.log_event(
                            event_type=EventType.ENTRY if action == ENTRY_ACTION else EventType.EXIT,
                            summary=f"TEST MODE: Simulated {action} order for {self.symbol}",
                            position_id=self.state.position_id,
                            data=order_data
                        )
                        logger.info(f"✓ Simulated order persisted to DB")
                    except Exception as e:
                        logger.error(f"✗ Error persisting simulated order: {e}")
            else:
                # Place real order via OpenAlgo API using smart order (position-aware)
                logger.info(f"📤 LIVE ORDER PLACEMENT ({action}): {self.symbol} x {quantity}")
                
                response = client.placesmartorder(
                    strategy=f"expiry_blast_{self.leg_type}",
                    symbol=self.symbol,
                    exchange=OPTION_EXCHANGE,
                    action=action,
                    price_type=PRICE_TYPE,
                    product=PRODUCT,
                    quantity=quantity,
                    position_size=quantity  # Smart order adjusts based on current position
                )
                
                if response.get('status') == 'success':
                    order_id = response.get('orderid')
                    logger.info(f"✅ Order placed successfully! OrderID: {order_id}")
                    
                    # Log real order to persistence
                    if persistence and self.state.position_id:
                        try:
                            order_data = {
                                'symbol': self.symbol,
                                'action': action,
                                'quantity': quantity,
                                'order_id': order_id,
                                'price_type': PRICE_TYPE,
                                'product': PRODUCT,
                                'response': response
                            }
                            persistence.log_event(
                                event_type=EventType.ENTRY if action == ENTRY_ACTION else EventType.EXIT,
                                summary=f"Live order placed: {action} {quantity}x {self.symbol} → {order_id}",
                                position_id=self.state.position_id,
                                data=order_data
                            )
                        except Exception as e:
                            logger.error(f"✗ Error logging order to persistence: {e}")
                else:
                    error_msg = response.get('message', 'Unknown error')
                    logger.error(f"✗ Order placement failed: {error_msg}")
                    logger.error(f"  Full response: {response}")
        
        except Exception as e:
            logger.error(f"✗ Exception in _place_order: {e}")
            import traceback
            traceback.print_exc()


# ==================== STRATEGY EXECUTION ====================

def run_full_monitor():
    """Main strategy execution function"""
    global ws_client
    
    now = datetime.now(TZ)
    logger.info(f"\n{'='*70}")
    logger.info(f"▶️  Strategy Triggered at {now.strftime('%Y-%m-%d %H:%M:%S IST')}")
    if TEST_MODE:
        logger.info(f"🧪 TEST MODE ACTIVE - All logic simulated, no real orders placed")
    logger.info(f"{'='*70}\n")
    
    # Initialize WebSocket if enabled
    if USE_WEBSOCKET and not ws_client:
        logger.info("📡 Initializing WebSocket connection for real-time LTP...")
        initialize_websocket()
        time.sleep(2)  # Give WebSocket time to connect
    
    # Start persistence session
    session_id = None
    if persistence:
        try:
            session_id = persistence.start_session(underlying=UNDERLYING)
            logger.info(f"✓ Persistence session started: {session_id}")
            if TEST_MODE:
                logger.info(f"  ℹ️  Session marked as TEST MODE")
            
            # Check for crashed positions from previous runs
            crashed = persistence.get_crashed_positions(underlying=UNDERLYING)
            if crashed:
                logger.warning(f"⚠️  Found {len(crashed)} crashed positions from previous runs!")
                for pos in crashed:
                    logger.warning(f"   - {pos.symbol} @ Entry: {pos.entry_price:.2f}, Stop: {pos.stop_price:.2f}")
        except Exception as e:
            logger.error(f"✗ Error starting persistence session: {e}")
    
    # Check if it's expiry day and within market hours
    if not is_expiry_day():
        logger.info("⏭️  Not an expiry day. Exiting.")
        if persistence and session_id:
            persistence.end_session('COMPLETED')
        return
    
    if not is_market_open():
        logger.info(f"⏹️  Market hours not active ({START_HOUR:02d}:{START_MINUTE:02d}-{END_HOUR:02d}:{END_MINUTE:02d} IST). Exiting.")
        if persistence and session_id:
            persistence.end_session('COMPLETED')
        return
    
    try:
        # Resolve ATM for CE and PE
        logger.info(f"🔍 Resolving ATM for {UNDERLYING}...\n")
        
        ce_symbol, expiry, ce_strike = resolve_option_symbol("CE")
        pe_symbol, _, pe_strike = resolve_option_symbol("PE")
        
        logger.info(f"\n✓ Symbols resolved. Starting monitoring...")
        
        # Subscribe to WebSocket if available
        if ws_client and ws_client.is_connected():
            logger.info(f"📡 Subscribing to WebSocket LTP for {ce_symbol} and {pe_symbol}...")
            
            try:
                # Subscribe to both symbols
                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                async def subscribe_all():
                    await ws_client.subscribe_ltp(ce_symbol, OPTION_EXCHANGE)
                    await ws_client.subscribe_ltp(pe_symbol, OPTION_EXCHANGE)
                
                loop.run_until_complete(subscribe_all())
            except Exception as e:
                logger.warning(f"⚠️  WebSocket subscription error: {e}")
        
        # Create monitors
        ce_monitor = LegMonitor(ce_symbol, "CE", ce_strike)
        pe_monitor = LegMonitor(pe_symbol, "PE", pe_strike)
        
        # Run CE and PE monitoring in parallel threads
        threads = []
        
        if MONITOR_CE:
            ce_thread = Thread(target=ce_monitor.monitor, daemon=False, name="CE-Monitor")
            threads.append(ce_thread)
            ce_thread.start()
        
        if MONITOR_PE:
            pe_thread = Thread(target=pe_monitor.monitor, daemon=False, name="PE-Monitor")
            threads.append(pe_thread)
            pe_thread.start()
        
        # Wait for all monitoring threads to complete
        for thread in threads:
            thread.join()
        
        logger.info(f"\n✓ Strategy execution completed at {datetime.now(TZ).strftime('%H:%M:%S IST')}")
        if persistence and session_id:
            persistence.end_session('COMPLETED')
    
    except KeyboardInterrupt:
        logger.info("\n⏹️  Strategy interrupted by user.")
        if persistence and session_id:
            persistence.end_session('MANUAL_STOP')
    except Exception as e:
        logger.error(f"\n✗ Strategy error: {e}")
        import traceback
        if persistence and session_id:
            persistence.handle_crash(str(e), "Strategy execution error")
            persistence.flush_all_pending()
            persistence.end_session('CRASHED')
        traceback.print_exc()


# ==================== SCHEDULER ====================

if __name__ == "__main__":
    # RUN_IMMEDIATELY_ON_STARTUP: Bypass scheduler and run strategy immediately (for testing)
    # TEST_MODE: Simulate order placement without actually sending orders to broker
    RUN_IMMEDIATELY_ON_STARTUP = config_loader.get('strategy', 'run_immediately_on_startup', False)
    
    scheduler = BackgroundScheduler(timezone="Asia/Kolkata")
    
    # Add event listeners to track job execution
    def job_listener(event):
        if event.exception:
            logger.error(f"❌ Job {event.job_id} failed with exception: {event.exception}")
        else:
            logger.info(f"✅ Job {event.job_id} executed successfully at {datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S IST')}")
    
    scheduler.add_listener(job_listener, EVENT_JOB_EXECUTED | EVENT_JOB_ERROR)
    
    # Add the scheduled job
    scheduler.add_job(
        run_full_monitor,
        'cron',
        day_of_week='mon-fri',
        hour=START_HOUR,
        minute=START_MINUTE,
        id='expiry_blast_job'
    )
    
    scheduler.start()
    
    logger.info(f"\n{'='*70}")
    logger.info(f"✓ Scheduler Started at {datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S IST')}")
    logger.info(f"{'='*70}")
    logger.info(f"  Daily trigger: {START_HOUR:02d}:{START_MINUTE:02d} IST (Mon-Fri)")
    logger.info(f"  Configuration: {config_loader.config_path}")
    logger.info(f"  Run Immediately: {RUN_IMMEDIATELY_ON_STARTUP}")
    if TEST_MODE:
        logger.info(f"  🟢 TEST MODE ENABLED (Order Simulation):")
        logger.info(f"     • All strategy logic will execute normally")
        logger.info(f"     • All entries/exits will be tracked in database")
        logger.info(f"     • NO actual orders will be placed to broker")
        logger.info(f"     • Perfect for multi-expiry backtesting")
    else:
        logger.info(f"  🔴 LIVE TRADING MODE - Orders will be placed to broker!")
    logger.info(f"{'='*70}\n")
    
    # Run immediately if configured (bypass scheduler)
    if RUN_IMMEDIATELY_ON_STARTUP:
        logger.info(f"⚡ RUN_IMMEDIATELY_ON_STARTUP enabled: Executing strategy now...\n")
        run_full_monitor()
    
    try:
        logger.info(f"📋 Next scheduled run: {START_HOUR:02d}:{START_MINUTE:02d} IST")
        logger.info(f"⏸️  Press Ctrl+C to stop scheduler\n")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("\n\n⏹️  Scheduler stopped by user.")
        scheduler.shutdown()
        logger.info("Goodbye!")
