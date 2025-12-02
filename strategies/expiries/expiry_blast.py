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
from database.expiry_blast_db import init_position_persistence_db, EventType
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

START_HOUR = int(config_loader.get('schedule', 'start_hour', 10))
START_MINUTE = int(config_loader.get('schedule', 'start_minute', 0))
END_HOUR = int(config_loader.get('schedule', 'end_hour', 15))
END_MINUTE = int(config_loader.get('schedule', 'end_minute', 30))
CHECK_SLEEP = int(config_loader.get('schedule', 'check_interval_seconds', 5))

MONITOR_INTERVAL = config_loader.get('monitoring', 'candle_interval', '3m')
LOOKBACK_CANDLES = int(config_loader.get('monitoring', 'lookback_candles', 5))
BREAKOUT_THRESHOLD_PCT = float(config_loader.get('monitoring', 'breakout_threshold_pct', 0.0))

ENTRY_QUANTITY = int(config_loader.get('position', 'entry_quantity', 75))
LOT_MULTIPLIER = int(config_loader.get('position', 'lot_multiplier', 1))
FINAL_QUANTITY = int(ENTRY_QUANTITY * LOT_MULTIPLIER)
INITIAL_STOP_PCT = float(config_loader.get('position', 'initial_stop_pct', 0.50))
TRAIL_PERCENT_STEP = float(config_loader.get('position', 'trail_step_pct', 0.01))
PROFIT_TARGET_PCT = float(config_loader.get('position', 'profit_target_pct', 1.00))
ATM_RECHECK_INTERVAL = int(config_loader.get('position', 'atm_recheck_interval', 5))

# Move SL to Cost Feature Configuration
MOVE_SL_TO_COST_ENABLED = config_loader.get('position', 'move_sl_to_cost_enabled', True)
MOVE_SL_PROFIT_THRESHOLD_PCT = float(config_loader.get('position', 'move_sl_profit_threshold_pct', 0.10))
SL_COST_OFFSET_PCT = float(config_loader.get('position', 'sl_cost_offset_pct', 0.02))

# Profit Trailing Feature Configuration
PROFIT_TRAILING_ENABLED = config_loader.get('position', 'profit_trailing_enabled', True)
PROFIT_TRAILING_STEP_PCT = float(config_loader.get('position', 'profit_trailing_step_pct', 0.05))
PROFIT_TRAILING_INCREMENT_PCT = float(config_loader.get('position', 'profit_trailing_increment_pct', 0.01))

AUTO_PLACE_ORDERS = config_loader.get('orders', 'auto_place_orders', False)
PRICE_TYPE = config_loader.get('orders', 'price_type', 'MARKET')
PRODUCT = config_loader.get('orders', 'product', 'NRML')
INSTRUMENT_TYPE = config_loader.get('orders', 'instrument_type', 'options')
ENTRY_ACTION = config_loader.get('orders', 'entry_action', 'BUY')
EXIT_ACTION = config_loader.get('orders', 'exit_action', 'SELL')
PLACE_SL_ORDER = config_loader.get('orders', 'place_sl_order', True)
SL_ORDER_TYPE = config_loader.get('orders', 'sl_order_type', 'SL')  # SL (Limit) - most brokers require this
SL_LIMIT_BUFFER = float(config_loader.get('orders', 'sl_limit_buffer', 0.50))  # Gap between trigger and limit price (in rupees)
SL_LIMIT_BUFFER_PERCENT = float(config_loader.get('orders', 'sl_limit_buffer_percent', 0.015))  # Buffer as percentage of stop price (1.5% default)
SL_USE_PERCENT_BUFFER = config_loader.get('orders', 'sl_use_percent_buffer', True)  # Use percentage instead of fixed buffer

PRINT_QUOTES_IMMEDIATELY = config_loader.get('logging', 'print_quotes_immediately', True)
PRINT_CANDLE_DATA = config_loader.get('logging', 'print_candle_data', True)
PRINT_POSITION_UPDATES = config_loader.get('logging', 'print_position_updates', True)

MONITOR_ATM_CHANGES = config_loader.get('atm_monitoring', 'monitor_atm_changes', True)
ATM_CHECK_INTERVAL_MINUTES = int(config_loader.get('atm_monitoring', 'atm_check_interval_minutes', 5))
CLOSE_ON_ATM_CHANGE = config_loader.get('atm_monitoring', 'close_on_atm_change', True)
ATM_STRIKE_CHANGE_BUFFER = float(config_loader.get('atm_monitoring', 'atm_strike_change_buffer', 20))

MONITOR_CE = config_loader.get('legs', 'monitor_ce', True)
MONITOR_PE = config_loader.get('legs', 'monitor_pe', True)
CE_STRIKE_SELECTION = config_loader.get('legs', 'ce_strike_selection', 'ATM').upper()
PE_STRIKE_SELECTION = config_loader.get('legs', 'pe_strike_selection', 'ATM').upper()

# ==================== WEBSOCKET CONFIGURATION ====================
USE_WEBSOCKET = config_loader.get('websocket', 'enabled', True)
WEBSOCKET_URL = os.getenv('WEBSOCKET_URL', config_loader.get('websocket', 'url', 'ws://127.0.0.1:8765'))
WEBSOCKET_TIMEOUT = int(config_loader.get('websocket', 'timeout_seconds', 10))
WEBSOCKET_RECONNECT_INTERVAL = int(config_loader.get('websocket', 'reconnect_interval_seconds', 5))

# ==================== TEST MODE CONFIGURATION ====================
# Set to True to simulate all strategy logic without placing actual orders
# All positions will be tracked in DB as if they were real trades
TEST_MODE = config_loader.get('strategy', 'test_mode', False)

TZ = pytz.timezone(config_loader.get('strategy', 'timezone', 'Asia/Kolkata'))

logger.info(f"\n{'='*60}")
logger.info(f"🔁 Expiry Blast Strategy - OpenAlgo")
logger.info(f"{'='*60}")
logger.info(f"Underlying: {UNDERLYING} | Exchange: {UNDERLYING_EXCHANGE}")
logger.info(f"Strike Selection: CE={CE_STRIKE_SELECTION} | PE={PE_STRIKE_SELECTION}")
logger.info(f"Start Time: {START_HOUR:02d}:{START_MINUTE:02d} IST | End Time: {END_HOUR:02d}:{END_MINUTE:02d} IST")
logger.info(f"Expiry Day Only: {RUN_ONLY_ON_EXPIRY_DAY}")
logger.info(f"Auto Place Orders: {AUTO_PLACE_ORDERS}")
logger.info(f"Monitor ATM Changes: {MONITOR_ATM_CHANGES} (Buffer: {ATM_STRIKE_CHANGE_BUFFER} points)")
logger.info(f"WebSocket: {'📡 ENABLED' if USE_WEBSOCKET else '🔄 DISABLED (REST API polling)'} @ {WEBSOCKET_URL if USE_WEBSOCKET else 'N/A'}")
logger.info(f"TEST MODE: {'🟢 ENABLED (Orders simulated, not placed)' if TEST_MODE else '🔴 DISABLED (Live trading)'}")
logger.info(f"Move SL to Cost: {'✅ ENABLED' if MOVE_SL_TO_COST_ENABLED else '❌ DISABLED'} (Threshold: {MOVE_SL_PROFIT_THRESHOLD_PCT*100:.0f}%, Offset: {SL_COST_OFFSET_PCT*100:.0f}%)")
logger.info(f"Profit Trailing: {'✅ ENABLED' if PROFIT_TRAILING_ENABLED else '❌ DISABLED'} (Step: {PROFIT_TRAILING_STEP_PCT*100:.0f}%, Increment: {PROFIT_TRAILING_INCREMENT_PCT*100:.0f}%)")
logger.info(f"{'='*60}\n")

# ==================== TICK SIZE UTILITIES ====================

def round_to_tick_size(price: float, tick_size: float = 0.05) -> float:
    """Round price to nearest tick size (default 0.05 for options)"""
    return round(price / tick_size) * tick_size

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

# Global cache for expiry date (fetched once per session)
_CACHED_EXPIRY: Optional[str] = None

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


def get_nearest_expiry(force_refresh: bool = False) -> str:
    """Fetch nearest weekly expiry and format it (cached for session)
    
    Args:
        force_refresh: If True, bypass cache and fetch fresh data
    
    Returns:
        Formatted expiry string (e.g., '02DEC25')
    """
    global _CACHED_EXPIRY
    
    # Return cached value if available and not forcing refresh
    if _CACHED_EXPIRY and not force_refresh:
        return _CACHED_EXPIRY
    
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
        
        # Cache for session
        _CACHED_EXPIRY = expiry_fmt
        return expiry_fmt
    except Exception as e:
        logger.error(f"✗ Error fetching expiry: {e}")
        raise


def resolve_option_symbol(
    option_type: str,
    expiry: Optional[str] = None,
    underlying: Optional[str] = None,
    strike_selection: str = 'ATM'
) -> Tuple[str, str, float]:
    """
    Resolve option symbol using optionchain API with configurable strike selection
    
    Args:
        option_type: "CE" or "PE"
        expiry: Pre-fetched expiry string (if None, will fetch from cache)
        underlying: Underlying symbol (if None, uses global UNDERLYING)
        strike_selection: Strike to select - "ATM", "ITM1"-"ITM5", "OTM1"-"OTM5"
                         ITM means in-the-money, OTM means out-of-the-money
    
    Returns:
        Tuple of (symbol, expiry, strike_price)
    """
    if underlying is None:
        underlying = UNDERLYING
    if expiry is None:
        # Use cached expiry (no API call if already fetched)
        expiry = get_nearest_expiry()
    
    try:
        # Parse strike selection (e.g., "ATM", "ITM1", "OTM2")
        strike_offset = 0
        strike_type = strike_selection.upper()
        
        if strike_type == 'ATM':
            strike_offset = 0
        elif strike_type.startswith('ITM'):
            # ITM1, ITM2, etc. - extract number
            try:
                strike_offset = -int(strike_type[3:])  # Negative for ITM (below spot for CE, above for PE)
            except (ValueError, IndexError):
                logger.warning(f"⚠️  Invalid strike selection '{strike_selection}', defaulting to ATM")
                strike_offset = 0
        elif strike_type.startswith('OTM'):
            # OTM1, OTM2, etc. - extract number
            try:
                strike_offset = int(strike_type[3:])  # Positive for OTM (above spot for CE, below for PE)
            except (ValueError, IndexError):
                logger.warning(f"⚠️  Invalid strike selection '{strike_selection}', defaulting to ATM")
                strike_offset = 0
        else:
            logger.warning(f"⚠️  Unknown strike selection '{strike_selection}', defaulting to ATM")
            strike_offset = 0
        
        # Fetch option chain - get more strikes to support ITM/OTM selection
        fetch_count = max(abs(strike_offset) + 2, 3)  # Fetch enough strikes
        
        resp = client.optionchain(
            underlying=underlying,
            exchange=UNDERLYING_EXCHANGE,
            expiry_date=expiry,
            strike_count=fetch_count
        )
        
        if resp.get('status') != 'success':
            raise RuntimeError(f"Option chain fetch failed: {resp}")
        
        atm_strike = float(resp.get('atm_strike', 0))
        chain = resp.get('chain', [])
        
        if not chain:
            raise RuntimeError(f"No chain data in response: {resp}")
        
        # Sort chain by strike
        chain = sorted(chain, key=lambda x: x['strike'])
        
        # Find ATM index
        atm_index = next((i for i, item in enumerate(chain) if item['strike'] == atm_strike), None)
        if atm_index is None:
            raise RuntimeError(f"ATM strike {atm_strike} not found in chain")
        
        # Calculate target strike index based on option type and offset
        # For CE: ITM = lower strikes, OTM = higher strikes
        # For PE: ITM = higher strikes, OTM = lower strikes
        if option_type.upper() == 'CE':
            target_index = atm_index + strike_offset
        else:  # PE
            target_index = atm_index - strike_offset
        
        # Ensure target index is within bounds
        target_index = max(0, min(target_index, len(chain) - 1))
        
        # Get the target strike item
        target_item = chain[target_index]
        target_strike = target_item['strike']
        
        # Extract symbol for the requested option type
        option_data = target_item.get(option_type.lower())
        if not option_data or not option_data.get('symbol'):
            raise RuntimeError(f"No {option_type} data for strike {target_strike}")
        
        symbol = option_data['symbol']
        resolved_expiry = resp.get('expiry_date', expiry)
        strike = target_strike
        
        # Log with strike selection info and fetch underlying + option LTP
        strike_desc = f"{strike_selection} (ATM: {atm_strike:.0f})"
        
        # Fetch underlying (spot) price
        try:
            underlying_quote = client.quotes(symbol=underlying, exchange=UNDERLYING_EXCHANGE)
            if isinstance(underlying_quote, dict) and 'data' in underlying_quote:
                if isinstance(underlying_quote['data'], dict):
                    spot_ltp = float(underlying_quote['data'].get('ltp', 0))
                elif isinstance(underlying_quote['data'], list) and len(underlying_quote['data']) > 0:
                    spot_ltp = float(underlying_quote['data'][0].get('ltp', 0))
                else:
                    spot_ltp = None
            else:
                spot_ltp = None
        except:
            spot_ltp = None
        
        # Fetch option LTP
        option_ltp = get_ltp(symbol)
        
        # Build log message
        spot_str = f"{underlying} Spot: {spot_ltp:.2f}" if spot_ltp else f"{underlying} Spot: N/A"
        option_str = f"Option: {option_ltp:.2f}" if option_ltp is not None else "Option: N/A"
        
        logger.info(f"✓ Resolved {option_type:2s} {strike_desc} @ Strike {strike:7.0f}: {symbol} | {spot_str} | {option_str}")
        
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
            logger.info(f"\n{'='*95}")
            logger.info(f"📊 [{symbol}] Last {LOOKBACK_CANDLES} {MONITOR_INTERVAL} candles:")
            logger.info(f"{'='*95}")
            
            # Header row with volume
            logger.info(f"  {'Timestamp':<19} | {'Open':>7} | {'High':>7} | {'Low':>7} | {'Close':>7} | {'Volume':>10}")
            logger.info(f"  {'-'*19}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*10}")
            
            # Format data with better alignment including volume
            for idx, row in df[['open', 'high', 'low', 'close', 'volume']].iterrows():
                timestamp_str = idx.strftime('%Y-%m-%d %H:%M:%S')
                logger.info(f"  {timestamp_str} | {row['open']:7.2f} | {row['high']:7.2f} | {row['low']:7.2f} | {row['close']:7.2f} | {row['volume']:10,.0f}")
            
            logger.info(f"{'='*95}")
            avg_price = df['close'].mean()
            logger.info(f"  📈 Highest High: {df['high'].max():.2f} | 📉 Lowest Low: {df['low'].min():.2f} | 📊 Average Price: {avg_price:.2f}")
            logger.info(f"{'='*95}\n")
        
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
        
        # Debug: Log the response structure
        logger.debug(f"Quote API response for {symbol}: {q}")
        
        # Handle different response formats
        if isinstance(q, dict):
            # Try different possible structures
            if 'data' in q and isinstance(q['data'], dict) and 'ltp' in q['data']:
                ltp = float(q['data']['ltp'])
            elif 'ltp' in q:
                ltp = float(q['ltp'])
            elif 'data' in q and isinstance(q['data'], list) and len(q['data']) > 0:
                ltp = float(q['data'][0].get('ltp', 0))
            else:
                logger.error(f"✗ Unexpected quote response format for {symbol}: {q}")
                return None
        else:
            logger.error(f"✗ Quote response is not a dict for {symbol}: {type(q)}")
            return None
        
        if PRINT_QUOTES_IMMEDIATELY:
            logger.info(f"💹 {symbol:15s} LTP: {ltp:8.2f} (REST)")
        
        return ltp
    
    except KeyError as e:
        logger.error(f"✗ KeyError fetching LTP for {symbol}: Missing key {e}")
        logger.error(f"   Response structure: {q if 'q' in locals() else 'N/A'}")
        return None
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
    sl_order_id: Optional[str] = None  # Broker-side stop-loss order ID
    sl_moved_to_cost: bool = False  # Track if SL has been moved to cost+offset
    profit_trailing_level: float = 0.0  # Track current profit trailing level
    exit_triggered: bool = False  # Prevent re-entry after SL/profit exit
    position_ever_opened: bool = False  # Track if position was ever opened in this session


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
            position_id=None,
            sl_order_id=None,
            sl_moved_to_cost=False,
            profit_trailing_level=0.0,
            exit_triggered=False
        )
        self.candle_count = 0
        self.highest_high = 0
        self.last_breakout_update = 0  # Track iterations since last breakout update
    
    def initialize(self):
        """Initialize by fetching initial candles and setting breakout level"""
        logger.info(f"\n{'='*60}")
        logger.info(f"🔍 Initializing {self.leg_type} Monitor: {self.symbol}")
        logger.info(f"{'='*60}")
        
        df = fetch_last_candles(self.symbol)
        self.highest_high = df['high'].max()
        
        logger.info(f"🎯 {self.leg_type} Breakout Level (Highest High): {self.highest_high:.2f}")
        
        self.candle_count = len(df)
    
    def update_breakout_levels(self):
        """Update breakout levels from latest candles (dynamic support/resistance)"""
        self.last_breakout_update += 1
        
        # Calculate update interval: candle_interval / check_interval
        # For 3m candles with 5s checks: 180s / 5s = 36 iterations = 1 candle
        interval_seconds = int(MONITOR_INTERVAL.replace('m', '')) * 60
        update_interval = interval_seconds // CHECK_SLEEP
        
        # Update breakout levels every candle
        if self.last_breakout_update >= update_interval:
            self.last_breakout_update = 0
            
            try:
                df = fetch_last_candles(self.symbol)
                old_high = self.highest_high
                
                self.highest_high = df['high'].max()
                
                # Log only if level changed significantly (avoid spam)
                if abs(self.highest_high - old_high) > 0.01:
                    logger.info(f"🔄 {self.leg_type} Breakout Level Updated: {old_high:.2f} → {self.highest_high:.2f}")
            
            except Exception as e:
                logger.error(f"⚠️  Error updating breakout levels for {self.leg_type}: {e}")
    
    def check_atm_changed(self) -> bool:
        """Check if ATM has changed during the day with buffer logic"""
        if not MONITOR_ATM_CHANGES:
            return False
        
        self.state.last_atm_check += 1
        
        if self.state.last_atm_check >= ATM_RECHECK_INTERVAL:
            self.state.last_atm_check = 0
            
            try:
                # Use cached expiry (no API call needed)
                cached_expiry = get_nearest_expiry()
                # Use appropriate strike selection based on leg type
                strike_sel = CE_STRIKE_SELECTION if self.leg_type == 'CE' else PE_STRIKE_SELECTION
                new_symbol, _, new_strike = resolve_option_symbol(
                    self.leg_type,
                    expiry=cached_expiry,
                    underlying=UNDERLYING,
                    strike_selection=strike_sel
                )
                
                if new_strike != self.state.current_strike:
                    # Calculate strike difference
                    strike_diff = abs(new_strike - self.state.current_strike)
                    
                    # Only trigger ATM change if difference exceeds buffer
                    if strike_diff > ATM_STRIKE_CHANGE_BUFFER:
                        logger.warning(f"\n⚠️  ATM CHANGED for {self.leg_type} (Strike moved {strike_diff:.0f} points > Buffer {ATM_STRIKE_CHANGE_BUFFER:.0f})")
                        logger.warning(f"   Old Strike: {self.state.current_strike} ({self.symbol})")
                        logger.warning(f"   New Strike: {new_strike} ({new_symbol})")
                        return True
                    else:
                        # Strike changed but within buffer - log but don't trigger reset
                        if self.candle_count % 20 == 0:  # Log occasionally to avoid spam
                            logger.debug(f"ℹ️  {self.leg_type} Strike changed {strike_diff:.0f} points (within buffer {ATM_STRIKE_CHANGE_BUFFER:.0f}) - Ignoring")
            
            except Exception as e:
                logger.error(f"⚠️  Error checking ATM change: {e}")
        
        return False
    
    def monitor(self, skip_init: bool = False):
        """Main monitoring loop for this leg - supports WebSocket or polling"""
        logger.info(f"▶️  Starting {self.leg_type} monitoring loop...")
        
        if not skip_init:
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
            
            # Get current LTP
            if ws_client and ws_client.is_connected() and price_event:
                # WebSocket mode: wait for price update or timeout
                price_event.clear()
                if price_event.wait(timeout=CHECK_SLEEP):
                    ltp = last_price_holder['price']
                    if iteration % 20 == 0:  # Log every 20 iterations to avoid spam
                        logger.debug(f"📡 {self.leg_type} WebSocket LTP: {ltp:.2f}")
                else:
                    # Timeout - fallback to REST API
                    if iteration % 10 == 0:  # Log occasionally
                        logger.warning(f"⚠️  {self.leg_type} WebSocket timeout, falling back to REST API")
                    ltp = get_ltp(self.symbol)
                    if ltp is None:
                        time.sleep(CHECK_SLEEP)
                        continue
            else:
                # Polling mode: fetch via REST API
                ltp = get_ltp(self.symbol)
                if ltp is None:
                    time.sleep(CHECK_SLEEP)
                    continue
                
                time.sleep(CHECK_SLEEP)
            
            iteration += 1
            
            # Log monitoring activity every 10 iterations
            if iteration % 10 == 0:
                logger.info(f"🔍 {self.leg_type} Monitor: LTP={ltp:.2f} | Breakout={self.highest_high:.2f} | Position={'OPEN' if self.state.position_open else 'WAITING'}")
            
            # ========== NO POSITION: CHECK ATM & ENTRY ==========
            if not self.state.position_open:
                # If position was already exited (SL/profit), don't re-enter
                if self.state.exit_triggered:
                    if iteration % 20 == 0:  # Log occasionally
                        logger.info(f"ℹ️  {self.leg_type} monitor idle (position already exited)")
                    time.sleep(CHECK_SLEEP)
                    continue
                
                # Update breakout levels dynamically (only when waiting for entry)
                self.update_breakout_levels()
                
                # Check if ATM has changed (only when no position)
                if self.check_atm_changed() and CLOSE_ON_ATM_CHANGE:
                    logger.info(f"🔄 ATM changed. Resetting {self.leg_type} monitor for new ATM strike.")
                    # Reset for new ATM - but preserve position_ever_opened flag
                    position_was_opened = self.state.position_ever_opened
                    self.initialize()
                    self.state.last_trail_level = 0
                    self.state.position_ever_opened = position_was_opened  # Preserve session-level flag
                    if position_was_opened:
                        self.state.exit_triggered = True  # Prevent re-entry if position was opened before
                        logger.info(f"⏸️ {self.leg_type} position was already opened this session. Staying in idle mode.")
                    continue
                
                # Entry logic: Check for breakout
                if ltp > self.highest_high * (1 + BREAKOUT_THRESHOLD_PCT):
                    self.state.entry_price = ltp
                    self.state.entry_time = datetime.now(TZ)
                    self.state.position_open = True
                    self.state.position_ever_opened = True  # Mark that position was opened in this session
                    self.state.stop_price = round_to_tick_size(ltp * (1 - INITIAL_STOP_PCT))
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
                    
                    # Calculate entry details
                    num_lots = LOT_MULTIPLIER
                    risk_per_unit = self.state.entry_price - self.state.stop_price
                    total_risk = risk_per_unit * FINAL_QUANTITY
                    total_value = self.state.entry_price * FINAL_QUANTITY
                    
                    entry_msg = (
                        f"\n{'='*70}\n"
                        f"✅ ENTRY SIGNAL: {self.leg_type}\n"
                        f"   Symbol: {self.symbol}\n"
                        f"   Entry Price: ₹{self.state.entry_price:.2f}\n"
                        f"   Stop Loss: ₹{self.state.stop_price:.2f} (Risk: ₹{risk_per_unit:.2f}/unit)\n"
                        f"   Quantity: {FINAL_QUANTITY} ({ENTRY_QUANTITY} x {num_lots} lot{'s' if num_lots > 1 else ''})\n"
                        f"   Total Value: ₹{total_value:,.2f}\n"
                        f"   Total Risk: ₹{total_risk:,.2f}\n"
                        f"   Time: {self.state.entry_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"{'='*70}\n"
                    )
                    logger.info(entry_msg)
                    
                    if AUTO_PLACE_ORDERS:
                        self._place_order(ENTRY_ACTION, FINAL_QUANTITY)
                        
                        # Place stop-loss order on broker for protection
                        if PLACE_SL_ORDER:
                            self.state.sl_order_id = self._place_sl_order(self.state.stop_price)
                            if self.state.sl_order_id:
                                logger.info(f"🛡️  Broker-side SL protection activated @ {self.state.stop_price:.2f}")
                            else:
                                logger.warning(f"⚠️  Failed to place broker-side SL order - using code-based monitoring only")
            
            # ========== POSITION OPEN: MANAGE STOP LOSS & PROFIT TARGET ==========
            elif self.state.position_open:
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
                
                # ========== NEW FEATURE: MOVE SL TO COST ==========
                # Once profit reaches threshold, move SL to entry + offset (e.g., cost + 2%)
                if MOVE_SL_TO_COST_ENABLED and not self.state.sl_moved_to_cost:
                    if profit_pct >= MOVE_SL_PROFIT_THRESHOLD_PCT:
                        cost_plus_offset = round_to_tick_size(self.state.entry_price * (1 + SL_COST_OFFSET_PCT))
                        
                        # Only move if new stop is higher than current stop
                        if cost_plus_offset > self.state.stop_price:
                            old_stop = self.state.stop_price
                            self.state.stop_price = cost_plus_offset
                            self.state.sl_moved_to_cost = True
                            
                            # Modify broker-side SL order
                            if self.state.sl_order_id:
                                logger.info(f"🔒 Moving SL to Cost+{SL_COST_OFFSET_PCT*100:.0f}% (Profit: {profit_pct*100:.1f}%)")
                                logger.info(f"   Old SL: {old_stop:.2f} → New SL: {self.state.stop_price:.2f}")
                                sl_modified = self._modify_sl_order(self.state.sl_order_id, self.state.stop_price)
                                if not sl_modified:
                                    logger.warning(f"⚠️  Failed to modify broker SL to cost - continuing with code monitoring")
                            else:
                                # No SL order ID but position is open - try to place new SL
                                logger.warning(f"⚠️  No broker SL order ID found - attempting to place new SL order at cost")
                                if PLACE_SL_ORDER:
                                    new_sl_id = self._place_sl_order(self.state.stop_price)
                                    if new_sl_id:
                                        self.state.sl_order_id = new_sl_id
                                        logger.info(f"✅ New SL order placed at cost: {new_sl_id} @ {self.state.stop_price:.2f}")
                                    else:
                                        logger.warning(f"⚠️  Failed to place new SL - using code-based monitoring only")
                            
                            if PRINT_POSITION_UPDATES:
                                logger.info(f"🔒 {self.leg_type} SL Moved to Cost+{SL_COST_OFFSET_PCT*100:.0f}%: {old_stop:.2f} → {self.state.stop_price:.2f}")
                
                # ========== NEW FEATURE: PROFIT TRAILING ==========
                # Calculate dynamic profit target based on profit levels
                current_profit_target = PROFIT_TARGET_PCT
                
                if PROFIT_TRAILING_ENABLED and profit_pct > 0:
                    # Calculate how many trailing steps have been achieved
                    trailing_steps = int(profit_pct / PROFIT_TRAILING_STEP_PCT)
                    
                    # Update profit target based on trailing steps
                    if trailing_steps > 0:
                        # Add increment for each step achieved
                        additional_target = trailing_steps * PROFIT_TRAILING_INCREMENT_PCT
                        current_profit_target = PROFIT_TARGET_PCT + additional_target
                        
                        # Log when we cross a new trailing level
                        if trailing_steps > self.state.profit_trailing_level:
                            self.state.profit_trailing_level = trailing_steps
                            if PRINT_POSITION_UPDATES:
                                logger.info(f"📈 {self.leg_type} Profit Trailing Level {trailing_steps}: Target moved to {current_profit_target*100:.1f}% (Profit: {profit_pct*100:.1f}%)")
                
                # Check for profit target exit (using dynamic target)
                if profit_pct >= current_profit_target:
                    # Calculate profit details
                    num_lots = LOT_MULTIPLIER
                    profit_per_unit = gain_points
                    total_profit = profit_per_unit * FINAL_QUANTITY
                    entry_value = self.state.entry_price * FINAL_QUANTITY
                    exit_value = ltp * FINAL_QUANTITY
                    
                    exit_msg = (
                        f"\n{'='*70}\n"
                        f"🎯 PROFIT TARGET HIT: {self.leg_type}\n"
                        f"   Symbol: {self.symbol}\n"
                        f"   Entry Price: ₹{self.state.entry_price:.2f} | Exit Price: ₹{ltp:.2f}\n"
                        f"   Profit per Unit: ₹{profit_per_unit:.2f} ({profit_pct*100:.1f}%)\n"
                        f"   Quantity: {FINAL_QUANTITY} ({ENTRY_QUANTITY} x {num_lots} lot{'s' if num_lots > 1 else ''})\n"
                        f"   Entry Value: ₹{entry_value:,.2f}\n"
                        f"   Exit Value: ₹{exit_value:,.2f}\n"
                        f"   Total Profit: ₹{total_profit:,.2f}\n"
                        f"   Exit Time: {datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"{'='*70}\n"
                    )
                    logger.info(exit_msg)
                    
                    # Cancel broker-side SL order before manual exit
                    if self.state.sl_order_id:
                        self._cancel_sl_order(self.state.sl_order_id)
                        self.state.sl_order_id = None
                    
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
                    self.state.exit_triggered = True  # Prevent re-entry
                    logger.info(f"✓ {self.leg_type} monitor entering idle mode (profit target reached)")
                    # Continue monitoring but don't re-enter
                    continue
                
                # Trailing stop logic - Move stop up by exactly the profit amount
                if profit_pct > 0:
                    # Calculate new stop: current_price - (entry_price * initial_stop_pct)
                    # This maintains a constant % buffer below price
                    new_stop = ltp - (self.state.entry_price * INITIAL_STOP_PCT)
                    
                    # Only update if stop has moved up
                    if new_stop > self.state.stop_price:
                        old_stop = self.state.stop_price
                        self.state.stop_price = round_to_tick_size(new_stop)
                        
                        # Modify broker-side SL order (use rounded stop price)
                        if self.state.sl_order_id:
                            logger.info(f"🔄 Attempting to modify broker SL order {self.state.sl_order_id}: {old_stop:.2f} → {self.state.stop_price:.2f}")
                            sl_modified = self._modify_sl_order(self.state.sl_order_id, self.state.stop_price)
                            if not sl_modified:
                                logger.warning(f"⚠️  Failed to modify broker SL - continuing with code monitoring")
                        else:
                            # No SL order ID but position is open - try to place new SL
                            logger.warning(f"⚠️  No broker SL order ID found - attempting to place new SL order")
                            if PLACE_SL_ORDER:
                                new_sl_id = self._place_sl_order(self.state.stop_price)
                                if new_sl_id:
                                    self.state.sl_order_id = new_sl_id
                                    logger.info(f"✅ New SL order placed: {new_sl_id} @ {self.state.stop_price:.2f}")
                                else:
                                    logger.warning(f"⚠️  Failed to place new SL - using code-based monitoring only")
                        
                        if PRINT_POSITION_UPDATES:
                            logger.info(f"📈 {self.leg_type} Trail Update: Stop {old_stop:.2f} → {self.state.stop_price:.2f} | Profit: {profit_pct*100:.1f}%")
                
                # Check for stop loss
                if ltp <= self.state.stop_price:
                    # Calculate loss details
                    num_lots = LOT_MULTIPLIER
                    loss_per_unit = abs(gain_points)
                    total_loss = loss_per_unit * FINAL_QUANTITY
                    entry_value = self.state.entry_price * FINAL_QUANTITY
                    exit_value = ltp * FINAL_QUANTITY
                    
                    exit_msg = (
                        f"\n{'='*70}\n"
                        f"🛑 STOP LOSS HIT: {self.leg_type}\n"
                        f"   Symbol: {self.symbol}\n"
                        f"   Entry Price: ₹{self.state.entry_price:.2f} | Exit Price: ₹{ltp:.2f}\n"
                        f"   Loss per Unit: ₹{loss_per_unit:.2f} ({profit_pct*100:.1f}%)\n"
                        f"   Quantity: {FINAL_QUANTITY} ({ENTRY_QUANTITY} x {num_lots} lot{'s' if num_lots > 1 else ''})\n"
                        f"   Entry Value: ₹{entry_value:,.2f}\n"
                        f"   Exit Value: ₹{exit_value:,.2f}\n"
                        f"   Total Loss: ₹{total_loss:,.2f}\n"
                        f"   Stop Price: {self.state.stop_price:.2f}\n"
                        f"   Exit Time: {datetime.now(TZ).strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"{'='*70}\n"
                    )
                    logger.info(exit_msg)
                    
                    # Note: Broker-side SL should execute automatically
                    # Cancel it only if code exits first (race condition)
                    if self.state.sl_order_id:
                        logger.info(f"ℹ️  Broker SL order {self.state.sl_order_id} should have triggered automatically")
                        self._cancel_sl_order(self.state.sl_order_id)  # Cancel if still pending
                        self.state.sl_order_id = None
                    
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
                    self.state.exit_triggered = True  # Prevent re-entry
                    logger.info(f"✓ {self.leg_type} monitor entering idle mode (stop loss hit)")
                    # Continue monitoring but don't re-enter
                    continue
            
            time.sleep(CHECK_SLEEP)
    
    def _verify_order_status(self, order_id: str, expected_status: str = 'open', max_retries: int = 3) -> bool:
        """Verify order status with broker (with retry logic)
        
        Args:
            order_id: Order ID to verify
            expected_status: Expected status ('open', 'complete', 'rejected', etc.)
            max_retries: Maximum number of retry attempts
        
        Returns:
            True if order exists and matches expected status, False otherwise
        """
        if not order_id or TEST_MODE:
            return True  # Skip verification in test mode
        
        for attempt in range(max_retries):
            try:
                # Add small delay between retries to allow order to sync
                if attempt > 0:
                    time.sleep(0.5)
                    logger.debug(f"Retry {attempt + 1}/{max_retries} for order {order_id}")
                
                response = client.orderbook()
                
                # Log full response for debugging (only on first attempt)
                if attempt == 0:
                    logger.debug(f"📋 Orderbook API Response Type: {type(response)}")
                    logger.debug(f"📋 Orderbook API Response: {response}")
                
                # Handle if response is a string (error message)
                if isinstance(response, str):
                    logger.warning(f"⚠️  Orderbook returned string response: {response}")
                    continue
                
                # Handle dict response
                if not isinstance(response, dict):
                    logger.warning(f"⚠️  Unexpected orderbook response type: {type(response)}")
                    continue
                
                if response.get('status') != 'success':
                    logger.warning(f"⚠️  Orderbook fetch failed: {response.get('message', 'Unknown error')}")
                    continue
                
                data = response.get('data', [])
                orders = []
                
                # Handle different response formats
                if isinstance(data, list):
                    # Standard format: data is a list of orders
                    orders = data
                elif isinstance(data, dict):
                    # Alternative format: data is a dict with 'orders' key
                    if 'orders' in data:
                        orders_data = data['orders']
                        orders = orders_data if isinstance(orders_data, list) else [orders_data]
                    else:
                        # Check if data itself is the order (has orderid)
                        if 'orderid' in data:
                            orders = [data]
                        else:
                            # Try to extract any dict values that look like orders
                            for key, value in data.items():
                                if isinstance(value, dict) and 'orderid' in value:
                                    if isinstance(value, list):
                                        orders.extend(value)
                                    else:
                                        orders.append(value)
                
                if not orders:
                    logger.debug(f"No orders found in orderbook (attempt {attempt + 1}/{max_retries})")
                    continue
                
                # Find order by ID (try exact match and string comparison)
                order = None
                for o in orders:
                    if isinstance(o, dict):
                        o_id = str(o.get('orderid', ''))
                        if o_id == str(order_id):
                            order = o
                            break
                
                if not order:
                    logger.debug(f"Order {order_id} not found in {len(orders)} orders (attempt {attempt + 1}/{max_retries})")
                    continue
                
                # Found the order - check status
                order_status = str(order.get('status', '')).lower()
                logger.info(f"✓ Order {order_id} found with status: {order_status}")
                
                # Check if order is in expected state (flexible matching)
                expected_lower = expected_status.lower()
                if (expected_lower in order_status or 
                    order_status in expected_lower or
                    order_status == expected_lower or
                    (expected_lower == 'open' and order_status in ['pending', 'trigger pending', 'open']) or
                    (expected_lower == 'complete' and order_status in ['complete', 'filled', 'executed'])):
                    return True
                else:
                    logger.warning(f"⚠️  Order {order_id} status '{order_status}' doesn't match expected '{expected_status}'")
                    return False
            
            except Exception as e:
                logger.error(f"✗ Exception verifying order status (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    import traceback
                    logger.debug(f"Final attempt failed. Traceback: {traceback.format_exc()}")
        
        # All retries exhausted
        logger.warning(f"⚠️  Failed to verify order {order_id} after {max_retries} attempts")
        return False
    
    def _place_sl_order(self, stop_price: float) -> Optional[str]:
        """Place a stop-loss order on the broker"""
        if not AUTO_PLACE_ORDERS or not PLACE_SL_ORDER:
            return None
        
        if TEST_MODE:
            logger.info(f"🧪 TEST MODE - Simulated SL order @ {stop_price:.2f}")
            return f"TEST_SL_{self.symbol}"
        
        try:
            # Round to tick size to avoid broker rejection
            rounded_trigger = round_to_tick_size(stop_price)
            
            # Calculate buffer (percentage-based or fixed)
            if SL_USE_PERCENT_BUFFER:
                buffer = rounded_trigger * SL_LIMIT_BUFFER_PERCENT
            else:
                buffer = SL_LIMIT_BUFFER
            
            # For SL order, limit price should be slightly worse than trigger (for SELL, lower)
            rounded_limit = round_to_tick_size(rounded_trigger - buffer)
            
            logger.info(f"📤 Placing SL-L order on broker: {self.symbol}")
            logger.info(f"   Trigger: {rounded_trigger:.2f} | Limit: {rounded_limit:.2f} (Buffer: {buffer:.2f} / {(buffer/rounded_trigger)*100:.2f}%)")
            
            response = client.placeorder(
                strategy=f"expiry_blast_{self.leg_type}_SL",
                symbol=self.symbol,
                exchange=OPTION_EXCHANGE,
                action=EXIT_ACTION,
                price_type="SL",  # Stop Loss Limit order
                product=PRODUCT,
                quantity=FINAL_QUANTITY,
                trigger_price=rounded_trigger,
                price=rounded_limit  # Limit price with buffer
            )
            
            # Log full response for debugging
            logger.debug(f"📤 SL PlaceOrder API Response Type: {type(response)}")
            logger.debug(f"📤 SL PlaceOrder API Response: {response}")
            
            if response.get('status') == 'success':
                order_id = response.get('orderid')
                logger.info(f"✅ SL order placed on broker! OrderID: {order_id} @ Trigger: {stop_price:.2f}")
                
                # Verify order was actually placed (wait 1s for order to appear in orderbook)
                time.sleep(1)
                if self._verify_order_status(order_id, 'open'):
                    logger.info(f"✓ SL order {order_id} verified in orderbook")
                    return order_id
                else:
                    logger.error(f"✗ SL order {order_id} not found in orderbook - may have been rejected")
                    return None
            else:
                logger.error(f"✗ SL order placement failed: {response.get('message')}")
                return None
        
        except Exception as e:
            logger.error(f"✗ Exception placing SL order: {e}")
            return None
    
    def _check_position_exists(self) -> bool:
        """Check if position exists for this symbol on the broker
        
        Returns:
            True if position exists, False otherwise
        """
        if TEST_MODE:
            return self.state.position_open
        
        try:
            response = client.positionbook()
            
            # Log full response for debugging
            logger.debug(f"📊 Positionbook API Response Type: {type(response)}")
            logger.debug(f"📊 Positionbook API Response: {response}")
            
            if isinstance(response, str):
                logger.debug(f"Positionbook returned string: {response}")
                return False
            
            if not isinstance(response, dict) or response.get('status') != 'success':
                logger.debug(f"Positionbook fetch failed: {response.get('message', 'Unknown error') if isinstance(response, dict) else 'Invalid response'}")
                return False
            
            data = response.get('data', [])
            positions = []
            
            # Handle different response formats
            if isinstance(data, list):
                positions = data
            elif isinstance(data, dict):
                if 'positions' in data:
                    positions_data = data['positions']
                    positions = positions_data if isinstance(positions_data, list) else [positions_data]
                elif 'symbol' in data:  # Single position as dict
                    positions = [data]
            
            # Check if our symbol exists in positions with non-zero quantity
            for pos in positions:
                if isinstance(pos, dict):
                    pos_symbol = str(pos.get('symbol', '')).upper()
                    quantity = int(pos.get('quantity', 0) or pos.get('netqty', 0) or 0)
                    
                    if pos_symbol == self.symbol.upper() and quantity != 0:
                        logger.debug(f"✓ Position found: {self.symbol} with quantity {quantity}")
                        return True
            
            logger.debug(f"No position found for {self.symbol}")
            return False
            
        except Exception as e:
            logger.error(f"✗ Exception checking position: {e}")
            return False
    
    def _modify_sl_order(self, order_id: str, new_stop_price: float) -> bool:
        """Modify existing stop-loss order on the broker
        
        Enhanced logic:
        1. Checks if order exists and is pending
        2. If order not pending (filled/rejected), checks if position exists
        3. If position exists but no pending SL, places new SL order
        4. This handles cases where initial entry order failed but position was created
        """
        if not AUTO_PLACE_ORDERS or not PLACE_SL_ORDER or not order_id:
            return False
        
        if TEST_MODE:
            logger.info(f"🧪 TEST MODE - Simulated SL modify to {new_stop_price:.2f}")
            return True
        
        # Verify order still exists and is pending before modifying
        order_is_pending = self._verify_order_status(order_id, 'open')
        
        if not order_is_pending:
            logger.warning(f"⚠️  SL order {order_id} not in pending state - checking if position exists...")
            
            # Check if position actually exists on broker
            position_exists = self._check_position_exists()
            
            if position_exists:
                logger.info(f"✓ Position exists for {self.symbol} - placing new SL order instead of modifying")
                
                # Clear old invalid SL order ID
                self.state.sl_order_id = None
                
                # Place a new SL order
                new_sl_order_id = self._place_sl_order(new_stop_price)
                
                if new_sl_order_id:
                    self.state.sl_order_id = new_sl_order_id
                    logger.info(f"✅ New SL order placed successfully: {new_sl_order_id} @ {new_stop_price:.2f}")
                    return True
                else:
                    logger.error(f"✗ Failed to place new SL order")
                    return False
            else:
                logger.warning(f"⚠️  No position found for {self.symbol} - SL order may have already executed")
                self.state.sl_order_id = None  # Clear invalid order ID
                return False
        
        try:
            # Round to tick size to avoid broker rejection
            rounded_trigger = round_to_tick_size(new_stop_price)
            
            # Calculate buffer (percentage-based or fixed)
            if SL_USE_PERCENT_BUFFER:
                buffer = rounded_trigger * SL_LIMIT_BUFFER_PERCENT
            else:
                buffer = SL_LIMIT_BUFFER
            
            # For SL order, limit price should be slightly worse than trigger (for SELL, lower)
            rounded_limit = round_to_tick_size(rounded_trigger - buffer)
            
            logger.info(f"📝 Modifying SL-L order {order_id}")
            logger.info(f"   New Trigger: {rounded_trigger:.2f} | New Limit: {rounded_limit:.2f} (Buffer: {buffer:.2f} / {(buffer/rounded_trigger)*100:.2f}%)")
            
            response = client.modifyorder(
                strategy=f"expiry_blast_{self.leg_type}_SL",
                symbol=self.symbol,
                exchange=OPTION_EXCHANGE,
                action=EXIT_ACTION,
                price_type="SL",  # Stop Loss Limit order
                product=PRODUCT,
                quantity=FINAL_QUANTITY,
                trigger_price=rounded_trigger,
                price=rounded_limit,  # Limit price with buffer
                order_id=order_id
            )
            
            # Log full response for debugging
            logger.debug(f"📝 ModifyOrder API Response Type: {type(response)}")
            logger.debug(f"📝 ModifyOrder API Response: {response}")
            
            if response.get('status') == 'success':
                logger.info(f"✅ SL order modified successfully! New trigger: {new_stop_price:.2f}")
                return True
            else:
                error_msg = response.get('message', 'Unknown error')
                logger.error(f"✗ SL order modification failed: {error_msg}")
                logger.error(f"   Full response: {response}")
                return False
        
        except Exception as e:
            logger.error(f"✗ Exception modifying SL order: {e}")
            return False
    
    def _cancel_sl_order(self, order_id: str) -> bool:
        """Cancel stop-loss order on the broker"""
        if not AUTO_PLACE_ORDERS or not PLACE_SL_ORDER or not order_id:
            return False
        
        if TEST_MODE:
            logger.info(f"🧪 TEST MODE - Simulated SL cancel")
            return True
        
        try:
            logger.info(f"🗑️ Canceling SL order {order_id}")
            
            # Use correct parameter name: order_id (not orderid)
            response = client.cancelorder(
                strategy=f"expiry_blast_{self.leg_type}_SL",
                order_id=order_id
            )
            
            if response.get('status') == 'success':
                logger.info(f"✅ SL order canceled successfully")
                return True
            else:
                logger.error(f"✗ SL order cancellation failed: {response.get('message')}")
                return False
        
        except Exception as e:
            logger.error(f"✗ Exception canceling SL order: {e}")
            return False
    
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
                        persistence._log_event(
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
                
                # Log full response for debugging
                logger.debug(f"📤 PlaceSmartOrder API Response Type: {type(response)}")
                logger.debug(f"📤 PlaceSmartOrder API Response: {response}")
                
                if response.get('status') == 'success':
                    order_id = response.get('orderid')
                    logger.info(f"✅ Order placed successfully! OrderID: {order_id}")
                    
                    # Verify entry order was accepted (wait 1s for confirmation)
                    time.sleep(1)
                    if not self._verify_order_status(order_id, 'complete'):
                        logger.error(f"⚠️  Entry order {order_id} may have been rejected - check orderbook")
                        # Don't return here - still log to persistence for tracking
                    
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
                            persistence._log_event(
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
        
        # Wait for WebSocket to connect (up to 10 seconds)
        wait_count = 0
        max_wait = 10
        while wait_count < max_wait and ws_client and not ws_client.is_connected():
            time.sleep(1)
            wait_count += 1
        
        if ws_client and ws_client.is_connected():
            logger.info(f"✓ WebSocket ready after {wait_count}s")
        else:
            logger.warning(f"⚠️  WebSocket not connected after {max_wait}s, will use REST API fallback")
    
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
        # Fetch expiry ONCE for the entire session (cached globally)
        logger.info(f"🔍 Resolving ATM for {UNDERLYING}...\n")
        expiry = get_nearest_expiry()
        logger.info(f"✓ Using expiry: {expiry} for entire session\n")
        
        # Resolve CE and PE using cached expiry and configured strike selection
        ce_symbol, _, ce_strike = resolve_option_symbol("CE", expiry=expiry, strike_selection=CE_STRIKE_SELECTION)
        pe_symbol, _, pe_strike = resolve_option_symbol("PE", expiry=expiry, strike_selection=PE_STRIKE_SELECTION)
        
        logger.info(f"\n✓ Symbols resolved. Starting monitoring...")
        
        # Subscribe to WebSocket if available
        if ws_client and ws_client.is_connected():
            logger.info(f"📡 Subscribing to WebSocket LTP for {ce_symbol} and {pe_symbol}...")
            
            try:
                # Subscribe using thread-safe method (WebSocket client handles async internally)
                ce_subscribed = ws_client.subscribe_ltp_sync(ce_symbol, OPTION_EXCHANGE)
                pe_subscribed = ws_client.subscribe_ltp_sync(pe_symbol, OPTION_EXCHANGE)
                
                if ce_subscribed and pe_subscribed:
                    logger.info(f"✓ WebSocket subscriptions successful for CE and PE")
                elif ce_subscribed or pe_subscribed:
                    logger.warning(f"⚠️  Partial subscription: CE={ce_subscribed}, PE={pe_subscribed}")
                else:
                    logger.warning(f"⚠️  WebSocket subscriptions failed, will use REST API fallback")
            except AttributeError:
                # Fallback: If sync method doesn't exist, subscriptions will happen on-demand
                logger.warning(f"⚠️  WebSocket sync methods not available")
                logger.info(f"ℹ️  Will use REST API fallback")
            except Exception as e:
                logger.warning(f"⚠️  WebSocket subscription error: {e}")
                logger.info(f"ℹ️  Will fallback to REST API if WebSocket fails")
        else:
            logger.info(f"ℹ️  WebSocket not connected, using REST API for LTP")
        
        # Create monitors
        ce_monitor = LegMonitor(ce_symbol, "CE", ce_strike)
        pe_monitor = LegMonitor(pe_symbol, "PE", pe_strike)
        
        # Initialize CE first to avoid race conditions with historical data API
        if MONITOR_CE:
            logger.info(f"\n🔵 Initializing CE monitor first...")
            ce_monitor.initialize()
            logger.info(f"✓ CE initialization complete\n")
        
        # Small delay to avoid API rate limiting / caching issues
        time.sleep(2)
        
        # Initialize PE second
        if MONITOR_PE:
            logger.info(f"\n🔴 Initializing PE monitor second...")
            pe_monitor.initialize()
            logger.info(f"✓ PE initialization complete\n")
        
        # Now start monitoring loops in parallel threads
        threads = []
        
        if MONITOR_CE:
            ce_thread = Thread(target=lambda: ce_monitor.monitor(skip_init=True), daemon=False, name="CE-Monitor")
            threads.append(ce_thread)
            ce_thread.start()
        
        if MONITOR_PE:
            pe_thread = Thread(target=lambda: pe_monitor.monitor(skip_init=True), daemon=False, name="PE-Monitor")
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
