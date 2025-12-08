"""
Market Data and Analysis Module
================================
Handles price fetching, candle analysis, and entry condition detection.
"""

import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta
import pandas as pd
import pytz

logger = logging.getLogger(__name__)


class MarketDataManager:
    """Manages market data fetching and analysis"""
    
    def __init__(self, client, config: Dict, websocket_client=None):
        """
        Initialize market data manager
        
        Args:
            client: OpenAlgo API client
            config: Configuration dictionary
            websocket_client: Optional WebSocket client for real-time data
        """
        self.client = client
        self.config = config
        self.websocket_client = websocket_client
        
        self.underlying = config.get('underlying', 'NIFTY')
        self.underlying_exchange = config.get('underlying_exchange', 'NSE_INDEX')
        self.candle_interval = config.get('candle_interval', '3m')
        self.lookback_candles = config.get('lookback_candles', 3)
        self.use_websocket = config.get('use_websocket', False)
    
    def get_quote(self, symbol: str, exchange: str = None, 
                  instrument_type: str = 'options') -> Optional[Dict]:
        """
        Get current quote for a symbol (expiry_blast style - simple and direct)

        Args:
            symbol: Symbol to fetch
            exchange: Exchange (defaults to config)
            instrument_type: Instrument type

        Returns:
            Quote data dict or None
        """
        # Try WebSocket first if enabled and connected
        if self.use_websocket and self.websocket_client and self.websocket_client.is_connected():
            ws_price = self._get_price_from_websocket(symbol)
            if ws_price:
                logger.debug(f"WebSocket LTP for {symbol}: {ws_price}")
                return {'ltp': ws_price}

        # Fallback to REST API (expiry_blast approach - simple, no retries)
        exchange = exchange or self.config.get('option_exchange', 'NFO')

        try:
            q = self.client.quotes(
                symbol=symbol,
                exchange=exchange
            )

            # Debug: Log the response structure
            logger.debug(f"Quote API response for {symbol}: {q}")

            # Handle different response formats (exactly like expiry_blast)
            if isinstance(q, dict):
                # Try different possible structures
                if 'data' in q and isinstance(q['data'], dict) and 'ltp' in q['data']:
                    return q['data']
                elif 'ltp' in q:
                    return q
                elif 'data' in q and isinstance(q['data'], list) and len(q['data']) > 0:
                    return q['data'][0]
                else:
                    logger.error(f"✗ Unexpected quote response format for {symbol}: {q}")
                    return None
            else:
                logger.error(f"✗ Quote response is not a dict for {symbol}: {type(q)}")
                return None

        except KeyError as e:
            logger.error(f"✗ KeyError fetching quote for {symbol}: Missing key {e}")
            logger.error(f"   Response structure: {q if 'q' in locals() else 'N/A'}")
            return None
        except Exception as e:
            logger.error(f"✗ Error fetching quote for {symbol}: {e}")
            return None

    def _get_price_from_websocket(self, symbol: str) -> Optional[float]:
        """Get price from WebSocket client"""
        if not self.websocket_client:
            return None
        
        try:
            # Correct method name is get_last_price(), not get_ltp()
            price = self.websocket_client.get_last_price(symbol)
            return float(price) if price else None
        except Exception as e:
            logger.warning(f"WebSocket price fetch failed for {symbol}: {e}")
            return None
    
    def get_underlying_price(self) -> Optional[float]:
        """Get current underlying price with multiple format support"""
        try:
            # Try WebSocket first if enabled and connected
            if self.use_websocket and self.websocket_client and self.websocket_client.is_connected():
                ws_price = self._get_price_from_websocket(self.underlying)
                if ws_price:
                    logger.debug(f"WebSocket LTP for {self.underlying}: {ws_price}")
                    return ws_price

            # Fallback to REST API
            quote = self.client.quotes(
                symbol=self.underlying,
                exchange=self.underlying_exchange
            )
            
            # Handle different response formats (like expiry_blast)
            if isinstance(quote, dict):
                # Format 1: {'data': {'ltp': ...}}
                if 'data' in quote and isinstance(quote['data'], dict):
                    ltp = quote['data'].get('ltp')
                    if ltp:
                        return float(ltp)

                # Format 2: {'ltp': ...} (direct)
                elif 'ltp' in quote:
                    return float(quote['ltp'])

                # Format 3: {'data': [{'ltp': ...}]} (list)
                elif 'data' in quote and isinstance(quote['data'], list) and len(quote['data']) > 0:
                    ltp = quote['data'][0].get('ltp')
                    if ltp:
                        return float(ltp)

            logger.warning(f"Could not extract LTP from underlying quote response: {quote}")
            return None
            
        except Exception as e:
            logger.error(f"Error getting underlying price: {e}")
            return None
    
    def get_candle_data(self, from_date: str, to_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch historical candle data
        
        Args:
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLC data or None if no data (holiday/weekend)
        """
        try:
            logger.info(f"Requesting history for {self.underlying} ({self.underlying_exchange}) "
                        f"from {from_date} to {to_date}, interval: {self.candle_interval}")
            
            df = self.client.history(
                symbol=self.underlying,
                exchange=self.underlying_exchange,
                start_date=from_date,
                end_date=to_date,
                interval=self.candle_interval
            )
            
            # Check if DataFrame is valid and has data
            if isinstance(df, pd.DataFrame) and not df.empty:
                logger.debug(f"Received {len(df)} candles for {from_date}")
                return df
            else:
                logger.debug(f"Empty DataFrame for {from_date} (likely holiday/weekend)")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching candle data: {e}")
            return None
    
    def get_previous_day_candles(self, tz) -> Optional[pd.DataFrame]:
        """
        Get previous day's last N candles, skipping weekends and holidays
        
        Goes back up to 10 days to find a trading day with data.
        Handles scenarios like:
        - Monday: fetches Friday's data (skips weekend)
        - After long weekends with holidays
        """
        try:
            today = datetime.now(tz).date()
            
            # Go back up to 10 days to find a trading day (handles long weekends + holidays)
            logger.info(f"Looking for previous trading day's data (today: {today})")
            
            for days_back in range(1, 11):
                check_date = today - timedelta(days=days_back)
                date_str = check_date.strftime('%Y-%m-%d')
                day_name = check_date.strftime('%A')
                
                logger.info(f"Attempting to fetch candles for {date_str} ({day_name})...")
                df = self.get_candle_data(date_str, date_str)
                
                if df is not None and len(df) > 0:
                    candle_count = len(df)
                    logger.info(f"✓ Found {candle_count} candles for {date_str} ({day_name})")
                    logger.info(f"✓ Using last {self.lookback_candles} candles for analysis")
                    return df.tail(self.lookback_candles)
                else:
                    logger.debug(f"✗ No data for {date_str} ({day_name}) - likely weekend/holiday")
            
            logger.error(f"❌ Could not find previous trading day data after checking 10 days back")
            return None
            
        except Exception as e:
            logger.error(f"Error getting previous day candles: {e}")
            return None
    
    def analyze_entry_condition(self, tz, cached_candles=None, cached_high=None, cached_low=None) -> Tuple[bool, Optional[str], float, float]:
        """
        Analyze entry condition based on previous day candles
        
        Args:
            tz: Timezone
            cached_candles: Pre-fetched candles (to avoid re-fetching)
            cached_high: Pre-calculated highest high (optimization)
            cached_low: Pre-calculated lowest low (optimization)
        
        Returns:
            Tuple of (should_enter, direction, highest_high, lowest_low)
        """
        try:
            # Use pre-calculated values if provided (optimization)
            if cached_high is not None and cached_low is not None:
                highest_high = cached_high
                lowest_low = cached_low
                logger.debug("Using pre-calculated high/low values (optimized)")
            elif cached_candles is not None:
                # Calculate from cached candles
                highest_high = float(cached_candles['high'].max())
                lowest_low = float(cached_candles['low'].min())
                logger.debug("Calculating high/low from cached candles")
            else:
                # Fetch and calculate
                logger.debug("Fetching previous day candles (no cached data)")
                df = self.get_previous_day_candles(tz)
                
                if df is None or len(df) == 0:
                    logger.warning("No candle data for entry analysis")
                    return False, None, 0.0, 0.0
                
                highest_high = float(df['high'].max())
                lowest_low = float(df['low'].min())
            
            logger.info(f"Previous day analysis: High={highest_high:.2f}, Low={lowest_low:.2f}")
            
            # Get current price
            current_price = self.get_underlying_price()
            if not current_price:
                logger.error("Could not get current price")
                return False, None, highest_high, lowest_low
            
            logger.info(f"Current price: {current_price:.2f}")
            
            # Determine direction
            if current_price > highest_high:
                logger.info(f"✓ Entry condition MET: Price {current_price:.2f} > High {highest_high:.2f} → CE")
                return True, "CE", highest_high, lowest_low
            elif current_price < lowest_low:
                logger.info(f"✓ Entry condition MET: Price {current_price:.2f} < Low {lowest_low:.2f} → PE")
                return True, "PE", highest_high, lowest_low
            else:
                logger.info(f"✗ Entry condition NOT met: {lowest_low:.2f} < {current_price:.2f} < {highest_high:.2f}")
                return False, None, highest_high, lowest_low
                
        except Exception as e:
            logger.error(f"Error analyzing entry condition: {e}")
            return False, None, 0.0, 0.0
    
    def check_wait_trade(self, reference_price: float, direction: str, 
                        threshold_pct: float) -> bool:
        """
        Check if Wait & Trade condition is met
        
        Args:
            reference_price: Starting reference price
            direction: "CE" or "PE"
            threshold_pct: Required percentage movement
            
        Returns:
            True if threshold is met
        """
        try:
            current_price = self.get_underlying_price()
            if not current_price:
                return False
            
            price_change_pct = ((current_price - reference_price) / reference_price) * 100
            
            logger.debug(f"Wait & Trade: Current={current_price:.2f}, "
                        f"Reference={reference_price:.2f}, Change={price_change_pct:.2f}%")
            
            # For CE: price should increase by threshold%
            # For PE: price should decrease by threshold%
            if direction == "CE" and price_change_pct >= threshold_pct:
                logger.info(f"✓ Wait & Trade confirmed: {price_change_pct:.2f}% ≥ {threshold_pct}%")
                return True
            elif direction == "PE" and price_change_pct <= -threshold_pct:
                logger.info(f"✓ Wait & Trade confirmed: {abs(price_change_pct):.2f}% ≥ {threshold_pct}%")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking wait & trade: {e}")
            return False
