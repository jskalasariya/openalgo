"""
Strategy Utilities
==================
Helper functions for strike calculation, symbol generation, etc.
"""

import logging
from typing import Optional
from datetime import datetime, timedelta
import pytz

logger = logging.getLogger(__name__)


def is_market_open(tz=pytz.timezone('Asia/Kolkata')) -> bool:
    # return True
    """Check if market is currently open"""
    now = datetime.now(tz)
    current_time = now.time()
    
    # Market hours: 09:15 to 15:30
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0).time()
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0).time()
    
    # Check if weekend
    if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
        return False
    
    return market_open <= current_time <= market_close


def is_expiry_day(underlying: str = "NIFTY", tz=pytz.timezone('Asia/Kolkata')) -> bool:
    """
    Check if today is expiry day for the underlying
    
    NIFTY: Thursday
    BANKNIFTY: Wednesday
    FINNIFTY: Tuesday
    """
    today = datetime.now(tz)
    weekday = today.weekday()
    
    expiry_days = {
        "NIFTY": 3,      # Thursday
        "BANKNIFTY": 2,  # Wednesday
        "FINNIFTY": 1,   # Tuesday
        "MIDCPNIFTY": 0  # Monday
    }
    
    expected_day = expiry_days.get(underlying.upper(), 3)
    return weekday == expected_day


def get_current_atm_strike(spot_price: float, underlying: str = "NIFTY") -> int:
    """
    Calculate ATM strike from spot price
    
    Args:
        spot_price: Current underlying price
        underlying: Underlying asset name
        
    Returns:
        Nearest ATM strike
    """
    strike_diff = 50  # Default for NIFTY
    
    # Adjust strike difference based on underlying
    if underlying.upper() == "BANKNIFTY":
        strike_diff = 100
    elif underlying.upper() in ["FINNIFTY", "MIDCPNIFTY"]:
        strike_diff = 50
    
    # Round to nearest strike
    atm_strike = round(spot_price / strike_diff) * strike_diff
    return int(atm_strike)


def get_option_symbol(strike: int, expiry_date: str, option_type: str,
                      underlying: str = "NIFTY") -> str:
    """
    Generate option symbol
    
    Args:
        strike: Strike price
        expiry_date: Expiry date (YYYYMMDD)
        option_type: "CE" or "PE"
        underlying: Underlying asset
        
    Returns:
        Option symbol string
    """
    # Format: NIFTY24NOV24700CE
    # Extract year, month, day from expiry_date
    year = expiry_date[2:4]
    month_num = int(expiry_date[4:6])
    
    months = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
              "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
    month = months[month_num - 1]
    
    symbol = f"{underlying}{year}{month}{strike}{option_type}"
    return symbol


def get_next_expiry_date(underlying: str = "NIFTY",
                         tz=pytz.timezone('Asia/Kolkata')) -> str:
    """
    Get next expiry date for the underlying
    
    Returns:
        Expiry date in YYYYMMDD format
    """
    today = datetime.now(tz)
    
    expiry_days = {
        "NIFTY": 3,      # Thursday
        "BANKNIFTY": 2,  # Wednesday
        "FINNIFTY": 1,   # Tuesday
        "MIDCPNIFTY": 0  # Monday
    }
    
    target_weekday = expiry_days.get(underlying.upper(), 3)
    current_weekday = today.weekday()
    
    # Calculate days until next expiry
    if current_weekday < target_weekday:
        days_ahead = target_weekday - current_weekday
    elif current_weekday == target_weekday:
        # If today is expiry day, return today
        days_ahead = 0
    else:
        days_ahead = (7 - current_weekday) + target_weekday
    
    expiry_date = today + timedelta(days=days_ahead)
    return expiry_date.strftime('%Y%m%d')


def calculate_itm_strike(atm_strike: int, itm_level: int, option_type: str,
                         underlying: str = "NIFTY") -> int:
    """
    Calculate ITM strike based on ITM level
    
    Args:
        atm_strike: ATM strike
        itm_level: ITM level (1, 2, 3, 4, etc.)
        option_type: "CE" or "PE"
        underlying: Underlying asset
        
    Returns:
        ITM strike
    """
    strike_diff = 50  # Default for NIFTY
    
    if underlying.upper() == "BANKNIFTY":
        strike_diff = 100
    elif underlying.upper() in ["FINNIFTY", "MIDCPNIFTY"]:
        strike_diff = 50
    
    # For CE: ITM means lower strike (ATM - level * diff)
    # For PE: ITM means higher strike (ATM + level * diff)
    if option_type == "CE":
        return atm_strike - (itm_level * strike_diff)
    else:  # PE
        return atm_strike + (itm_level * strike_diff)
