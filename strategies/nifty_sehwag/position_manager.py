"""
Position Management Module
==========================
Handles position tracking, SL management, and exit logic.
"""

import logging
from typing import Optional, Dict
from datetime import datetime

from .models import LegPosition, StrategyState
from .order_manager import OrderManager

logger = logging.getLogger(__name__)


class PositionManager:
    """Manages position lifecycle and risk management"""
    
    def __init__(self, order_manager: OrderManager, config: Dict):
        """
        Initialize position manager
        
        Args:
            order_manager: OrderManager instance
            config: Configuration dictionary
        """
        self.order_manager = order_manager
        self.config = config
        self.entry_action = config.get('entry_action', 'BUY')
        self.exit_action = config.get('exit_action', 'SELL')
    
    def manage_position(self, leg_num: int, leg_config: Dict, 
                       state: StrategyState, current_price: float) -> None:
        """
        Generic position management - handles all leg types dynamically
        
        Args:
            leg_num: Leg number
            leg_config: Leg configuration
            state: Strategy state
            current_price: Current market price
        """
        leg = state.get_position(leg_num)
        if not leg or not leg.is_active:
            return
        
        pnl, pnl_pct = leg.calculate_pnl(current_price, self.entry_action)
        
        leg_name = leg_config.get('name', f'Leg {leg_num}')
        logger.debug(f"{leg_name}: PnL {pnl_pct:.2f}%, SL: {leg.current_sl:.2f}, LTP: {current_price:.2f}")
        
        # Check exit condition - SL breach
        if current_price <= leg.current_sl:
            logger.warning(f"⚠️  {leg_name} SL breached at {current_price:.2f}")
            self.exit_position(leg, current_price, "SL_BREACH")
            return
        
        management_type = leg_config.get('management_type', 'fixed')
        
        # ===== ESCALATING PROFIT MANAGEMENT =====
        if management_type == 'escalating_profit':
            self._manage_escalating_profit(leg, leg_config, leg_name, current_price, pnl_pct)
        
        # ===== DYNAMIC SL MANAGEMENT =====
        elif management_type == 'dynamic_sl':
            self._manage_dynamic_sl(leg, leg_config, leg_name, current_price, pnl_pct)
        
        # ===== FIXED MANAGEMENT =====
        else:  # fixed
            self._manage_fixed(leg, leg_config, leg_name, current_price, pnl_pct)
    
    def _manage_escalating_profit(self, leg: LegPosition, leg_config: Dict,
                                   leg_name: str, current_price: float, pnl_pct: float):
        """Handle escalating profit management logic"""
        profit_increase_step = leg_config.get('profit_increase_step', 1.0)
        profit_step_threshold = leg_config.get('profit_step_threshold', 2.0)
        
        # Update lock profit based on profit level
        if pnl_pct >= leg.lock_profit_pct and pnl_pct >= leg.profit_level_for_lock_increase:
            leg.lock_profit_pct += profit_increase_step
            leg.profit_level_for_lock_increase += profit_step_threshold
            logger.info(f"🔒 {leg_name} lock profit increased to {leg.lock_profit_pct:.2f}%")
        
        # Check profit target exit
        if pnl_pct >= leg.lock_profit_pct and pnl_pct >= 2.0:  # Minimum 2% before lock triggers
            logger.info(f"✅ {leg_name} profit target reached: {pnl_pct:.2f}%")
            self.exit_position(leg, current_price, "PROFIT_TARGET")
    
    def _manage_dynamic_sl(self, leg: LegPosition, leg_config: Dict,
                           leg_name: str, current_price: float, pnl_pct: float):
        """Handle dynamic SL management logic"""
        dynamic_sl_decrease = leg_config.get('dynamic_sl_decrease', 1.0)
        initial_sl_pct = leg_config.get('initial_sl_pct', 7.0)
        
        # SL tightens as profit increases
        if pnl_pct > 0:
            # New SL = Entry * (1 - (InitialSL - CurrentProfit) / 100)
            new_sl = leg.entry_price * (1 - (initial_sl_pct - pnl_pct * dynamic_sl_decrease) / 100)
            if new_sl > leg.current_sl:
                leg.current_sl = new_sl
                logger.info(f"📈 {leg_name} SL updated to {leg.current_sl:.2f} (PnL: {pnl_pct:.2f}%)")
        
        # Check auto-close at profit target
        auto_close_pct = leg_config.get('auto_close_profit_pct')
        if auto_close_pct and pnl_pct >= auto_close_pct:
            logger.info(f"✅ {leg_name} auto-close at {auto_close_pct}% profit: {pnl_pct:.2f}%")
            self.exit_position(leg, current_price, f"AUTO_CLOSE_{auto_close_pct}PCT")
    
    def _manage_fixed(self, leg: LegPosition, leg_config: Dict,
                      leg_name: str, current_price: float, pnl_pct: float):
        """Handle fixed SL and target management"""
        lock_profit_pct = leg_config.get('lock_profit_pct', 5.0)
        if pnl_pct >= lock_profit_pct:
            logger.info(f"✅ {leg_name} profit target reached: {pnl_pct:.2f}%")
            self.exit_position(leg, current_price, "PROFIT_TARGET")
    
    def exit_position(self, leg: LegPosition, exit_price: float, reason: str) -> None:
        """
        Exit a position
        
        Args:
            leg: Leg position to exit
            exit_price: Exit price
            reason: Exit reason
        """
        if not leg.is_active:
            return
        
        logger.info(f"🚪 Exiting Leg {leg.leg_id}: {reason} at {exit_price:.2f}")
        
        # Place exit order
        order_id = self.order_manager.place_order(
            leg.symbol, 
            leg.quantity, 
            self.exit_action
        )
        
        if order_id:
            leg.is_active = False
            leg.exit_price = exit_price
            leg.exit_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            leg.pnl, leg.pnl_pct = leg.calculate_pnl(exit_price, self.entry_action)
            
            logger.info(f"✅ Leg {leg.leg_id} exited - PnL: {leg.pnl_pct:.2f}% (₹{leg.pnl:.2f})")
