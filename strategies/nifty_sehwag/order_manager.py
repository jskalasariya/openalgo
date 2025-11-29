"""
Order Execution Module
======================
Handles order placement and execution tracking.
"""

import logging
from typing import Optional, Dict
import time

logger = logging.getLogger(__name__)


class OrderManager:
    """Manages order placement and execution"""
    
    def __init__(self, client, config: Dict):
        """
        Initialize order manager
        
        Args:
            client: OpenAlgo API client
            config: Configuration dictionary
        """
        self.client = client
        self.config = config
        
        self.test_mode = config.get('test_mode', False)
        self.auto_place_orders = config.get('auto_place_orders', False)
        self.exchange = config.get('option_exchange', 'NFO')
        self.price_type = config.get('price_type', 'MARKET')
        self.product = config.get('product', 'NRML')
        self.instrument_type = config.get('instrument_type', 'options')
        self.entry_action = config.get('entry_action', 'BUY')
        self.exit_action = config.get('exit_action', 'SELL')
    
    def place_order(self, symbol: str, quantity: int, action: str) -> Optional[str]:
        """
        Place an order
        
        Args:
            symbol: Symbol to trade
            quantity: Quantity to trade
            action: "BUY" or "SELL"
            
        Returns:
            Order ID or None if failed
        """
        try:
            logger.info(f"📊 Placing {action} order: {symbol}, Qty: {quantity}")
            
            if self.test_mode:
                logger.info(f"🧪 TEST MODE - Order NOT placed (simulated): {action} {quantity} {symbol}")
                return f"TEST_ORDER_{symbol}_{int(time.time())}"
            
            if not self.auto_place_orders:
                logger.warning(f"⚠️  AUTO_PLACE_ORDERS is False - Order simulated: {action} {quantity} {symbol}")
                return f"SIM_ORDER_{symbol}_{int(time.time())}"
            
            resp = self.client.place_order(
                symbol=symbol,
                exchange=self.exchange,
                action=action,
                quantity=quantity,
                price_type=self.price_type,
                product=self.product,
                pricetype='MARKET',
                instrumenttype=self.instrument_type
            )
            
            if resp.get('status') == 'success':
                order_id = resp.get('orderid')
                logger.info(f"✓ Order placed successfully: {order_id}")
                return order_id
            else:
                logger.error(f"✗ Order placement failed: {resp}")
                return None
        
        except Exception as e:
            logger.error(f"✗ Error placing order: {e}")
            return None
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """
        Get status of an order
        
        Args:
            order_id: Order ID to check
            
        Returns:
            Order status dict or None
        """
        try:
            if self.test_mode or order_id.startswith("TEST_") or order_id.startswith("SIM_"):
                return {
                    'status': 'COMPLETE',
                    'order_id': order_id,
                    'filled_quantity': 0,
                    'average_price': 0.0
                }
            
            resp = self.client.order_status(orderid=order_id)
            
            if resp.get('status') == 'success':
                return resp.get('data')
            else:
                logger.error(f"Failed to get order status for {order_id}: {resp}")
                return None
                
        except Exception as e:
            logger.error(f"Error getting order status for {order_id}: {e}")
            return None
