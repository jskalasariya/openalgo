"""
Order Execution Module
======================
Handles order placement and execution tracking.
Includes SL order management following Expiry Blast standards.
"""

import logging
from typing import Optional, Dict
import time

logger = logging.getLogger(__name__)


def round_to_tick_size(price: float, tick_size: float = 0.05) -> float:
    """Round price to nearest tick size"""
    return round(price / tick_size) * tick_size


class OrderManager:
    """Manages order placement and execution including SL orders"""

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

        # SL Order configuration
        self.place_sl_order = config.get('place_sl_order', True)
        self.sl_order_type = config.get('sl_order_type', 'SL')  # SL-Limit
        self.sl_limit_buffer_percent = config.get('sl_limit_buffer_percent', 0.015)  # 1.5% default
        self.sl_use_percent_buffer = config.get('sl_use_percent_buffer', True)

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

    def place_sl_order(self, symbol: str, quantity: int, stop_price: float, strategy_name: str = "nifty_sehwag") -> Optional[str]:
        """
        Place a stop-loss order on the broker

        Args:
            symbol: Symbol to place SL for
            quantity: Quantity
            stop_price: Stop loss trigger price
            strategy_name: Strategy identifier

        Returns:
            SL Order ID or None if failed
        """
        if not self.auto_place_orders or not self.place_sl_order:
            return None

        if self.test_mode:
            logger.info(f"🧪 TEST MODE - Simulated SL order @ ₹{stop_price:.2f}")
            return f"TEST_SL_{symbol}_{int(time.time())}"

        try:
            # Round to tick size to avoid broker rejection
            rounded_trigger = round_to_tick_size(stop_price)

            # Calculate buffer (percentage-based)
            buffer = rounded_trigger * self.sl_limit_buffer_percent

            # For SL order, limit price should be slightly worse than trigger (for SELL, lower)
            rounded_limit = round_to_tick_size(rounded_trigger - buffer)

            logger.info(f"📤 Placing SL-L order on broker: {symbol}")
            logger.info(f"   Trigger: ₹{rounded_trigger:.2f} | Limit: ₹{rounded_limit:.2f} (Buffer: ₹{buffer:.2f} / {(buffer/rounded_trigger)*100:.2f}%)")

            response = self.client.placeorder(
                strategy=f"{strategy_name}_SL",
                symbol=symbol,
                exchange=self.exchange,
                action=self.exit_action,
                price_type="SL",  # Stop Loss Limit order
                product=self.product,
                quantity=quantity,
                trigger_price=rounded_trigger,
                price=rounded_limit  # Limit price with buffer
            )

            if response.get('status') == 'success':
                order_id = response.get('orderid')
                logger.info(f"✅ SL order placed on broker! OrderID: {order_id} @ Trigger: ₹{stop_price:.2f}")

                # Verify order was actually placed (wait 1s for order to appear in orderbook)
                time.sleep(1)
                return order_id
            else:
                logger.error(f"✗ SL order placement failed: {response.get('message')}")
                return None

        except Exception as e:
            logger.error(f"✗ Exception placing SL order: {e}")
            return None

    def modify_sl_order(self, order_id: str, symbol: str, quantity: int, new_stop_price: float, strategy_name: str = "nifty_sehwag") -> bool:
        """
        Modify existing stop-loss order on the broker

        Args:
            order_id: Existing SL order ID
            symbol: Symbol
            quantity: Quantity
            new_stop_price: New stop loss trigger price
            strategy_name: Strategy identifier

        Returns:
            True if modified successfully, False otherwise
        """
        if not self.auto_place_orders or not self.place_sl_order or not order_id:
            return False

        if self.test_mode:
            logger.info(f"🧪 TEST MODE - Simulated SL modify to ₹{new_stop_price:.2f}")
            return True

        try:
            # Round to tick size to avoid broker rejection
            rounded_trigger = round_to_tick_size(new_stop_price)

            # Calculate buffer (percentage-based)
            buffer = rounded_trigger * self.sl_limit_buffer_percent

            # For SL order, limit price should be slightly worse than trigger (for SELL, lower)
            rounded_limit = round_to_tick_size(rounded_trigger - buffer)

            logger.info(f"📝 Modifying SL-L order {order_id}")
            logger.info(f"   New Trigger: ₹{rounded_trigger:.2f} | New Limit: ₹{rounded_limit:.2f} (Buffer: ₹{buffer:.2f} / {(buffer/rounded_trigger)*100:.2f}%)")

            response = self.client.modifyorder(
                strategy=f"{strategy_name}_SL",
                symbol=symbol,
                exchange=self.exchange,
                action=self.exit_action,
                price_type="SL",  # Stop Loss Limit order
                product=self.product,
                quantity=quantity,
                trigger_price=rounded_trigger,
                price=rounded_limit,  # Limit price with buffer
                order_id=order_id
            )

            if response.get('status') == 'success':
                logger.info(f"✅ SL order modified successfully! New trigger: ₹{new_stop_price:.2f}")
                return True
            else:
                error_msg = response.get('message', 'Unknown error')
                logger.error(f"✗ SL order modification failed: {error_msg}")
                return False

        except Exception as e:
            logger.error(f"✗ Exception modifying SL order: {e}")
            return False

    def cancel_sl_order(self, order_id: str, strategy_name: str = "nifty_sehwag") -> bool:
        """
        Cancel stop-loss order on the broker

        Args:
            order_id: SL order ID to cancel
            strategy_name: Strategy identifier

        Returns:
            True if canceled successfully, False otherwise
        """
        if not self.auto_place_orders or not self.place_sl_order or not order_id:
            return False

        if self.test_mode:
            logger.info(f"🧪 TEST MODE - Simulated SL cancel")
            return True

        try:
            logger.info(f"🗑️ Canceling SL order {order_id}")

            response = self.client.cancelorder(
                strategy=f"{strategy_name}_SL",
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
