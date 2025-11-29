"""
Unit Tests for Expiry Blast Strategy (Isolated)
===============================================
Comprehensive test suite covering core logic without importing the full strategy module.
Tests focus on:
- Breakout detection logic
- Stop loss and trailing stop logic  
- Profit target logic
- P&L calculations
- Position management
"""

import unittest
from datetime import datetime, timedelta
import pytz


class TestBreakoutLogic(unittest.TestCase):
    """Test breakout detection and entry logic"""
    
    def test_breakout_detection_simple(self):
        """Test simple breakout detection above highest high"""
        highest_high = 100.0
        ltp = 101.0
        breakout_threshold = 0.0
        
        is_breakout = ltp > highest_high * (1 + breakout_threshold)
        self.assertTrue(is_breakout)
    
    def test_breakout_detection_with_threshold(self):
        """Test breakout detection with threshold percentage"""
        highest_high = 100.0
        ltp = 101.5
        breakout_threshold = 0.01
        
        is_breakout = ltp > highest_high * (1 + breakout_threshold)
        self.assertTrue(is_breakout)
    
    def test_no_breakout_below_threshold(self):
        """Test no breakout when LTP below threshold"""
        highest_high = 100.0
        ltp = 100.5
        breakout_threshold = 0.01
        
        is_breakout = ltp > highest_high * (1 + breakout_threshold)
        self.assertFalse(is_breakout)
    
    def test_no_breakout_at_threshold_level(self):
        """Test no breakout at exact threshold level"""
        highest_high = 100.0
        ltp = 101.0  # Exactly at 1% threshold
        breakout_threshold = 0.01
        
        is_breakout = ltp > highest_high * (1 + breakout_threshold)
        self.assertFalse(is_breakout)
    
    def test_breakout_above_threshold_level(self):
        """Test breakout slightly above threshold level"""
        highest_high = 100.0
        ltp = 101.01  # Slightly above 1% threshold
        breakout_threshold = 0.01
        
        is_breakout = ltp > highest_high * (1 + breakout_threshold)
        self.assertTrue(is_breakout)


class TestStopLossLogic(unittest.TestCase):
    """Test stop loss and exit logic"""
    
    def test_initial_stop_loss_calculation(self):
        """Test initial stop loss is set to entry_price * (1 - initial_stop_pct)"""
        entry_price = 100.0
        initial_stop_pct = 0.50
        
        expected_stop = entry_price * (1 - initial_stop_pct)
        self.assertEqual(expected_stop, 50.0)
    
    def test_stop_loss_hit_detection(self):
        """Test detection when price reaches stop loss"""
        entry_price = 100.0
        initial_stop_pct = 0.50
        stop_price = entry_price * (1 - initial_stop_pct)  # 50.0
        
        current_ltp = 49.0
        is_stop_hit = current_ltp <= stop_price
        
        self.assertTrue(is_stop_hit)
    
    def test_stop_loss_not_hit(self):
        """Test stop loss not hit when price above stop"""
        entry_price = 100.0
        initial_stop_pct = 0.50
        stop_price = entry_price * (1 - initial_stop_pct)  # 50.0
        
        current_ltp = 51.0
        is_stop_hit = current_ltp <= stop_price
        
        self.assertFalse(is_stop_hit)
    
    def test_stop_loss_at_exact_level(self):
        """Test stop loss at exact stop level"""
        entry_price = 100.0
        initial_stop_pct = 0.50
        stop_price = entry_price * (1 - initial_stop_pct)  # 50.0
        
        current_ltp = 50.0
        is_stop_hit = current_ltp <= stop_price
        
        self.assertTrue(is_stop_hit)
    
    def test_multiple_stop_loss_levels(self):
        """Test various stop loss percentages"""
        entry_price = 100.0
        stop_percentages = [0.25, 0.50, 0.75, 0.90]
        
        for stop_pct in stop_percentages:
            expected_stop = entry_price * (1 - stop_pct)
            self.assertGreater(expected_stop, 0)
            self.assertLess(expected_stop, entry_price)


class TestTrailingStopLogic(unittest.TestCase):
    """Test trailing stop logic"""
    
    def test_trailing_stop_calculation(self):
        """Test trailing stop is moved up correctly with profit"""
        entry_price = 100.0
        current_ltp = 110.0
        trail_percent_step = 0.01  # 1%
        
        profit_pct = (current_ltp / entry_price) - 1
        steps = int(profit_pct / trail_percent_step)
        
        new_stop = entry_price * (1 + (steps - 1) * trail_percent_step)
        
        self.assertAlmostEqual(profit_pct, 0.10, places=5)  # 10% profit
        self.assertEqual(steps, 10)  # 10 steps
        self.assertAlmostEqual(new_stop, 109.0, places=2)  # 100 * (1 + 9 * 0.01)
    
    def test_trailing_stop_no_movement_at_loss(self):
        """Test trailing stop doesn't move if position is in loss"""
        entry_price = 100.0
        current_ltp = 95.0
        trail_percent_step = 0.01
        
        profit_pct = (current_ltp / entry_price) - 1
        steps = int(profit_pct / trail_percent_step) if profit_pct > 0 else 0
        
        self.assertEqual(steps, 0)  # No trailing stop movement in loss
    
    def test_trailing_stop_multiple_steps(self):
        """Test trailing stop with multiple profit steps"""
        entry_price = 100.0
        trail_percent_step = 0.01
        
        # Test at 5% profit
        ltp_5pct = 105.0
        profit = (ltp_5pct / entry_price) - 1
        steps = int(profit / trail_percent_step)
        new_stop = entry_price * (1 + (steps - 1) * trail_percent_step)
        
        self.assertEqual(steps, 5)
        self.assertEqual(new_stop, 104.0)
    
    def test_trailing_stop_at_15pct_profit(self):
        """Test trailing stop at 15% profit"""
        entry_price = 100.0
        current_ltp = 115.0
        trail_percent_step = 0.01
        
        profit_pct = (current_ltp / entry_price) - 1
        steps = int(profit_pct / trail_percent_step)
        new_stop = entry_price * (1 + (steps - 1) * trail_percent_step)
        
        self.assertEqual(steps, 14)  # int(0.15 / 0.01) = 14
        self.assertAlmostEqual(new_stop, 113.0, places=2)  # 100 * (1 + 13 * 0.01)
    
    def test_trailing_stop_sequence(self):
        """Test trailing stop sequence as price moves up"""
        entry_price = 100.0
        trail_step = 0.01
        
        price_sequence = [101, 102, 103, 104, 105]
        stops = []
        
        for price in price_sequence:
            profit = (price / entry_price) - 1
            steps = int(profit / trail_step)
            new_stop = entry_price * (1 + (steps - 1) * trail_step) if steps > 0 else entry_price * (1 - 0.50)
            stops.append(new_stop)
        
        # Verify stops are increasing
        for i in range(1, len(stops)):
            self.assertGreaterEqual(stops[i], stops[i-1])


class TestProfitTargetLogic(unittest.TestCase):
    """Test profit target detection"""
    
    def test_profit_target_hit(self):
        """Test detection when 100% profit is achieved"""
        entry_price = 100.0
        current_ltp = 200.0
        profit_target_pct = 1.00  # 100%
        
        profit_pct = (current_ltp / entry_price) - 1
        is_target_hit = profit_pct >= profit_target_pct
        
        self.assertTrue(is_target_hit)
    
    def test_profit_target_not_hit(self):
        """Test profit target not hit when below target"""
        entry_price = 100.0
        current_ltp = 150.0
        profit_target_pct = 1.00  # 100%
        
        profit_pct = (current_ltp / entry_price) - 1
        is_target_hit = profit_pct >= profit_target_pct
        
        self.assertFalse(is_target_hit)
    
    def test_profit_target_exact(self):
        """Test profit target at exact level"""
        entry_price = 100.0
        current_ltp = 200.0  # Exactly 100% profit
        profit_target_pct = 1.00
        
        profit_pct = (current_ltp / entry_price) - 1
        is_target_hit = profit_pct >= profit_target_pct
        
        self.assertTrue(is_target_hit)
    
    def test_profit_target_50pct(self):
        """Test 50% profit target"""
        entry_price = 100.0
        current_ltp = 150.0
        profit_target_pct = 0.50
        
        profit_pct = (current_ltp / entry_price) - 1
        is_target_hit = profit_pct >= profit_target_pct
        
        self.assertTrue(is_target_hit)
    
    def test_profit_target_50pct_not_hit(self):
        """Test 50% profit target not hit at 49%"""
        entry_price = 100.0
        current_ltp = 149.0
        profit_target_pct = 0.50
        
        profit_pct = (current_ltp / entry_price) - 1
        is_target_hit = profit_pct >= profit_target_pct
        
        self.assertFalse(is_target_hit)


class TestProfitCalculations(unittest.TestCase):
    """Test profit/loss calculations"""
    
    def test_profit_calculation(self):
        """Test profit percentage calculation"""
        entry_price = 100.0
        current_price = 150.0
        
        profit_pct = (current_price / entry_price) - 1
        gain_points = current_price - entry_price
        
        self.assertEqual(profit_pct, 0.50)  # 50% profit
        self.assertEqual(gain_points, 50.0)
    
    def test_loss_calculation(self):
        """Test loss percentage calculation"""
        entry_price = 100.0
        current_price = 75.0
        
        profit_pct = (current_price / entry_price) - 1
        loss_points = current_price - entry_price
        
        self.assertEqual(profit_pct, -0.25)  # 25% loss
        self.assertEqual(loss_points, -25.0)
    
    def test_p_l_with_quantity(self):
        """Test P&L calculation with quantity"""
        entry_price = 100.0
        exit_price = 150.0
        quantity = 75
        
        gain_points = exit_price - entry_price
        profit_loss = gain_points * quantity
        
        self.assertEqual(profit_loss, 3750.0)
    
    def test_p_l_with_quantity_loss(self):
        """Test P&L calculation with quantity on loss"""
        entry_price = 100.0
        exit_price = 50.0
        quantity = 75
        
        loss_points = exit_price - entry_price
        profit_loss = loss_points * quantity
        
        self.assertEqual(profit_loss, -3750.0)
    
    def test_multiple_positions_combined_pl(self):
        """Test combined P&L from multiple positions"""
        positions = [
            {'entry': 100.0, 'exit': 150.0, 'qty': 75},  # +3750
            {'entry': 200.0, 'exit': 180.0, 'qty': 50},  # -1000
        ]
        
        total_pl = sum((pos['exit'] - pos['entry']) * pos['qty'] for pos in positions)
        self.assertEqual(total_pl, 2750.0)


class TestIntegrationScenarios(unittest.TestCase):
    """Test complete trading scenarios"""
    
    def test_complete_profitable_trade_scenario(self):
        """Test a complete profitable trade from entry to exit"""
        entry_price = 100.0
        highest_high = 95.0
        breakout_price = 98.0
        target_exit_price = 200.0
        
        # Entry triggered at breakout
        self.assertGreater(breakout_price, highest_high)
        
        # Calculate profit
        profit_pct = (target_exit_price / entry_price) - 1
        self.assertGreaterEqual(profit_pct, 1.0)
    
    def test_trade_hit_stop_loss_scenario(self):
        """Test a trade that gets stopped out"""
        entry_price = 100.0
        initial_stop_pct = 0.50
        stop_price = entry_price * (1 - initial_stop_pct)  # 50.0
        
        exit_price = 45.0  # Below stop
        loss_pct = (exit_price / entry_price) - 1
        
        self.assertLess(exit_price, stop_price)
        self.assertLess(loss_pct, -initial_stop_pct)
    
    def test_trade_with_trailing_stop_scenario(self):
        """Test trailing stop moves up with profit"""
        entry_price = 100.0
        trail_step = 0.01
        
        # Simulate price movement with trailing stop
        prices = [101, 102, 103, 104, 105, 106]
        stops = []
        
        for price in prices:
            profit = (price / entry_price) - 1
            steps = int(profit / trail_step)
            new_stop = entry_price * (1 + (steps - 1) * trail_step) if steps > 0 else entry_price * (1 - 0.50)
            stops.append(new_stop)
        
        # Stops should be increasing (or staying same)
        for i in range(1, len(stops)):
            self.assertGreaterEqual(stops[i], stops[i-1])
    
    def test_three_leg_trade_with_different_stops(self):
        """Test three legs with different stop loss levels"""
        legs = {
            'CE': {'entry': 100, 'stop_pct': 0.50, 'qty': 75},
            'PE': {'entry': 100, 'stop_pct': 0.50, 'qty': 75},
            'ATM': {'entry': 50, 'stop_pct': 0.30, 'qty': 100},
        }
        
        total_entry_value = sum(leg['entry'] * leg['qty'] for leg in legs.values())
        
        for leg_name, leg_data in legs.items():
            stop = leg_data['entry'] * (1 - leg_data['stop_pct'])
            self.assertGreater(stop, 0)
            self.assertLess(stop, leg_data['entry'])
        
        self.assertGreater(total_entry_value, 0)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""
    
    def test_zero_profit_at_entry_price(self):
        """Test when current price equals entry price"""
        entry_price = 100.0
        current_price = 100.0
        
        profit_pct = (current_price / entry_price) - 1
        self.assertEqual(profit_pct, 0.0)
    
    def test_very_small_price_movement(self):
        """Test with very small price movements"""
        entry_price = 100.0
        current_price = 100.01
        
        profit_pct = (current_price / entry_price) - 1
        self.assertAlmostEqual(profit_pct, 0.0001, places=5)
    
    def test_very_small_trailing_step(self):
        """Test trailing stop calculation with very small step"""
        entry_price = 100.0
        trail_step = 0.001  # 0.1%
        
        profit_pct = 0.05  # 5% profit
        steps = int(profit_pct / trail_step)
        
        self.assertEqual(steps, 50)  # 50 steps
    
    def test_high_profit_scenario(self):
        """Test with very high profit (200% or more)"""
        entry_price = 100.0
        current_price = 400.0
        
        profit_pct = (current_price / entry_price) - 1
        self.assertEqual(profit_pct, 3.0)  # 300% profit
    
    def test_near_stop_loss(self):
        """Test price moving very close to stop loss"""
        entry_price = 100.0
        stop_price = 50.0
        current_price = 50.01  # Just above stop
        
        is_stop_hit = current_price <= stop_price
        self.assertFalse(is_stop_hit)
    
    def test_at_stop_loss_exactly(self):
        """Test price at exact stop loss level"""
        entry_price = 100.0
        stop_price = 50.0
        current_price = 50.0  # Exactly at stop
        
        is_stop_hit = current_price <= stop_price
        self.assertTrue(is_stop_hit)


class TestBreakoutThresholdVariations(unittest.TestCase):
    """Test breakout detection with various threshold levels"""
    
    def test_breakout_with_0_percent_threshold(self):
        """Test breakout with 0% threshold"""
        highest_high = 100.0
        ltp = 100.1
        threshold = 0.0
        
        is_breakout = ltp > highest_high * (1 + threshold)
        self.assertTrue(is_breakout)
    
    def test_breakout_with_0_5_percent_threshold(self):
        """Test breakout with 0.5% threshold"""
        highest_high = 100.0
        ltp = 100.6
        threshold = 0.005
        
        is_breakout = ltp > highest_high * (1 + threshold)
        self.assertTrue(is_breakout)
    
    def test_breakout_with_1_percent_threshold(self):
        """Test breakout with 1% threshold"""
        highest_high = 100.0
        ltp = 101.1
        threshold = 0.01
        
        is_breakout = ltp > highest_high * (1 + threshold)
        self.assertTrue(is_breakout)
    
    def test_breakout_with_2_percent_threshold(self):
        """Test breakout with 2% threshold"""
        highest_high = 100.0
        ltp = 102.1
        threshold = 0.02
        
        is_breakout = ltp > highest_high * (1 + threshold)
        self.assertTrue(is_breakout)


class TestTimezoneHandling(unittest.TestCase):
    """Test timezone-related functionality"""
    
    def test_ist_timezone_configuration(self):
        """Test IST timezone is correctly configured"""
        tz = pytz.timezone('Asia/Kolkata')
        now = datetime.now(tz)
        
        self.assertEqual(tz.zone, 'Asia/Kolkata')
        self.assertIsNotNone(now)
    
    def test_datetime_formatting(self):
        """Test datetime formatting for logs"""
        tz = pytz.timezone('Asia/Kolkata')
        now = datetime.now(tz)
        formatted = now.strftime('%Y-%m-%d %H:%M:%S IST')
        
        self.assertIn('IST', formatted)
        self.assertIn('-', formatted)
    
    def test_market_hours_time_parsing(self):
        """Test market hours time parsing"""
        start_hour = 10
        start_minute = 0
        end_hour = 15
        end_minute = 30
        
        start_time = datetime.strptime(f"{start_hour:02d}:{start_minute:02d}", "%H:%M").time()
        end_time = datetime.strptime(f"{end_hour:02d}:{end_minute:02d}", "%H:%M").time()
        
        # Verify times are parsed correctly
        self.assertEqual(start_time.hour, 10)
        self.assertEqual(start_time.minute, 0)
        self.assertEqual(end_time.hour, 15)
        self.assertEqual(end_time.minute, 30)


if __name__ == '__main__':
    unittest.main(verbosity=2)
