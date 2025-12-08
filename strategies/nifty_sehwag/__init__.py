"""
Nifty Sehwag Strategy Package
==============================
Multi-leg options strategy for NIFTY trading (runs every trading day).

Refactored modular architecture (v2.0):
- models.py: Data structures (LegPosition, StrategyState, LegSchedule)
- market_data.py: Market data management (MarketDataManager)
- order_manager.py: Order execution (OrderManager)
- position_manager.py: Risk management (PositionManager)
- utils.py: Helper functions
- strategy.py: Main orchestrator (NiftySehwagStrategy)
- main.py: Entry point

Legacy:
- nifty_sehwag.py: Original monolithic implementation (deprecated)

Entry:
    python -m strategies.nifty_sehwag.main
"""

from .strategy import NiftySehwagStrategy
from .models import LegPosition, StrategyState, LegSchedule

__version__ = "2.0.0"
__author__ = "OpenAlgo Team"

__all__ = [
    'NiftySehwagStrategy',
    'LegPosition',
    'StrategyState',
    'LegSchedule'
]
__all__ = ["run_strategy_main_loop"]
