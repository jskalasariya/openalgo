"""
Nifty Sehwag Strategy - Entry Point (Refactored v2.0)
======================================================
Multi-leg options strategy for NIFTY trading (runs every day).

⚠️  REFACTORED: This file now uses modular architecture (v2.0)
    For details, see: REFACTORED_ARCHITECTURE.md

New Architecture:
  - models.py: Data structures
  - market_data.py: Market data management
  - order_manager.py: Order execution
  - position_manager.py: Risk management
  - utils.py: Helper functions
  - strategy.py: Main orchestrator

Usage:
  Recommended: python -m strategies.nifty_sehwag.main
  Also works:  python strategies/nifty_sehwag/nifty_sehwag.py

Configuration: nifty_sehwag_config.yaml
Version: 2.0.0
"""

import sys
import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from openalgo import api
from strategies.nifty_sehwag.strategy import NiftySehwagStrategy


def setup_logging():
    """Configure logging to both file and console with daily log files"""
    # Create logs directory if it doesn't exist
    log_dir = Path(__file__).parent / 'logs'
    log_dir.mkdir(exist_ok=True)
    
    # Generate log filename with date and timestamp
    log_filename = log_dir / f"nifty_sehwag_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Create logger
    logger = logging.getLogger('NiftySehwag')
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


def load_config():
    """Load strategy configuration"""
    config_file = Path(__file__).parent / 'nifty_sehwag_config.yaml'
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def initialize_clients(config, logger):
    """Initialize API and WebSocket clients"""
    # Get API key from environment variable or config (priority: env var > config)
    api_key = os.getenv(
        "OPENALGO_APIKEY",
        config.get('openalgo', {}).get('api_key')
    )
    
    if not api_key or api_key == 'YOUR_API_KEY_HERE':
        raise ValueError(
            "Please configure valid API key:\n"
            "  1. Set environment variable: OPENALGO_APIKEY=your_key\n"
            "  2. Or update nifty_sehwag_config.yaml"
        )
    
    # Initialize API client
    client = api(api_key=api_key)
    
    # Initialize WebSocket client if enabled
    websocket_client = None
    if config.get('websocket', {}).get('enabled', False):
        try:
            from utils.websocket_ltp_client import create_websocket_client
            websocket_client = create_websocket_client(
                ws_url=os.getenv('WEBSOCKET_URL', config.get('websocket', {}).get('url', 'ws://127.0.0.1:8765')),
                api_key=api_key,
                logger_instance=logger
            )
            websocket_client.start_background()
            logger.info("✓ WebSocket client initialized")
        except Exception as e:
            logger.warning(f"WebSocket client initialization failed: {e}")
            logger.warning("Will use REST API fallback")
    
    return client, websocket_client


def run_strategy_main_loop():
    """
    Main strategy execution (legacy function name for backwards compatibility)
    """
    logger, log_file = setup_logging()
    
    logger.info("\n" + "="*70)
    logger.info("🚀 Nifty Sehwag Strategy v2.0 (Refactored)")
    logger.info("="*70)
    logger.info(f"📝 Log file: {log_file}")
    logger.info("="*70 + "\n")
    
    try:
        # Initialize database
        try:
            from database.nifty_sehwag_db import init_db
            init_db()
            logger.info("✓ Database initialized")
        except Exception as e:
            logger.warning(f"⚠️  Database initialization failed: {e}")
            logger.warning("Strategy will continue without database persistence")
        
        # Load configuration
        logger.info("📋 Loading configuration...")
        config = load_config()
        logger.info("✓ Configuration loaded")
        
        # Initialize clients
        logger.info("🔌 Initializing API clients...")
        client, websocket_client = initialize_clients(config, logger)
        logger.info("✓ API clients initialized")
        
        # Initialize strategy
        logger.info("⚙️  Initializing strategy...")
        strategy = NiftySehwagStrategy(client, config, websocket_client)
        logger.info("✓ Strategy initialized")
        
        # Run strategy
        strategy.run()
        
        logger.info("\n✅ Strategy completed successfully\n")
        
    except KeyboardInterrupt:
        logger.warning("\n⚠️  Strategy interrupted by user\n")
    except Exception as e:
        logger.error(f"\n❌ Strategy failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main entry point"""
    run_strategy_main_loop()


if __name__ == "__main__":
    main()
