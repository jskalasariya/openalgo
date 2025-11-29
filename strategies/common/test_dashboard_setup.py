"""
Test Generic Strategy Dashboard
================================
Quick test to verify dashboard configuration loads correctly.
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

def test_dashboard_imports():
    """Test that all required imports work"""
    print("Testing imports...")
    
    try:
        import streamlit as st
        print("✓ Streamlit imported")
    except ImportError as e:
        print(f"✗ Streamlit import failed: {e}")
        return False
    
    try:
        import plotly.graph_objects as go
        print("✓ Plotly imported")
    except ImportError as e:
        print(f"✗ Plotly import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ Pandas imported")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    try:
        import yaml
        print("✓ PyYAML imported")
    except ImportError as e:
        print(f"✗ PyYAML import failed: {e}")
        return False
    
    try:
        from sqlalchemy import create_engine
        print("✓ SQLAlchemy imported")
    except ImportError as e:
        print(f"✗ SQLAlchemy import failed: {e}")
        return False
    
    return True

def test_config_loading():
    """Test configuration loading"""
    print("\nTesting unified configuration...")
    
    import yaml
    from pathlib import Path
    
    config_dir = Path(__file__).parent
    
    # Test unified config file
    unified_config = config_dir / "dashboard_config.yaml"
    if unified_config.exists():
        try:
            with open(unified_config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Check global config
            if 'global' in config:
                print(f"✓ Global config loaded")
            else:
                print("⚠ Global config section not found")
            
            # Check strategies config
            if 'strategies' in config:
                strategies = config['strategies']
                print(f"✓ Found {len(strategies)} strategy configurations")
                
                # Check for expected strategies
                if 'expiry_blast' in strategies:
                    print(f"  ✓ Expiry Blast: {strategies['expiry_blast'].get('display_name')}")
                else:
                    print("  ⚠ Expiry Blast config not found")
                
                if 'nifty_sehwag' in strategies:
                    print(f"  ✓ Nifty Sehwag: {strategies['nifty_sehwag'].get('display_name')}")
                else:
                    print("  ⚠ Nifty Sehwag config not found")
            else:
                print("✗ Strategies config section not found")
                return False
            
            return True
        except Exception as e:
            print(f"✗ Failed to load unified config: {e}")
            return False
    else:
        print("✗ Unified config file 'dashboard_config.yaml' not found")
        return False

def test_database_models():
    """Test database models import"""
    print("\nTesting database models...")
    
    try:
        from database.expiry_blast_db import (
            StrategySession,
            StrategyPosition,
            PositionEvent,
            SessionStatus,
            PositionStatus,
            EventType,
        )
        print("✓ Database models imported")
        return True
    except ImportError as e:
        print(f"✗ Failed to import database models: {e}")
        return False

def test_launcher_scripts():
    """Test that launcher scripts exist"""
    print("\nTesting launcher scripts...")
    
    common_dir = Path(__file__).parent
    
    # Check common launchers (new centralized location)
    ps1_launcher = common_dir / "launch_dashboard.ps1"
    sh_launcher = common_dir / "launch_dashboard.sh"
    bat_launcher = common_dir / "launch_dashboard.bat"
    
    all_exist = True
    
    if ps1_launcher.exists():
        print("✓ PowerShell launcher exists (launch_dashboard.ps1)")
    else:
        print("✗ PowerShell launcher missing (launch_dashboard.ps1)")
        all_exist = False
    
    if sh_launcher.exists():
        print("✓ Bash launcher exists (launch_dashboard.sh)")
    else:
        print("✗ Bash launcher missing (launch_dashboard.sh)")
        all_exist = False
    
    if bat_launcher.exists():
        print("✓ CMD launcher exists (launch_dashboard.bat)")
    else:
        print("✗ CMD launcher missing (launch_dashboard.bat)")
        all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("=" * 60)
    print("Generic Strategy Dashboard - Setup Verification")
    print("=" * 60)
    
    all_passed = True
    
    # Test imports
    if not test_dashboard_imports():
        all_passed = False
        print("\n⚠️ Some imports failed. Run: pip install -r requirements.txt")
    
    # Test config loading
    if not test_config_loading():
        all_passed = False
    
    # Test database models
    if not test_database_models():
        all_passed = False
    
    # Test launcher scripts
    test_launcher_scripts()
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed! Dashboard is ready to use.")
        print("\nTo launch dashboard:")
        print("  Windows:       .\\launch_dashboard.ps1")
        print("  Linux/Mac:     ./launch_dashboard.sh")
        print("  With menu:     .\\launch_dashboard.ps1 (no args)")
        print("  Direct:        .\\launch_dashboard.ps1 -Strategy expiry_blast")
        print("                 ./launch_dashboard.sh nifty_sehwag")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
    print("=" * 60)

if __name__ == "__main__":
    main()
