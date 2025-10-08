#!/usr/bin/env python3
"""
Test script for SportbexProvider class.

This script tests the SportbexProvider implementation with the existing
SPORTBEX_API_KEY to ensure it works correctly.
"""

import os
import sys
import logging
from api_providers import SportbexProvider, SportType, create_sportbex_provider

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_sportbex_provider():
    """Test the SportbexProvider implementation."""
    
    # Use the same API key as in api_server.py
    api_key = 'NZLDw8ZXFv0O8elaPq0wjbP4zxb2gCwJDsArWQUF'
    
    try:
        # Create provider instance
        logger.info("Creating SportbexProvider instance...")
        provider = SportbexProvider(api_key=api_key)
        
        # Test 1: Health check
        logger.info("Testing health check...")
        health_response = provider.health_check()
        print(f"Health Check: {'PASS' if health_response.success else 'FAIL'}")
        if not health_response.success:
            print(f"  Error: {health_response.error_message}")
        else:
            print(f"  Response time: {health_response.data.get('response_time', 'N/A')}s")
        
        # Test 2: Get tennis competitions
        logger.info("Testing get_competitions for tennis...")
        tennis_response = provider.get_competitions(sport=SportType.TENNIS)
        print(f"Tennis Competitions: {'PASS' if tennis_response.success else 'FAIL'}")
        if tennis_response.success:
            data = tennis_response.data
            if isinstance(data, dict) and 'data' in data:
                competitions = data['data']
            else:
                competitions = data if isinstance(data, list) else []
            print(f"  Found {len(competitions)} tennis competitions")
        else:
            print(f"  Error: {tennis_response.error_message}")
        
        # Test 3: Get basketball props
        logger.info("Testing get_props for basketball...")
        basketball_response = provider.get_props(sport=SportType.BASKETBALL)
        print(f"Basketball Props: {'PASS' if basketball_response.success else 'FAIL'}")
        if basketball_response.success:
            data = basketball_response.data
            if isinstance(data, dict) and 'data' in data:
                props = data['data']
            else:
                props = data if isinstance(data, list) else []
            print(f"  Found {len(props)} basketball competitions/props")
        else:
            print(f"  Error: {basketball_response.error_message}")
        
        # Test 4: Get odds (general request)
        logger.info("Testing get_odds...")
        odds_response = provider.get_odds()
        print(f"Odds Data: {'PASS' if odds_response.success else 'FAIL'}")
        if odds_response.success:
            data = odds_response.data
            if isinstance(data, dict) and 'data' in data:
                odds = data['data']
            else:
                odds = data if isinstance(data, list) else []
            print(f"  Found {len(odds)} odds entries")
        else:
            print(f"  Error: {odds_response.error_message}")
        
        # Test 5: Test factory function
        logger.info("Testing factory function...")
        factory_provider = create_sportbex_provider(api_key=api_key)
        factory_health = factory_provider.health_check()
        print(f"Factory Function: {'PASS' if factory_health.success else 'FAIL'}")
        
        # Summary
        tests = [health_response, tennis_response, basketball_response, odds_response, factory_health]
        passed = sum(1 for test in tests if test.success)
        total = len(tests)
        
        print(f"\nTest Summary: {passed}/{total} tests passed")
        
        if passed == total:
            print("✅ All tests passed! SportbexProvider is working correctly.")
            return True
        else:
            print("❌ Some tests failed. Check the logs above for details.")
            return False
        
    except Exception as e:
        logger.error(f"Test failed with exception: {e}")
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = test_sportbex_provider()
    sys.exit(0 if success else 1)