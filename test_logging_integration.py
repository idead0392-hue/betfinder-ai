#!/usr/bin/env python3
"""
Test script to verify the comprehensive logging and monitoring integration
Tests the complete workflow with OpenAI Agent Router and Enhanced Sport Agent
"""

import logging
from datetime import datetime
from typing import Dict, List

# Import the integrated components
from agent_integration import EnhancedSportAgent
from openai_agent_router import OpenAIAgentRouter
from agent_analytics_tracker import AgentAnalyticsTracker

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_props() -> List[Dict]:
    """Create sample prop data for testing"""
    return [
        {
            "player": "LeBron James",
            "prop": "Points",
            "line": 25.5,
            "odds": -110,
            "opponent": "Boston Celtics",
            "sport": "basketball",
            "matchup": "Lakers @ Celtics"
        },
        {
            "player": "Anthony Davis",
            "prop": "Rebounds",
            "line": 11.5,
            "odds": -115,
            "opponent": "Boston Celtics",
            "sport": "basketball",
            "matchup": "Lakers @ Celtics"
        },
        {
            "player": "Jayson Tatum",
            "prop": "Points",
            "line": 27.5,
            "odds": -105,
            "opponent": "Los Angeles Lakers",
            "sport": "basketball",
            "matchup": "Lakers @ Celtics"
        }
    ]

def test_enhanced_sport_agent():
    """Test the enhanced sport agent with comprehensive logging"""
    
    print("🏀 Testing Enhanced Sport Agent with Comprehensive Logging...")
    print("=" * 60)
    
    try:
        # Initialize enhanced agent (will fallback to local since no real OpenAI key)
        agent = EnhancedSportAgent(
            sport="basketball",
            use_openai=True,  # Will fallback to local agent
            openai_api_key=None  # No real key for testing
        )
        
        print(f"✅ Enhanced agent initialized for {agent.sport}")
        print(f"   OpenAI enabled: {agent.openai_enabled}")
        print(f"   Monitoring initialized: {agent.monitor is not None}")
        print(f"   Analytics initialized: {agent.analytics is not None}")
        print()
        
        # Create test props
        test_props = create_test_props()
        print(f"📊 Created {len(test_props)} test props:")
        for i, prop in enumerate(test_props, 1):
            print(f"   {i}. {prop['player']} - {prop['prop']} {prop['line']} ({prop['odds']})")
        print()
        
        # Analyze props
        print("🔍 Analyzing props...")
        result = agent.analyze_props(
            props_data=test_props,
            include_stats=True,
            include_context=True,
            force_local=True  # Force local for testing
        )
        
        print("✅ Analysis completed!")
        print(f"   Success: {result.get('success', 'Unknown')}")
        print(f"   Method: {result.get('analysis_metadata', {}).get('method', 'Unknown')}")
        print(f"   Duration: {result.get('analysis_metadata', {}).get('duration_ms', 0):.2f}ms")
        print(f"   Picks generated: {len(result.get('picks', []))}")
        print()
        
        # Display agent statistics
        print("📈 Agent Performance Statistics:")
        print(f"   Total requests: {agent.request_count}")
        print(f"   Successful requests: {agent.success_count}")
        print(f"   OpenAI usage: {agent.openai_usage_count}")
        print(f"   Local usage: {agent.local_usage_count}")
        
        if agent.request_count > 0:
            success_rate = (agent.success_count / agent.request_count) * 100
            print(f"   Success rate: {success_rate:.1f}%")
        print()
        
        # Check if monitoring is working
        print("📊 Monitoring Status:")
        print(f"   Monitor data directory: {agent.monitor.data_dir}")
        print(f"   Analytics data directory: {agent.analytics.data_dir}")
        print()
        
        # Test logging export
        print("📝 Testing Log Export...")
        log_data = agent.agent_logger.export_logs(
            start_date=datetime.now().replace(hour=0, minute=0, second=0),
            format="dict"
        )
        print(f"   Exported {len(log_data)} log entries")
        
        if log_data:
            print(f"   Latest log entry type: {log_data[-1].get('log_type', 'unknown')}")
            print(f"   Latest operation: {log_data[-1].get('operation_name', 'unknown')}")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.error(f"Enhanced sport agent test failed: {e}")
        return False

def test_openai_agent_router():
    """Test the OpenAI Agent Router with monitoring"""
    
    print("🤖 Testing OpenAI Agent Router with Monitoring...")
    print("=" * 60)
    
    try:
        # Initialize router (will work with mock data)
        router = OpenAIAgentRouter(
            api_key="test_key",  # Mock key for testing
            enable_logging=True
        )
        
        print("✅ OpenAI Agent Router initialized")
        print()
        
        # Test sport detection
        test_prop_data = {
            "player": "Patrick Mahomes",
            "prop": "Passing Yards",
            "line": 275.5,
            "sport": "football"
        }
        
        detected_sport = router.detect_sport(test_prop_data)
        print("🎯 Sport Detection Test:")
        print(f"   Input: {test_prop_data['prop']} for {test_prop_data['player']}")
        print(f"   Detected sport: {detected_sport}")
        print()
        
        # Test agent status
        print("📊 Agent Status:")
        status = router.get_agent_status()
        print(f"   Total agents configured: {status['total_agents']}")
        print(f"   Supported sports: {len(status['supported_sports'])}")
        print(f"   API key configured: {status['api_key_configured']}")
        print(f"   Monitoring session active: {status['monitoring']['session_active']}")
        print(f"   Success rate: {status['monitoring']['success_rate']:.3f}")
        print()
        
        # Test supported sports
        supported_sports = router.get_supported_sports()
        print(f"🏈 Supported Sports ({len(supported_sports)}):")
        for sport in supported_sports[:10]:  # Show first 10
            agent_name = router.get_agent_for_sport(sport)
            print(f"   - {sport} → {agent_name}")
        if len(supported_sports) > 10:
            print(f"   ... and {len(supported_sports) - 10} more")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.error(f"OpenAI agent router test failed: {e}")
        return False

def test_analytics_dashboard():
    """Test analytics components"""
    
    print("📈 Testing Analytics and Monitoring Components...")
    print("=" * 60)
    
    try:
        # Initialize analytics tracker
        analytics = AgentAnalyticsTracker(
            data_dir="analytics_data/test"
        )
        
        print("✅ Analytics tracker initialized")
        print(f"   Data directory: {analytics.data_dir}")
        print()
        
        # Test basic functionality
        print("📊 Analytics Components:")
        print(f"   Analysis window: {analytics.analysis_window} days")
        print(f"   Data directory exists: {analytics.data_dir.exists()}")
        print()
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        logger.error(f"Analytics test failed: {e}")
        return False

def main():
    """Run all integration tests"""
    
    print("🚀 BetFinder AI - Comprehensive Logging Integration Test")
    print("=" * 80)
    print()
    
    # Track test results
    test_results = {}
    
    # Run tests
    test_results['enhanced_agent'] = test_enhanced_sport_agent()
    test_results['router'] = test_openai_agent_router()
    test_results['analytics'] = test_analytics_dashboard()
    
    # Summary
    print("📋 Test Summary:")
    print("=" * 40)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    print()
    print(f"🏁 Overall Result: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Comprehensive logging integration is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the logs for details.")
    
    print()
    print("📁 Log files created in:")
    print("   - logs/agent_router/")
    print("   - logs/integration/basketball/")
    print("   - logs/test/")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)