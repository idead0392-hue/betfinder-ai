"""
Comprehensive test suite for OpenAI Agent Router system
Tests routing, data formatting, integration, and error handling
"""

import unittest
import json
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import sys

# Add project root to path
sys.path.append('/workspaces/betfinder-ai')

# Import modules to test
from openai_agent_router import (
    OpenAIAgentRouter, PropData, StatsData, ContextData, create_agent_router
)
from sport_data_formatters import (
    SportFormatterFactory, format_props_for_sport, extract_prop_insights,
    NBAFormatter, NFLFormatter, SoccerFormatter, EsportsFormatter
)
from agent_integration import EnhancedSportAgent, AgentManager, create_agent_manager

class TestOpenAIAgentRouter(unittest.TestCase):
    """Test OpenAI Agent Router functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.router = OpenAIAgentRouter(api_key="test_key_123", enable_logging=False)
        
        self.sample_prop = PropData(
            player="LeBron James",
            prop="Over 24.5 Points",
            line=24.5,
            odds=-110,
            opponent="Warriors",
            sport="basketball",
            matchup="Lakers vs Warriors"
        )
    
    def test_sport_detection(self):
        """Test sport detection from various inputs"""
        # Test PropData input
        sport = self.router.detect_sport(self.sample_prop)
        self.assertEqual(sport, "basketball")
        
        # Test dict input
        prop_dict = {"sport": "NFL", "prop": "passing yards"}
        sport = self.router.detect_sport(prop_dict)
        self.assertEqual(sport, "nfl")
        
        # Test string input
        sport = self.router.detect_sport("NBA basketball points")
        self.assertEqual(sport, "basketball")
        
        # Test unknown sport
        sport = self.router.detect_sport("unknown sport type")
        self.assertEqual(sport, "unknown")
    
    def test_agent_mapping(self):
        """Test sport to agent mapping"""
        # Test valid sports
        self.assertEqual(self.router.get_agent_for_sport("basketball"), "NBA_Analysis_Agent")
        self.assertEqual(self.router.get_agent_for_sport("football"), "NFL_Analysis_Agent")
        self.assertEqual(self.router.get_agent_for_sport("csgo"), "CSGO_Analysis_Agent")
        
        # Test invalid sport
        self.assertIsNone(self.router.get_agent_for_sport("invalid_sport"))
    
    def test_input_formatting(self):
        """Test agent input formatting"""
        props = [self.sample_prop]
        stats = StatsData(season_avg=25.5, last_5_avg=28.0)
        context = ContextData(injuries=["Minor ankle sprain"], weather="Clear")
        
        formatted = self.router.format_agent_input(props, stats, context)
        
        self.assertIn("analysis_type", formatted)
        self.assertIn("props", formatted)
        self.assertIn("stats", formatted)
        self.assertIn("context", formatted)
        self.assertEqual(len(formatted["props"]), 1)
        self.assertEqual(formatted["props"][0]["player"], "LeBron James")
    
    def test_prompt_creation(self):
        """Test analysis prompt creation"""
        agent_input = {
            "props": [{"player": "Test Player", "prop": "Over 10.5 Points", "line": 10.5}],
            "stats": {"season_avg": 12.0},
            "context": {"injuries": []}
        }
        
        prompt = self.router.create_analysis_prompt(agent_input, "NBA")
        
        self.assertIn("NBA Prop Betting Analysis", prompt)
        self.assertIn("Test Player", prompt)
        self.assertIn("Over 10.5 Points", prompt)
        self.assertIn("Statistical Data", prompt)
        self.assertIn("Additional Context", prompt)
    
    def test_supported_sports(self):
        """Test supported sports list"""
        sports = self.router.get_supported_sports()
        
        self.assertIn("basketball", sports)
        self.assertIn("football", sports)
        self.assertIn("soccer", sports)
        self.assertIn("csgo", sports)
        self.assertTrue(len(sports) > 10)
    
    def test_agent_status(self):
        """Test agent status reporting"""
        status = self.router.get_agent_status()
        
        self.assertIn("total_agents", status)
        self.assertIn("supported_sports", status)
        self.assertIn("agent_mapping", status)
        self.assertIn("api_key_configured", status)
        self.assertTrue(status["api_key_configured"])

class TestSportDataFormatters(unittest.TestCase):
    """Test sport-specific data formatters"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.sample_nba_props = [
            {
                "Name": "LeBron James",
                "Prop": "Over 24.5 Points",
                "Points": 24.5,
                "team": "LAL",
                "opponent": "GSW"
            },
            {
                "Name": "Stephen Curry",
                "Prop": "Over 5.5 Assists",
                "Points": 5.5,
                "team": "GSW",
                "opponent": "LAL"
            }
        ]
    
    def test_formatter_factory(self):
        """Test formatter factory creates correct formatters"""
        nba_formatter = SportFormatterFactory.create_formatter("NBA")
        self.assertIsInstance(nba_formatter, NBAFormatter)
        
        nfl_formatter = SportFormatterFactory.create_formatter("NFL")
        self.assertIsInstance(nfl_formatter, NFLFormatter)
        
        soccer_formatter = SportFormatterFactory.create_formatter("soccer")
        self.assertIsInstance(soccer_formatter, SoccerFormatter)
        
        csgo_formatter = SportFormatterFactory.create_formatter("csgo")
        self.assertIsInstance(csgo_formatter, EsportsFormatter)
    
    def test_nba_formatting(self):
        """Test NBA-specific formatting"""
        formatted = format_props_for_sport("NBA", self.sample_nba_props)
        
        self.assertEqual(formatted["sport"], "NBA")
        self.assertEqual(formatted["total_props"], 2)
        self.assertIn("props", formatted)
        
        # Check prop details
        prop1 = formatted["props"][0]
        self.assertEqual(prop1["player"], "LeBron James")
        self.assertEqual(prop1["line"], 24.5)
        self.assertEqual(prop1["sport"], "NBA")
    
    def test_prop_insights_extraction(self):
        """Test extraction of insights from props"""
        formatted = format_props_for_sport("NBA", self.sample_nba_props)
        insights = extract_prop_insights(formatted["props"])
        
        self.assertEqual(insights["total_props"], 2)
        self.assertIn("LeBron James", insights["players"])
        self.assertIn("Stephen Curry", insights["players"])
        self.assertIn("average_line", insights)
        self.assertIn("line_range", insights)
    
    def test_esports_formatter(self):
        """Test esports-specific formatting"""
        csgo_formatter = EsportsFormatter("CSGO")
        self.assertEqual(csgo_formatter.sport_name, "CSGO Esports")
        self.assertIn("kills", csgo_formatter.stat_categories)
        self.assertIn("adr", csgo_formatter.stat_categories)
        
        lol_formatter = EsportsFormatter("LoL")
        self.assertIn("cs", lol_formatter.stat_categories)
        self.assertIn("gold", lol_formatter.stat_categories)

class TestAgentIntegration(unittest.TestCase):
    """Test agent integration and management"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.enhanced_agent = EnhancedSportAgent(
            sport="basketball",
            use_openai=False,  # Use local only for testing
            openai_api_key=None
        )
        
        self.sample_props = [
            {
                "Name": "Test Player",
                "Prop": "Over 10.5 Points",
                "Points": 10.5
            }
        ]
    
    def test_enhanced_agent_initialization(self):
        """Test enhanced agent initialization"""
        self.assertEqual(self.enhanced_agent.sport, "basketball")
        self.assertFalse(self.enhanced_agent.openai_enabled)
        self.assertIsNotNone(self.enhanced_agent.local_agent)
    
    @patch('agent_integration.BasketballAgent')
    def test_local_agent_analysis(self, mock_basketball_agent):
        """Test analysis using local agent"""
        # Mock the local agent
        mock_agent_instance = Mock()
        mock_agent_instance.make_picks.return_value = [
            {
                "player_name": "Test Player",
                "confidence": 75,
                "expected_value": 5.0
            }
        ]
        mock_basketball_agent.return_value = mock_agent_instance
        
        # Recreate agent with mock
        agent = EnhancedSportAgent("basketball", use_openai=False)
        agent.local_agent = mock_agent_instance
        
        result = agent.analyze_props(self.sample_props)
        
        self.assertTrue(result["success"])
        self.assertIn("picks", result)
        self.assertEqual(len(result["picks"]), 1)
        self.assertIn("analysis_metadata", result)
    
    def test_performance_tracking(self):
        """Test performance statistics tracking"""
        # Simulate some requests
        self.enhanced_agent.request_count = 10
        self.enhanced_agent.success_count = 8
        self.enhanced_agent.local_usage_count = 8
        
        stats = self.enhanced_agent.get_performance_stats()
        
        self.assertEqual(stats["total_requests"], 10)
        self.assertEqual(stats["successful_requests"], 8)
        self.assertEqual(stats["success_rate"], 80.0)
        self.assertEqual(stats["local_usage"], 8)
    
    def test_agent_manager(self):
        """Test agent manager functionality"""
        manager = AgentManager(use_openai=False)
        
        # Test agent retrieval
        basketball_agent = manager.get_agent("basketball")
        self.assertIsNotNone(basketball_agent)
        self.assertEqual(basketball_agent.sport, "basketball")
        
        # Test unsupported sport
        unknown_agent = manager.get_agent("unknown_sport")
        self.assertIsNone(unknown_agent)
        
        # Test supported sports
        self.assertIn("basketball", manager.supported_sports)
        self.assertIn("football", manager.supported_sports)

class TestErrorHandling(unittest.TestCase):
    """Test error handling and edge cases"""
    
    def test_invalid_api_key(self):
        """Test handling of invalid API key"""
        with self.assertRaises(ValueError):
            OpenAIAgentRouter(api_key=None)
    
    def test_empty_props_list(self):
        """Test handling of empty props list"""
        router = OpenAIAgentRouter(api_key="test_key", enable_logging=False)
        
        result = router.route_to_sport_agent("basketball", [])
        self.assertFalse(result["success"])
        self.assertIn("error", result)
    
    def test_unsupported_sport(self):
        """Test handling of unsupported sport"""
        router = OpenAIAgentRouter(api_key="test_key", enable_logging=False)
        
        prop = PropData(
            player="Test", prop="Test", line=10, odds=-110,
            opponent="Test", sport="unknown", matchup="Test"
        )
        
        result = router.route_to_sport_agent("unknown_sport", [prop])
        self.assertFalse(result["success"])
        self.assertIn("available_sports", result)
    
    def test_malformed_prop_data(self):
        """Test handling of malformed prop data"""
        enhanced_agent = EnhancedSportAgent("basketball", use_openai=False)
        
        # Test with malformed data
        malformed_props = [
            {"invalid": "data"},
            None,
            "string_instead_of_dict"
        ]
        
        # Should handle gracefully and not crash
        result = enhanced_agent.analyze_props(malformed_props)
        self.assertIn("success", result)

class TestIntegrationScenarios(unittest.TestCase):
    """Test real-world integration scenarios"""
    
    def setUp(self):
        """Set up integration test scenarios"""
        self.manager = create_agent_manager(use_openai=False)
    
    def test_full_analysis_workflow(self):
        """Test complete analysis workflow"""
        # Sample props from different sports
        nba_props = [{"Name": "LeBron James", "Prop": "Over 24.5 Points", "Points": 24.5}]
        nfl_props = [{"Name": "Tom Brady", "Prop": "Over 250.5 Passing Yards", "Points": 250.5}]
        
        # Analyze NBA props
        nba_result = self.manager.analyze_props("basketball", nba_props)
        self.assertTrue(nba_result.get("success", False))
        
        # Analyze NFL props
        nfl_result = self.manager.analyze_props("football", nfl_props)
        self.assertTrue(nfl_result.get("success", False))
        
        # Check performance stats
        stats = self.manager.get_all_performance_stats()
        self.assertIn("overall_stats", stats)
        self.assertGreater(stats["overall_stats"]["total_requests"], 0)
    
    def test_mode_switching(self):
        """Test switching between OpenAI and local modes"""
        # Start in local mode
        self.manager.switch_to_local_mode()
        
        # Verify all agents are in local mode
        for agent in self.manager.agents.values():
            self.assertFalse(agent.openai_enabled)
        
        # Switch to OpenAI mode (will fail without real API key, but should not crash)
        self.manager.switch_to_openai_mode()
        
        # Should complete without errors
        self.assertIsNotNone(self.manager)

def run_test_suite():
    """Run the complete test suite"""
    
    # Create test suite
    test_classes = [
        TestOpenAIAgentRouter,
        TestSportDataFormatters,
        TestAgentIntegration,
        TestErrorHandling,
        TestIntegrationScenarios
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return test results
    return {
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "success_rate": ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    }

if __name__ == "__main__":
    print("🧪 Running OpenAI Agent Router Test Suite")
    print("=" * 50)
    
    results = run_test_suite()
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"Tests Run: {results['tests_run']}")
    print(f"Failures: {results['failures']}")
    print(f"Errors: {results['errors']}")
    print(f"Success Rate: {results['success_rate']:.1f}%")
    
    if results['failures'] == 0 and results['errors'] == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed. Check the output above for details.")