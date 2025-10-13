"""
Memory Integration Demo for BetFinder AI

Demonstrates how agents use mem0ai for personalization and learning:
- Store pick results and user preferences
- Retrieve personalized insights
- Adapt recommendations based on user patterns
- Learn from betting sessions
"""

import sys
import os
sys.path.append('/workspaces/betfinder-ai')

from memory_manager import global_memory
from sport_agents import BasketballAgent, CSGOAgent

def demo_memory_integration():
    print("🧠 Memory Integration Demo")
    print("=" * 50)
    
    # Initialize agents
    print("\n1. Initializing agents with memory...")
    nba_agent = BasketballAgent()
    csgo_agent = CSGOAgent()
    
    # Sample user
    user_id = "demo_user_123"
    
    # Sample props
    nba_prop = {
        "player_name": "LeBron James",
        "stat_type": "points",
        "line": 28.5,
        "over_under": "over",
        "sport": "basketball",
        "odds": -110
    }
    
    csgo_prop = {
        "player_name": "s1mple",
        "stat_type": "kills",
        "line": 18.5,
        "over_under": "over", 
        "sport": "csgo",
        "odds": -105
    }
    
    print("✅ Agents initialized with memory capabilities")
    
    # Demo 1: Store pick results
    print("\n2. Storing pick results...")
    
    # Store some NBA results
    nba_agent.store_pick_result(user_id, nba_prop, "won", 85.0, "Strong performance against weak defense")
    nba_agent.store_pick_result(user_id, nba_prop, "won", 78.0, "Home court advantage")
    nba_agent.store_pick_result(user_id, nba_prop, "lost", 72.0, "Unexpected rest day")
    
    # Store some CSGO results  
    csgo_agent.store_pick_result(user_id, csgo_prop, "won", 90.0, "Excellent form on favorite map")
    csgo_agent.store_pick_result(user_id, csgo_prop, "won", 83.0, "Team synergy strong")
    
    print("✅ Stored pick results for personalization")
    
    # Demo 2: Store user preferences
    print("\n3. Storing user preferences...")
    
    nba_agent.store_user_feedback(user_id, "favorite_stat", "points", "User prefers scoring props")
    nba_agent.store_user_feedback(user_id, "risk_tolerance", "moderate", "Likes 70-80% confidence picks")
    csgo_agent.store_user_feedback(user_id, "favorite_stat", "kills", "Prefers frag-based props")
    
    print("✅ Stored user preferences")
    
    # Demo 3: Get personalized insights
    print("\n4. Getting personalized insights...")
    
    nba_insights = nba_agent.get_personalized_insights(user_id, nba_prop)
    csgo_insights = csgo_agent.get_personalized_insights(user_id, csgo_prop)
    
    print(f"📊 NBA insights: {nba_insights.get('stat_pattern', {}).get('pattern', 'none')} pattern")
    print(f"📊 CSGO insights: {csgo_insights.get('stat_pattern', {}).get('pattern', 'none')} pattern")
    
    # Demo 4: Enhanced picks with memory
    print("\n5. Generating memory-enhanced picks...")
    
    # Create sample picks
    nba_pick = {
        "player_name": "LeBron James",
        "stat_type": "points", 
        "confidence": 75.0,
        "sport": "basketball"
    }
    
    csgo_pick = {
        "player_name": "s1mple",
        "stat_type": "kills",
        "confidence": 80.0, 
        "sport": "csgo"
    }
    
    # Enhance with memory
    enhanced_nba = nba_agent.enhance_pick_with_memory(user_id, nba_pick)
    enhanced_csgo = csgo_agent.enhance_pick_with_memory(user_id, csgo_pick)
    
    print(f"🎯 NBA pick confidence: {nba_pick['confidence']}% → {enhanced_nba.get('confidence', 0)}%")
    if enhanced_nba.get('memory_adjustment'):
        print(f"   Adjustment: {enhanced_nba['memory_adjustment']}")
    
    print(f"🎯 CSGO pick confidence: {csgo_pick['confidence']}% → {enhanced_csgo.get('confidence', 0)}%")
    if enhanced_csgo.get('memory_adjustment'):
        print(f"   Adjustment: {enhanced_csgo['memory_adjustment']}")
    
    # Demo 5: Agent learning from sessions
    print("\n6. Learning from betting sessions...")
    
    session_results = [
        {"outcome": "won", "sport": "basketball", "stat_type": "points"},
        {"outcome": "won", "sport": "basketball", "stat_type": "assists"},
        {"outcome": "lost", "sport": "csgo", "stat_type": "kills"},
        {"outcome": "won", "sport": "csgo", "stat_type": "deaths"}
    ]
    
    nba_agent.learn_from_session(session_results)
    print("✅ Session learnings stored")
    
    # Demo 6: Cross-agent insights
    print("\n7. Cross-agent insights...")
    
    global_memory.add_cross_agent_insight(
        "Users show strong preference for scoring-based props across all sports",
        {"insight_type": "user_behavior", "confidence": "high"}
    )
    
    insights = global_memory.get_cross_agent_insights("user_behavior")
    print(f"📈 Found {len(insights)} cross-agent insights")
    
    print("\n🎉 Memory Integration Demo Complete!")
    print("\n📋 Key Features Demonstrated:")
    print("   ✅ Pick result storage and retrieval")
    print("   ✅ User preference tracking")
    print("   ✅ Pattern analysis and personalization")
    print("   ✅ Confidence adjustment based on history")
    print("   ✅ Session-based learning")
    print("   ✅ Cross-agent insight sharing")
    print("\n🚀 Ready for production with intelligent memory!")

if __name__ == "__main__":
    # Set environment for demo
    os.environ.pop('DISABLE_MEMORY', None)  # Enable memory
    demo_memory_integration()