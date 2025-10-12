"""
Memory Manager for BetFinder AI Agents

Provides intelligent memory and personalization using mem0ai package.
Agents can store and retrieve user patterns, successful strategies, and learnings.

Key features:
- User-specific memory for personalized recommendations
- Agent-specific learnings and strategy adaptation
- Session-based insights and trend tracking
- Safe fallbacks when mem0ai is unavailable

Environment toggles:
- DISABLE_MEMORY=1 to bypass memory storage and use simple fallbacks
"""
from __future__ import annotations

import os
import json
from typing import Dict, List, Optional, Any
from datetime import datetime

DISABLE = os.getenv("DISABLE_MEMORY", "0") == "1"

try:
    from mem0 import Memory
    _HAS_MEM0 = True
except Exception as e:
    Memory = None  # type: ignore
    _HAS_MEM0 = False


class AgentMemoryManager:
    """Memory manager for agent learning and user personalization"""
    
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.memory = None
        self.fallback_storage = {}  # Simple dict fallback
        
        if not DISABLE and _HAS_MEM0:
            try:
                self.memory = Memory()
                self.use_mem0 = True
                print(f"[Memory] Initialized mem0ai for {agent_type}")
            except Exception as e:
                print(f"[Memory] Failed to initialize mem0ai: {e}")
                self.use_mem0 = False
        else:
            self.use_mem0 = False
            if DISABLE:
                print(f"[Memory] Disabled for {agent_type}")
    
    def add_pick_result(self, user_id: str, prop: Dict[str, Any], outcome: str, 
                       confidence: float, reasoning: str = "") -> bool:
        """Store a pick result for learning and personalization"""
        try:
            text = self._format_pick_memory(prop, outcome, confidence, reasoning)
            metadata = {
                "category": "pick_result",
                "agent": self.agent_type,
                "outcome": outcome,
                "sport": prop.get("sport", "unknown"),
                "stat_type": prop.get("stat_type", "unknown"),
                "confidence": confidence,
                "timestamp": datetime.now().isoformat()
            }
            
            return self._add_memory(text, user_id, metadata)
        except Exception as e:
            print(f"[Memory] Error adding pick result: {e}")
            return False
    
    def add_user_preference(self, user_id: str, preference_type: str, 
                           preference_value: str, context: str = "") -> bool:
        """Store user preference or feedback"""
        try:
            text = f"User preference: {preference_type} = {preference_value}"
            if context:
                text += f" (context: {context})"
            
            metadata = {
                "category": "user_preference",
                "agent": self.agent_type,
                "preference_type": preference_type,
                "preference_value": preference_value,
                "timestamp": datetime.now().isoformat()
            }
            
            return self._add_memory(text, user_id, metadata)
        except Exception as e:
            print(f"[Memory] Error adding user preference: {e}")
            return False
    
    def add_agent_learning(self, insight: str, context: Dict[str, Any]) -> bool:
        """Store agent-specific learning or strategy insight"""
        try:
            text = f"{self.agent_type} learning: {insight}"
            metadata = {
                "category": "agent_learning",
                "agent": self.agent_type,
                "insight_type": context.get("type", "general"),
                "context": context,
                "timestamp": datetime.now().isoformat()
            }
            
            return self._add_memory(text, "global", metadata)
        except Exception as e:
            print(f"[Memory] Error adding agent learning: {e}")
            return False
    
    def get_user_pick_history(self, user_id: str, sport: str = None, 
                             days_back: int = 30) -> List[Dict[str, Any]]:
        """Retrieve user's pick history for personalization"""
        try:
            query = f"{self.agent_type} pick"
            if sport:
                query += f" {sport}"
            
            results = self._search_memory(query, user_id, limit=50)
            
            # Filter by category and timeframe if using mem0
            filtered = []
            for result in results:
                metadata = result.get("metadata", {})
                if metadata.get("category") == "pick_result":
                    # Add basic time filtering logic here if needed
                    filtered.append(result)
            
            return filtered[:20]  # Return last 20 picks
        except Exception as e:
            print(f"[Memory] Error retrieving pick history: {e}")
            return []
    
    def get_user_preferences(self, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve user preferences and feedback"""
        try:
            results = self._search_memory("user preference", user_id, limit=20)
            return [r for r in results if r.get("metadata", {}).get("category") == "user_preference"]
        except Exception as e:
            print(f"[Memory] Error retrieving preferences: {e}")
            return []
    
    def get_agent_learnings(self, insight_type: str = None) -> List[Dict[str, Any]]:
        """Retrieve agent-specific learnings and strategies"""
        try:
            query = f"{self.agent_type} learning"
            if insight_type:
                query += f" {insight_type}"
            
            results = self._search_memory(query, "global", limit=30)
            filtered = [r for r in results if r.get("metadata", {}).get("category") == "agent_learning"]
            
            if insight_type:
                filtered = [r for r in filtered if r.get("metadata", {}).get("insight_type") == insight_type]
            
            return filtered
        except Exception as e:
            print(f"[Memory] Error retrieving agent learnings: {e}")
            return []
    
    def check_user_pattern(self, user_id: str, prop_type: str) -> Dict[str, Any]:
        """Check if user has patterns with specific prop types"""
        try:
            results = self._search_memory(f"{prop_type} pick", user_id, limit=10)
            
            if not results:
                return {"pattern": "none", "sample_size": 0}
            
            # Analyze outcomes
            outcomes = []
            for result in results:
                metadata = result.get("metadata", {})
                if metadata.get("category") == "pick_result":
                    outcomes.append(metadata.get("outcome"))
            
            if not outcomes:
                return {"pattern": "none", "sample_size": 0}
            
            win_rate = outcomes.count("won") / len(outcomes)
            
            if win_rate >= 0.7:
                pattern = "strong_positive"
            elif win_rate >= 0.6:
                pattern = "positive"
            elif win_rate <= 0.3:
                pattern = "negative"
            elif win_rate <= 0.4:
                pattern = "weak_negative"
            else:
                pattern = "neutral"
            
            return {
                "pattern": pattern,
                "win_rate": win_rate,
                "sample_size": len(outcomes),
                "recent_picks": outcomes[-5:]  # Last 5 outcomes
            }
            
        except Exception as e:
            print(f"[Memory] Error checking user pattern: {e}")
            return {"pattern": "unknown", "sample_size": 0}
    
    def _add_memory(self, text: str, user_id: str, metadata: Dict[str, Any]) -> bool:
        """Add memory using mem0ai or fallback storage"""
        if self.use_mem0 and self.memory:
            try:
                self.memory.add(text, user_id=user_id, metadata=metadata)
                return True
            except Exception as e:
                print(f"[Memory] mem0ai add failed: {e}")
                # Fall through to fallback
        
        # Fallback storage
        key = f"{user_id}_{self.agent_type}"
        if key not in self.fallback_storage:
            self.fallback_storage[key] = []
        
        self.fallback_storage[key].append({
            "text": text,
            "user_id": user_id,
            "metadata": metadata,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 100 entries per user-agent
        self.fallback_storage[key] = self.fallback_storage[key][-100:]
        return True
    
    def _search_memory(self, query: str, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search memory using mem0ai or fallback storage"""
        if self.use_mem0 and self.memory:
            try:
                results = self.memory.search(query, user_id=user_id)
                return results[:limit] if results else []
            except Exception as e:
                print(f"[Memory] mem0ai search failed: {e}")
                # Fall through to fallback
        
        # Fallback search
        key = f"{user_id}_{self.agent_type}"
        if key not in self.fallback_storage:
            return []
        
        # Simple text matching for fallback
        results = []
        query_words = query.lower().split()
        
        for entry in self.fallback_storage[key]:
            text = entry["text"].lower()
            if any(word in text for word in query_words):
                results.append(entry)
        
        return results[-limit:]  # Return most recent matches
    
    def _format_pick_memory(self, prop: Dict[str, Any], outcome: str, 
                           confidence: float, reasoning: str) -> str:
        """Format pick information for memory storage"""
        player = prop.get("player_name", "Unknown")
        stat = prop.get("stat_type", "unknown")
        line = prop.get("line", "?")
        ou = prop.get("over_under", "?")
        sport = prop.get("sport", "unknown")
        
        text = f"User {outcome} on {sport} pick: {ou.title()} {line} {stat} - {player}"
        if confidence:
            text += f" (confidence: {confidence:.0f}%)"
        if reasoning:
            text += f" - {reasoning}"
        
        return text


class GlobalMemoryManager:
    """Global memory manager that coordinates across all agents"""
    
    def __init__(self):
        self.agents = {}
        
    def get_agent_memory(self, agent_type: str) -> AgentMemoryManager:
        """Get or create memory manager for specific agent"""
        if agent_type not in self.agents:
            self.agents[agent_type] = AgentMemoryManager(agent_type)
        return self.agents[agent_type]
    
    def add_cross_agent_insight(self, insight: str, context: Dict[str, Any]) -> bool:
        """Add insight that applies to multiple agents"""
        try:
            if not DISABLE and _HAS_MEM0:
                memory = Memory()
                metadata = {
                    "category": "cross_agent_insight",
                    "context": context,
                    "timestamp": datetime.now().isoformat()
                }
                memory.add(f"Cross-agent insight: {insight}", user_id="global", metadata=metadata)
                return True
        except Exception as e:
            print(f"[Memory] Error adding cross-agent insight: {e}")
        return False
    
    def get_cross_agent_insights(self, topic: str = None) -> List[Dict[str, Any]]:
        """Retrieve insights that apply across agents"""
        try:
            if not DISABLE and _HAS_MEM0:
                memory = Memory()
                query = "cross-agent insight"
                if topic:
                    query += f" {topic}"
                
                results = memory.search(query, user_id="global")
                return [r for r in results if r.get("metadata", {}).get("category") == "cross_agent_insight"]
        except Exception as e:
            print(f"[Memory] Error retrieving cross-agent insights: {e}")
        return []


# Global instance for use across the application
global_memory = GlobalMemoryManager()