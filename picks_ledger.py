"""
Picks Ledger Module for BetFinder AI

This module provides persistent storage and analytics for all betting picks
made by sport agents, enabling performance tracking and machine learning
from historical results.
"""

import json
import os
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict
import statistics


class PicksLedger:
    """
    Centralized ledger for tracking all betting picks and their outcomes
    Provides analytics and learning capabilities for sport agents
    """
    
    def __init__(self, ledger_file: str = "picks_ledger.json"):
        """
        Initialize the picks ledger
        
        Args:
            ledger_file (str): Path to the JSON file storing picks data
        """
        self.ledger_file = ledger_file
        self.picks = []
        self.load_picks()
    
    def load_picks(self) -> None:
        """Load picks from the JSON file"""
        try:
            if os.path.exists(self.ledger_file):
                with open(self.ledger_file, 'r') as f:
                    data = json.load(f)
                    self.picks = data.get('picks', [])
                    print(f"Loaded {len(self.picks)} picks from {self.ledger_file}")
            else:
                self.picks = []
                print("No existing ledger file found. Starting fresh.")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading picks ledger: {e}")
            self.picks = []
    
    def save_picks(self) -> None:
        """Save picks to the JSON file"""
        try:
            data = {
                'picks': self.picks,
                'last_updated': datetime.now().isoformat(),
                'total_picks': len(self.picks)
            }
            
            # Create backup before saving
            if os.path.exists(self.ledger_file):
                backup_file = f"{self.ledger_file}.backup"
                with open(self.ledger_file, 'r') as src, open(backup_file, 'w') as dst:
                    dst.write(src.read())
            
            with open(self.ledger_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            print(f"Saved {len(self.picks)} picks to {self.ledger_file}")
        except IOError as e:
            print(f"Error saving picks ledger: {e}")
    
    def log_pick(self, pick_data: Dict[str, Any]) -> str:
        """
        Log a new pick to the ledger
        
        Args:
            pick_data (Dict): Pick information including all required fields
            
        Returns:
            str: Unique pick ID
        """
        # Generate unique pick ID
        pick_id = f"{pick_data.get('sport', 'unknown')}_{int(time.time() * 1000)}"
        
        # Create standardized pick entry
        pick_entry = {
            'pick_id': pick_id,
            'timestamp': datetime.now().isoformat(),
            'sport': pick_data.get('sport', 'unknown'),
            'agent_type': pick_data.get('agent_type', 'unknown'),
            'event_id': pick_data.get('game_id', pick_id),
            'event_start_time': pick_data.get('event_start_time', ''),
            'matchup': pick_data.get('matchup', 'TBD vs TBD'),
            'player_name': pick_data.get('player_name', 'Unknown'),
            'pick_type': pick_data.get('pick_type', 'player_prop'),
            'pick_description': pick_data.get('pick', ''),
            'over_under': pick_data.get('over_under', ''),
            'line': pick_data.get('line', 0),
            'stat_type': pick_data.get('stat_type', ''),
            'odds': pick_data.get('odds', -110),
            'confidence': pick_data.get('confidence', 0),
            'reasoning': pick_data.get('reasoning', ''),
            'detailed_reasoning': pick_data.get('detailed_reasoning', {}),
            'expected_value': pick_data.get('expected_value', 0),
            'sportsbook': pick_data.get('sportsbook', 'Multiple'),
            'outcome': 'pending',  # pending/won/lost/push/cancelled
            'actual_result': None,
            'profit_loss': None,
            'bet_amount': pick_data.get('bet_amount', 0),
            'updated_at': datetime.now().isoformat()
        }
        
        self.picks.append(pick_entry)
        self.save_picks()
        
        return pick_id
    
    def update_pick_outcome(self, pick_id: str, outcome: str, 
                           actual_result: Optional[float] = None,
                           profit_loss: Optional[float] = None) -> bool:
        """
        Update the outcome of a pick after the event completes
        
        Args:
            pick_id (str): Unique pick identifier
            outcome (str): Result - 'won', 'lost', 'push', or 'cancelled'
            actual_result (float, optional): Actual statistical result
            profit_loss (float, optional): Profit or loss amount
            
        Returns:
            bool: True if pick was found and updated, False otherwise
        """
        for pick in self.picks:
            if pick['pick_id'] == pick_id:
                pick['outcome'] = outcome
                pick['actual_result'] = actual_result
                pick['profit_loss'] = profit_loss
                pick['updated_at'] = datetime.now().isoformat()
                
                self.save_picks()
                return True
        
        print(f"Pick with ID {pick_id} not found")
        return False
    
    def get_picks_by_agent(self, agent_type: str, 
                          days_back: Optional[int] = None) -> List[Dict]:
        """
        Get all picks made by a specific agent
        
        Args:
            agent_type (str): Type of agent (e.g., 'basketball_agent')
            days_back (int, optional): Only include picks from last N days
            
        Returns:
            List[Dict]: List of picks made by the agent
        """
        agent_picks = [pick for pick in self.picks if pick['agent_type'] == agent_type]
        
        if days_back:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            agent_picks = [
                pick for pick in agent_picks 
                if datetime.fromisoformat(pick['timestamp']) >= cutoff_date
            ]
        
        return agent_picks
    
    def get_picks_by_sport(self, sport: str, 
                          days_back: Optional[int] = None) -> List[Dict]:
        """
        Get all picks for a specific sport
        
        Args:
            sport (str): Sport name
            days_back (int, optional): Only include picks from last N days
            
        Returns:
            List[Dict]: List of picks for the sport
        """
        sport_picks = [pick for pick in self.picks if pick['sport'] == sport]
        
        if days_back:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            sport_picks = [
                pick for pick in sport_picks 
                if datetime.fromisoformat(pick['timestamp']) >= cutoff_date
            ]
        
        return sport_picks
    
    def get_performance_metrics(self, agent_type: Optional[str] = None,
                               sport: Optional[str] = None,
                               days_back: Optional[int] = None) -> Dict[str, Any]:
        """
        Get comprehensive performance metrics
        
        Args:
            agent_type (str, optional): Filter by agent type
            sport (str, optional): Filter by sport
            days_back (int, optional): Only include picks from last N days
            
        Returns:
            Dict: Performance metrics and analytics
        """
        # Filter picks based on criteria
        filtered_picks = self.picks.copy()
        
        if agent_type:
            filtered_picks = [p for p in filtered_picks if p['agent_type'] == agent_type]
        
        if sport:
            filtered_picks = [p for p in filtered_picks if p['sport'] == sport]
        
        if days_back:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            filtered_picks = [
                p for p in filtered_picks 
                if datetime.fromisoformat(p['timestamp']) >= cutoff_date
            ]
        
        # Calculate metrics
        total_picks = len(filtered_picks)
        settled_picks = [p for p in filtered_picks if p['outcome'] in ['won', 'lost']]
        won_picks = [p for p in settled_picks if p['outcome'] == 'won']
        
        if not settled_picks:
            return {
                'total_picks': total_picks,
                'settled_picks': 0,
                'win_rate': 0,
                'average_confidence': 0,
                'profit_loss': 0,
                'roi': 0,
                'best_bet_types': [],
                'confidence_analysis': {},
                'recent_form': []
            }
        
        win_rate = len(won_picks) / len(settled_picks) * 100
        avg_confidence = statistics.mean([p['confidence'] for p in filtered_picks])
        total_profit_loss = sum([p.get('profit_loss', 0) for p in settled_picks if p.get('profit_loss')])
        
        # Calculate ROI (assuming standard bet amounts)
        total_wagered = len(settled_picks) * 100  # Assume $100 per bet
        roi = (total_profit_loss / total_wagered * 100) if total_wagered > 0 else 0
        
        # Analyze best performing bet types
        bet_type_performance = defaultdict(lambda: {'total': 0, 'won': 0})
        for pick in settled_picks:
            bet_key = f"{pick['stat_type']} {pick['over_under']}"
            bet_type_performance[bet_key]['total'] += 1
            if pick['outcome'] == 'won':
                bet_type_performance[bet_key]['won'] += 1
        
        best_bet_types = []
        for bet_type, stats in bet_type_performance.items():
            if stats['total'] >= 3:  # Minimum sample size
                win_rate_bt = stats['won'] / stats['total'] * 100
                best_bet_types.append({
                    'bet_type': bet_type,
                    'win_rate': win_rate_bt,
                    'sample_size': stats['total']
                })
        
        best_bet_types.sort(key=lambda x: x['win_rate'], reverse=True)
        
        # Confidence analysis
        confidence_buckets = {'60-70': [], '70-80': [], '80-90': [], '90-100': []}
        for pick in settled_picks:
            conf = pick['confidence']
            if 60 <= conf < 70:
                confidence_buckets['60-70'].append(pick['outcome'] == 'won')
            elif 70 <= conf < 80:
                confidence_buckets['70-80'].append(pick['outcome'] == 'won')
            elif 80 <= conf < 90:
                confidence_buckets['80-90'].append(pick['outcome'] == 'won')
            elif conf >= 90:
                confidence_buckets['90-100'].append(pick['outcome'] == 'won')
        
        confidence_analysis = {}
        for bucket, outcomes in confidence_buckets.items():
            if outcomes:
                confidence_analysis[bucket] = {
                    'win_rate': sum(outcomes) / len(outcomes) * 100,
                    'sample_size': len(outcomes)
                }
        
        # Recent form (last 10 settled picks)
        recent_picks = sorted(settled_picks, key=lambda x: x['timestamp'], reverse=True)[:10]
        recent_form = [p['outcome'] == 'won' for p in recent_picks]
        
        return {
            'total_picks': total_picks,
            'settled_picks': len(settled_picks),
            'pending_picks': total_picks - len(settled_picks),
            'win_rate': round(win_rate, 2),
            'average_confidence': round(avg_confidence, 2),
            'profit_loss': round(total_profit_loss, 2),
            'roi': round(roi, 2),
            'best_bet_types': best_bet_types[:5],
            'confidence_analysis': confidence_analysis,
            'recent_form': recent_form,
            'recent_form_percentage': round(sum(recent_form) / len(recent_form) * 100, 2) if recent_form else 0
        }
    
    def get_learning_insights(self, agent_type: str, min_sample_size: int = 5) -> Dict[str, Any]:
        """
        Get insights for machine learning and strategy adjustment
        
        Args:
            agent_type (str): Type of agent to analyze
            min_sample_size (int): Minimum picks needed for statistical significance
            
        Returns:
            Dict: Learning insights and recommendations
        """
        agent_picks = self.get_picks_by_agent(agent_type)
        settled_picks = [p for p in agent_picks if p['outcome'] in ['won', 'lost']]
        
        if len(settled_picks) < min_sample_size:
            return {
                'insufficient_data': True,
                'message': f"Need at least {min_sample_size} settled picks for analysis. Current: {len(settled_picks)}"
            }
        
        insights = {
            'insufficient_data': False,
            'optimal_confidence_threshold': None,
            'best_stat_types': [],
            'best_over_under_preference': None,
            'time_based_patterns': {},
            'sportsbook_performance': {},
            'recommendations': []
        }
        
        # Find optimal confidence threshold
        confidence_thresholds = [60, 65, 70, 75, 80, 85, 90]
        threshold_performance = {}
        
        for threshold in confidence_thresholds:
            high_conf_picks = [p for p in settled_picks if p['confidence'] >= threshold]
            if len(high_conf_picks) >= 3:  # Minimum sample
                win_rate = sum(1 for p in high_conf_picks if p['outcome'] == 'won') / len(high_conf_picks) * 100
                threshold_performance[threshold] = {
                    'win_rate': win_rate,
                    'sample_size': len(high_conf_picks)
                }
        
        if threshold_performance:
            best_threshold = max(threshold_performance.items(), 
                               key=lambda x: x[1]['win_rate'] if x[1]['sample_size'] >= 3 else 0)
            insights['optimal_confidence_threshold'] = {
                'threshold': best_threshold[0],
                'win_rate': best_threshold[1]['win_rate'],
                'sample_size': best_threshold[1]['sample_size']
            }
        
        # Analyze stat type performance
        stat_performance = defaultdict(lambda: {'total': 0, 'won': 0})
        for pick in settled_picks:
            stat_type = pick['stat_type']
            stat_performance[stat_type]['total'] += 1
            if pick['outcome'] == 'won':
                stat_performance[stat_type]['won'] += 1
        
        best_stats = []
        for stat, stats in stat_performance.items():
            if stats['total'] >= 3:
                win_rate = stats['won'] / stats['total'] * 100
                best_stats.append({
                    'stat_type': stat,
                    'win_rate': win_rate,
                    'sample_size': stats['total']
                })
        
        insights['best_stat_types'] = sorted(best_stats, key=lambda x: x['win_rate'], reverse=True)[:3]
        
        # Over/Under preference analysis
        over_picks = [p for p in settled_picks if p['over_under'] == 'over']
        under_picks = [p for p in settled_picks if p['over_under'] == 'under']
        
        if over_picks and under_picks:
            over_win_rate = sum(1 for p in over_picks if p['outcome'] == 'won') / len(over_picks) * 100
            under_win_rate = sum(1 for p in under_picks if p['outcome'] == 'won') / len(under_picks) * 100
            
            if abs(over_win_rate - under_win_rate) > 10:  # Significant difference
                insights['best_over_under_preference'] = {
                    'preference': 'over' if over_win_rate > under_win_rate else 'under',
                    'win_rate': max(over_win_rate, under_win_rate),
                    'over_sample': len(over_picks),
                    'under_sample': len(under_picks)
                }
        
        # Generate recommendations
        recommendations = []
        
        if insights['optimal_confidence_threshold']:
            threshold_data = insights['optimal_confidence_threshold']
            if threshold_data['win_rate'] > 60:
                recommendations.append(
                    f"Focus on picks with confidence >= {threshold_data['threshold']}% "
                    f"(Win rate: {threshold_data['win_rate']:.1f}%)"
                )
        
        if insights['best_stat_types']:
            best_stat = insights['best_stat_types'][0]
            if best_stat['win_rate'] > 60:
                recommendations.append(
                    f"Prioritize {best_stat['stat_type']} props "
                    f"(Win rate: {best_stat['win_rate']:.1f}%)"
                )
        
        if insights['best_over_under_preference']:
            pref_data = insights['best_over_under_preference']
            recommendations.append(
                f"Show preference for {pref_data['preference']} bets "
                f"(Win rate: {pref_data['win_rate']:.1f}%)"
            )
        
        insights['recommendations'] = recommendations
        
        return insights
    
    def export_data(self, format_type: str = 'json', 
                   agent_type: Optional[str] = None) -> str:
        """
        Export picks data in various formats
        
        Args:
            format_type (str): Export format ('json', 'csv')
            agent_type (str, optional): Filter by agent type
            
        Returns:
            str: Filename of exported data
        """
        picks_to_export = self.picks
        
        if agent_type:
            picks_to_export = [p for p in picks_to_export if p['agent_type'] == agent_type]
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format_type == 'json':
            filename = f"picks_export_{agent_type or 'all'}_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(picks_to_export, f, indent=2, default=str)
        
        elif format_type == 'csv':
            import csv
            filename = f"picks_export_{agent_type or 'all'}_{timestamp}.csv"
            
            if picks_to_export:
                with open(filename, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=picks_to_export[0].keys())
                    writer.writeheader()
                    writer.writerows(picks_to_export)
        
        return filename
    
    def cleanup_old_picks(self, days_to_keep: int = 90) -> int:
        """
        Remove picks older than specified days to manage file size
        
        Args:
            days_to_keep (int): Number of days of picks to retain
            
        Returns:
            int: Number of picks removed
        """
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)
        
        old_count = len(self.picks)
        self.picks = [
            pick for pick in self.picks 
            if datetime.fromisoformat(pick['timestamp']) >= cutoff_date
        ]
        new_count = len(self.picks)
        
        removed_count = old_count - new_count
        
        if removed_count > 0:
            self.save_picks()
            print(f"Removed {removed_count} old picks, keeping {new_count} picks")
        
        return removed_count


# Singleton instance for global access
picks_ledger = PicksLedger()