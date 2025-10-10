"""
Sport Agents Module for BetFinder AI

This module contains the base SportAgent class and individual sport-specific agents
for analyzing props and making picks with confidence scores. Includes comprehensive
PicksLedger integration for performance tracking and machine learning from historical results.
"""

import time
import random
import statistics
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from abc import ABC, abstractmethod
from collections import defaultdict, Counter

# Import the picks ledger for logging and analytics
from picks_ledger import picks_ledger


class SportAgent(ABC):
    """
    Base class for all sport-specific agents with PicksLedger integration
    and machine learning capabilities from historical performance
    """
    
    def __init__(self, sport_name: str):
        """
        Initialize the sport agent
        
        Args:
            sport_name (str): Name of the sport this agent handles
        """
        self.sport_name = sport_name
        self.agent_type = f"{sport_name}_agent"
        self.props_data = []
        self.picks = []
        self.learning_insights = {}
        self.performance_metrics = {}
        
        # Load historical insights on initialization
        self.learn_from_history()
    
    def fetch_props(self, max_props: int = 50) -> List[Dict]:
        """
        Fetch props data for the sport
        
        Args:
            max_props (int): Maximum number of props to fetch
            
        Returns:
            List[Dict]: List of props data
        """
        # This would typically fetch from an API or data source
        # For now, we'll generate mock data specific to each sport
        return self._generate_mock_props(max_props)
    
    @abstractmethod
    def _generate_mock_props(self, max_props: int) -> List[Dict]:
        """
        Generate mock props data specific to the sport
        Must be implemented by each sport agent
        """
        pass
    
    def make_picks(self, props_data: Optional[List[Dict]] = None) -> List[Dict]:
        """
        Analyze props and make over/under picks with confidence scores
        Includes detailed multi-factor reasoning and PicksLedger integration
        
        Args:
            props_data (List[Dict], optional): Props data to analyze
            
        Returns:
            List[Dict]: List of picks with confidence scores
        """
        if props_data is None:
            props_data = self.fetch_props()
        
        self.props_data = props_data
        picks = []
        
        # Update learning insights before making picks
        self.learn_from_history()
        
        for prop in props_data:
            pick = self._analyze_prop_and_make_pick(prop)
            if pick:
                # Log pick to ledger and get pick_id
                pick_id = picks_ledger.log_pick(pick)
                pick['pick_id'] = pick_id
                picks.append(pick)
        
        # Sort picks by time and confidence
        sorted_picks = self.sort_picks_by_time_and_confidence(picks)
        self.picks = sorted_picks
        
        return sorted_picks
    
    def _analyze_prop_and_make_pick(self, prop: Dict) -> Optional[Dict]:
        """
        Analyze a single prop and make a pick decision with detailed reasoning
        
        Args:
            prop (Dict): Individual prop data
            
        Returns:
            Dict: Pick data with confidence score and detailed reasoning, or None if no pick
        """
        # Extract basic prop information
        line = prop.get('line', 0)
        player_name = prop.get('player_name', 'Unknown')
        stat_type = prop.get('stat_type', 'Unknown')
        odds = prop.get('odds', -110)
        
        # Perform comprehensive analysis
        analysis_factors = self._perform_detailed_analysis(prop)
        
        # Calculate confidence based on multiple factors
        confidence = self._calculate_enhanced_confidence(prop, analysis_factors)
        
        # Apply learning insights to adjust confidence and decision
        confidence = self._apply_learning_insights(prop, confidence, analysis_factors)
        
        # Only make picks that meet confidence threshold
        min_confidence = self._get_minimum_confidence_threshold()
        if confidence < min_confidence:
            return None
        
        # Determine over/under based on analysis
        over_under = self._determine_over_under_with_insights(prop, confidence, analysis_factors)
        
        # Generate detailed reasoning
        detailed_reasoning = self._generate_detailed_reasoning(prop, analysis_factors, over_under, confidence)
        
        # Calculate expected value
        expected_value = self._calculate_expected_value(confidence, odds)
        
        pick = {
            'game_id': prop.get('game_id', f"{self.sport_name}_{time.time()}"),
            'sport': self.sport_name,
            'agent_type': self.agent_type,
            'player_name': player_name,
            'pick': f"{player_name} {over_under.title()} {line} {stat_type}",
            'pick_type': 'player_prop',
            'over_under': over_under,
            'line': line,
            'stat_type': stat_type,
            'confidence': round(confidence, 1),
            'odds': odds,
            'event_start_time': prop.get('event_start_time', ''),
            'matchup': prop.get('matchup', 'TBD vs TBD'),
            'sportsbook': prop.get('sportsbook', 'Multiple'),
            'reasoning': detailed_reasoning['summary'],
            'detailed_reasoning': detailed_reasoning,
            'expected_value': round(expected_value, 2),
            'analysis_factors': analysis_factors,
            'bet_amount': self._calculate_bet_size(confidence, expected_value)
        }
        
        return pick
    
    def _perform_detailed_analysis(self, prop: Dict) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of a prop using multiple factors
        
        Args:
            prop (Dict): Prop data to analyze
            
        Returns:
            Dict: Analysis factors and scores
        """
        factors = {
            'player_form': self._analyze_player_form(prop),
            'matchup_analysis': self._analyze_matchup(prop),
            'injury_impact': self._analyze_injury_impact(prop),
            'historical_performance': self._analyze_historical_performance(prop),
            'situational_factors': self._analyze_situational_factors(prop),
            'line_value': self._analyze_line_value(prop),
            'weather_conditions': self._analyze_weather_conditions(prop),
            'team_dynamics': self._analyze_team_dynamics(prop)
        }
        
        # Calculate overall factor score
        factor_scores = [f['score'] for f in factors.values() if 'score' in f]
        factors['overall_score'] = statistics.mean(factor_scores) if factor_scores else 5.0
        
        return factors
    
    def _analyze_player_form(self, prop: Dict) -> Dict[str, Any]:
        """Analyze player's recent form and performance trends"""
        recent_form = prop.get('recent_form', random.uniform(5, 9))
        
        # Simulate form analysis
        form_trend = random.choice(['improving', 'declining', 'stable'])
        consistency = random.uniform(0.6, 0.95)
        
        score = recent_form
        if form_trend == 'improving':
            score += 0.5
        elif form_trend == 'declining':
            score -= 0.5
        
        score += (consistency - 0.75) * 2  # Bonus for consistency
        
        return {
            'recent_form': recent_form,
            'form_trend': form_trend,
            'consistency': consistency,
            'score': max(1, min(10, score)),
            'reasoning': f"Player showing {form_trend} form with {consistency:.1%} consistency"
        }
    
    def _analyze_matchup(self, prop: Dict) -> Dict[str, Any]:
        """Analyze matchup difficulty and opponent strength"""
        matchup_difficulty = prop.get('matchup_difficulty', random.uniform(3, 8))
        
        # Simulate matchup analysis
        opponent_defense_rank = random.randint(1, 32)
        historical_h2h = random.choice(['favorable', 'neutral', 'unfavorable'])
        pace_factor = random.uniform(0.8, 1.2)
        
        # Calculate matchup score (lower difficulty = higher score)
        score = 10 - matchup_difficulty
        
        if historical_h2h == 'favorable':
            score += 1
        elif historical_h2h == 'unfavorable':
            score -= 1
        
        score *= pace_factor
        
        return {
            'difficulty': matchup_difficulty,
            'opponent_defense_rank': opponent_defense_rank,
            'historical_h2h': historical_h2h,
            'pace_factor': pace_factor,
            'score': max(1, min(10, score)),
            'reasoning': f"Matchup difficulty {matchup_difficulty}/10, {historical_h2h} H2H history"
        }
    
    def _analyze_injury_impact(self, prop: Dict) -> Dict[str, Any]:
        """Analyze injury status and impact on performance"""
        injury_status = prop.get('injury_status', 'healthy')
        
        status_scores = {
            'healthy': 10,
            'probable': 8,
            'questionable': 6,
            'doubtful': 3,
            'out': 0,
            'day-to-day': 7,
            'minor knock': 8,
            'upper-body': 6,
            'lower-body': 5
        }
        
        score = status_scores.get(injury_status, 7)
        
        # Factor in injury impact on specific stat types
        if injury_status in ['lower-body', 'questionable'] and prop.get('stat_type') in ['rushing_yards', 'steals']:
            score -= 2
        
        return {
            'status': injury_status,
            'score': max(0, min(10, score)),
            'reasoning': f"Injury status: {injury_status}"
        }
    
    def _analyze_historical_performance(self, prop: Dict) -> Dict[str, Any]:
        """Analyze historical performance against the line"""
        stat_type = prop.get('stat_type', '')
        line = prop.get('line', 0)
        
        # Simulate historical analysis
        season_average = line * random.uniform(0.85, 1.15)
        l10_average = line * random.uniform(0.8, 1.2)
        vs_opponent_average = line * random.uniform(0.75, 1.25)
        
        # Calculate how often player hits over the line
        over_rate = random.uniform(0.3, 0.7)
        
        score = 5.0  # Base score
        
        # Adjust based on averages vs line
        if season_average > line:
            score += (season_average - line) / line * 3
        else:
            score -= (line - season_average) / line * 3
        
        if l10_average > line:
            score += 1
        else:
            score -= 1
        
        return {
            'season_average': round(season_average, 1),
            'l10_average': round(l10_average, 1),
            'vs_opponent_average': round(vs_opponent_average, 1),
            'over_rate': round(over_rate, 2),
            'score': max(1, min(10, score)),
            'reasoning': f"Season avg: {season_average:.1f}, L10: {l10_average:.1f} vs line {line}"
        }
    
    def _analyze_situational_factors(self, prop: Dict) -> Dict[str, Any]:
        """Analyze situational factors like game importance, venue, etc."""
        # Simulate situational analysis
        game_importance = random.choice(['low', 'medium', 'high', 'playoff'])
        venue = random.choice(['home', 'away', 'neutral'])
        rest_days = random.randint(0, 5)
        motivation_level = random.uniform(0.7, 1.0)
        
        score = 5.0  # Base score
        
        # Game importance factor
        importance_scores = {'low': -0.5, 'medium': 0, 'high': 1, 'playoff': 2}
        score += importance_scores[game_importance]
        
        # Venue factor
        if venue == 'home':
            score += 0.5
        elif venue == 'away':
            score -= 0.3
        
        # Rest factor
        if rest_days == 0:  # Back-to-back
            score -= 1
        elif rest_days >= 3:
            score += 0.5
        
        score += (motivation_level - 0.85) * 3
        
        return {
            'game_importance': game_importance,
            'venue': venue,
            'rest_days': rest_days,
            'motivation_level': motivation_level,
            'score': max(1, min(10, score)),
            'reasoning': f"{game_importance.title()} importance game, {venue} venue, {rest_days} rest days"
        }
    
    def _analyze_line_value(self, prop: Dict) -> Dict[str, Any]:
        """Analyze if the line offers good value"""
        line = prop.get('line', 0)
        odds = prop.get('odds', -110)
        
        # Calculate implied probability from odds
        if odds > 0:
            implied_prob = 100 / (odds + 100)
        else:
            implied_prob = abs(odds) / (abs(odds) + 100)
        
        # Simulate "true" probability estimation
        true_prob = random.uniform(0.35, 0.65)
        
        # Calculate edge
        edge = true_prob - implied_prob
        
        score = 5.0 + (edge * 20)  # Convert edge to score
        
        return {
            'line': line,
            'odds': odds,
            'implied_probability': round(implied_prob, 3),
            'estimated_true_probability': round(true_prob, 3),
            'edge': round(edge, 3),
            'score': max(1, min(10, score)),
            'reasoning': f"Edge: {edge:.1%} (True: {true_prob:.1%} vs Implied: {implied_prob:.1%})"
        }
    
    def _analyze_weather_conditions(self, prop: Dict) -> Dict[str, Any]:
        """Analyze weather impact (mainly for outdoor sports)"""
        # For indoor sports, return neutral
        if self.sport_name in ['basketball', 'hockey', 'esports']:
            return {
                'conditions': 'indoor',
                'score': 5.0,
                'reasoning': 'Indoor sport - weather not applicable'
            }
        
        # Simulate weather conditions
        conditions = random.choice(['clear', 'cloudy', 'light_rain', 'heavy_rain', 'windy', 'cold'])
        
        condition_scores = {
            'clear': 6,
            'cloudy': 5,
            'light_rain': 4,
            'heavy_rain': 3,
            'windy': 4,
            'cold': 4
        }
        
        score = condition_scores.get(conditions, 5)
        
        # Weather can affect different stats differently
        stat_type = prop.get('stat_type', '')
        if conditions in ['heavy_rain', 'windy'] and stat_type in ['passing_yards', 'receiving_yards']:
            score -= 1
        
        return {
            'conditions': conditions,
            'score': score,
            'reasoning': f"Weather: {conditions.replace('_', ' ')}"
        }
    
    def _analyze_team_dynamics(self, prop: Dict) -> Dict[str, Any]:
        """Analyze team dynamics and coaching factors"""
        # Simulate team analysis
        team_form = random.uniform(4, 9)
        coaching_advantage = random.choice(['strong', 'neutral', 'weak'])
        offensive_efficiency = random.uniform(0.8, 1.2)
        
        score = team_form * 0.7
        
        coaching_scores = {'strong': 1, 'neutral': 0, 'weak': -1}
        score += coaching_scores[coaching_advantage]
        
        score += (offensive_efficiency - 1) * 3
        
        return {
            'team_form': team_form,
            'coaching_advantage': coaching_advantage,
            'offensive_efficiency': offensive_efficiency,
            'score': max(1, min(10, score)),
            'reasoning': f"Team form {team_form:.1f}/10, {coaching_advantage} coaching"
        }
    
    def _calculate_enhanced_confidence(self, prop: Dict, analysis_factors: Dict) -> float:
        """
        Calculate confidence score based on comprehensive analysis
        
        Args:
            prop (Dict): Prop data
            analysis_factors (Dict): Detailed analysis factors
            
        Returns:
            float: Confidence score (0-100)
        """
        # Base confidence from overall factor score
        base_confidence = analysis_factors['overall_score'] * 10
        
        # Apply weights to different factors
        factor_weights = {
            'player_form': 0.25,
            'matchup_analysis': 0.20,
            'historical_performance': 0.20,
            'injury_impact': 0.15,
            'line_value': 0.10,
            'situational_factors': 0.05,
            'weather_conditions': 0.03,
            'team_dynamics': 0.02
        }
        
        weighted_score = 0
        for factor_name, weight in factor_weights.items():
            if factor_name in analysis_factors:
                weighted_score += analysis_factors[factor_name]['score'] * weight
        
        confidence = weighted_score * 10  # Convert to 0-100 scale
        
        # Add some randomness for realism
        confidence += random.uniform(-5, 5)
        
        # Ensure confidence is within bounds
        return max(30, min(95, confidence))
    
    def _apply_learning_insights(self, prop: Dict, confidence: float, analysis_factors: Dict) -> float:
        """
        Apply historical learning insights to adjust confidence
        
        Args:
            prop (Dict): Prop data
            confidence (float): Base confidence score
            analysis_factors (Dict): Analysis factors
            
        Returns:
            float: Adjusted confidence score
        """
        if not self.learning_insights or self.learning_insights.get('insufficient_data'):
            return confidence
        
        adjusted_confidence = confidence
        
        # Apply optimal confidence threshold learning
        optimal_threshold = self.learning_insights.get('optimal_confidence_threshold')
        if optimal_threshold and confidence >= optimal_threshold['threshold']:
            # Boost confidence for picks that meet the learned optimal threshold
            adjusted_confidence *= 1.05
        
        # Apply stat type preferences
        best_stats = self.learning_insights.get('best_stat_types', [])
        stat_type = prop.get('stat_type', '')
        
        for stat_info in best_stats:
            if stat_info['stat_type'] == stat_type and stat_info['win_rate'] > 65:
                adjusted_confidence *= 1.03
                break
        
        # Apply over/under preferences
        over_under_pref = self.learning_insights.get('best_over_under_preference')
        if over_under_pref:
            prop_over_under = self._determine_over_under_with_insights(prop, confidence, analysis_factors)
            if prop_over_under == over_under_pref['preference']:
                adjusted_confidence *= 1.02
        
        return max(30, min(95, adjusted_confidence))
    
    def _get_minimum_confidence_threshold(self) -> float:
        """Get the minimum confidence threshold for making picks"""
        # Use learned optimal threshold if available
        if (self.learning_insights and 
            not self.learning_insights.get('insufficient_data') and
            self.learning_insights.get('optimal_confidence_threshold')):
            return self.learning_insights['optimal_confidence_threshold']['threshold']
        
        # Default threshold
        return 60.0
    
    def _determine_over_under_with_insights(self, prop: Dict, confidence: float, analysis_factors: Dict) -> str:
        """
        Determine whether to pick over or under using analysis and insights
        """
        # Check if we have a learned preference
        over_under_pref = self.learning_insights.get('best_over_under_preference')
        
        # Analyze factors that suggest over or under
        over_factors = 0
        under_factors = 0
        
        # Player form analysis
        player_form = analysis_factors.get('player_form', {})
        if player_form.get('form_trend') == 'improving':
            over_factors += 1
        elif player_form.get('form_trend') == 'declining':
            under_factors += 1
        
        # Historical performance
        hist_perf = analysis_factors.get('historical_performance', {})
        season_avg = hist_perf.get('season_average', 0)
        line = prop.get('line', 0)
        
        if season_avg > line:
            over_factors += 1
        else:
            under_factors += 1
        
        # Matchup analysis
        matchup = analysis_factors.get('matchup_analysis', {})
        if matchup.get('difficulty', 5) < 5:  # Easy matchup
            over_factors += 1
        else:
            under_factors += 1
        
        # Line value analysis
        line_value = analysis_factors.get('line_value', {})
        edge = line_value.get('edge', 0)
        true_prob = line_value.get('estimated_true_probability', 0.5)
        
        if true_prob > 0.5:
            over_factors += 1
        else:
            under_factors += 1
        
        # Make decision based on factors and insights
        if over_factors > under_factors:
            base_decision = 'over'
        elif under_factors > over_factors:
            base_decision = 'under'
        else:
            # Tie - use learned preference or random
            if over_under_pref and over_under_pref['win_rate'] > 60:
                base_decision = over_under_pref['preference']
            else:
                base_decision = random.choice(['over', 'under'])
        
        return base_decision
    
    def _generate_detailed_reasoning(self, prop: Dict, analysis_factors: Dict, 
                                   over_under: str, confidence: float) -> Dict[str, Any]:
        """
        Generate comprehensive reasoning for the pick
        """
        player = prop.get('player_name', 'Player')
        stat = prop.get('stat_type', 'stat')
        line = prop.get('line', 0)
        
        # Extract key insights from analysis
        player_form = analysis_factors.get('player_form', {})
        matchup = analysis_factors.get('matchup_analysis', {})
        hist_perf = analysis_factors.get('historical_performance', {})
        injury = analysis_factors.get('injury_impact', {})
        line_value = analysis_factors.get('line_value', {})
        
        # Build detailed reasoning
        summary_parts = []
        
        # Main pick reasoning
        if over_under == 'over':
            summary_parts.append(f"{player} to exceed {line} {stat}")
        else:
            summary_parts.append(f"{player} to stay under {line} {stat}")
        
        # Key supporting factors
        if player_form.get('score', 5) > 6:
            summary_parts.append(f"Strong form ({player_form.get('form_trend', 'stable')})")
        
        if matchup.get('score', 5) > 6:
            summary_parts.append("Favorable matchup")
        elif matchup.get('score', 5) < 4:
            summary_parts.append("Challenging matchup supports under")
        
        if hist_perf.get('score', 5) > 6:
            summary_parts.append("Historical trends support pick")
        
        if line_value.get('edge', 0) > 0.05:
            summary_parts.append("Positive expected value")
        
        # Confidence qualifier
        if confidence > 80:
            confidence_desc = "High confidence"
        elif confidence > 70:
            confidence_desc = "Good confidence"
        else:
            confidence_desc = "Moderate confidence"
        
        summary = f"{summary_parts[0]}. {'. '.join(summary_parts[1:])}. {confidence_desc} pick."
        
        return {
            'summary': summary,
            'key_factors': summary_parts[1:] if len(summary_parts) > 1 else [],
            'confidence_level': confidence_desc,
            'analysis_breakdown': {
                'player_form': player_form.get('reasoning', ''),
                'matchup': matchup.get('reasoning', ''),
                'historical': hist_perf.get('reasoning', ''),
                'injury_status': injury.get('reasoning', ''),
                'line_value': line_value.get('reasoning', ''),
                'weather': analysis_factors.get('weather_conditions', {}).get('reasoning', ''),
                'team_dynamics': analysis_factors.get('team_dynamics', {}).get('reasoning', '')
            },
            'decision_factors': {
                'over_indicators': [],
                'under_indicators': [],
                'neutral_factors': []
            }
        }
    
    def _calculate_expected_value(self, confidence: float, odds: int) -> float:
        """Calculate expected value of the bet"""
        # Convert confidence to win probability
        win_prob = confidence / 100
        
        # Convert odds to decimal
        if odds > 0:
            decimal_odds = (odds / 100) + 1
        else:
            decimal_odds = (100 / abs(odds)) + 1
        
        # Calculate expected value
        expected_value = (win_prob * decimal_odds) - 1
        
        return expected_value
    
    def _calculate_bet_size(self, confidence: float, expected_value: float) -> float:
        """Calculate optimal bet size based on confidence and EV"""
        # Simple Kelly Criterion approximation
        base_bet = 100  # Base bet amount
        
        if expected_value <= 0:
            return 0
        
        # Kelly fraction (simplified)
        kelly_fraction = expected_value / 4  # Conservative approach
        
        # Apply confidence modifier
        confidence_modifier = confidence / 100
        
        bet_size = base_bet * kelly_fraction * confidence_modifier
        
        # Cap bet size for risk management
        return min(bet_size, base_bet * 2)
    
    def learn_from_history(self) -> None:
        """
        Analyze historical picks to identify winning patterns and adjust strategy
        """
        try:
            # Get learning insights from the picks ledger
            insights = picks_ledger.get_learning_insights(self.agent_type, min_sample_size=5)
            self.learning_insights = insights
            
            # Get performance metrics
            metrics = picks_ledger.get_performance_metrics(agent_type=self.agent_type, days_back=30)
            self.performance_metrics = metrics
            
            # Log insights for debugging
            if not insights.get('insufficient_data'):
                print(f"\n=== Learning Insights for {self.agent_type} ===")
                
                if insights.get('optimal_confidence_threshold'):
                    threshold_data = insights['optimal_confidence_threshold']
                    print(f"Optimal confidence threshold: {threshold_data['threshold']}% "
                          f"(Win rate: {threshold_data['win_rate']:.1f}%)")
                
                if insights.get('best_stat_types'):
                    print("Best performing stat types:")
                    for stat in insights['best_stat_types'][:3]:
                        print(f"  - {stat['stat_type']}: {stat['win_rate']:.1f}% "
                              f"({stat['sample_size']} picks)")
                
                if insights.get('recommendations'):
                    print("Recommendations:")
                    for rec in insights['recommendations']:
                        print(f"  - {rec}")
        
        except Exception as e:
            print(f"Error in learn_from_history for {self.agent_type}: {e}")
            self.learning_insights = {'insufficient_data': True}
            self.performance_metrics = {}
    
    def update_pick_outcome(self, pick_id: str, outcome: str, 
                           actual_result: Optional[float] = None) -> bool:
        """
        Update the outcome of a pick after the event completes
        
        Args:
            pick_id (str): Unique pick identifier
            outcome (str): Result - 'won', 'lost', 'push', or 'cancelled'
            actual_result (float, optional): Actual statistical result
            
        Returns:
            bool: True if pick was found and updated, False otherwise
        """
        # Calculate profit/loss if outcome is provided
        profit_loss = None
        if outcome in ['won', 'lost']:
            # Find the pick to get bet amount and odds
            for pick in self.picks:
                if pick.get('pick_id') == pick_id:
                    bet_amount = pick.get('bet_amount', 100)
                    odds = pick.get('odds', -110)
                    
                    if outcome == 'won':
                        if odds > 0:
                            profit_loss = bet_amount * (odds / 100)
                        else:
                            profit_loss = bet_amount * (100 / abs(odds))
                    else:  # lost
                        profit_loss = -bet_amount
                    break
        
        # Update in the ledger
        success = picks_ledger.update_pick_outcome(pick_id, outcome, actual_result, profit_loss)
        
        if success:
            # Trigger re-learning after outcome update
            self.learn_from_history()
        
        return success
    
    def sort_picks_by_time_and_confidence(self, picks: List[Dict]) -> List[Dict]:
        """
        Sort picks by event start time (ascending) then by confidence (descending)
        
        Args:
            picks (List[Dict]): List of picks to sort
            
        Returns:
            List[Dict]: Sorted picks
        """
        def sort_key(pick):
            # Get event time - handle various time formats
            event_time = pick.get('event_start_time', '')
            
            # Convert to sortable format
            if isinstance(event_time, str) and event_time:
                try:
                    if 'T' in event_time or ' ' in event_time:
                        # Full datetime string
                        parsed_time = datetime.fromisoformat(event_time.replace('Z', '+00:00'))
                        time_sort_key = parsed_time.timestamp()
                    elif ':' in event_time:
                        # Just time, assume today
                        time_part = f"{datetime.now().strftime('%Y-%m-%d')} {event_time}"
                        parsed_time = datetime.fromisoformat(time_part)
                        time_sort_key = parsed_time.timestamp()
                    else:
                        # Invalid format, use current time
                        time_sort_key = time.time()
                except (ValueError, TypeError):
                    time_sort_key = time.time()
            else:
                time_sort_key = time.time()
            
            # Get confidence (higher confidence = lower sort value for descending order)
            confidence = pick.get('confidence', 0)
            confidence_sort_key = -confidence  # Negative for descending order
            
            return (time_sort_key, confidence_sort_key)
        
        try:
            return sorted(picks, key=sort_key)
        except Exception:
            # If sorting fails, return original list
            return picks
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of this agent's performance
        
        Returns:
            Dict: Performance summary including metrics and insights
        """
        return {
            'agent_type': self.agent_type,
            'sport': self.sport_name,
            'performance_metrics': self.performance_metrics,
            'learning_insights': self.learning_insights,
            'recent_picks_count': len(self.picks),
            'has_sufficient_data': not self.learning_insights.get('insufficient_data', True)
        }


class TennisAgent(SportAgent):
    """Tennis-specific agent for analyzing tennis props with enhanced analytics"""
    
    def __init__(self):
        super().__init__("tennis")
        self.sport_specific_factors = {
            'surface_preferences': {},  # Player preferences for different surfaces
            'stamina_factors': {},      # Player endurance in longer matches
            'mental_toughness': {}      # Clutch performance metrics
        }
    
    def _generate_mock_props(self, max_props: int) -> List[Dict]:
        """Generate mock tennis props with enhanced data"""
        players = [
            "Novak Djokovic", "Carlos Alcaraz", "Daniil Medvedev", "Jannik Sinner",
            "Stefanos Tsitsipas", "Andrey Rublev", "Alexander Zverev", "Holger Rune"
        ]
        
        stat_types = ["games_won", "sets_won", "aces", "double_faults", "break_points_converted"]
        surfaces = ["hard", "clay", "grass"]
        
        props = []
        
        for i in range(min(max_props, 20)):
            player = random.choice(players)
            stat_type = random.choice(stat_types)
            surface = random.choice(surfaces)
            
            # Tennis-specific line ranges
            if stat_type == "games_won":
                line = random.uniform(8.5, 15.5)
            elif stat_type == "sets_won":
                line = random.choice([1.5, 2.5])
            elif stat_type == "aces":
                line = random.uniform(3.5, 12.5)
            elif stat_type == "double_faults":
                line = random.uniform(1.5, 4.5)
            else:  # break_points_converted
                line = random.uniform(1.5, 5.5)
            
            # Tennis-specific enhancements
            prop = {
                'game_id': f"tennis_{i}",
                'player_name': player,
                'stat_type': stat_type,
                'line': round(line, 1),
                'odds': random.choice([-110, -105, -115, +100, +105]),
                'event_start_time': (datetime.now() + timedelta(hours=random.randint(1, 48))).isoformat(),
                'matchup': f"{player} vs {random.choice(players)}",
                'sportsbook': random.choice(['DraftKings', 'FanDuel', 'BetMGM']),
                'recent_form': random.uniform(5, 9),
                'matchup_difficulty': random.uniform(3, 8),
                'injury_status': random.choice(['healthy', 'minor knock']),
                # Tennis-specific factors
                'surface': surface,
                'tournament_round': random.choice(['R1', 'R2', 'R3', 'QF', 'SF', 'F']),
                'match_format': random.choice(['best_of_3', 'best_of_5']),
                'head_to_head_record': f"{random.randint(0, 5)}-{random.randint(0, 5)}",
                'surface_win_rate': random.uniform(0.5, 0.8),
                'serve_stats': {
                    'first_serve_percentage': random.uniform(0.55, 0.75),
                    'ace_rate': random.uniform(0.05, 0.15)
                }
            }
            
            props.append(prop)
        
        return props
    
    def _analyze_tennis_specific_factors(self, prop: Dict) -> Dict[str, Any]:
        """Analyze tennis-specific factors"""
        surface = prop.get('surface', 'hard')
        tournament_round = prop.get('tournament_round', 'R1')
        match_format = prop.get('match_format', 'best_of_3')
        surface_win_rate = prop.get('surface_win_rate', 0.6)
        
        score = 5.0
        
        # Surface preference analysis
        if surface_win_rate > 0.7:
            score += 2
        elif surface_win_rate < 0.5:
            score -= 2
        
        # Tournament round pressure
        round_pressure = {'R1': 0, 'R2': 0.5, 'R3': 1, 'QF': 1.5, 'SF': 2, 'F': 2.5}
        pressure_adjustment = round_pressure.get(tournament_round, 0)
        
        # Different players handle pressure differently
        if random.random() > 0.5:  # Simulate clutch player
            score += pressure_adjustment * 0.5
        else:
            score -= pressure_adjustment * 0.3
        
        # Match format consideration for stamina-related stats
        stat_type = prop.get('stat_type', '')
        if match_format == 'best_of_5' and stat_type in ['games_won', 'sets_won']:
            score += 0.5  # Longer matches favor certain stats
        
        return {
            'surface_preference': surface_win_rate,
            'tournament_pressure': pressure_adjustment,
            'match_format_factor': match_format,
            'score': max(1, min(10, score)),
            'reasoning': f"Surface win rate: {surface_win_rate:.1%}, {tournament_round} round pressure"
        }
    
    def _perform_detailed_analysis(self, prop: Dict) -> Dict[str, Any]:
        """Enhanced analysis including tennis-specific factors"""
        # Get base analysis
        factors = super()._perform_detailed_analysis(prop)
        
        # Add tennis-specific analysis
        factors['tennis_specific'] = self._analyze_tennis_specific_factors(prop)
        
        # Recalculate overall score including tennis factors
        factor_scores = [f['score'] for f in factors.values() if isinstance(f, dict) and 'score' in f]
        factors['overall_score'] = statistics.mean(factor_scores) if factor_scores else 5.0
        
        return factors


class BasketballAgent(SportAgent):
    """Basketball-specific agent for analyzing basketball props with advanced NBA analytics"""
    
    def __init__(self):
        super().__init__("basketball")
        self.sport_specific_factors = {
            'pace_factors': {},        # Team pace impact on player stats
            'usage_rates': {},         # Player usage rate in different situations
            'matchup_advantages': {}   # Positional matchup advantages
        }
    
    def _generate_mock_props(self, max_props: int) -> List[Dict]:
        """Generate mock basketball props with advanced NBA analytics"""
        players = [
            "LeBron James", "Stephen Curry", "Giannis Antetokounmpo", "Luka Doncic",
            "Jayson Tatum", "Nikola Jokic", "Joel Embiid", "Damian Lillard",
            "Kawhi Leonard", "Jimmy Butler", "Anthony Davis", "Devin Booker"
        ]
        
        stat_types = ["points", "rebounds", "assists", "steals", "blocks", "threes_made"]
        props = []
        
        for i in range(min(max_props, 25)):
            player = random.choice(players)
            stat_type = random.choice(stat_types)
            
            # Basketball-specific line ranges
            if stat_type == "points":
                line = random.uniform(18.5, 32.5)
            elif stat_type == "rebounds":
                line = random.uniform(5.5, 13.5)
            elif stat_type == "assists":
                line = random.uniform(3.5, 11.5)
            elif stat_type == "steals":
                line = random.uniform(0.5, 2.5)
            elif stat_type == "blocks":
                line = random.uniform(0.5, 2.5)
            else:  # threes_made
                line = random.uniform(1.5, 5.5)
            
            # Enhanced NBA analytics
            prop = {
                'game_id': f"basketball_{i}",
                'player_name': player,
                'stat_type': stat_type,
                'line': round(line, 1),
                'odds': random.choice([-110, -105, -115, +100, +105]),
                'event_start_time': (datetime.now() + timedelta(hours=random.randint(2, 72))).isoformat(),
                'matchup': f"{random.choice(['LAL', 'GSW', 'MIL', 'DAL', 'BOS', 'DEN', 'PHI', 'POR'])} vs {random.choice(['LAC', 'MIA', 'PHX', 'BRK', 'NYK'])}",
                'sportsbook': random.choice(['DraftKings', 'FanDuel', 'BetMGM', 'Caesars']),
                'recent_form': random.uniform(6, 9),
                'matchup_difficulty': random.uniform(4, 8),
                'injury_status': random.choice(['healthy', 'questionable']),
                # Basketball-specific analytics
                'minutes_projection': random.uniform(28, 38),
                'usage_rate': random.uniform(0.18, 0.35),
                'team_pace': random.uniform(95, 110),
                'opponent_defense_rating': random.uniform(105, 118),
                'back_to_back': random.choice([True, False]),
                'home_away': random.choice(['home', 'away']),
                'days_rest': random.randint(0, 4),
                'opponent_stat_allowed': {
                    'points': random.uniform(20, 30),
                    'rebounds': random.uniform(8, 12),
                    'assists': random.uniform(6, 10)
                }
            }
            
            props.append(prop)
        
        return props
    
    def _analyze_basketball_specific_factors(self, prop: Dict) -> Dict[str, Any]:
        """Analyze basketball-specific factors like pace, usage, matchups"""
        minutes_proj = prop.get('minutes_projection', 32)
        usage_rate = prop.get('usage_rate', 0.25)
        team_pace = prop.get('team_pace', 100)
        opp_def_rating = prop.get('opponent_defense_rating', 110)
        back_to_back = prop.get('back_to_back', False)
        days_rest = prop.get('days_rest', 1)
        
        score = 5.0
        
        # Minutes projection impact
        if minutes_proj > 35:
            score += 1.5
        elif minutes_proj < 30:
            score -= 1.5
        
        # Usage rate considerations
        stat_type = prop.get('stat_type', '')
        if stat_type in ['points', 'assists'] and usage_rate > 0.28:
            score += 1
        elif stat_type == 'rebounds' and usage_rate < 0.22:
            score += 0.5  # Low usage players often get more rebounds
        
        # Pace factor
        pace_multiplier = team_pace / 100
        if stat_type in ['points', 'assists', 'steals']:
            score += (pace_multiplier - 1) * 2
        
        # Defense rating (lower is better defense)
        if opp_def_rating > 115:  # Poor defense
            if stat_type in ['points', 'assists']:
                score += 1
        elif opp_def_rating < 108:  # Great defense
            if stat_type in ['points', 'assists']:
                score -= 1
        
        # Rest factors
        if back_to_back:
            score -= 1
        elif days_rest >= 3:
            score += 0.5
        
        return {
            'minutes_impact': minutes_proj,
            'usage_factor': usage_rate,
            'pace_factor': pace_multiplier,
            'defense_matchup': opp_def_rating,
            'rest_situation': f"{days_rest} days rest, B2B: {back_to_back}",
            'score': max(1, min(10, score)),
            'reasoning': f"Pace: {team_pace}, Usage: {usage_rate:.1%}, vs {opp_def_rating} DefRtg"
        }
    
    def _perform_detailed_analysis(self, prop: Dict) -> Dict[str, Any]:
        """Enhanced analysis including basketball-specific factors"""
        # Get base analysis
        factors = super()._perform_detailed_analysis(prop)
        
        # Add basketball-specific analysis
        factors['basketball_specific'] = self._analyze_basketball_specific_factors(prop)
        
        # Recalculate overall score
        factor_scores = [f['score'] for f in factors.values() if isinstance(f, dict) and 'score' in f]
        factors['overall_score'] = statistics.mean(factor_scores) if factor_scores else 5.0
        
        return factors


class FootballAgent(SportAgent):
    """Football-specific agent for analyzing NFL props with advanced analytics"""
    
    def __init__(self):
        super().__init__("football")
        self.sport_specific_factors = {
            'weather_impact': {},      # Weather effects on different stats
            'target_share': {},        # Receiver target share analysis
            'red_zone_efficiency': {}  # Red zone usage patterns
        }
    
    def _generate_mock_props(self, max_props: int) -> List[Dict]:
        """Generate mock football props with NFL analytics"""
        players = [
            "Josh Allen", "Patrick Mahomes", "Lamar Jackson", "Joe Burrow",
            "Derrick Henry", "Christian McCaffrey", "Travis Kelce", "Tyreek Hill",
            "Stefon Diggs", "Cooper Kupp", "Aaron Donald", "T.J. Watt"
        ]
        
        stat_types = ["passing_yards", "rushing_yards", "receiving_yards", "touchdowns", "receptions"]
        props = []
        
        for i in range(min(max_props, 20)):
            player = random.choice(players)
            stat_type = random.choice(stat_types)
            
            # Football-specific line ranges
            if stat_type == "passing_yards":
                line = random.uniform(225.5, 325.5)
            elif stat_type == "rushing_yards":
                line = random.uniform(45.5, 125.5)
            elif stat_type == "receiving_yards":
                line = random.uniform(35.5, 95.5)
            elif stat_type == "touchdowns":
                line = random.uniform(0.5, 2.5)
            else:  # receptions
                line = random.uniform(3.5, 8.5)
            
            # Enhanced NFL analytics
            prop = {
                'game_id': f"football_{i}",
                'player_name': player,
                'stat_type': stat_type,
                'line': round(line, 1),
                'odds': random.choice([-110, -105, -115, +100, +105]),
                'event_start_time': (datetime.now() + timedelta(days=random.randint(1, 7))).isoformat(),
                'matchup': f"{random.choice(['KC', 'BUF', 'CIN', 'DAL', 'SF', 'PHI'])} vs {random.choice(['MIA', 'NYJ', 'LV', 'DEN'])}",
                'sportsbook': random.choice(['DraftKings', 'FanDuel', 'BetMGM', 'PointsBet']),
                'recent_form': random.uniform(6, 9),
                'matchup_difficulty': random.uniform(3, 9),
                'injury_status': random.choice(['healthy', 'questionable', 'probable']),
                # Football-specific analytics
                'target_share': random.uniform(0.15, 0.35) if stat_type in ['receiving_yards', 'receptions'] else None,
                'red_zone_touches': random.randint(1, 8) if stat_type in ['touchdowns', 'rushing_yards'] else None,
                'opponent_yards_allowed': random.uniform(180, 280),
                'game_script': random.choice(['passing_game_script', 'rushing_game_script', 'balanced']),
                'weather_conditions': random.choice(['dome', 'clear', 'wind', 'rain', 'cold']),
                'vegas_total': random.uniform(42.5, 55.5),
                'spread': random.uniform(-14, 14),
                'snap_count_projection': random.uniform(0.65, 0.95)
            }
            
            props.append(prop)
        
        return props
    
    def _analyze_football_specific_factors(self, prop: Dict) -> Dict[str, Any]:
        """Analyze football-specific factors"""
        stat_type = prop.get('stat_type', '')
        target_share = prop.get('target_share')
        red_zone_touches = prop.get('red_zone_touches')
        game_script = prop.get('game_script', 'balanced')
        weather = prop.get('weather_conditions', 'clear')
        vegas_total = prop.get('vegas_total', 47)
        spread = prop.get('spread', 0)
        snap_count = prop.get('snap_count_projection', 0.8)
        
        score = 5.0
        
        # Target share analysis for receivers
        if target_share and stat_type in ['receiving_yards', 'receptions']:
            if target_share > 0.25:
                score += 2
            elif target_share < 0.18:
                score -= 1
        
        # Red zone usage for TDs and rushing
        if red_zone_touches and stat_type in ['touchdowns', 'rushing_yards']:
            if red_zone_touches > 5:
                score += 1.5
            elif red_zone_touches < 3:
                score -= 1
        
        # Game script analysis
        if game_script == 'passing_game_script' and stat_type in ['passing_yards', 'receiving_yards', 'receptions']:
            score += 1
        elif game_script == 'rushing_game_script' and stat_type == 'rushing_yards':
            score += 1
        
        # Weather impact
        weather_impact = {
            'dome': 0,
            'clear': 0,
            'wind': -1 if stat_type in ['passing_yards', 'receiving_yards'] else 0,
            'rain': -1.5 if stat_type in ['passing_yards', 'receiving_yards'] else 0.5,
            'cold': -0.5
        }
        score += weather_impact.get(weather, 0)
        
        # Vegas total (higher total = more opportunities)
        if vegas_total > 50:
            score += 1
        elif vegas_total < 44:
            score -= 1
        
        # Spread considerations
        abs_spread = abs(spread)
        if abs_spread > 7:  # Blowout potential
            if stat_type == 'rushing_yards':
                score += 0.5  # More rushing in blowouts
            elif stat_type in ['passing_yards', 'receiving_yards']:
                score -= 0.5  # Less passing in blowouts
        
        # Snap count projection
        score += (snap_count - 0.75) * 2
        
        return {
            'target_share': target_share,
            'red_zone_usage': red_zone_touches,
            'game_script': game_script,
            'weather_impact': weather,
            'total_implications': vegas_total,
            'spread_impact': spread,
            'snap_count': snap_count,
            'score': max(1, min(10, score)),
            'reasoning': f"Game script: {game_script}, Weather: {weather}, Total: {vegas_total}"
        }
    
    def _perform_detailed_analysis(self, prop: Dict) -> Dict[str, Any]:
        """Enhanced analysis including football-specific factors"""
        # Get base analysis
        factors = super()._perform_detailed_analysis(prop)
        
        # Add football-specific analysis
        factors['football_specific'] = self._analyze_football_specific_factors(prop)
        
        # Recalculate overall score
        factor_scores = [f['score'] for f in factors.values() if isinstance(f, dict) and 'score' in f]
        factors['overall_score'] = statistics.mean(factor_scores) if factor_scores else 5.0
        
        return factors


class BaseballAgent(SportAgent):
    """Baseball-specific agent for analyzing baseball props"""
    
    def __init__(self):
        super().__init__("baseball")
    
    def _generate_mock_props(self, max_props: int) -> List[Dict]:
        """Generate mock baseball props"""
        players = [
            "Mike Trout", "Aaron Judge", "Ronald Acuna Jr.", "Mookie Betts",
            "Juan Soto", "Vladimir Guerrero Jr.", "Fernando Tatis Jr.", "Freddie Freeman"
        ]
        
        stat_types = ["hits", "runs", "rbis", "home_runs", "stolen_bases", "strikeouts"]
        props = []
        
        for i in range(min(max_props, 15)):
            player = random.choice(players)
            stat_type = random.choice(stat_types)
            
            # Baseball-specific line ranges
            if stat_type == "hits":
                line = random.uniform(0.5, 2.5)
            elif stat_type == "runs":
                line = random.uniform(0.5, 1.5)
            elif stat_type == "rbis":
                line = random.uniform(0.5, 2.5)
            elif stat_type == "home_runs":
                line = random.uniform(0.5, 1.5)
            elif stat_type == "stolen_bases":
                line = random.uniform(0.5, 1.5)
            else:  # strikeouts (for pitchers)
                line = random.uniform(4.5, 9.5)
            
            props.append({
                'game_id': f"baseball_{i}",
                'player_name': player,
                'stat_type': stat_type,
                'line': round(line, 1),
                'odds': random.choice([-110, -105, -115, +100, +105]),
                'event_start_time': (datetime.now() + timedelta(hours=random.randint(6, 48))).isoformat(),
                'matchup': f"Team vs Team",
                'sportsbook': random.choice(['DraftKings', 'FanDuel', 'BetMGM']),
                'recent_form': random.uniform(5, 9),
                'matchup_difficulty': random.uniform(4, 8),
                'injury_status': random.choice(['healthy', 'day-to-day']),
                # Baseball-specific factors
                'ballpark_factor': random.uniform(0.85, 1.15),
                'pitcher_handedness': random.choice(['L', 'R']),
                'wind_direction': random.choice(['in', 'out', 'cross', 'calm'])
            })
        
        return props


class HockeyAgent(SportAgent):
    """Hockey-specific agent for analyzing hockey props"""
    
    def __init__(self):
        super().__init__("hockey")
    
    def _generate_mock_props(self, max_props: int) -> List[Dict]:
        """Generate mock hockey props"""
        players = [
            "Connor McDavid", "Leon Draisaitl", "Nathan MacKinnon", "Erik Karlsson",
            "David Pastrnak", "Auston Matthews", "Mikko Rantanen", "Alex Ovechkin"
        ]
        
        stat_types = ["goals", "assists", "points", "shots_on_goal", "penalty_minutes"]
        props = []
        
        for i in range(min(max_props, 15)):
            player = random.choice(players)
            stat_type = random.choice(stat_types)
            
            # Hockey-specific line ranges
            if stat_type == "goals":
                line = random.uniform(0.5, 1.5)
            elif stat_type == "assists":
                line = random.uniform(0.5, 2.5)
            elif stat_type == "points":
                line = random.uniform(0.5, 2.5)
            elif stat_type == "shots_on_goal":
                line = random.uniform(2.5, 5.5)
            else:  # penalty_minutes
                line = random.uniform(0.5, 2.5)
            
            props.append({
                'game_id': f"hockey_{i}",
                'player_name': player,
                'stat_type': stat_type,
                'line': round(line, 1),
                'odds': random.choice([-110, -105, -115, +100, +105]),
                'event_start_time': (datetime.now() + timedelta(hours=random.randint(4, 72))).isoformat(),
                'matchup': f"Team vs Team",
                'sportsbook': random.choice(['DraftKings', 'FanDuel', 'BetMGM']),
                'recent_form': random.uniform(6, 9),
                'matchup_difficulty': random.uniform(4, 8),
                'injury_status': random.choice(['healthy', 'upper-body', 'lower-body']),
                # Hockey-specific factors
                'pp_time_projection': random.uniform(2, 6),
                'line_chemistry': random.uniform(0.7, 1.0)
            })
        
        return props


class SoccerAgent(SportAgent):
    """Soccer-specific agent for analyzing soccer props"""
    
    def __init__(self):
        super().__init__("soccer")
    
    def _generate_mock_props(self, max_props: int) -> List[Dict]:
        """Generate mock soccer props"""
        players = [
            "Lionel Messi", "Kylian Mbappe", "Erling Haaland", "Mohamed Salah",
            "Kevin De Bruyne", "Vinicius Jr.", "Harry Kane", "Karim Benzema"
        ]
        
        stat_types = ["goals", "assists", "shots", "shots_on_target", "cards"]
        props = []
        
        for i in range(min(max_props, 15)):
            player = random.choice(players)
            stat_type = random.choice(stat_types)
            
            # Soccer-specific line ranges
            if stat_type == "goals":
                line = random.uniform(0.5, 1.5)
            elif stat_type == "assists":
                line = random.uniform(0.5, 1.5)
            elif stat_type == "shots":
                line = random.uniform(2.5, 4.5)
            elif stat_type == "shots_on_target":
                line = random.uniform(1.5, 3.5)
            else:  # cards
                line = random.uniform(0.5, 1.5)
            
            props.append({
                'game_id': f"soccer_{i}",
                'player_name': player,
                'stat_type': stat_type,
                'line': round(line, 1),
                'odds': random.choice([-110, -105, -115, +100, +105]),
                'event_start_time': (datetime.now() + timedelta(hours=random.randint(12, 168))).isoformat(),
                'matchup': f"Team vs Team",
                'sportsbook': random.choice(['DraftKings', 'FanDuel', 'BetMGM']),
                'recent_form': random.uniform(5, 9),
                'matchup_difficulty': random.uniform(3, 9),
                'injury_status': random.choice(['healthy', 'minor knock']),
                # Soccer-specific factors
                'competition': random.choice(['Premier League', 'Champions League', 'La Liga']),
                'home_away': random.choice(['home', 'away'])
            })
        
        return props


class EsportsAgent(SportAgent):
    """Base esports agent for analyzing esports props"""
    
    def __init__(self, game_title: str = "esports"):
        super().__init__(game_title)
        self.game_title = game_title
    
    def _generate_mock_props(self, max_props: int) -> List[Dict]:
        """Generate mock esports props - to be overridden by specific games"""
        return []
    
    def _analyze_esports_specific_factors(self, prop: Dict) -> Dict[str, Any]:
        """Base esports analysis - to be overridden by specific games"""
        return {
            'game_meta': 'unknown',
            'patch_impact': 0,
            'score': 5.0,
            'reasoning': 'Base esports analysis'
        }


class CSGOAgent(EsportsAgent):
    """Counter-Strike: Global Offensive specific agent"""
    
    def __init__(self):
        super().__init__("csgo")
    
    def _generate_mock_props(self, max_props: int) -> List[Dict]:
        """Generate CSGO-specific props"""
        players = [
            "s1mple", "ZywOo", "sh1ro", "electronic", "Ax1Le",
            "nafany", "jks", "stavn", "blameF", "ropz"
        ]
        
        stat_types = ["kills", "deaths", "adr", "rating", "headshot_percentage", "first_kills"]
        maps = ["Dust2", "Mirage", "Inferno", "Cache", "Overpass", "Vertigo", "Ancient"]
        
        props = []
        
        for i in range(min(max_props, 12)):
            player = random.choice(players)
            stat_type = random.choice(stat_types)
            
            # CSGO-specific line ranges
            if stat_type == "kills":
                line = random.uniform(15.5, 25.5)
            elif stat_type == "deaths":
                line = random.uniform(12.5, 20.5)
            elif stat_type == "adr":  # Average Damage per Round
                line = random.uniform(65.5, 85.5)
            elif stat_type == "rating":
                line = random.uniform(0.95, 1.35)
            elif stat_type == "headshot_percentage":
                line = random.uniform(35.5, 55.5)
            else:  # first_kills
                line = random.uniform(2.5, 6.5)
            
            props.append({
                'game_id': f"csgo_{i}",
                'player_name': player,
                'stat_type': stat_type,
                'line': round(line, 1),
                'odds': random.choice([-110, -105, -115, +100, +105]),
                'event_start_time': (datetime.now() + timedelta(hours=random.randint(6, 72))).isoformat(),
                'matchup': f"{random.choice(['NAVI', 'G2', 'FaZe', 'Vitality'])} vs {random.choice(['Astralis', 'FURIA', 'NIP', 'Heroic'])}",
                'sportsbook': random.choice(['DraftKings', 'Betway', 'GGBET']),
                'recent_form': random.uniform(6, 9),
                'matchup_difficulty': random.uniform(4, 8),
                'injury_status': 'healthy',
                # CSGO-specific factors
                'map': random.choice(maps),
                'side_preference': random.choice(['T', 'CT', 'balanced']),
                'team_chemistry': random.uniform(0.6, 1.0),
                'recent_map_performance': random.uniform(0.4, 0.8),
                'opponent_map_ban_rate': random.uniform(0.1, 0.4)
            })
        
        return props
    
    def _analyze_esports_specific_factors(self, prop: Dict) -> Dict[str, Any]:
        """Analyze CSGO-specific factors"""
        map_name = prop.get('map', 'unknown')
        side_pref = prop.get('side_preference', 'balanced')
        team_chemistry = prop.get('team_chemistry', 0.8)
        map_performance = prop.get('recent_map_performance', 0.6)
        
        score = 5.0
        
        # Map-specific performance
        if map_performance > 0.7:
            score += 1.5
        elif map_performance < 0.5:
            score -= 1.5
        
        # Side preference impact
        if side_pref != 'balanced':
            score += 0.5  # Specialists often perform better
        
        # Team chemistry factor
        score += (team_chemistry - 0.7) * 3
        
        return {
            'map': map_name,
            'side_preference': side_pref,
            'team_chemistry': team_chemistry,
            'map_performance': map_performance,
            'score': max(1, min(10, score)),
            'reasoning': f"Map: {map_name}, Side: {side_pref}, Team chemistry: {team_chemistry:.1%}"
        }
    
    def _perform_detailed_analysis(self, prop: Dict) -> Dict[str, Any]:
        """Enhanced analysis including CSGO-specific factors"""
        # Get base analysis
        factors = super()._perform_detailed_analysis(prop)
        
        # Add CSGO-specific analysis
        factors['csgo_specific'] = self._analyze_esports_specific_factors(prop)
        
        # Recalculate overall score
        factor_scores = [f['score'] for f in factors.values() if isinstance(f, dict) and 'score' in f]
        factors['overall_score'] = statistics.mean(factor_scores) if factor_scores else 5.0
        
        return factors


class LeagueOfLegendsAgent(EsportsAgent):
    """League of Legends specific agent"""
    
    def __init__(self):
        super().__init__("league_of_legends")
    
    def _generate_mock_props(self, max_props: int) -> List[Dict]:
        """Generate League of Legends props"""
        players = [
            "Faker", "Caps", "Jankos", "Rekkles", "Perkz", "Bjergsen", 
            "Canyon", "ShowMaker", "Chovy", "Deft", "Gumayusi", "Keria"
        ]
        
        stat_types = ["kills", "deaths", "assists", "cs", "gold", "damage_dealt", "vision_score"]
        roles = ["Top", "Jungle", "Mid", "ADC", "Support"]
        
        props = []
        
        for i in range(min(max_props, 15)):
            player = random.choice(players)
            stat_type = random.choice(stat_types)
            role = random.choice(roles)
            
            # LoL-specific line ranges based on role
            if stat_type == "kills":
                base = 3.5 if role == "Support" else 5.5
                line = random.uniform(base, base + 4)
            elif stat_type == "deaths":
                line = random.uniform(1.5, 4.5)
            elif stat_type == "assists":
                base = 8.5 if role == "Support" else 6.5
                line = random.uniform(base, base + 6)
            elif stat_type == "cs":
                base = 50 if role == "Support" else 180
                line = random.uniform(base, base + 100)
            elif stat_type == "gold":
                base = 8000 if role == "Support" else 12000
                line = random.uniform(base, base + 6000)
            elif stat_type == "damage_dealt":
                base = 15000 if role == "Support" else 25000
                line = random.uniform(base, base + 15000)
            else:  # vision_score
                base = 25 if role == "Support" else 15
                line = random.uniform(base, base + 20)
            
            props.append({
                'game_id': f"lol_{i}",
                'player_name': player,
                'stat_type': stat_type,
                'line': round(line, 1),
                'odds': random.choice([-110, -105, -115, +100, +105]),
                'event_start_time': (datetime.now() + timedelta(hours=random.randint(6, 72))).isoformat(),
                'matchup': f"{random.choice(['T1', 'DK', 'GEN', 'DRX'])} vs {random.choice(['KT', 'LSB', 'HLE', 'BRO'])}",
                'sportsbook': random.choice(['DraftKings', 'FanDuel', 'Betway']),
                'recent_form': random.uniform(6, 9),
                'matchup_difficulty': random.uniform(4, 8),
                'injury_status': 'healthy',
                # LoL-specific factors
                'role': role,
                'champion_pool': random.choice(['meta', 'off-meta', 'comfort']),
                'patch_adaptation': random.uniform(0.6, 1.0),
                'team_playstyle': random.choice(['aggressive', 'passive', 'balanced']),
                'average_game_time': random.uniform(25, 35)
            })
        
        return props
    
    def _analyze_esports_specific_factors(self, prop: Dict) -> Dict[str, Any]:
        """Analyze League of Legends specific factors"""
        role = prop.get('role', 'unknown')
        champion_pool = prop.get('champion_pool', 'meta')
        patch_adaptation = prop.get('patch_adaptation', 0.8)
        team_style = prop.get('team_playstyle', 'balanced')
        avg_game_time = prop.get('average_game_time', 30)
        
        score = 5.0
        
        # Champion pool meta alignment
        pool_scores = {'meta': 1, 'comfort': 0.5, 'off-meta': -0.5}
        score += pool_scores.get(champion_pool, 0)
        
        # Patch adaptation
        score += (patch_adaptation - 0.8) * 3
        
        # Team playstyle synergy with role
        if role in ['Jungle', 'Mid'] and team_style == 'aggressive':
            score += 0.5
        elif role == 'Support' and team_style == 'passive':
            score += 0.5
        
        # Game time impact on different stats
        stat_type = prop.get('stat_type', '')
        if avg_game_time > 32 and stat_type in ['cs', 'gold']:
            score += 0.5
        elif avg_game_time < 28 and stat_type in ['kills', 'deaths']:
            score += 0.5
        
        return {
            'role': role,
            'champion_pool': champion_pool,
            'patch_adaptation': patch_adaptation,
            'team_playstyle': team_style,
            'game_time_factor': avg_game_time,
            'score': max(1, min(10, score)),
            'reasoning': f"Role: {role}, Pool: {champion_pool}, Style: {team_style}, Avg time: {avg_game_time}min"
        }
    
    def _perform_detailed_analysis(self, prop: Dict) -> Dict[str, Any]:
        """Enhanced analysis including League of Legends specific factors"""
        # Get base analysis
        factors = super()._perform_detailed_analysis(prop)
        
        # Add LoL-specific analysis
        factors['league_of_legends_specific'] = self._analyze_esports_specific_factors(prop)
        
        # Recalculate overall score
        factor_scores = [f['score'] for f in factors.values() if isinstance(f, dict) and 'score' in f]
        factors['overall_score'] = statistics.mean(factor_scores) if factor_scores else 5.0
        
        return factors


class Dota2Agent(EsportsAgent):
    """Dota 2 specific agent"""
    
    def __init__(self):
        super().__init__("dota2")
    
    def _generate_mock_props(self, max_props: int) -> List[Dict]:
        """Generate Dota 2 props"""
        players = [
            "Topson", "Ceb", "N0tail", "ana", "JerAx",
            "Puppey", "Nisha", "MATUMBAMAN", "zai", "YapzOr"
        ]
        
        stat_types = ["kills", "deaths", "assists", "last_hits", "gpm", "xpm", "hero_damage"]
        positions = ["Position 1", "Position 2", "Position 3", "Position 4", "Position 5"]
        
        props = []
        
        for i in range(min(max_props, 12)):
            player = random.choice(players)
            stat_type = random.choice(stat_types)
            position = random.choice(positions)
            
            # Dota 2 line ranges
            if stat_type == "kills":
                base = 2.5 if position in ["Position 4", "Position 5"] else 4.5
                line = random.uniform(base, base + 6)
            elif stat_type == "deaths":
                line = random.uniform(2.5, 6.5)
            elif stat_type == "assists":
                base = 8.5 if position in ["Position 4", "Position 5"] else 6.5
                line = random.uniform(base, base + 8)
            elif stat_type == "last_hits":
                base = 50 if position in ["Position 4", "Position 5"] else 200
                line = random.uniform(base, base + 150)
            elif stat_type == "gpm":  # Gold per minute
                base = 300 if position == "Position 5" else 450
                line = random.uniform(base, base + 200)
            elif stat_type == "xpm":  # Experience per minute
                base = 350 if position == "Position 5" else 500
                line = random.uniform(base, base + 200)
            else:  # hero_damage
                base = 15000 if position in ["Position 4", "Position 5"] else 25000
                line = random.uniform(base, base + 20000)
            
            props.append({
                'game_id': f"dota2_{i}",
                'player_name': player,
                'stat_type': stat_type,
                'line': round(line, 1),
                'odds': random.choice([-110, -105, -115, +100, +105]),
                'event_start_time': (datetime.now() + timedelta(hours=random.randint(6, 72))).isoformat(),
                'matchup': f"{random.choice(['OG', 'Secret', 'EG', 'VP'])} vs {random.choice(['Liquid', 'Alliance', 'Nigma', 'NaVi'])}",
                'sportsbook': random.choice(['DraftKings', 'Betway', 'GG.Bet']),
                'recent_form': random.uniform(6, 9),
                'matchup_difficulty': random.uniform(4, 8),
                'injury_status': 'healthy',
                # Dota 2 specific factors
                'position': position,
                'hero_pool': random.choice(['meta', 'comfort', 'versatile']),
                'draft_priority': random.uniform(0.3, 0.9),
                'team_coordination': random.uniform(0.6, 1.0)
            })
        
        return props
    
    def _analyze_esports_specific_factors(self, prop: Dict) -> Dict[str, Any]:
        """Analyze Dota 2 specific factors"""
        position = prop.get('position', 'unknown')
        hero_pool = prop.get('hero_pool', 'meta')
        draft_priority = prop.get('draft_priority', 0.6)
        
        score = 5.0
        pool_scores = {'versatile': 1, 'meta': 0.5, 'comfort': 0}
        score += pool_scores.get(hero_pool, 0)
        score += (draft_priority - 0.6) * 2
        
        return {
            'position': position,
            'hero_pool': hero_pool,
            'draft_priority': draft_priority,
            'score': max(1, min(10, score)),
            'reasoning': f"Position: {position}, Pool: {hero_pool}, Draft priority: {draft_priority:.1%}"
        }
    
    def _perform_detailed_analysis(self, prop: Dict) -> Dict[str, Any]:
        """Enhanced analysis including Dota 2 specific factors"""
        factors = super()._perform_detailed_analysis(prop)
        factors['dota2_specific'] = self._analyze_esports_specific_factors(prop)
        factor_scores = [f['score'] for f in factors.values() if isinstance(f, dict) and 'score' in f]
        factors['overall_score'] = statistics.mean(factor_scores) if factor_scores else 5.0
        return factors


class VALORANTAgent(EsportsAgent):
    """VALORANT specific agent"""
    
    def __init__(self):
        super().__init__("valorant")
    
    def _generate_mock_props(self, max_props: int) -> List[Dict]:
        """Generate VALORANT props"""
        players = [
            "TenZ", "Sick", "dapr", "ShahZaM", "zombs",
            "ScreaM", "Nivera", "Jamppi", "soulcas", "L1NK"
        ]
        
        stat_types = ["kills", "deaths", "assists", "acs", "first_kills", "clutch_rate"]
        agents = ["Jett", "Reyna", "Phoenix", "Sage", "Cypher", "Sova", "Breach", "Omen"]
        maps = ["Bind", "Haven", "Split", "Ascent", "Icebox", "Breeze", "Fracture"]
        
        props = []
        
        for i in range(min(max_props, 10)):
            player = random.choice(players)
            stat_type = random.choice(stat_types)
            
            # VALORANT line ranges
            if stat_type == "kills":
                line = random.uniform(14.5, 22.5)
            elif stat_type == "deaths":
                line = random.uniform(11.5, 18.5)
            elif stat_type == "assists":
                line = random.uniform(3.5, 8.5)
            elif stat_type == "acs":  # Average Combat Score
                line = random.uniform(180.5, 280.5)
            elif stat_type == "first_kills":
                line = random.uniform(2.5, 6.5)
            else:  # clutch_rate
                line = random.uniform(0.15, 0.45)
            
            props.append({
                'game_id': f"valorant_{i}",
                'player_name': player,
                'stat_type': stat_type,
                'line': round(line, 1),
                'odds': random.choice([-110, -105, -115, +100, +105]),
                'event_start_time': (datetime.now() + timedelta(hours=random.randint(6, 72))).isoformat(),
                'matchup': f"{random.choice(['SEN', 'TSM', 'C9', 'NV'])} vs {random.choice(['100T', 'FaZe', 'XSET', 'LG'])}",
                'sportsbook': random.choice(['DraftKings', 'FanDuel', 'Betway']),
                'recent_form': random.uniform(6, 9),
                'matchup_difficulty': random.uniform(4, 8),
                'injury_status': 'healthy',
                # VALORANT specific factors
                'agent_pool': random.choice(agents),
                'map': random.choice(maps),
                'role': random.choice(['Duelist', 'Controller', 'Initiator', 'Sentinel']),
                'team_strategy': random.choice(['aggressive', 'tactical', 'adaptive'])
            })
        
        return props
    
    def _analyze_esports_specific_factors(self, prop: Dict) -> Dict[str, Any]:
        """Analyze VALORANT specific factors"""
        agent_pool = prop.get('agent_pool', 'unknown')
        map_name = prop.get('map', 'unknown')
        role = prop.get('role', 'unknown')
        team_strategy = prop.get('team_strategy', 'balanced')
        
        score = 5.0
        
        # Role synergy with team strategy
        if role == 'Duelist' and team_strategy == 'aggressive':
            score += 1
        elif role == 'Controller' and team_strategy == 'tactical':
            score += 1
        
        return {
            'agent_pool': agent_pool,
            'map': map_name,
            'role': role,
            'team_strategy': team_strategy,
            'score': max(1, min(10, score)),
            'reasoning': f"Agent: {agent_pool}, Map: {map_name}, Role: {role}, Strategy: {team_strategy}"
        }
    
    def _perform_detailed_analysis(self, prop: Dict) -> Dict[str, Any]:
        """Enhanced analysis including VALORANT specific factors"""
        factors = super()._perform_detailed_analysis(prop)
        factors['valorant_specific'] = self._analyze_esports_specific_factors(prop)
        factor_scores = [f['score'] for f in factors.values() if isinstance(f, dict) and 'score' in f]
        factors['overall_score'] = statistics.mean(factor_scores) if factor_scores else 5.0
        return factors


class OverwatchAgent(EsportsAgent):
    """Overwatch specific agent"""
    
    def __init__(self):
        super().__init__("overwatch")
    
    def _generate_mock_props(self, max_props: int) -> List[Dict]:
        """Generate Overwatch props"""
        players = [
            "Profit", "Gesture", "Bdosin", "Fury", "Fleta",
            "Carpe", "Alarm", "Poko", "EQO", "FunnyAstro"
        ]
        
        stat_types = ["eliminations", "deaths", "damage", "healing", "objective_time", "final_blows"]
        roles = ["Tank", "Damage", "Support"]
        
        props = []
        
        for i in range(min(max_props, 8)):
            player = random.choice(players)
            stat_type = random.choice(stat_types)
            role = random.choice(roles)
            
            # Overwatch line ranges by role
            if stat_type == "eliminations":
                base = 15 if role == "Support" else 25
                line = random.uniform(base, base + 15)
            elif stat_type == "deaths":
                line = random.uniform(4.5, 8.5)
            elif stat_type == "damage":
                base = 5000 if role == "Support" else 12000
                line = random.uniform(base, base + 8000)
            elif stat_type == "healing":
                base = 8000 if role == "Support" else 0
                line = random.uniform(base, base + 4000) if role == "Support" else 0
            elif stat_type == "objective_time":
                line = random.uniform(45.5, 90.5)
            else:  # final_blows
                base = 8 if role == "Support" else 15
                line = random.uniform(base, base + 10)
            
            if line > 0:  # Skip healing for non-support roles
                props.append({
                    'game_id': f"overwatch_{i}",
                    'player_name': player,
                    'stat_type': stat_type,
                    'line': round(line, 1),
                    'odds': random.choice([-110, -105, -115, +100, +105]),
                    'event_start_time': (datetime.now() + timedelta(hours=random.randint(6, 72))).isoformat(),
                    'matchup': f"{random.choice(['London', 'Seoul', 'Philly'])} vs {random.choice(['Dallas', 'SF', 'Boston'])}",
                    'sportsbook': random.choice(['DraftKings', 'Betway']),
                    'recent_form': random.uniform(6, 9),
                    'matchup_difficulty': random.uniform(4, 8),
                    'injury_status': 'healthy',
                    # Overwatch specific factors
                    'role': role,
                    'hero_flexibility': random.uniform(0.6, 1.0),
                    'team_synergy': random.uniform(0.7, 1.0)
                })
        
        return props[:max_props]
    
    def _analyze_esports_specific_factors(self, prop: Dict) -> Dict[str, Any]:
        """Analyze Overwatch specific factors"""
        role = prop.get('role', 'unknown')
        hero_flexibility = prop.get('hero_flexibility', 0.8)
        team_synergy = prop.get('team_synergy', 0.8)
        
        score = 5.0
        score += (hero_flexibility - 0.8) * 3
        score += (team_synergy - 0.8) * 2
        
        return {
            'role': role,
            'hero_flexibility': hero_flexibility,
            'team_synergy': team_synergy,
            'score': max(1, min(10, score)),
            'reasoning': f"Role: {role}, Flexibility: {hero_flexibility:.1%}, Synergy: {team_synergy:.1%}"
        }
    
    def _perform_detailed_analysis(self, prop: Dict) -> Dict[str, Any]:
        """Enhanced analysis including Overwatch specific factors"""
        factors = super()._perform_detailed_analysis(prop)
        factors['overwatch_specific'] = self._analyze_esports_specific_factors(prop)
        factor_scores = [f['score'] for f in factors.values() if isinstance(f, dict) and 'score' in f]
        factors['overall_score'] = statistics.mean(factor_scores) if factor_scores else 5.0
        return factors


class RocketLeagueAgent(EsportsAgent):
    """Rocket League specific agent"""
    
    def __init__(self):
        super().__init__("rocket_league")
    
    def _generate_mock_props(self, max_props: int) -> List[Dict]:
        """Generate Rocket League props"""
        players = [
            "jstn", "GarrettG", "Squishy", "Turbo", "Kaydop",
            "Fairy Peak", "Alpha54", "Extra", "Monkey Moon", "BeastMode"
        ]
        
        stat_types = ["goals", "saves", "assists", "shots", "score", "demos"]
        
        props = []
        
        for i in range(min(max_props, 8)):
            player = random.choice(players)
            stat_type = random.choice(stat_types)
            
            # Rocket League line ranges
            if stat_type == "goals":
                line = random.uniform(1.5, 3.5)
            elif stat_type == "saves":
                line = random.uniform(2.5, 5.5)
            elif stat_type == "assists":
                line = random.uniform(1.5, 3.5)
            elif stat_type == "shots":
                line = random.uniform(4.5, 8.5)
            elif stat_type == "score":
                line = random.uniform(350.5, 550.5)
            else:  # demos
                line = random.uniform(0.5, 2.5)
            
            props.append({
                'game_id': f"rocket_league_{i}",
                'player_name': player,
                'stat_type': stat_type,
                'line': round(line, 1),
                'odds': random.choice([-110, -105, -115, +100, +105]),
                'event_start_time': (datetime.now() + timedelta(hours=random.randint(6, 72))).isoformat(),
                'matchup': f"{random.choice(['NRG', 'G2', 'SSG', 'C9'])} vs {random.choice(['BDS', 'Vitality', 'Endpoint', 'DIG'])}",
                'sportsbook': random.choice(['DraftKings', 'Betway']),
                'recent_form': random.uniform(6, 9),
                'matchup_difficulty': random.uniform(4, 8),
                'injury_status': 'healthy',
                # Rocket League specific factors
                'playstyle': random.choice(['mechanical', 'positional', 'aggressive']),
                'team_rotation': random.uniform(0.7, 1.0),
                'boost_management': random.uniform(0.6, 1.0)
            })
        
        return props
    
    def _analyze_esports_specific_factors(self, prop: Dict) -> Dict[str, Any]:
        """Analyze Rocket League specific factors"""
        playstyle = prop.get('playstyle', 'balanced')
        team_rotation = prop.get('team_rotation', 0.8)
        boost_management = prop.get('boost_management', 0.8)
        
        score = 5.0
        
        # Playstyle impact on stats
        stat_type = prop.get('stat_type', '')
        if playstyle == 'mechanical' and stat_type in ['goals', 'shots']:
            score += 1
        elif playstyle == 'positional' and stat_type == 'saves':
            score += 1
        
        score += (team_rotation - 0.8) * 2
        score += (boost_management - 0.8) * 1.5
        
        return {
            'playstyle': playstyle,
            'team_rotation': team_rotation,
            'boost_management': boost_management,
            'score': max(1, min(10, score)),
            'reasoning': f"Style: {playstyle}, Rotation: {team_rotation:.1%}, Boost: {boost_management:.1%}"
        }
    
    def _perform_detailed_analysis(self, prop: Dict) -> Dict[str, Any]:
        """Enhanced analysis including Rocket League specific factors"""
        factors = super()._perform_detailed_analysis(prop)
        factors['rocket_league_specific'] = self._analyze_esports_specific_factors(prop)
        factor_scores = [f['score'] for f in factors.values() if isinstance(f, dict) and 'score' in f]
        factors['overall_score'] = statistics.mean(factor_scores) if factor_scores else 5.0
        return factors


class CollegeFootballAgent(SportAgent):
    """College Football-specific agent for analyzing college football props"""
    
    def __init__(self):
        super().__init__("college_football")
    
    def _generate_mock_props(self, max_props: int) -> List[Dict]:
        """Generate mock college football props"""
        players = [
            "Caleb Williams", "Drake Maye", "Bo Nix", "Michael Penix Jr.",
            "Rome Odunze", "Marvin Harrison Jr.", "Malik Nabers", "Brock Bowers"
        ]
        
        stat_types = ["passing_yards", "rushing_yards", "receiving_yards", "touchdowns", "receptions"]
        props = []
        
        for i in range(min(max_props, 12)):
            player = random.choice(players)
            stat_type = random.choice(stat_types)
            
            # College football-specific line ranges (generally lower than NFL)
            if stat_type == "passing_yards":
                line = random.uniform(185.5, 285.5)
            elif stat_type == "rushing_yards":
                line = random.uniform(35.5, 105.5)
            elif stat_type == "receiving_yards":
                line = random.uniform(25.5, 85.5)
            elif stat_type == "touchdowns":
                line = random.uniform(0.5, 2.5)
            else:  # receptions
                line = random.uniform(2.5, 7.5)
            
            props.append({
                'game_id': f"college_football_{i}",
                'player_name': player,
                'stat_type': stat_type,
                'line': round(line, 1),
                'odds': random.choice([-110, -105, -115, +100, +105]),
                'event_start_time': (datetime.now() + timedelta(days=random.randint(1, 14))).isoformat(),
                'matchup': f"University vs University",
                'sportsbook': random.choice(['DraftKings', 'FanDuel', 'BetMGM']),
                'recent_form': random.uniform(5, 9),
                'matchup_difficulty': random.uniform(3, 9),
                'injury_status': random.choice(['healthy', 'questionable']),
                # College-specific factors
                'conference_game': random.choice([True, False]),
                'rivalry_game': random.choice([True, False])
            })
        
        return props


# Factory function to create agents
def create_sport_agent(sport_name: str) -> SportAgent:
    """
    Factory function to create the appropriate sport agent
    
    Args:
        sport_name (str): Name of the sport
        
    Returns:
        SportAgent: Instance of the appropriate sport agent
    """
    sport_agents = {
        'tennis': TennisAgent,
        'basketball': BasketballAgent,
        'football': FootballAgent,
        'baseball': BaseballAgent,
        'hockey': HockeyAgent,
        'soccer': SoccerAgent,
        'esports': EsportsAgent,  # Keep for backward compatibility
        'college_football': CollegeFootballAgent,
        # New esports-specific agents
        'csgo': CSGOAgent,
        'league_of_legends': LeagueOfLegendsAgent,
        'dota2': Dota2Agent,
        'valorant': VALORANTAgent,
        'overwatch': OverwatchAgent,
        'rocket_league': RocketLeagueAgent
    }
    
    agent_class = sport_agents.get(sport_name.lower())
    if agent_class:
        return agent_class()
    else:
        raise ValueError(f"No agent available for sport: {sport_name}")


def demonstrate_esports_agents():
    """
    Demonstrate the new esports-specific agents
    """
    print("=== Testing New Esports Agents ===\n")
    
    esports_games = ['csgo', 'league_of_legends', 'dota2', 'valorant', 'overwatch', 'rocket_league']
    
    for game in esports_games:
        print(f"Testing {game.upper().replace('_', ' ')} Agent...")
        try:
            agent = create_sport_agent(game)
            picks = agent.make_picks()
            
            print(f"  - Generated {len(picks)} picks")
            
            if picks:
                best_pick = max(picks, key=lambda x: x['confidence'])
                print(f"  - Best pick: {best_pick['pick']} ({best_pick['confidence']:.1f}% confidence)")
                
                if 'detailed_reasoning' in best_pick and 'analysis_breakdown' in best_pick['detailed_reasoning']:
                    game_specific = best_pick.get('analysis_factors', {}).get(f'{game}_specific')
                    if game_specific:
                        print(f"  - Game-specific analysis: {game_specific.get('reasoning', 'N/A')}")
            
            print()
        
        except Exception as e:
            print(f"  - Error: {e}\n")
    
    return True


def demonstrate_picks_ledger_integration():
    """
    Demonstrate the PicksLedger integration and analytics capabilities
    """
    print("=== BetFinder AI Sport Agents with PicksLedger Integration ===\n")
    
    # Create a basketball agent as example
    agent = create_sport_agent('basketball')
    
    print(f"Created {agent.agent_type} for {agent.sport_name}")
    print(f"Learning insights available: {not agent.learning_insights.get('insufficient_data', True)}\n")
    
    # Make some picks
    print("Making picks...")
    picks = agent.make_picks()
    
    print(f"Generated {len(picks)} picks")
    
    if picks:
        # Show example pick with detailed reasoning
        example_pick = picks[0]
        print(f"\n=== Example Pick ===")
        print(f"Pick: {example_pick['pick']}")
        print(f"Confidence: {example_pick['confidence']:.1f}%")
        print(f"Expected Value: {example_pick['expected_value']:.2f}")
        print(f"Pick ID: {example_pick.get('pick_id', 'N/A')}")
        print(f"Reasoning: {example_pick['reasoning']}")
        
        if 'detailed_reasoning' in example_pick:
            print(f"\nDetailed Analysis:")
            detailed = example_pick['detailed_reasoning']
            if 'analysis_breakdown' in detailed:
                for factor, reasoning in detailed['analysis_breakdown'].items():
                    if reasoning:
                        print(f"  - {factor.title()}: {reasoning}")
        
        # Simulate updating pick outcome
        print(f"\n=== Simulating Pick Outcome Update ===")
        pick_id = example_pick.get('pick_id')
        if pick_id:
            # Simulate a win
            success = agent.update_pick_outcome(pick_id, 'won', actual_result=25.5)
            print(f"Updated pick outcome: {'Success' if success else 'Failed'}")
    
    # Show performance summary
    print(f"\n=== Performance Summary ===")
    summary = agent.get_performance_summary()
    
    if summary['performance_metrics']:
        metrics = summary['performance_metrics']
        print(f"Total Picks: {metrics.get('total_picks', 0)}")
        print(f"Win Rate: {metrics.get('win_rate', 0):.1f}%")
        print(f"Average Confidence: {metrics.get('average_confidence', 0):.1f}%")
        print(f"ROI: {metrics.get('roi', 0):.1f}%")
    
    if summary['learning_insights'] and not summary['learning_insights'].get('insufficient_data'):
        insights = summary['learning_insights']
        print(f"\nLearning Insights:")
        
        if insights.get('optimal_confidence_threshold'):
            threshold = insights['optimal_confidence_threshold']
            print(f"  - Optimal confidence threshold: {threshold['threshold']}%")
        
        if insights.get('recommendations'):
            print(f"  - Recommendations: {len(insights['recommendations'])} available")
    
    return agent, picks


# Example usage and testing
if __name__ == "__main__":
    # Demonstrate the enhanced system
    agent, picks = demonstrate_picks_ledger_integration()
    
    print(f"\n=== Testing All Agents ===")
    # Test all agents briefly
    sports = ['tennis', 'basketball', 'football', 'baseball', 'hockey', 'soccer', 'college_football', 
              'csgo', 'league_of_legends', 'dota2', 'valorant', 'overwatch', 'rocket_league']
    
    total_picks = 0
    for sport in sports:
        try:
            print(f"\nTesting {sport.title()} Agent...")
            sport_agent = create_sport_agent(sport)
            
            # Generate fewer picks for testing
            test_picks = sport_agent.make_picks()
            sport_agent.props_data = sport_agent.fetch_props(5)  # Limit props for testing
            test_picks = sport_agent.make_picks()
            
            total_picks += len(test_picks)
            print(f"  - Generated {len(test_picks)} picks")
            
            if test_picks:
                best_pick = max(test_picks, key=lambda x: x['confidence'])
                print(f"  - Best pick: {best_pick['pick']} ({best_pick['confidence']:.1f}% confidence)")
        
        except Exception as e:
            print(f"  - Error testing {sport}: {e}")
    
    print(f"\n=== Test Complete ===")
    print(f"Total picks generated across all sports: {total_picks}")
    print(f"PicksLedger integration: {'✓' if total_picks > 0 else '✗'}")
    print(f"All picks logged with detailed reasoning and analytics")
    
    # Show picks ledger summary
    try:
        from picks_ledger import picks_ledger
        all_picks_count = len(picks_ledger.picks)
        print(f"Total picks in ledger: {all_picks_count}")
        
        if all_picks_count > 0:
            print("Ledger successfully tracking all agent picks with comprehensive analytics")
    except Exception as e:
        print(f"Picks ledger summary error: {e}")