"""
Sport Agents Module for BetFinder AI

This module contains the base SportAgent class and individual sport-specific agents
for analyzing props and making picks with confidence scores. Includes comprehensive
PicksLedger integration for performance tracking and machine learning from historical results.
"""

import time
import random
import statistics
import numpy as np
import json
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Any
from abc import ABC

# Import the picks ledger for logging and analytics

def generate_half_increment_line(min_val: float, max_val: float) -> float:
    """Generate a random line value that uses only 0.5 increments."""
    # Convert to half-point scale (multiply by 2)
    min_half = int(min_val * 2)
    max_half = int(max_val * 2)
    
    # Generate random half-point value
    random_half = random.randint(min_half, max_half)
    
    # Convert back to decimal (divide by 2)
    return random_half / 2.0
from picks_ledger import picks_ledger


class PropValueMLModel:
    """Machine Learning model for predicting prop value/edge/ROI using historical data."""
    def __init__(self):
        self.weights = {
            'confidence_factor': 0.25,
            'historical_performance': 0.30,
            'stat_type_success': 0.20,
            'over_under_preference': 0.15,
            'recency_factor': 0.10
        }
        self.model_version = "1.0"
        self.last_training_time = None
        self.training_sample_size = 0
        self.feature_importance = {}
        self.model_accuracy = 0.0
        # Load existing model if available
        self.load_model()
    
    def load_model(self) -> None:
        """Load pre-trained model weights and parameters"""
        try:
            model_file = "prop_value_model.json"
            if os.path.exists(model_file):
                with open(model_file, 'r') as f:
                    model_data = json.load(f)
                    self.weights = model_data.get('weights', self.weights)
                    self.model_version = model_data.get('version', self.model_version)
                    self.last_training_time = model_data.get('last_training_time')
                    self.training_sample_size = model_data.get('training_sample_size', 0)
                    self.feature_importance = model_data.get('feature_importance', {})
                    self.model_accuracy = model_data.get('accuracy', 0.0)
                    print(f"✅ Loaded ML model v{self.model_version} (accuracy: {self.model_accuracy:.1%})")
        except Exception as e:
            print(f"⚠️ Error loading ML model: {e}")
    
    def save_model(self) -> None:
        """Save trained model weights and parameters"""
        try:
            model_data = {
                'weights': self.weights,
                'version': self.model_version,
                'last_training_time': datetime.now().isoformat(),
                'training_sample_size': self.training_sample_size,
                'feature_importance': self.feature_importance,
                'accuracy': self.model_accuracy
            }
            
            with open("prop_value_model.json", 'w') as f:
                json.dump(model_data, f, indent=2)
            
            print(f"💾 Saved ML model v{self.model_version}")
        except Exception as e:
            print(f"⚠️ Error saving ML model: {e}")
    
    def train_model(self, historical_picks: List[Dict]) -> None:
        """
        Train/update the model using historical pick outcomes.

        Args:
            historical_picks: List of historical picks with outcomes
        """
        # Cold-start tolerant training
        is_cold_start = False
        if len(historical_picks) < 10:
            print("ℹ️ ML cold-start: sparse historical data (" + str(len(historical_picks)) + " picks). Using heuristics + incremental learning.")
            is_cold_start = True
        # Filter for completed picks only
        completed_picks = [p for p in historical_picks if p.get('outcome') in ['won', 'lost']]
        if len(completed_picks) < 5:
            # Proceed in heuristic-only mode; keep existing weights, compute default metrics
            print("ℹ️ ML cold-start: insufficient completed picks; retaining baseline weights.")
            self.training_sample_size = len(completed_picks)
            self.feature_importance = {}
            self.model_accuracy = 0.0
            # Save current model state for consistency
            self.save_model()
            return
        self.training_sample_size = len(completed_picks)
        # Extract features and outcomes
        features = []
        outcomes = []
        for pick in completed_picks:
            feature_vector = self._extract_features(pick)
            outcome = 1 if pick['outcome'] == 'won' else 0
            features.append(list(feature_vector.values()))
            outcomes.append(outcome)
        # Simple logistic regression-style weight updates
        features_array = np.array(features)
        outcomes_array = np.array(outcomes)
        # Calculate feature importance
        self.feature_importance = self._calculate_feature_importance(features_array, outcomes_array)
        # Update weights based on correlation with success
        self._update_weights(features_array, outcomes_array)
        # Calculate model accuracy
        self.model_accuracy = self._calculate_accuracy(features_array, outcomes_array)
        # Save updated model
        self.save_model()
        # Report training status
        status = "(cold-start) " if is_cold_start else ""
        print(f"🤖 ML model {status}trained on {self.training_sample_size} picks (accuracy: {self.model_accuracy:.1%})")
    
    def predict_value(self, prop: Dict, agent_context: Dict = None) -> Dict[str, float]:
        """Predict value/edge/ROI for a given prop.
        Args:
            prop: Prop data dictionary
            agent_context: Additional context from the sport agent
        Returns:
            Dict with predicted_value, confidence, edge, expected_roi
        """
        try:
                # Extract features for prediction
                features = self._extract_features_for_prediction(prop, agent_context)
            
                # Calculate weighted score
                predicted_value = sum(
                    features.get(feature, 0.5) * weight 
                    for feature, weight in self.weights.items()
                )
            
                # Convert to probability
                probability = self._sigmoid(predicted_value)
            
                # Calculate edge (value over market)
                odds = prop.get('odds', -110)
                implied_probability = self._american_odds_to_probability(odds)
                edge = probability - implied_probability
            
                # Calculate expected ROI
                if edge > 0:
                    expected_roi = edge * 100  # Simple ROI calculation
                else:
                    expected_roi = edge * 50   # Penalty for negative edge
            
                return {
                    'predicted_value': round(predicted_value, 3),
                    'confidence': round(probability * 100, 1),
                    'edge': round(edge, 3),
                    'expected_roi': round(expected_roi, 2),
                    'model_version': self.model_version,
                    'prediction_time': datetime.now().isoformat()
                }
            
        except Exception as e:
            print(f"⚠️ Error in ML prediction: {e}")
            return {
                'predicted_value': 0.5,
                'confidence': 50.0,
                'edge': 0.0,
                'expected_roi': 0.0,
                'model_version': self.model_version,
                'prediction_time': datetime.now().isoformat(),
                'error': str(e)
            }
        try:
            # Extract features for prediction
            features = self._extract_features_for_prediction(prop, agent_context)
            
            # Calculate weighted score
            predicted_value = sum(
                features.get(feature, 0.5) * weight 
                for feature, weight in self.weights.items()
            )
            
            # Convert to probability
            probability = self._sigmoid(predicted_value)
            
            # Calculate edge (value over market)
            odds = prop.get('odds', -110)
            implied_probability = self._american_odds_to_probability(odds)
            edge = probability - implied_probability
            
            # Calculate expected ROI
            if edge > 0:
                expected_roi = edge * 100  # Simple ROI calculation
            else:
                expected_roi = edge * 50   # Penalty for negative edge
            
            return {
                'predicted_value': round(predicted_value, 3),
                'confidence': round(probability * 100, 1),
                'edge': round(edge, 3),
                'expected_roi': round(expected_roi, 2),
                'model_version': self.model_version,
                'prediction_time': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"⚠️ Error in ML prediction: {e}")
            return {
                'predicted_value': 0.5,
                'confidence': 50.0,
                'edge': 0.0,
                'expected_roi': 0.0,
                'model_version': self.model_version,
                'prediction_time': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def _extract_features(self, pick: Dict) -> Dict[str, float]:
        """Extract numerical features from a historical pick."""
        return {
            'confidence_factor': pick.get('confidence', 50) / 100,
            'historical_performance': 0.6,  # Will be enhanced with more data
            'stat_type_success': 0.5,      # Will be enhanced with stat-specific data
            'over_under_preference': 0.5,   # Will be enhanced with O/U analysis
            'recency_factor': 0.8           # Recent picks weighted more
        }
    
    def _extract_features_for_prediction(self, prop: Dict, agent_context: Dict = None) -> Dict[str, float]:
        """Extract features for making predictions on new props."""
        confidence = agent_context.get('confidence', 50) if agent_context else 50
        # Get historical performance for this stat type
        stat_type_performance = 0.5
        over_under_performance = 0.5
        if agent_context and agent_context.get('learning_insights'):
            insights = agent_context['learning_insights']
            # Stat type performance
            best_stats = insights.get('best_stat_types', [])
            for stat_info in best_stats:
                if stat_info.get('stat_type') == prop.get('stat_type'):
                    stat_type_performance = stat_info.get('win_rate', 50) / 100
                    break
            # Over/under preference
            over_under_pref = insights.get('best_over_under_preference')
            if over_under_pref:
                over_under_performance = over_under_pref.get('win_rate', 50) / 100
        return {
            'confidence_factor': confidence / 100,
            'historical_performance': agent_context.get('overall_win_rate', 50) / 100 if agent_context else 0.5,
            'stat_type_success': stat_type_performance,
            'over_under_preference': over_under_performance,
            'recency_factor': 0.8  # Default high recency factor
        }
    
    def _calculate_feature_importance(self, features: np.ndarray, outcomes: np.ndarray) -> Dict[str, float]:
        """Calculate feature importance using correlation with outcomes."""
        importance = {}
        feature_names = list(self.weights.keys())
        for i, feature_name in enumerate(feature_names):
            if features.shape[1] > i:
                correlation = np.corrcoef(features[:, i], outcomes)[0, 1]
                importance[feature_name] = abs(correlation) if not np.isnan(correlation) else 0.0
        return importance
    
    def _update_weights(self, features: np.ndarray, outcomes: np.ndarray) -> None:
        """Update model weights based on feature importance."""
        feature_names = list(self.weights.keys())
        for i, feature_name in enumerate(feature_names):
            if features.shape[1] > i:
                importance = self.feature_importance.get(feature_name, 0.0)
                # Adjust weights based on importance (learning rate = 0.1)
                self.weights[feature_name] = 0.9 * self.weights[feature_name] + 0.1 * importance
        # Normalize weights to sum to 1
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}
    
    def _calculate_accuracy(self, features: np.ndarray, outcomes: np.ndarray) -> float:
        """Calculate model accuracy on training data."""
        predictions = []
        for feature_vector in features:
            feature_dict = dict(zip(self.weights.keys(), feature_vector))
            predicted_value = sum(
                feature_dict.get(feature, 0.5) * weight 
                for feature, weight in self.weights.items()
            )
            probability = self._sigmoid(predicted_value)
            predictions.append(1 if probability > 0.5 else 0)
        correct = sum(p == o for p, o in zip(predictions, outcomes))
        return correct / len(outcomes) if outcomes else 0.0
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _american_odds_to_probability(self, odds: int) -> float:
        """Convert American odds to implied probability."""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)


# Global ML model instance shared across all agents
ml_model = PropValueMLModel()


class SportAgent(ABC):
    """Base class for all sport-specific agents with PicksLedger integration and machine learning capabilities from historical performance."""

    def __init__(self, sport_name: str):
        """Initialize the sport agent. Args: sport_name (str): Name of the sport this agent handles"""
        self.sport_name = sport_name
        self.agent_type = f"{sport_name}_agent"
        self.props_data = []
        self.picks = []
        self.learning_insights = {}
        self.performance_metrics = {}
        # Load historical insights and train ML model on initialization
        self.learn_from_history()
        if os.getenv('DISABLE_ML_TRAINING_IN_UI') == '1':
            print(f"Skipping ML training in UI for {self.agent_type} (DISABLE_ML_TRAINING_IN_UI=1)")
        else:
            self.train_ml_model()
    
    def train_ml_model(self) -> None:
        """Train the shared ML model using this agent's historical data."""
        try:
            # Get historical picks for this agent
            historical_picks = picks_ledger.get_picks_by_agent(self.agent_type, days_back=90)
            if len(historical_picks) >= 5:
                print(f"🤖 Training ML model for {self.agent_type} with {len(historical_picks)} historical picks")
                ml_model.train_model(historical_picks)
            else:
                print(f"⚠️ Insufficient historical data for {self.agent_type} ML training ({len(historical_picks)} picks)")
        except Exception as e:
            print(f"⚠️ Error training ML model for {self.agent_type}: {e}")
    
    def get_ml_prediction(self, prop: Dict) -> Dict[str, float]:
        """Get ML model prediction for a prop's value/edge/ROI. Args: prop: Prop data dictionary Returns: Dict with ML predictions and confidence metrics"""
        agent_context = {
            'confidence': prop.get('confidence', 50),
            'learning_insights': self.learning_insights,
            'overall_win_rate': self.performance_metrics.get('win_rate', 50)
        }
        return ml_model.predict_value(prop, agent_context)
    
    def fetch_props(self, max_props: int = 50) -> List[Dict]:
        """Fetch props exclusively from PrizePicks board (CSV produced by the scraper). No mock props. No alternate providers. Strictly PrizePicks-displayed lines. Reads CSV from PRIZEPICKS_CSV env var or 'prizepicks_props.csv'. Returns only rows that map to this agent's sport."""
        import os
        import pandas as pd

        csv_path = os.environ.get('PRIZEPICKS_CSV', 'prizepicks_props.csv')
        if not os.path.exists(csv_path):
            return []

        try:
            df = pd.read_csv(csv_path)
        except Exception:
            return []

        # Flexible column detection
        def _cols(df_):
            m = {
                'player': None,
                'line': None,
                'prop': None,
                'sport': None,
                'league': None,
                'game': None,
                'game_date': None,
                'game_time': None,
                'last_updated': None,
            }
            for c in df_.columns:
                lc = str(c).strip().lower()
                if lc in ['name', 'player', 'player_name']:
                    m['player'] = c
                elif lc in ['points', 'line', 'value']:
                    m['line'] = c
                elif lc in ['prop', 'stat', 'type']:
                    m['prop'] = c
                elif lc in ['sport', 'category']:
                    m['sport'] = c
                elif lc == 'league':
                    m['league'] = c
                elif lc in ['game', 'matchup']:
                    m['game'] = c
                elif lc == 'game_date':
                    m['game_date'] = c
                elif lc == 'game_time':
                    m['game_time'] = c
                elif lc == 'last_updated':
                    m['last_updated'] = c
            return m

        cols = _cols(df)

        sport_key = self.sport_name.lower()

        # If CSV doesn't carry sport/league metadata, reuse the UI's grouping logic
        if not cols['sport'] and not cols['league']:
            try:
                from page_utils import load_prizepicks_csv_grouped
                grouped = load_prizepicks_csv_grouped(csv_path)
                props = grouped.get(sport_key, [])
                # Ensure standard fields
                for p in props:
                    p.setdefault('sportsbook', 'PrizePicks')
                    p.setdefault('odds', -110)
                    p.setdefault('confidence', 70.0)
                    p.setdefault('expected_value', 0.0)
                if max_props and len(props) > max_props:
                    props = props[:max_props]
                return props
            except Exception:
                # Fall through to stat-only parsing below
                pass

        def _norm_sport(val: str) -> str:
            s = str(val or '').strip().lower()
            aliases = {
                'nba': 'basketball', 'wnba': 'basketball', 'cbb': 'basketball',
                'nfl': 'football', 'cfb': 'college_football', 'ncaa football': 'college_football',
                'mlb': 'baseball', 'nhl': 'hockey', 'epl': 'soccer', 'soccer': 'soccer',
                'csgo': 'csgo', 'cs:go': 'csgo', 'cs2': 'csgo', 'counter-strike': 'csgo', 'counter strike': 'csgo', 'counter-strike 2': 'csgo',
                'league of legends': 'league_of_legends', 'lol': 'league_of_legends',
                'valorant': 'valorant', 'dota2': 'dota2', 'dota 2': 'dota2',
                'overwatch': 'overwatch', 'rocket league': 'rocket_league', 'rocket_league': 'rocket_league',
            }
            return aliases.get(s, s)

        def _to_float(val):
            if val is None:
                return None
            try:
                s = str(val).strip()
                if not s:
                    return None
                # Remove common noise like '+' signs
                s = s.replace('+', '')
                return float(s)
            except Exception:
                return None

        def _parse_event_start(game_date: str, game_time_et: str) -> str:
            try:
                if not (game_date and game_time_et):
                    return ''
                t = str(game_time_et).replace('ET', '').strip()
                # Prefer 12-hour clock, fallback to 24-hour
                try:
                    dt_local = datetime.strptime(f"{game_date} {t}", "%Y-%m-%d %I:%M %p")
                except Exception:
                    dt_local = datetime.strptime(f"{game_date} {t}", "%Y-%m-%d %H:%M")
                dt_et = dt_local.replace(tzinfo=ZoneInfo('America/New_York'))
                return dt_et.astimezone(timezone.utc).isoformat().replace('+00:00', 'Z')
            except Exception:
                return ''

        def _map(row):
            player_name = str(row.get(cols['player'], '')).strip()
            line_val = _to_float(row.get(cols['line'])) if cols['line'] else None
            stat_raw = str(row.get(cols['prop'], '')).strip().lower()
            sport_raw = row.get(cols['sport']) or row.get(cols['league'])
            game_val = str(row.get(cols['game'], '')).strip() if cols['game'] else ''
            sport_norm = _norm_sport(sport_raw)
            game_date = str(row.get(cols['game_date'], '')).strip() if cols['game_date'] else ''
            game_time_et = str(row.get(cols['game_time'], '')).strip() if cols['game_time'] else ''
            last_updated = str(row.get(cols['last_updated'], '')).strip() if cols['last_updated'] else ''
            event_start = _parse_event_start(game_date, game_time_et)
            return {
                'game_id': f"pp_{hash((player_name, stat_raw, line_val, game_val)) & 0xffffffff}",
                'sport': sport_norm,
                'player_name': player_name,
                'stat_type': stat_raw,
                'line': line_val,
                'odds': -110,
                'event_start_time': event_start,
                'event_date': game_date,
                'event_time_et': game_time_et,
                'last_updated': last_updated,
                'matchup': game_val,
                'sportsbook': 'PrizePicks',
                'confidence': 70.0,
                'expected_value': 0.0,
            }

        props = []
        for _, r in df.iterrows():
            item = _map(r)
            # Require a numeric line to avoid downstream issues
            if item.get('line') is None:
                continue
            props.append(item)
        # Filter by this agent's sport
        props = [p for p in props if p.get('sport') == sport_key]
        if max_props and len(props) > max_props:
            props = props[:max_props]

        return props
    
    def _generate_mock_props(self, max_props: int) -> List[Dict]:
        """
        Deprecated: Mock props generation is disabled.
        Agents must use PrizePicks props via fetch_props().
        """
        return []
    
    def make_picks(self, props_data: Optional[List[Dict]] = None, log_to_ledger: bool = True) -> List[Dict]:
        """
        Analyze props and make over/under picks with confidence scores
        Includes detailed multi-factor reasoning, PicksLedger integration, and ML predictions
        
        Args:
            props_data (List[Dict], optional): Props data to analyze
            log_to_ledger (bool): When False, skip logging picks to the persistent ledger
            
        Returns:
            List[Dict]: List of picks with confidence scores and ML predictions
        """
        if props_data is None:
            props_data = self.fetch_props()
        
        self.props_data = props_data
        picks = []
        
        # Update learning insights; training already handled in __init__ or controlled by env
        self.learn_from_history()
        
        for prop in props_data:
            pick = self._analyze_prop_and_make_pick(prop)
            if pick:
                # Get ML prediction for this prop
                ml_prediction = self.get_ml_prediction(prop)
                pick['ml_prediction'] = ml_prediction
                
                # Enhance pick with ML insights
                pick = self._enhance_pick_with_ml(pick, ml_prediction)
                
                # Log pick to ledger and get pick_id (optional)
                if log_to_ledger:
                    pick_id = picks_ledger.log_pick(pick)
                    pick['pick_id'] = pick_id
                picks.append(pick)
        
        # Sort picks by ML-enhanced confidence and expected value
        sorted_picks = self.sort_picks_by_ml_value(picks)
        self.picks = sorted_picks
        
        return sorted_picks
    
    def _enhance_pick_with_ml(self, pick: Dict, ml_prediction: Dict) -> Dict:
        """
        Enhance pick with ML prediction insights
        
        Args:
            pick: Original pick data
            ml_prediction: ML model predictions
            
        Returns:
            Enhanced pick with ML insights
        """
        # Adjust confidence based on ML prediction
        original_confidence = pick['confidence']
        ml_confidence = ml_prediction.get('confidence', 50)
        
        # Weighted average: 60% original analysis, 40% ML prediction
        enhanced_confidence = 0.6 * original_confidence + 0.4 * ml_confidence
        
        # Update pick with ML enhancements
        pick['confidence'] = round(enhanced_confidence, 1)
        pick['ml_edge'] = ml_prediction.get('edge', 0.0)
        pick['ml_expected_roi'] = ml_prediction.get('expected_roi', 0.0)
        pick['ml_model_version'] = ml_prediction.get('model_version', 'unknown')
        
        # Add ML recommendation flags
        edge = ml_prediction.get('edge', 0.0)
        if edge > 0.05:
            pick['ml_recommendation'] = 'strong_value'
        elif edge > 0.02:
            pick['ml_recommendation'] = 'moderate_value'
        elif edge > -0.02:
            pick['ml_recommendation'] = 'fair_value'
        else:
            pick['ml_recommendation'] = 'poor_value'
        
        # Update expected value with ML insights
        original_ev = pick.get('expected_value', 0)
        ml_ev = ml_prediction.get('expected_roi', 0)
        pick['expected_value'] = round(0.7 * original_ev + 0.3 * ml_ev, 2)
        
        return pick
    
    def sort_picks_by_ml_value(self, picks: List[Dict]) -> List[Dict]:
        """
        Sort picks by ML-enhanced value metrics
        
        Args:
            picks: List of picks to sort
            
        Returns:
            Sorted list prioritizing high ML value picks
        """
        def sort_key(pick):
            ml_edge = pick.get('ml_edge', 0.0)
            confidence = pick.get('confidence', 0.0)
            expected_roi = pick.get('ml_expected_roi', 0.0)
            
            # Combined score: edge (40%) + confidence (35%) + expected ROI (25%)
            score = 0.4 * ml_edge + 0.35 * (confidence / 100) + 0.25 * (expected_roi / 100)
            return score
        
        return sorted(picks, key=sort_key, reverse=True)
    
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
            'event_date': prop.get('event_date', ''),
            'event_time_et': prop.get('event_time_et', ''),
            'last_updated': prop.get('last_updated', ''),
            'matchup': prop.get('matchup', 'TBD vs TBD'),
            'sportsbook': prop.get('sportsbook', 'Multiple'),
            'reasoning': detailed_reasoning['summary'],
            'detailed_reasoning': detailed_reasoning,
            'prizepicks_classification': detailed_reasoning.get('prizepicks_classification', {}),
            'expected_value': round(expected_value, 2),
            'analysis_factors': analysis_factors,
            'bet_amount': self._calculate_bet_size(confidence, expected_value),
            'allow_under': prop.get('allow_under', True),
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
        _stat_type = prop.get('stat_type', '')  # unused, for debugging/placeholder
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
        _base_confidence = analysis_factors['overall_score'] * 10  # unused, for debugging/placeholder
        
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
        _edge = line_value.get('edge', 0)  # unused, for debugging/placeholder
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

        # Enforce over-only props: never return 'under' if not allowed
        allow_under_flag = prop.get('allow_under')
        if allow_under_flag is False and base_decision == 'under':
            base_decision = 'over'

        return base_decision
    
    def _classify_pick_with_prizepicks_terms(self, prop: Dict, confidence: float, line_value_edge: float, over_under: str) -> Dict[str, str]:
        """
        Classify picks using PrizePicks terminology (demon, goblin, discount, etc.)
        
        Returns:
            Dict with 'classification', 'emoji', and 'description'
        """
        line = prop.get('line', 0)
        stat = prop.get('stat_type', 'stat')
        player = prop.get('player_name', 'Player')
        
        # PrizePicks classification logic
        if confidence >= 85:
            if line_value_edge > 0.15:
                demon_terms = [
                    f'Elite play - {player} {over_under} {line} {stat} is a demon lock',
                    f'Absolute unit - {player} {over_under} {line} {stat} demon mode',
                    f'Free money - {player} {over_under} {line} {stat} is cooked'
                ]
                return {
                    'classification': 'DEMON 👹',
                    'emoji': '👹',
                    'description': random.choice(demon_terms)
                }
            else:
                lock_terms = [
                    f'High confidence lock on {player} {over_under} {line} {stat}',
                    f'Bank it - {player} {over_under} {line} {stat} locked',
                    f'Chalk play - {player} {over_under} {line} {stat} easy'
                ]
                return {
                    'classification': 'LOCK 🔒',
                    'emoji': '🔒', 
                    'description': random.choice(lock_terms)
                }
        elif confidence >= 75:
            if line_value_edge > 0.10:
                discount_terms = [
                    f'Great value - {player} {over_under} {line} {stat} at a discount',
                    f'Steal alert - {player} {over_under} {line} {stat} undervalued',
                    f'Line error - {player} {over_under} {line} {stat} free money'
                ]
                return {
                    'classification': 'DISCOUNT 💰',
                    'emoji': '💰',
                    'description': random.choice(discount_terms)
                }
            else:
                solid_terms = [
                    f'Solid pick on {player} {over_under} {line} {stat}',
                    f'Good spot - {player} {over_under} {line} {stat}',
                    f'Core play - {player} {over_under} {line} {stat}'
                ]
                return {
                    'classification': 'SOLID 💪',
                    'emoji': '💪',
                    'description': random.choice(solid_terms)
                }
        elif confidence >= 65:
            # Check if it's a Tuesday for "Taco Tuesday" reference
            is_tuesday = datetime.now().weekday() == 1
            if is_tuesday:
                tuesday_terms = [
                    f'Taco Tuesday special - {player} {over_under} {line} {stat}',
                    f'Tuesday vibes - {player} {over_under} {line} {stat} hits different',
                    f'Taco time - {player} {over_under} {line} {stat} spicy play'
                ]
                return {
                    'classification': 'TACO TUESDAY 🌮',
                    'emoji': '🌮',
                    'description': random.choice(tuesday_terms)
                }
            else:
                decent_terms = [
                    f'Decent spot for {player} {over_under} {line} {stat}',
                    f'Playable - {player} {over_under} {line} {stat}',
                    f'Worth a look - {player} {over_under} {line} {stat}'
                ]
                return {
                    'classification': 'DECENT ✅',
                    'emoji': '✅',
                    'description': random.choice(decent_terms)
                }
        else:
            # Lower confidence picks get goblin treatment
            goblin_terms = [
                f'Goblin mode - risky but {player} {over_under} {line} {stat} could hit',
                f'Sweat play - {player} {over_under} {line} {stat} sketchy but possible',
                f'Degenerate special - {player} {over_under} {line} {stat} pray'
            ]
            return {
                'classification': 'GOBLIN 👺',
                'emoji': '👺',
                'description': random.choice(goblin_terms)
            }

    def _generate_detailed_reasoning(self, prop: Dict, analysis_factors: Dict, 
                                   over_under: str, confidence: float) -> Dict[str, Any]:
        """
        Generate comprehensive reasoning for the pick using PrizePicks terminology
        """
    
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
    
    def render_prop_card(self, prop: Dict, include_ml_prediction: bool = True) -> Dict[str, Any]:
        """
        Render prop card with ML value highlighting and recommendations
        
        Args:
            prop: Prop data dictionary
            include_ml_prediction: Whether to include ML predictions
            
        Returns:
            Dict with card rendering data and ML highlights
        """
        try:
            # Get ML prediction if requested
            ml_prediction = None
            if include_ml_prediction:
                ml_prediction = self.get_ml_prediction(prop)
            
            # Determine card styling based on ML prediction
            card_style = self._get_card_styling(ml_prediction)
            
            # Extract basic prop info
            player_name = prop.get('player_name', 'Unknown Player')
            stat_type = prop.get('stat_type', 'Unknown Stat')
            line = prop.get('line', 0)
            odds = prop.get('odds', -110)
            
            # Build card data
            card_data = {
                'player_name': player_name,
                'stat_type': stat_type.replace('_', ' ').title(),
                'line': line,
                'odds': odds,
                'matchup': prop.get('matchup', 'TBD vs TBD'),
                'event_time': prop.get('event_start_time', 'TBD'),
                'sportsbook': prop.get('sportsbook', 'Multiple'),
                'card_style': card_style,
                'ml_prediction': ml_prediction,
                'recommendation_badge': self._get_recommendation_badge(ml_prediction),
                'value_indicators': self._get_value_indicators(ml_prediction)
            }
            
            # Add warning flags for missing/stale predictions
            if not ml_prediction or ml_prediction.get('error'):
                card_data['warning'] = {
                    'type': 'missing_prediction',
                    'message': 'ML prediction unavailable',
                    'style': 'opacity: 0.6; border: 2px dashed #ffc107;'
                }
            elif self._is_prediction_stale(ml_prediction):
                card_data['warning'] = {
                    'type': 'stale_prediction',
                    'message': 'Prediction may be outdated',
                    'style': 'border-left: 4px solid #fd7e14;'
                }
            
            return card_data
            
        except Exception as e:
            print(f"⚠️ Error rendering prop card: {e}")
            return {
                'player_name': prop.get('player_name', 'Unknown'),
                'stat_type': prop.get('stat_type', 'Unknown'),
                'line': prop.get('line', 0),
                'odds': prop.get('odds', -110),
                'card_style': {'background': '#f8f9fa', 'border': '1px solid #dee2e6'},
                'warning': {
                    'type': 'render_error',
                    'message': 'Unable to process prop data',
                    'style': 'opacity: 0.5;'
                }
            }
    
    def _get_card_styling(self, ml_prediction: Optional[Dict]) -> Dict[str, str]:
        """Get CSS styling for prop card based on ML prediction"""
        if not ml_prediction:
            return {
                'background': '#f8f9fa',
                'border': '1px solid #dee2e6',
                'box_shadow': 'none'
            }
        
        edge = ml_prediction.get('edge', 0.0)
        confidence = ml_prediction.get('confidence', 50)
        
        if edge > 0.05 and confidence > 70:
            # Strong positive value - green highlight
            return {
                'background': 'linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%)',
                'border': '2px solid #28a745',
                'box_shadow': '0 4px 12px rgba(40, 167, 69, 0.3)',
                'animation': 'pulse-green 2s infinite'
            }
        elif edge > 0.02 and confidence > 60:
            # Moderate positive value - light green
            return {
                'background': 'linear-gradient(135deg, #e8f5e8 0%, #d4edda 100%)',
                'border': '2px solid #20c997',
                'box_shadow': '0 2px 8px rgba(32, 201, 151, 0.2)'
            }
        elif edge < -0.03 or confidence < 40:
            # Poor value - red highlight
            return {
                'background': 'linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%)',
                'border': '2px solid #dc3545',
                'box_shadow': '0 2px 8px rgba(220, 53, 69, 0.2)'
            }
        else:
            # Neutral/fair value - default styling
            return {
                'background': 'linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%)',
                'border': '1px solid #ffc107',
                'box_shadow': '0 2px 6px rgba(255, 193, 7, 0.1)'
            }
    
    def _get_recommendation_badge(self, ml_prediction: Optional[Dict]) -> Dict[str, str]:
        """Get recommendation badge based on ML prediction"""
        if not ml_prediction:
            return {'text': 'NO DATA', 'style': 'badge-secondary'}
        
        edge = ml_prediction.get('edge', 0.0)
        confidence = ml_prediction.get('confidence', 50)
        
        if edge > 0.05 and confidence > 70:
            return {'text': 'STRONG VALUE', 'style': 'badge-success'}
        elif edge > 0.02 and confidence > 60:
            return {'text': 'GOOD VALUE', 'style': 'badge-info'}
        elif edge > -0.02 and confidence > 50:
            return {'text': 'FAIR VALUE', 'style': 'badge-warning'}
        else:
            return {'text': 'POOR VALUE', 'style': 'badge-danger'}
    
    def _get_value_indicators(self, ml_prediction: Optional[Dict]) -> List[Dict]:
        """Get value indicator metrics for display"""
        if not ml_prediction:
            return []
        
        indicators = []
        
        # Edge indicator
        edge = ml_prediction.get('edge', 0.0)
        edge_color = 'success' if edge > 0.02 else 'danger' if edge < -0.02 else 'warning'
        indicators.append({
            'label': 'Edge',
            'value': f"{edge:+.1%}",
            'color': edge_color
        })
        
        # Confidence indicator
        confidence = ml_prediction.get('confidence', 50)
        conf_color = 'success' if confidence > 70 else 'warning' if confidence > 50 else 'danger'
        indicators.append({
            'label': 'ML Confidence',
            'value': f"{confidence:.0f}%",
            'color': conf_color
        })
        
        # Expected ROI indicator
        roi = ml_prediction.get('expected_roi', 0.0)
        roi_color = 'success' if roi > 5 else 'warning' if roi > 0 else 'danger'
        indicators.append({
            'label': 'Expected ROI',
            'value': f"{roi:+.1f}%",
            'color': roi_color
        })
        
        return indicators
    
    def _is_prediction_stale(self, ml_prediction: Dict) -> bool:
        """Check if ML prediction is stale (older than 1 hour)"""
        try:
            prediction_time = ml_prediction.get('prediction_time')
            if not prediction_time:
                return True

            pred_dt = datetime.fromisoformat(prediction_time.replace('Z', '+00:00'))
            age = datetime.now() - pred_dt.replace(tzinfo=None)

            return age.total_seconds() > 3600  # 1 hour threshold
        except Exception:
            return True
    def learn_from_history(self) -> None:
        """
        Analyze historical picks to identify winning patterns and adjust strategy
        Also triggers ML model training with latest data
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
    """Tennis-specific agent for analyzing tennis props with enhanced analytics."""
    
    def __init__(self):
        super().__init__("tennis")
        self.sport_specific_factors = {
            'surface_preferences': {},  # Player preferences for different surfaces
            'stamina_factors': {},      # Player endurance in longer matches
            'mental_toughness': {}      # Clutch performance metrics
        }
    
    def _generate_mock_props(self, max_props: int) -> List[Dict]:
        """Generate mock props for tennis agent (currently returns empty list)."""
        return []
    
    def _analyze_tennis_specific_factors(self, prop: Dict) -> Dict[str, Any]:
        """Analyze tennis-specific factors."""
        _surface = prop.get('surface', 'hard')  # unused, for debugging/placeholder
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
        """Enhanced analysis including tennis-specific factors."""
        # Get base analysis
        factors = super()._perform_detailed_analysis(prop)
        
        # Add tennis-specific analysis
        factors['tennis_specific'] = self._analyze_tennis_specific_factors(prop)
        
        # Recalculate overall score including tennis factors
        factor_scores = [f['score'] for f in factors.values() if isinstance(f, dict) and 'score' in f]
        factors['overall_score'] = statistics.mean(factor_scores) if factor_scores else 5.0
        
        return factors


class BasketballAgent(SportAgent):
    """Basketball-specific agent for analyzing basketball props with advanced NBA analytics."""
    
    def __init__(self):
        super().__init__("basketball")
        self.sport_specific_factors = {
            'pace_factors': {},        # Team pace impact on player stats
            'usage_rates': {},         # Player usage rate in different situations
            'matchup_advantages': {}   # Positional matchup advantages
        }
    
    def _generate_mock_props(self, max_props: int) -> List[Dict]:
        return []
    
    def _analyze_basketball_specific_factors(self, prop: Dict) -> Dict[str, Any]:
        """Analyze basketball-specific factors like pace, usage, matchups"""
        minutes_proj = prop.get('minutes_projection', 32)
        usage_rate = prop.get('usage_rate', 0.25)
        team_pace = prop.get('team_pace', 100)
        opp_def_rating = prop.get('opponent_defense_rating', 110)
        def _generate_mock_props(self, max_props: int) -> List[Dict]:
            return []

        def _analyze_basketball_specific_factors(self, prop: Dict) -> Dict[str, Any]:
            """Analyze basketball-specific factors like pace, usage, matchups."""
            _minutes_proj = prop.get('minutes_projection', 32)  # unused, for debugging/placeholder
            _usage_rate = prop.get('usage_rate', 0.25)  # unused, for debugging/placeholder
            _team_pace = prop.get('team_pace', 100)  # unused, for debugging/placeholder
            _opp_def_rating = prop.get('opponent_defense_rating', 110)  # unused, for debugging/placeholder
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
        """Enhanced analysis including basketball-specific factors and NBA history enrichment"""
        # Try to enrich prop with recent averages using nbastatpy (best-effort, safe to fail)
        try:
            from nba_history import enrich_prop_data_with_history
            enrich_prop_data_with_history(prop)
        except Exception:
            pass

        # Get base analysis
        factors = super()._perform_detailed_analysis(prop)
        
        # Add basketball-specific analysis
        factors['basketball_specific'] = self._analyze_basketball_specific_factors(prop)
        
        # Recalculate overall score
        factor_scores = [f['score'] for f in factors.values() if isinstance(f, dict) and 'score' in f]
        factors['overall_score'] = statistics.mean(factor_scores) if factor_scores else 5.0
        
        return factors

    def _analyze_historical_performance(self, prop: Dict) -> Dict[str, Any]:
        """Analyze historical performance using nbastatpy game logs when available.

        Falls back to base implementation on failure.
        """
        stat_type = str(prop.get('stat_type', '')).lower()
        line = prop.get('line', 0) or 0
        player_name = prop.get('player_name') or prop.get('player')

        # Map common PrizePicks stat labels to NBA stat columns
        stat_map = {
            'points': 'PTS', 'pts': 'PTS',
            'assists': 'AST', 'ast': 'AST',
            'rebounds': 'REB', 'reb': 'REB',
            'steals': 'STL', 'stl': 'STL',
            'blocks': 'BLK', 'blk': 'BLK',
            '3pt': 'FG3M', '3pt made': 'FG3M', 'three': 'FG3M', '3pm': 'FG3M',
            'pra': None, 'pts+rebs+asts': None,  # Composite not directly in logs
        }

        try:
            # If no player name or unsupported stat, defer to base logic
            if not player_name:
                raise ValueError('missing player')

            # Choose the best-matching stat column
            stat_col = None
            for key, col in stat_map.items():
                if key in stat_type:
                    stat_col = col
                    break

            if stat_col is None:
                # For composites, attempt to compute from components if possible
                composite = None
                if 'pra' in stat_type or 'pts+rebs+asts' in stat_type:
                    composite = ('PTS', 'REB', 'AST')
                # Load logs and compute accordingly
                from nba_history import get_player_history
                data = get_player_history(player_name)
                logs = data.get('game_logs')
                if logs is None or len(logs) == 0:
                    raise ValueError('no logs')

                # Ensure most recent games are last; try common date columns
                date_cols = [c for c in ['GAME_DATE', 'Game Date', 'DATE', 'Date'] if c in logs.columns]
                if date_cols:
                    logs = logs.sort_values(by=date_cols[0])

                if composite:
                    # Compute PRA per game
                    comp_cols = [c for c in composite if c in logs.columns]
                    if len(comp_cols) == 3:
                        series = logs[comp_cols[0]] + logs[comp_cols[1]] + logs[comp_cols[2]]
                    else:
                        raise ValueError('missing composite columns')
                else:
                    raise ValueError('unsupported stat type')

                season_avg = float(series.mean()) if len(series) else 0.0
                l10_avg = float(series.tail(10).mean()) if len(series) else 0.0
                over_rate = float((series > float(line)).mean()) if len(series) else 0.0

                score = 5.0
                try:
                    if line:
                        if season_avg > line:
                            score += min(3.0, (season_avg - line) / max(1e-6, line) * 3)
                        else:
                            score -= min(3.0, (line - season_avg) / max(1e-6, line) * 3)
                        if l10_avg > line:
                            score += 1
                        else:
                            score -= 1
                except Exception:
                    pass

                return {
                    'season_average': round(season_avg, 2),
                    'l10_average': round(l10_avg, 2),
                    'vs_opponent_average': None,
                    'over_rate': round(over_rate, 2),
                    'score': max(1, min(10, score)),
                    'reasoning': f"Hist avg {season_avg:.1f}, L10 {l10_avg:.1f} vs line {line}",
                }

            # Load logs for simple single-stat analysis
            from nba_history import get_player_history
            data = get_player_history(player_name)
            logs = data.get('game_logs')
            if logs is None or len(logs) == 0 or stat_col not in logs.columns:
                raise ValueError('no logs or stat col')

            # Sort chronologically if possible
            date_cols = [c for c in ['GAME_DATE', 'Game Date', 'DATE', 'Date'] if c in logs.columns]
            if date_cols:
                logs = logs.sort_values(by=date_cols[0])

            series = logs[stat_col]

            # Opponent-specific average if we can parse matchup/opponent
            vs_opponent_avg = None
            try:
                # Common columns: 'MATCHUP' like 'LAL @ BOS' or separate 'OPP'
                if 'OPP' in logs.columns:
                    opp_series = logs['OPP']
                elif 'MATCHUP' in logs.columns:
                    # Extract opponent from matchup: 'LAL vs BOS' -> 'BOS'
                    opp_series = logs['MATCHUP'].astype(str).str.extract(r"[vV]s\s+([A-Z]{2,3})|@\s+([A-Z]{2,3})").ffill(axis=1).iloc[:, 0]
                else:
                    opp_series = None
                if opp_series is not None and isinstance(opp_series, type(series)):
                    matchup = str(prop.get('matchup', ''))
                    # Try to find 3-letter code in matchup
                    import re
                    m = re.findall(r"\b[A-Z]{2,3}\b", matchup.upper())
                    target_opp = m[-1] if m else None
                    if target_opp:
                        vs_mask = opp_series.astype(str).str.upper().str.contains(target_opp)
                        if vs_mask.any():
                            vs_opponent_avg = float(series[vs_mask].mean())
            except Exception:
                pass

            season_avg = float(series.mean()) if len(series) else 0.0
            l10_avg = float(series.tail(10).mean()) if len(series) else 0.0
            over_rate = float((series > float(line)).mean()) if len(series) else 0.0

            score = 5.0
            try:
                if line:
                    if season_avg > line:
                        score += min(3.0, (season_avg - line) / max(1e-6, line) * 3)
                    else:
                        score -= min(3.0, (line - season_avg) / max(1e-6, line) * 3)
                    if l10_avg > line:
                        score += 1
                    else:
                        score -= 1
            except Exception:
                pass

            return {
                'season_average': round(season_avg, 2),
                'l10_average': round(l10_avg, 2),
                'vs_opponent_average': round(vs_opponent_avg, 2) if vs_opponent_avg is not None else None,
                'over_rate': round(over_rate, 2),
                'score': max(1, min(10, score)),
                'reasoning': f"Hist avg {season_avg:.1f}, L10 {l10_avg:.1f} vs line {line}",
            }
        except Exception:
            # Fallback to simulated analysis if history fetch fails
            return super()._analyze_historical_performance(prop)


class FootballAgent(SportAgent):
    """Football-specific agent for analyzing NFL props with advanced analytics."""
    
    def __init__(self):
        super().__init__("football")
        self.sport_specific_factors = {
            'weather_impact': {},      # Weather effects on different stats
            'target_share': {},        # Receiver target share analysis
            'red_zone_efficiency': {}  # Red zone usage patterns
        }
    
    def _generate_mock_props(self, max_props: int) -> List[Dict]:
        return []
    
    def _analyze_football_specific_factors(self, prop: Dict) -> Dict[str, Any]:
        """Analyze football-specific factors"""
        stat_type = prop.get('stat_type', '')
        def _generate_mock_props(self, max_props: int) -> List[Dict]:
            return []

        def _analyze_football_specific_factors(self, prop: Dict) -> Dict[str, Any]:
            """Analyze football-specific factors."""
            _stat_type = prop.get('stat_type', '')  # unused, for debugging/placeholder
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
    """Baseball-specific agent for analyzing baseball props."""
    
    def __init__(self):
        super().__init__("baseball")
    
    def _generate_mock_props(self, max_props: int) -> List[Dict]:
        return []
        def _generate_mock_props(self, max_props: int) -> List[Dict]:
            return []


class HockeyAgent(SportAgent):
    """Hockey-specific agent for analyzing hockey props."""
    
    def __init__(self):
        super().__init__("hockey")
    
    def _generate_mock_props(self, max_props: int) -> List[Dict]:
        return []
        def _generate_mock_props(self, max_props: int) -> List[Dict]:
            return []


class SoccerAgent(SportAgent):
    """Soccer-specific agent for analyzing soccer props."""
    
    def __init__(self):
        super().__init__("soccer")
    
    def _generate_mock_props(self, max_props: int) -> List[Dict]:
        return []
        def _generate_mock_props(self, max_props: int) -> List[Dict]:
            return []


class EsportsAgent(SportAgent):
    """Base esports agent for analyzing esports props."""
    
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
        return []
        def _generate_mock_props(self, max_props: int) -> List[Dict]:
            return []
    
    def _analyze_esports_specific_factors(self, prop: Dict) -> Dict[str, Any]:
        """Analyze CSGO-specific factors"""
        map_name = prop.get('map', 'unknown')
        side_pref = prop.get('side_preference', 'balanced')
        team_chemistry = prop.get('team_chemistry', 0.8)
        map_performance = prop.get('recent_map_performance', 0.6)
        def _generate_mock_props(self, max_props: int) -> List[Dict]:
            return []

        def _analyze_esports_specific_factors(self, prop: Dict) -> Dict[str, Any]:
            """Analyze CSGO-specific factors."""
            _map_name = prop.get('map', 'unknown')  # unused, for debugging/placeholder
            _side_pref = prop.get('side_preference', 'balanced')  # unused, for debugging/placeholder
            _team_chemistry = prop.get('team_chemistry', 0.8)  # unused, for debugging/placeholder
            _map_performance = prop.get('recent_map_performance', 0.6)  # unused, for debugging/placeholder
        
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
        """Enhanced analysis including CSGO-specific factors and CS2 context"""
        # Try to enrich prop with CS2 live context (best-effort)
        try:
            from cs2_data import enrich_csgo_prop_with_cs2
            enrich_csgo_prop_with_cs2(prop)
        except Exception:
            pass

        # Get base analysis
        factors = super()._perform_detailed_analysis(prop)
        
        # Add CSGO-specific analysis
        factors['csgo_specific'] = self._analyze_esports_specific_factors(prop)

        # Incorporate CS2 context factor
        try:
            status = prop.get('cs2_match_status')
            if status == 'live':
                # Slight boost if match is live (more volatility/volume for certain props)
                factors['cs2_context'] = {'status': status, 'score': 0.5, 'reasoning': 'Live match context via cs2api'}
            elif status:
                factors['cs2_context'] = {'status': status, 'score': 0.0, 'reasoning': 'Scheduled match context via cs2api'}
        except Exception:
            pass
        
        # Recalculate overall score
        factor_scores = [f['score'] for f in factors.values() if isinstance(f, dict) and 'score' in f]
        factors['overall_score'] = statistics.mean(factor_scores) if factor_scores else 5.0
        
        return factors


class LeagueOfLegendsAgent(EsportsAgent):
    """League of Legends specific agent"""
    
    def __init__(self):
        super().__init__("league_of_legends")
    
    def _generate_mock_props(self, max_props: int) -> List[Dict]:
        return []
    
    def _analyze_esports_specific_factors(self, prop: Dict) -> Dict[str, Any]:
        """Analyze League of Legends specific factors"""
        role = prop.get('role', 'unknown')
        def _generate_mock_props(self, max_props: int) -> List[Dict]:
            return []

        def _analyze_esports_specific_factors(self, prop: Dict) -> Dict[str, Any]:
            """Analyze League of Legends specific factors."""
            _role = prop.get('role', 'unknown')  # unused, for debugging/placeholder
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
        return []
        def _generate_mock_props(self, max_props: int) -> List[Dict]:
            return []
    
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
        return []
    
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
        return []
    
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


class GolfAgent(SportAgent):
    """Golf specific agent"""
    
    def __init__(self):
        super().__init__("golf")
    
    def _generate_mock_props(self, max_props: int) -> List[Dict]:
        return []
    
    def _analyze_golf_specific_factors(self, prop: Dict) -> Dict[str, Any]:
        """Analyze Golf specific factors"""
        driving_accuracy = prop.get('driving_accuracy', 0.75)
        putting_average = prop.get('putting_average', 1.9)
        course_history = prop.get('course_history', 0.75)
        
        score = 5.0
        
        # Driving accuracy impact
        score += (driving_accuracy - 0.75) * 3
        
        # Putting impact (lower is better)
        score += (1.9 - putting_average) * 2
        
        # Course history impact
        score += (course_history - 0.75) * 2
        
        return {
            'driving_accuracy': driving_accuracy,
            'putting_average': putting_average,
            'course_history': course_history,
            'score': max(1, min(10, score)),
            'reasoning': f"Driving: {driving_accuracy:.1%}, Putting: {putting_average:.1f}, History: {course_history:.1%}"
        }
    
    def _perform_detailed_analysis(self, prop: Dict) -> Dict[str, Any]:
        """Enhanced analysis including Golf specific factors"""
        factors = super()._perform_detailed_analysis(prop)
        factors['golf_specific'] = self._analyze_golf_specific_factors(prop)
        factor_scores = [f['score'] for f in factors.values() if isinstance(f, dict) and 'score' in f]
        factors['overall_score'] = statistics.mean(factor_scores) if factor_scores else 5.0
        return factors


class CollegeFootballAgent(SportAgent):
    """College Football-specific agent for analyzing college football props"""
    
    def __init__(self):
        super().__init__("college_football")
    
    def _generate_mock_props(self, max_props: int) -> List[Dict]:
        return []


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
        'golf': GolfAgent
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
        print("\n=== Example Pick ===")
        print(f"Pick: {example_pick['pick']}")
        print(f"Confidence: {example_pick['confidence']:.1f}%")
        print(f"Expected Value: {example_pick['expected_value']:.2f}")
        print(f"Pick ID: {example_pick.get('pick_id', 'N/A')}")
        print(f"Reasoning: {example_pick['reasoning']}")
        
        if 'detailed_reasoning' in example_pick:
            print("\nDetailed Analysis:")
            detailed = example_pick['detailed_reasoning']
            if 'analysis_breakdown' in detailed:
                for factor, reasoning in detailed['analysis_breakdown'].items():
                    if reasoning:
                        print(f"  - {factor.title()}: {reasoning}")
        
        # Simulate updating pick outcome
        print("\n=== Simulating Pick Outcome Update ===")
        pick_id = example_pick.get('pick_id')
        if pick_id:
            # Simulate a win
            success = agent.update_pick_outcome(pick_id, 'won', actual_result=25.5)
            print(f"Updated pick outcome: {'Success' if success else 'Failed'}")
    
    # Show performance summary
    print("\n=== Performance Summary ===")
    summary = agent.get_performance_summary()
    
    if summary['performance_metrics']:
        metrics = summary['performance_metrics']
        print(f"Total Picks: {metrics.get('total_picks', 0)}")
        print(f"Win Rate: {metrics.get('win_rate', 0):.1f}%")
        print(f"Average Confidence: {metrics.get('average_confidence', 0):.1f}%")
        print(f"ROI: {metrics.get('roi', 0):.1f}%")
    
    if summary['learning_insights'] and not summary['learning_insights'].get('insufficient_data'):
        insights = summary['learning_insights']
        print("\nLearning Insights:")
        
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
    
    print("\n=== Testing All Agents ===")
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
    
    print("\n=== Test Complete ===")
    print(f"Total picks generated across all sports: {total_picks}")
    print(f"PicksLedger integration: {'✓' if total_picks > 0 else '✗'}")
    print("All picks logged with detailed reasoning and analytics")
    
    # Show picks ledger summary
    try:
        from picks_ledger import picks_ledger
        all_picks_count = len(picks_ledger.picks)
        print(f"Total picks in ledger: {all_picks_count}")
        
        if all_picks_count > 0:
            print("Ledger successfully tracking all agent picks with comprehensive analytics")
    except Exception as e:
        print(f"Picks ledger summary error: {e}")

class ApexAgent(EsportsAgent):
    """Apex Legends specific agent"""
    
    def __init__(self):
        super().__init__("apex")
    
    def _generate_mock_props(self, max_props: int) -> List[Dict]:
        return []
    
    def _analyze_esports_specific_factors(self, prop: Dict) -> Dict[str, Any]:
        """Analyze Apex Legends specific factors"""
        legend_pick = prop.get('legend', 'unknown')
        team_composition = prop.get('team_comp', 'balanced')
        map_rotation = prop.get('current_map', 'unknown')
        ring_strategy = prop.get('ring_positioning', 'edge')
        playstyle = prop.get('playstyle', 'balanced')
        
        score = 5.0
        
        # Legend meta alignment
        meta_legends = ['Wraith', 'Bloodhound', 'Lifeline', 'Octane', 'Pathfinder']
        if legend_pick in meta_legends:
            score += 0.8
        elif legend_pick != 'unknown':
            score += 0.3  # Any specific legend is better than unknown
        
        # Team composition synergy
        if team_composition == 'meta':
            score += 0.7
        elif team_composition == 'balanced':
            score += 0.3
        
        # Map-specific performance
        if map_rotation != 'unknown':
            score += 0.5
        
        # Playstyle vs stat type alignment
        stat_type = prop.get('stat_type', '').lower()
        if 'kills' in stat_type:
            if playstyle == 'aggressive':
                score += 1.0
            elif playstyle == 'passive':
                score -= 0.5
        elif 'damage' in stat_type:
            if playstyle in ['aggressive', 'balanced']:
                score += 0.5
        elif 'placement' in stat_type:
            if playstyle == 'passive':
                score += 0.8
        
        return {
            'legend': legend_pick,
            'team_composition': team_composition,
            'current_map': map_rotation,
            'ring_strategy': ring_strategy,
            'playstyle': playstyle,
            'score': max(1, min(10, score)),
            'reasoning': f"Legend: {legend_pick}, Comp: {team_composition}, Style: {playstyle}, Map: {map_rotation}"
        }
    
    def _perform_detailed_analysis(self, prop: Dict) -> Dict[str, Any]:
        """Enhanced analysis including Apex-specific factors"""
        # Get base analysis
        factors = super()._perform_detailed_analysis(prop)
        
        # Add Apex-specific analysis
        factors['apex_specific'] = self._analyze_esports_specific_factors(prop)
        
        # Recalculate overall score
        factor_scores = [f['score'] for f in factors.values() if isinstance(f, dict) and 'score' in f]
        factors['overall_score'] = statistics.mean(factor_scores) if factor_scores else 5.0
        
        return factors