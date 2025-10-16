# src/core/sport_agents.py
import time
import random
import statistics
import numpy as np
import json
import os
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Any
from abc import ABC

# Configure logging
logging.basicConfig(level=logging.INFO)

# Import the picks ledger for logging and analytics
from .picks_ledger import picks_ledger


class PropValueMLModel:
    _last_successful_prop = None
    """AutoML-powered model for predicting prop value/edge/ROI using historical data"""

    def __init__(self):
        # Import AutoML engine
        try:
            from .automl_engine import AutoMLEngine
            self.automl = AutoMLEngine("global")
            self.use_automl = True
        except Exception as e:
            print(f"⚠️ AutoML unavailable, using fallback: {e}")
            self.automl = None
            self.use_automl = False
        
        # Fallback weights for when AutoML is unavailable
        self.weights = {
            'confidence_factor': 0.25,
            'historical_performance': 0.30,
            'stat_type_success': 0.20,
            'over_under_preference': 0.15,
            'recency_factor': 0.10,
        }
        self.model_version = "automl_2.0" if self.use_automl else "weights_1.0"
        self.last_training_time: Optional[str] = None
        self.training_sample_size: int = 0
        self.feature_importance: Dict[str, float] = {}
        self.model_accuracy: float = 0.0

        # Load existing model if available
        if not self.use_automl:
            self.load_model()

    def load_model(self) -> None:
        """Load pre-trained model weights and parameters"""
        try:
            model_file = "data/prop_value_model.json"
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
                'accuracy': self.model_accuracy,
            }

            with open("data/prop_value_model.json", 'w') as f:
                json.dump(model_data, f, indent=2)

            print(f"💾 Saved ML model v{self.model_version}")
        except Exception as e:
            print(f"⚠️ Error saving ML model: {e}")

    def train_model(self, historical_picks: List[Dict]) -> None:
        """Train/update the model using historical pick outcomes"""
        if self.use_automl and self.automl:
            # Use AutoML training
            success = self.automl.train_model(historical_picks)
            if success:
                self.training_sample_size = self.automl.training_sample_size
                self.model_accuracy = self.automl.model_accuracy
                self.last_training_time = self.automl.last_training_time
                print(f"🤖 AutoML training complete: {self.training_sample_size} picks, {self.model_accuracy:.1%} accuracy")
                return
            else:
                print("⚠️ AutoML training failed, falling back to weights-based approach")
                self.use_automl = False
        
        # Fallback to original weights-based training
        is_cold_start = False
        if len(historical_picks) < 10:
            print(f"ℹ️ ML cold-start: sparse historical data ({len(historical_picks)} picks).")
            is_cold_start = True

        completed_picks = [p for p in historical_picks if p.get('outcome') in ['won', 'lost']]

        if len(completed_picks) < 5:
            print("ℹ️ ML cold-start: insufficient completed picks; retaining baseline weights.")
            self.training_sample_size = len(completed_picks)
            self.feature_importance = {}
            self.model_accuracy = 0.0
            self.save_model()
            return

        self.training_sample_size = len(completed_picks)

        features: List[List[float]] = []
        outcomes: List[int] = []

        for pick in completed_picks:
            feature_vector_dict = self._extract_features(pick)
            feature_vector = [feature_vector_dict.get(k, 0.5) for k in self.weights.keys()]
            outcome = 1 if pick['outcome'] == 'won' else 0
            features.append(feature_vector)
            outcomes.append(outcome)

        features_array = np.array(features)
        outcomes_array = np.array(outcomes)

        self.feature_importance = self._calculate_feature_importance(features_array, outcomes_array)
        self._update_weights(features_array, outcomes_array)
        self.model_accuracy = self._calculate_accuracy(features_array, outcomes_array)
        self.save_model()

        status = "(cold-start) " if is_cold_start else ""
        print(f"🤖 ML model {status}trained on {self.training_sample_size} picks (accuracy: {self.model_accuracy:.1%})")

    def predict_value(self, prop: Dict, agent_context: Dict = None) -> Dict[str, float]:
        logger = logging.getLogger("PropValueMLModel")
        MAX_RETRIES = 10
        RETRY_DELAY = 2.0
        for attempt in range(MAX_RETRIES):
            try:
                if self.use_automl and self.automl:
                    features = {
                        'confidence': prop.get('confidence', 50),
                        'line': prop.get('line', 0),
                        'odds': prop.get('odds', -110),
                        'stat_type': prop.get('stat_type', ''),
                        'over_under': prop.get('over_under', ''),
                        'sport': prop.get('sport', ''),
                    }
                    automl_result = self.automl.predict(features)
                    probability = automl_result.get('predicted_probability', 0.5)
                    odds = prop.get('odds', -110)
                    implied_probability = self._american_odds_to_probability(odds)
                    edge = probability - implied_probability
                    expected_roi = edge * 100 if edge > 0 else edge * 50
                    result = {
                        'predicted_value': round(probability, 3),
                        'confidence': round(probability * 100, 1),
                        'edge': round(edge, 3),
                        'expected_roi': round(expected_roi, 2),
                        'model_version': self.model_version,
                        'model_type': automl_result.get('model_type', 'automl'),
                        'features_used': automl_result.get('features_used', 0),
                        'prediction_time': datetime.now().isoformat(),
                    }
                    PropValueMLModel._last_successful_prop = result
                    return result

                features = self._extract_features_for_prediction(prop, agent_context)
                predicted_value = sum(
                    features.get(feature, 0.5) * weight
                    for feature, weight in self.weights.items()
                )
                probability = self._sigmoid(predicted_value)
                odds = prop.get('odds', -110)
                implied_probability = self._american_odds_to_probability(odds)
                edge = probability - implied_probability
                expected_roi = edge * 100 if edge > 0 else edge * 50
                result = {
                    'predicted_value': round(predicted_value, 3),
                    'confidence': round(probability * 100, 1),
                    'edge': round(edge, 3),
                    'expected_roi': round(expected_roi, 2),
                    'model_version': self.model_version,
                    'model_type': 'weights',
                    'prediction_time': datetime.now().isoformat(),
                }
                PropValueMLModel._last_successful_prop = result
                return result
            except Exception as e:
                logger.warning(f"Retry {attempt+1}/{MAX_RETRIES} fetching real data for prediction: {e}")
                time.sleep(RETRY_DELAY)
        logger.warning(f"FALLBACK ACTIVATED: Using last successful prop {PropValueMLModel._last_successful_prop} after {MAX_RETRIES} retries for {prop.get('player_name', 'Unknown')}")
        return PropValueMLModel._last_successful_prop or {
            'predicted_value': 0.5, 'confidence': 50.0, 'edge': 0.0, 'expected_roi': 0.0,
            'model_version': self.model_version, 'prediction_time': datetime.now().isoformat(),
            'error': 'No successful prediction available.'
        }

    def _extract_features(self, pick: Dict) -> Dict[str, float]:
        return {
            'confidence_factor': float(pick.get('confidence', 50)) / 100.0,
            'historical_performance': 0.6,
            'stat_type_success': 0.5,
            'over_under_preference': 0.5,
            'recency_factor': 0.8,
        }

    def _extract_features_for_prediction(self, prop: Dict, agent_context: Dict = None) -> Dict[str, float]:
        confidence = agent_context.get('confidence', 50) if agent_context else 50
        stat_type_performance = 0.5
        over_under_performance = 0.5

        if agent_context and agent_context.get('learning_insights'):
            insights = agent_context['learning_insights']
            best_stats = insights.get('best_stat_types', [])
            for stat_info in best_stats:
                if stat_info.get('stat_type') == prop.get('stat_type'):
                    stat_type_performance = float(stat_info.get('win_rate', 50)) / 100.0
                    break
            over_under_pref = insights.get('best_over_under_preference')
            if over_under_pref:
                over_under_performance = float(over_under_pref.get('win_rate', 50)) / 100.0

        return {
            'confidence_factor': float(confidence) / 100.0,
            'historical_performance': float(agent_context.get('overall_win_rate', 50)) / 100.0 if agent_context else 0.5,
            'stat_type_success': stat_type_performance,
            'over_under_preference': over_under_performance,
            'recency_factor': 0.8,
        }

    def _calculate_feature_importance(self, features: np.ndarray, outcomes: np.ndarray) -> Dict[str, float]:
        importance: Dict[str, float] = {}
        if features.size == 0 or outcomes.size == 0 or features.ndim != 2:
            return importance
        for i, feature_name in enumerate(self.weights.keys()):
            if features.shape[1] > i:
                try:
                    correlation = np.corrcoef(features[:, i], outcomes)[0, 1]
                    importance[feature_name] = abs(correlation) if not np.isnan(correlation) else 0.0
                except Exception:
                    importance[feature_name] = 0.0
        return importance

    def _update_weights(self, features: np.ndarray, outcomes: np.ndarray) -> None:
        if not self.feature_importance:
            return
        for feature_name in self.weights.keys():
            importance = self.feature_importance.get(feature_name, 0.0)
            self.weights[feature_name] = 0.9 * self.weights[feature_name] + 0.1 * importance
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}

    def _calculate_accuracy(self, features: np.ndarray, outcomes: np.ndarray) -> float:
        if features.size == 0 or outcomes.size == 0:
            return 0.0
        predictions: List[int] = []
        for row in features:
            feature_dict = dict(zip(self.weights.keys(), row))
            predicted_value = sum(feature_dict.get(feature, 0.5) * weight for feature, weight in self.weights.items())
            probability = self._sigmoid(predicted_value)
            predictions.append(1 if probability > 0.5 else 0)
        correct = int(sum(p == o for p, o in zip(predictions, outcomes)))
        return float(correct) / float(len(outcomes)) if len(outcomes) else 0.0

    def _sigmoid(self, x: float) -> float:
        return float(1 / (1 + np.exp(-np.clip(x, -500, 500))))

    def _american_odds_to_probability(self, odds: int) -> float:
        if odds > 0:
            return 100.0 / (float(odds) + 100.0)
        else:
            return float(abs(odds)) / (float(abs(odds)) + 100.0)
