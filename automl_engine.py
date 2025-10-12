"""
AutoML Engine for BetFinder AI Agents

Provides automatic machine learning capabilities using auto-machine-learning package.
Agents can train, select, and predict with best-in-class models automatically.

Key features:
- Automatic model selection and hyperparameter tuning
- Data preparation from picks_ledger format
- Model persistence and loading
- Safe fallbacks when AutoML is disabled or unavailable

Environment toggles:
- DISABLE_AUTOML=1 to bypass AutoML and use simple fallbacks
"""
from __future__ import annotations

import os
import json
import pickle
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

DISABLE = os.getenv("DISABLE_AUTOML", "0") == "1"

try:
    from auto_machine_learning.supervised import SupervisedLearning
    _HAS_AUTOML = True
except Exception as e:
    SupervisedLearning = None  # type: ignore
    _HAS_AUTOML = False


class AutoMLEngine:
    """AutoML wrapper for agent prop prediction and outcome classification"""
    
    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        self.model = None
        self.feature_columns: List[str] = []
        self.target_column: str = "hit"
        self.model_version = "automl_1.0"
        self.last_training_time: Optional[str] = None
        self.training_sample_size = 0
        self.model_accuracy = 0.0
        self.model_file = f"automl_{agent_type}_model.pkl"
        self.metadata_file = f"automl_{agent_type}_metadata.json"
        
        # Load existing model if available
        self.load_model()
    
    def prepare_data_from_picks(self, picks: List[Dict]) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Convert picks_ledger format to ML-ready DataFrame"""
        if not picks:
            return pd.DataFrame(), None
        
        # Convert to DataFrame
        df = pd.DataFrame(picks)
        
        # Create target variable (hit = 1 for won, 0 for lost)
        if 'outcome' in df.columns:
            df['hit'] = df['outcome'].map({'won': 1, 'lost': 0})
            # Filter only completed picks
            df = df[df['hit'].notna()].copy()
        else:
            # No outcomes available - return features only
            return df, None
        
        if df.empty:
            return pd.DataFrame(), None
        
        # Extract features
        feature_cols = []
        
        # Numeric features
        if 'confidence' in df.columns:
            df['confidence_norm'] = pd.to_numeric(df['confidence'], errors='coerce') / 100.0
            feature_cols.append('confidence_norm')
        
        if 'line' in df.columns:
            df['line_value'] = pd.to_numeric(df['line'], errors='coerce')
            feature_cols.append('line_value')
        
        if 'odds' in df.columns:
            df['odds_value'] = pd.to_numeric(df['odds'], errors='coerce')
            feature_cols.append('odds_value')
        
        # Categorical features (one-hot encoded)
        categorical_cols = ['stat_type', 'over_under', 'sport']
        for col in categorical_cols:
            if col in df.columns:
                # One-hot encode
                dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df = pd.concat([df, dummies], axis=1)
                feature_cols.extend(dummies.columns.tolist())
        
        # Time-based features
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            feature_cols.extend(['hour', 'day_of_week'])
        
        # Store feature columns for later use
        self.feature_columns = feature_cols
        
        # Return features and target
        X = df[feature_cols].fillna(0)  # Fill NaN with 0
        y = df[['hit']]
        
        return X, y
    
    def train_model(self, historical_picks: List[Dict]) -> bool:
        """Train AutoML model on historical picks data"""
        if DISABLE or not _HAS_AUTOML:
            print(f"[AutoML] Disabled or unavailable for {self.agent_type}")
            return False
        
        print(f"[AutoML] Preparing data for {self.agent_type}...")
        X, y = self.prepare_data_from_picks(historical_picks)
        
        if X is None or X.empty or y is None or y.empty:
            print(f"[AutoML] Insufficient data for {self.agent_type} (need completed picks)")
            return False
        
        if len(X) < 10:
            print(f"[AutoML] Insufficient samples for {self.agent_type} ({len(X)} picks)")
            return False
        
        self.training_sample_size = len(X)
        
        try:
            print(f"[AutoML] Training model for {self.agent_type} with {len(X)} samples...")
            
            # Initialize AutoML
            automl = SupervisedLearning(X, y.values.ravel())
            
            # Get best model (auto-selects, tunes, and optimizes)
            self.model = automl.get_best_model()
            
            # Calculate accuracy on training data (basic metric)
            y_pred = self.model.predict(X)
            self.model_accuracy = float(np.mean(y_pred == y.values.ravel()))
            
            self.last_training_time = datetime.now().isoformat()
            
            # Save model
            self.save_model()
            
            print(f"[AutoML] Training complete for {self.agent_type} (accuracy: {self.model_accuracy:.2%})")
            return True
            
        except Exception as e:
            print(f"[AutoML] Training failed for {self.agent_type}: {e}")
            return False
    
    def predict(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Make prediction on new prop using trained AutoML model"""
        if DISABLE or not _HAS_AUTOML or self.model is None:
            # Fallback to simple heuristic
            confidence = features.get('confidence', 50)
            return {
                'predicted_probability': confidence / 100.0,
                'confidence': confidence,
                'model_type': 'fallback',
                'features_used': 0
            }
        
        try:
            # Convert features to DataFrame format expected by model
            feature_df = pd.DataFrame([features])
            
            # Ensure all training features are present
            for col in self.feature_columns:
                if col not in feature_df.columns:
                    feature_df[col] = 0
            
            # Select only training features in correct order
            X_pred = feature_df[self.feature_columns].fillna(0)
            
            # Make prediction
            if hasattr(self.model, 'predict_proba'):
                proba = self.model.predict_proba(X_pred)[0]
                if len(proba) > 1:
                    predicted_prob = float(proba[1])  # Probability of class 1 (hit)
                else:
                    predicted_prob = float(proba[0])
            else:
                pred = self.model.predict(X_pred)[0]
                predicted_prob = float(pred)
            
            return {
                'predicted_probability': predicted_prob,
                'confidence': predicted_prob * 100,
                'model_type': 'automl',
                'features_used': len(self.feature_columns),
                'model_version': self.model_version,
                'training_samples': self.training_sample_size
            }
            
        except Exception as e:
            print(f"[AutoML] Prediction failed for {self.agent_type}: {e}")
            # Fallback
            confidence = features.get('confidence', 50)
            return {
                'predicted_probability': confidence / 100.0,
                'confidence': confidence,
                'model_type': 'fallback_error',
                'error': str(e)
            }
    
    def save_model(self) -> None:
        """Save trained AutoML model and metadata"""
        if self.model is None:
            return
        
        try:
            # Save model
            with open(self.model_file, 'wb') as f:
                pickle.dump(self.model, f)
            
            # Save metadata
            metadata = {
                'agent_type': self.agent_type,
                'feature_columns': self.feature_columns,
                'target_column': self.target_column,
                'model_version': self.model_version,
                'last_training_time': self.last_training_time,
                'training_sample_size': self.training_sample_size,
                'model_accuracy': self.model_accuracy
            }
            
            with open(self.metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"[AutoML] Saved model for {self.agent_type}")
            
        except Exception as e:
            print(f"[AutoML] Error saving model for {self.agent_type}: {e}")
    
    def load_model(self) -> bool:
        """Load previously trained AutoML model"""
        if not os.path.exists(self.model_file) or not os.path.exists(self.metadata_file):
            return False
        
        try:
            # Load metadata first
            with open(self.metadata_file, 'r') as f:
                metadata = json.load(f)
            
            self.feature_columns = metadata.get('feature_columns', [])
            self.target_column = metadata.get('target_column', 'hit')
            self.model_version = metadata.get('model_version', 'automl_1.0')
            self.last_training_time = metadata.get('last_training_time')
            self.training_sample_size = metadata.get('training_sample_size', 0)
            self.model_accuracy = metadata.get('model_accuracy', 0.0)
            
            # Load model
            with open(self.model_file, 'rb') as f:
                self.model = pickle.load(f)
            
            print(f"[AutoML] Loaded model for {self.agent_type} (accuracy: {self.model_accuracy:.2%})")
            return True
            
        except Exception as e:
            print(f"[AutoML] Error loading model for {self.agent_type}: {e}")
            return False


def export_picks_to_ml_format(picks: List[Dict], filename: str) -> pd.DataFrame:
    """Export picks_ledger data to CSV format suitable for external ML tools"""
    engine = AutoMLEngine("export")
    X, y = engine.prepare_data_from_picks(picks)
    
    if X is not None and y is not None and not X.empty and not y.empty:
        # Combine features and target
        df = pd.concat([X, y], axis=1)
        df.to_csv(filename, index=False)
        print(f"[AutoML] Exported {len(df)} rows to {filename}")
        return df
    else:
        print(f"[AutoML] No data to export to {filename}")
        return pd.DataFrame()