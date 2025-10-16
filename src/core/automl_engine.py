# src/core/automl_engine.py
from typing import Dict, List
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

class AutoMLEngine:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_path = f"data/{model_name}.pkl"
        self.model = self.load_model()
        self.training_sample_size = 0
        self.model_accuracy = 0.0
        self.last_training_time = None

    def load_model(self):
        """Load a trained model from a file."""
        if os.path.exists(self.model_path):
            return joblib.load(self.model_path)
        return RandomForestClassifier()

    def save_model(self):
        """Save the trained model to a file."""
        joblib.dump(self.model, self.model_path)

    def train_model(self, historical_picks: List[Dict]) -> bool:
        """Train the model on historical pick data."""
        if len(historical_picks) < 10:
            return False
        
        df = pd.DataFrame(historical_picks)
        df = df.dropna(subset=['outcome'])
        df['outcome'] = df['outcome'].apply(lambda x: 1 if x == 'won' else 0)

        features = ['confidence', 'line', 'odds']
        X = df[features]
        y = df['outcome']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.model.fit(X_train, y_train)
        predictions = self.model.predict(X_test)
        
        self.model_accuracy = accuracy_score(y_test, predictions)
        self.training_sample_size = len(df)
        self.last_training_time = pd.Timestamp.now().isoformat()
        
        self.save_model()
        return True

    def predict(self, features: Dict) -> Dict:
        """Make a prediction based on a set of features."""
        df = pd.DataFrame([features])
        prediction = self.model.predict(df)
        probability = self.model.predict_proba(df)[0][1]
        
        return {
            'prediction': int(prediction[0]),
            'predicted_probability': float(probability)
        }
