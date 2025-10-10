#!/usr/bin/env python3
"""
Test script for ML model integration in sport_agents.py
"""

import sys
import json
from datetime import datetime, timedelta

# Import our sport agents
from sport_agents import BasketballAgent, PropValueMLModel, ml_model

def test_ml_model_basic():
    """Test basic ML model functionality"""
    print("Testing PropValueMLModel basic functionality...")
    
    # Create a new model instance
    model = PropValueMLModel()
    
    # Test feature extraction
    sample_pick = {
        'confidence': 75,
        'pick_id': 'test_123',
        'agent_type': 'nba_props',
        'stat_type': 'points',
        'bet_type': 'over',
        'timestamp': datetime.now().isoformat()
    }
    
    features = model._extract_features(sample_pick)
    print(f"Extracted features: {features}")
    assert len(features) == 5, "Should extract 5 features"
    assert isinstance(features, dict), "Features should be returned as dict"
    
    # Test prediction
    prediction_result = model.predict_value(sample_pick)
    print(f"Value prediction: {prediction_result}")
    assert 'predicted_value' in prediction_result, "Should return prediction dict"
    assert 0 <= prediction_result['predicted_value'] <= 1, "Prediction should be between 0 and 1"
    
    print("✓ Basic ML model tests passed!")

def test_sport_agent_ml_integration():
    """Test SportAgent ML integration"""
    print("\nTesting SportAgent ML integration...")
    
    # Create a sport agent
    agent = BasketballAgent()
    
    # Override some attributes for testing
    agent.confidence_threshold = 70
    agent.max_picks = 5
    
    # Test ML model access
    assert ml_model is not None, "Global ML model should exist"
    
    # Test that agent can access ML model through global instance
    assert hasattr(agent, 'get_ml_prediction'), "Agent should have get_ml_prediction method"
    
    # Test ML prediction method
    sample_pick = {
        'confidence': 80,
        'pick_id': 'test_456',
        'agent_type': 'nba_props',
        'stat_type': 'rebounds',
        'bet_type': 'under',
        'odds': -110,
        'timestamp': datetime.now().isoformat()
    }
    
    ml_prediction = agent.get_ml_prediction(sample_pick)
    print(f"ML prediction result: {ml_prediction}")
    
    # Check prediction structure
    expected_keys = ['predicted_value', 'edge', 'confidence', 'expected_roi']
    for key in expected_keys:
        assert key in ml_prediction, f"Missing key: {key}"
    
    print("✓ SportAgent ML integration tests passed!")

def test_prop_card_rendering():
    """Test prop card rendering with ML insights"""
    print("\nTesting prop card rendering...")
    
    agent = BasketballAgent()
    
    # Override some attributes for testing
    agent.confidence_threshold = 70
    agent.max_picks = 5
    
    # Create a sample pick with ML prediction
    sample_pick = {
        'player_name': 'LeBron James',  # Changed from 'player' to 'player_name'
        'stat_type': 'points',
        'line': 25.5,
        'bet_type': 'over',
        'odds': -110,
        'confidence': 82,
        'reasoning': 'Strong matchup advantage',
        'game': 'Lakers vs Warriors',
        'event_start_time': '8:00 PM ET',
        'pick_id': 'test_render_123',
        'agent_type': 'nba_props',
        'timestamp': datetime.now().isoformat()
    }
    
    # Get ML prediction
    ml_prediction = agent.get_ml_prediction(sample_pick)
    sample_pick['ml_prediction'] = ml_prediction
    
    # Test card rendering
    card_data = agent.render_prop_card(sample_pick)
    print(f"Generated card data keys: {list(card_data.keys())}")
    
    # Check for key elements in card data
    assert 'player_name' in card_data, "Should contain player_name"
    assert 'LeBron James' in card_data['player_name'], "Should contain player name"
    assert 'ml_prediction' in card_data, "Should contain ML prediction"
    assert 'card_style' in card_data, "Should contain card styling"
    assert 'recommendation_badge' in card_data, "Should contain recommendation badge"
    
    # Check ML prediction is included
    assert card_data['ml_prediction'] is not None, "ML prediction should be included"
    
    # Check for ML styling based on edge
    if ml_prediction['edge'] > 0.05:  # 5% edge threshold
        card_style = card_data['card_style']
        assert 'background' in card_style, "High value picks should have background styling"
    
    print("✓ Prop card rendering tests passed!")

def test_model_persistence():
    """Test ML model save/load functionality"""
    print("\nTesting ML model persistence...")
    
    # Create model with some training data
    model = PropValueMLModel()
    
    # Add some mock training data
    training_data = [
        {
            'confidence_factor': 0.8,
            'historical_performance': 0.7,
            'stat_type_success': 0.6,
            'over_under_preference': 0.5,
            'recency_factor': 0.9,
            'outcome': 1
        },
        {
            'confidence_factor': 0.3,
            'historical_performance': 0.4,
            'stat_type_success': 0.2,
            'over_under_preference': 0.6,
            'recency_factor': 0.1,
            'outcome': 0
        }
    ]
    
    # Train model
    model.train_model(training_data)
    original_weights = model.weights.copy()
    
    print(f"Original weights type: {type(original_weights)}")
    print(f"Original weights: {original_weights}")
    
    # Save model
    model.save_model()
    print("Model saved successfully")
    
    # Create new model and load
    new_model = PropValueMLModel()
    new_model.load_model()
    
    print(f"New weights type: {type(new_model.weights)}")
    print(f"New weights: {new_model.weights}")
    
    # Compare weights properly
    if isinstance(original_weights, dict) and isinstance(new_model.weights, dict):
        # Compare dictionary weights
        weights_match = True
        for key in original_weights:
            if key not in new_model.weights:
                weights_match = False
                break
            if abs(original_weights[key] - new_model.weights[key]) > 1e-6:
                weights_match = False
                break
    else:
        # Compare list weights
        weights_match = all(abs(a - b) < 1e-6 for a, b in zip(original_weights, new_model.weights))
    
    assert weights_match, "Loaded weights should match saved weights"
    
    print("✓ Model persistence tests passed!")

def main():
    """Run all ML integration tests"""
    print("=== BetFinder AI ML Integration Tests ===\n")
    
    try:
        test_ml_model_basic()
        test_sport_agent_ml_integration()
        test_prop_card_rendering()
        test_model_persistence()
        
        print("\n🎉 All ML integration tests passed successfully!")
        print("\nML model integration is ready for use with:")
        print("- Historical data training")
        print("- Value/edge/ROI predictions")
        print("- Visual highlighting in prop cards")
        print("- Shared parameters across all agents")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()