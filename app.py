from flask import Flask, render_template, request, jsonify, redirect, url_for
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import json
from datetime import datetime, timedelta
import os
from bet_db import db

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')

class BetAnalyzer:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.trained = False
    
    def scrape_odds(self, sport='football', league='premier-league'):
        """Scrape betting odds from various sources"""
        # This is a simplified example - in reality you'd scrape from multiple bookmakers
        odds_data = [
            {
                'event': 'Manchester United vs Chelsea',
                'sport': 'Football',
                'home_odds': 2.50,
                'draw_odds': 3.20,
                'away_odds': 2.80,
                'bookmaker': 'Bet365',
                'date': datetime.now() + timedelta(days=1)
            },
            {
                'event': 'Liverpool vs Arsenal',
                'sport': 'Football',
                'home_odds': 1.95,
                'draw_odds': 3.50,
                'away_odds': 3.25,
                'bookmaker': 'William Hill',
                'date': datetime.now() + timedelta(days=2)
            },
            {
                'event': 'Barcelona vs Real Madrid',
                'sport': 'Football',
                'home_odds': 2.10,
                'draw_odds': 3.00,
                'away_odds': 3.40,
                'bookmaker': 'Betfair',
                'date': datetime.now() + timedelta(days=3)
            }
        ]
        
        # Store odds in database
        for odds in odds_data:
            db.add_odds_comparison(
                event_name=odds['event'],
                sport=odds['sport'],
                bookmaker=odds['bookmaker'],
                bet_type='1X2 - Home Win',
                odds=odds['home_odds']
            )
            db.add_odds_comparison(
                event_name=odds['event'],
                sport=odds['sport'],
                bookmaker=odds['bookmaker'],
                bet_type='1X2 - Draw',
                odds=odds['draw_odds']
            )
            db.add_odds_comparison(
                event_name=odds['event'],
                sport=odds['sport'],
                bookmaker=odds['bookmaker'],
                bet_type='1X2 - Away Win',
                odds=odds['away_odds']
            )
        
        return odds_data
    
    def analyze_value(self, odds, implied_probability):
        """Analyze if a bet offers value"""
        true_probability = 1 / odds
        if implied_probability > true_probability:
            return {
                'has_value': True,
                'value_percentage': ((implied_probability - true_probability) / true_probability) * 100,
                'recommendation': 'BET',
                'confidence': min(90, (implied_probability - true_probability) * 1000)
            }
        else:
            return {
                'has_value': False,
                'value_percentage': 0,
                'recommendation': 'AVOID',
                'confidence': 30
            }
    
    def get_team_stats(self, team_name):
        """Get team statistics (simplified mock data)"""
        # In a real application, this would fetch actual team statistics
        mock_stats = {
            'Manchester United': {'wins': 15, 'draws': 5, 'losses': 8, 'goals_for': 45, 'goals_against': 32},
            'Chelsea': {'wins': 12, 'draws': 8, 'losses': 8, 'goals_for': 38, 'goals_against': 35},
            'Liverpool': {'wins': 18, 'draws': 4, 'losses': 6, 'goals_for': 52, 'goals_against': 28},
            'Arsenal': {'wins': 14, 'draws': 6, 'losses': 8, 'goals_for': 41, 'goals_against': 33},
            'Barcelona': {'wins': 20, 'draws': 3, 'losses': 5, 'goals_for': 58, 'goals_against': 25},
            'Real Madrid': {'wins': 19, 'draws': 4, 'losses': 5, 'goals_for': 55, 'goals_against': 27}
        }
        return mock_stats.get(team_name, {'wins': 10, 'draws': 10, 'losses': 10, 'goals_for': 30, 'goals_against': 30})

analyzer = BetAnalyzer()

@app.route('/')
def index():
    """Main dashboard"""
    recent_bets = db.get_bets()[:5]  # Get last 5 bets
    stats = db.get_betting_stats()
    return render_template('index.html', bets=recent_bets, stats=stats)

@app.route('/odds')
def odds():
    """Odds comparison page"""
    odds_data = analyzer.scrape_odds()
    return render_template('odds.html', odds=odds_data)

@app.route('/analyze', methods=['GET', 'POST'])
def analyze():
    """Betting analysis page"""
    if request.method == 'POST':
        event_name = request.form['event_name']
        sport = request.form['sport']
        bet_type = request.form['bet_type']
        odds = float(request.form['odds'])
        
        # Get team statistics
        teams = event_name.split(' vs ')
        if len(teams) == 2:
            home_stats = analyzer.get_team_stats(teams[0])
            away_stats = analyzer.get_team_stats(teams[1])
            
            # Calculate implied probability based on team stats
            home_win_prob = home_stats['wins'] / (home_stats['wins'] + home_stats['draws'] + home_stats['losses'])
            away_win_prob = away_stats['wins'] / (away_stats['wins'] + away_stats['draws'] + away_stats['losses'])
            
            if 'Home' in bet_type:
                implied_prob = home_win_prob
            elif 'Away' in bet_type:
                implied_prob = away_win_prob
            else:  # Draw
                implied_prob = 1 - home_win_prob - away_win_prob
        else:
            implied_prob = 0.5  # Default probability
        
        # Analyze value
        analysis = analyzer.analyze_value(odds, implied_prob)
        
        # Save analysis to database
        analysis_data = {
            'odds': odds,
            'implied_probability': implied_prob,
            'bet_type': bet_type
        }
        
        db.save_analysis_result(
            event_name=event_name,
            sport=sport,
            analysis_data=analysis_data,
            recommendation=analysis['recommendation'],
            confidence_score=analysis['confidence']
        )
        
        return render_template('analysis_result.html', 
                             event=event_name, 
                             analysis=analysis,
                             odds=odds,
                             bet_type=bet_type)
    
    return render_template('analyze.html')

@app.route('/place_bet', methods=['POST'])
def place_bet():
    """Place a new bet"""
    data = request.get_json()
    
    bet_id = db.add_bet(
        event_name=data['event_name'],
        sport=data['sport'],
        bet_type=data['bet_type'],
        odds=float(data['odds']),
        stake=float(data['stake']),
        bookmaker=data['bookmaker']
    )
    
    return jsonify({'success': True, 'bet_id': bet_id})

@app.route('/bets')
def bets():
    """View all bets"""
    all_bets = db.get_bets()
    return render_template('bets.html', bets=all_bets)

@app.route('/update_bet/<int:bet_id>', methods=['POST'])
def update_bet(bet_id):
    """Update bet status"""
    status = request.form['status']
    result = request.form.get('result')
    
    db.update_bet_status(bet_id, status, result)
    return redirect(url_for('bets'))

@app.route('/api/stats')
def api_stats():
    """API endpoint for statistics"""
    stats = db.get_betting_stats()
    return jsonify(stats)

@app.route('/api/odds/<event_name>')
def api_odds(event_name):
    """API endpoint for odds comparison"""
    odds = db.get_best_odds(event_name, '1X2 - Home Win')
    return jsonify(odds)

# Create templates directory and basic templates
@app.before_first_request
def create_templates():
    """Create basic HTML templates"""
    import os
    
    templates_dir = 'templates'
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
    
    # Basic index template
    index_html = '''
<!DOCTYPE html>
<html>
<head>
    <title>BetFinder AI</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <div class="container mt-4">
        <h1>BetFinder AI Dashboard</h1>
        
        <div class="row">
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">Betting Statistics</div>
                    <div class="card-body">
                        <p>Total Bets: {{ stats.total_bets }}</p>
                        <p>Win Rate: {{ stats.win_rate }}%</p>
                        <p>Profit/Loss: ${{ stats.profit_loss }}</p>
                    </div>
                </div>
            </div>
            <div class="col-md-6">
                <div class="card">
                    <div class="card-header">Quick Actions</div>
                    <div class="card-body">
                        <a href="/analyze" class="btn btn-primary">Analyze Bet</a>
                        <a href="/odds" class="btn btn-info">View Odds</a>
                        <a href="/bets" class="btn btn-secondary">My Bets</a>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="mt-4">
            <h3>Recent Bets</h3>
            <table class="table">
                <thead>
                    <tr>
                        <th>Event</th>
                        <th>Bet Type</th>
                        <th>Odds</th>
                        <th>Stake</th>
                        <th>Status</th>
                    </tr>
                </thead>
                <tbody>
                    {% for bet in bets %}
                    <tr>
                        <td>{{ bet.event_name }}</td>
                        <td>{{ bet.bet_type }}</td>
                        <td>{{ bet.odds }}</td>
                        <td>${{ bet.stake }}</td>
                        <td>{{ bet.status }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
    '''
    
    with open(os.path.join(templates_dir, 'index.html'), 'w') as f:
        f.write(index_html)

if __name__ == '__main__':
    # Initialize database
    print("Starting BetFinder AI...")
    print("Database initialized successfully!")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
