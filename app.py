import streamlit as st
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

# Streamlit App
st.set_page_config(page_title="BetFinder AI", page_icon="🎲", layout="wide")

st.title("🎲 BetFinder AI Dashboard")

# Sidebar navigation
page = st.sidebar.selectbox("Navigate", ["Dashboard", "Odds Comparison", "Analyze Bet", "My Bets"])

if page == "Dashboard":
    st.header("Betting Statistics")
    
    recent_bets = db.get_bets()[:5]
    stats = db.get_betting_stats()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Bets", stats.get('total_bets', 0))
    with col2:
        st.metric("Win Rate", f"{stats.get('win_rate', 0)}%")
    with col3:
        st.metric("Profit/Loss", f"${stats.get('profit_loss', 0)}")
    
    st.subheader("Recent Bets")
    if recent_bets:
        df = pd.DataFrame(recent_bets)
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No bets recorded yet.")

elif page == "Odds Comparison":
    st.header("Odds Comparison")
    
    if st.button("Refresh Odds"):
        with st.spinner("Fetching latest odds..."):
            odds_data = analyzer.scrape_odds()
        st.success("Odds updated!")
    
    odds_data = analyzer.scrape_odds()
    
    for odds in odds_data:
        with st.expander(f"{odds['event']} - {odds['bookmaker']}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Home Win", odds['home_odds'])
            with col2:
                st.metric("Draw", odds['draw_odds'])
            with col3:
                st.metric("Away Win", odds['away_odds'])
            st.caption(f"Date: {odds['date'].strftime('%Y-%m-%d %H:%M')}")

elif page == "Analyze Bet":
    st.header("Betting Analysis")
    
    with st.form("analysis_form"):
        event_name = st.text_input("Event Name", placeholder="e.g., Manchester United vs Chelsea")
        sport = st.selectbox("Sport", ["Football", "Basketball", "Tennis", "Other"])
        bet_type = st.selectbox("Bet Type", ["1X2 - Home Win", "1X2 - Draw", "1X2 - Away Win", "Over/Under", "Handicap"])
        odds = st.number_input("Odds", min_value=1.01, max_value=100.0, value=2.0, step=0.1)
        
        submitted = st.form_submit_button("Analyze")
        
        if submitted and event_name:
            teams = event_name.split(' vs ')
            if len(teams) == 2:
                home_stats = analyzer.get_team_stats(teams[0])
                away_stats = analyzer.get_team_stats(teams[1])
                
                home_win_prob = home_stats['wins'] / (home_stats['wins'] + home_stats['draws'] + home_stats['losses'])
                away_win_prob = away_stats['wins'] / (away_stats['wins'] + away_stats['draws'] + away_stats['losses'])
                
                if 'Home' in bet_type:
                    implied_prob = home_win_prob
                elif 'Away' in bet_type:
                    implied_prob = away_win_prob
                else:
                    implied_prob = 1 - home_win_prob - away_win_prob
            else:
                implied_prob = 0.5
            
            analysis = analyzer.analyze_value(odds, implied_prob)
            
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
            
            st.subheader("Analysis Results")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Recommendation", analysis['recommendation'])
                st.metric("Confidence", f"{analysis['confidence']:.1f}%")
            with col2:
                st.metric("Has Value", "Yes" if analysis['has_value'] else "No")
                if analysis['has_value']:
                    st.metric("Value %", f"{analysis['value_percentage']:.2f}%")
            
            if analysis['recommendation'] == 'BET':
                st.success("This bet appears to offer good value!")
            else:
                st.warning("Consider avoiding this bet.")

elif page == "My Bets":
    st.header("My Bets")
    
    all_bets = db.get_bets()
    
    if all_bets:
        df = pd.DataFrame(all_bets)
        st.dataframe(df, use_container_width=True)
        
        st.subheader("Place New Bet")
        with st.form("place_bet_form"):
            event_name = st.text_input("Event Name")
            sport = st.selectbox("Sport", ["Football", "Basketball", "Tennis", "Other"])
            bet_type = st.selectbox("Bet Type", ["1X2 - Home Win", "1X2 - Draw", "1X2 - Away Win", "Over/Under", "Handicap"])
            odds = st.number_input("Odds", min_value=1.01, value=2.0, step=0.1)
            stake = st.number_input("Stake ($)", min_value=1.0, value=10.0, step=1.0)
            bookmaker = st.text_input("Bookmaker", placeholder="e.g., Bet365")
            
            place_bet = st.form_submit_button("Place Bet")
            
            if place_bet and event_name and bookmaker:
                bet_id = db.add_bet(
                    event_name=event_name,
                    sport=sport,
                    bet_type=bet_type,
                    odds=odds,
                    stake=stake,
                    bookmaker=bookmaker
                )
                st.success(f"Bet placed successfully! Bet ID: {bet_id}")
                st.rerun()
    else:
        st.info("No bets recorded yet. Place your first bet below!")
        
        st.subheader("Place New Bet")
        with st.form("place_bet_form"):
            event_name = st.text_input("Event Name")
            sport = st.selectbox("Sport", ["Football", "Basketball", "Tennis", "Other"])
            bet_type = st.selectbox("Bet Type", ["1X2 - Home Win", "1X2 - Draw", "1X2 - Away Win", "Over/Under", "Handicap"])
            odds = st.number_input("Odds", min_value=1.01, value=2.0, step=0.1)
            stake = st.number_input("Stake ($)", min_value=1.0, value=10.0, step=1.0)
            bookmaker = st.text_input("Bookmaker", placeholder="e.g., Bet365")
            
            place_bet = st.form_submit_button("Place Bet")
            
            if place_bet and event_name and bookmaker:
                bet_id = db.add_bet(
                    event_name=event_name,
                    sport=sport,
                    bet_type=bet_type,
                    odds=odds,
                    stake=stake,
                    bookmaker=bookmaker
                )
                st.success(f"Bet placed successfully! Bet ID: {bet_id}")
                st.rerun()
