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
                'event': 'Lakers vs Warriors',
                'sport': 'Basketball',
                'home_odds': 1.85,
                'draw_odds': None,
                'away_odds': 1.95,
                'bookmaker': 'DraftKings',
                'date': datetime.now() + timedelta(days=1)
            },
            {
                'event': 'Yankees vs Red Sox',
                'sport': 'Baseball',
                'home_odds': 2.10,
                'draw_odds': None,
                'away_odds': 1.75,
                'bookmaker': 'FanDuel',
                'date': datetime.now() + timedelta(days=1)
            }
        ]
        return odds_data
    
    def calculate_value_bets(self, odds_data):
        """Calculate potential value bets based on odds comparison"""
        value_bets = []
        for odds in odds_data:
            # Simple value calculation - compare with average market odds
            implied_prob_home = 1 / odds['home_odds'] if odds['home_odds'] else 0
            implied_prob_away = 1 / odds['away_odds'] if odds['away_odds'] else 0
            
            if odds['draw_odds']:
                implied_prob_draw = 1 / odds['draw_odds']
                total_prob = implied_prob_home + implied_prob_draw + implied_prob_away
            else:
                total_prob = implied_prob_home + implied_prob_away
            
            margin = (total_prob - 1) * 100
            
            value_bets.append({
                'event': odds['event'],
                'sport': odds['sport'],
                'bookmaker': odds['bookmaker'],
                'best_bet': 'Home' if odds['home_odds'] > odds['away_odds'] else 'Away',
                'odds': max(odds['home_odds'], odds['away_odds']),
                'margin': round(margin, 2),
                'date': odds['date']
            })
        
        return value_bets

# Odds JSON and records builder
odds_json = [
    {'event': 'Manchester United vs Chelsea', 'sport': 'Football', 'home_odds': 2.50, 'draw_odds': 3.20, 'away_odds': 2.80, 'bookmaker': 'Bet365', 'league': 'Premier League'},
    {'event': 'Liverpool vs Arsenal', 'sport': 'Football', 'home_odds': 1.95, 'draw_odds': 3.50, 'away_odds': 3.25, 'bookmaker': 'William Hill', 'league': 'Premier League'},
    {'event': 'Real Madrid vs Barcelona', 'sport': 'Football', 'home_odds': 2.20, 'draw_odds': 3.30, 'away_odds': 3.10, 'bookmaker': 'Bet365', 'league': 'La Liga'},
    {'event': 'Bayern Munich vs Dortmund', 'sport': 'Football', 'home_odds': 1.80, 'draw_odds': 3.80, 'away_odds': 4.20, 'bookmaker': 'Betway', 'league': 'Bundesliga'},
    {'event': 'Lakers vs Warriors', 'sport': 'Basketball', 'home_odds': 1.85, 'draw_odds': None, 'away_odds': 1.95, 'bookmaker': 'DraftKings', 'league': 'NBA'},
    {'event': 'Celtics vs Nets', 'sport': 'Basketball', 'home_odds': 1.75, 'draw_odds': None, 'away_odds': 2.05, 'bookmaker': 'FanDuel', 'league': 'NBA'},
    {'event': 'Yankees vs Red Sox', 'sport': 'Baseball', 'home_odds': 2.10, 'draw_odds': None, 'away_odds': 1.75, 'bookmaker': 'FanDuel', 'league': 'MLB'},
    {'event': 'Dodgers vs Giants', 'sport': 'Baseball', 'home_odds': 1.90, 'draw_odds': None, 'away_odds': 1.90, 'bookmaker': 'DraftKings', 'league': 'MLB'},
    {'event': 'PSG vs Lyon', 'sport': 'Football', 'home_odds': 1.65, 'draw_odds': 3.90, 'away_odds': 5.50, 'bookmaker': 'Betway', 'league': 'Ligue 1'},
    {'event': 'Juventus vs AC Milan', 'sport': 'Football', 'home_odds': 2.35, 'draw_odds': 3.25, 'away_odds': 2.95, 'bookmaker': 'William Hill', 'league': 'Serie A'}
]

records = []
for odd in odds_json:
    records.append({
        'Event': odd['event'],
        'Sport': odd['sport'],
        'League': odd['league'],
        'Home Odds': odd['home_odds'],
        'Draw Odds': odd['draw_odds'] if odd['draw_odds'] else 'N/A',
        'Away Odds': odd['away_odds'],
        'Bookmaker': odd['bookmaker']
    })

def main():
    st.set_page_config(page_title="BetFinder AI", page_icon="🎲", layout="wide")
    
    st.title("🎲 BetFinder AI - Smart Betting Analytics")
    st.markdown("### Find the best betting opportunities with AI-powered analysis")
    
    # Sidebar filters
    st.sidebar.header("Filters")
    
    # Sport filter
    all_sports = sorted(list(set([r['Sport'] for r in records])))
    selected_sports = st.sidebar.multiselect(
        "Select Sports",
        options=all_sports,
        default=all_sports
    )
    
    # League filter
    all_leagues = sorted(list(set([r['League'] for r in records])))
    selected_leagues = st.sidebar.multiselect(
        "Select Leagues",
        options=all_leagues,
        default=all_leagues
    )
    
    # Bookmaker filter
    all_bookmakers = sorted(list(set([r['Bookmaker'] for r in records])))
    selected_bookmakers = st.sidebar.multiselect(
        "Select Bookmakers",
        options=all_bookmakers,
        default=all_bookmakers
    )
    
    # Odds range filter
    st.sidebar.subheader("Odds Range")
    min_odds = st.sidebar.number_input("Minimum Odds", min_value=1.0, max_value=10.0, value=1.0, step=0.1)
    max_odds = st.sidebar.number_input("Maximum Odds", min_value=1.0, max_value=10.0, value=10.0, step=0.1)
    
    # Filtered live odds dataframe display
    st.header("📊 Live Odds Board")
    
    # Filter records
    filtered_records = []
    for record in records:
        if (record['Sport'] in selected_sports and 
            record['League'] in selected_leagues and 
            record['Bookmaker'] in selected_bookmakers):
            
            # Check odds range
            home_odds = record['Home Odds']
            away_odds = record['Away Odds']
            
            if (min_odds <= home_odds <= max_odds or 
                min_odds <= away_odds <= max_odds):
                filtered_records.append(record)
    
    if filtered_records:
        df_odds = pd.DataFrame(filtered_records)
        st.dataframe(df_odds, use_container_width=True, hide_index=True)
        st.success(f"Showing {len(filtered_records)} betting opportunities")
    else:
        st.warning("No odds match your current filters. Try adjusting the filters.")
    
    # Initialize analyzer
    analyzer = BetAnalyzer()
    
    # Create tabs for different features
    tab1, tab2, tab3 = st.tabs(["📈 Value Bets", "🔍 Odds Comparison", "💾 Bet History"])
    
    with tab1:
        st.subheader("Potential Value Bets")
        
        if st.button("Analyze Current Odds"):
            with st.spinner("Analyzing odds..."):
                odds_data = analyzer.scrape_odds()
                value_bets = analyzer.calculate_value_bets(odds_data)
                
                if value_bets:
                    df = pd.DataFrame(value_bets)
                    st.dataframe(df, use_container_width=True)
                else:
                    st.info("No value bets found at the moment.")
    
    with tab2:
        st.subheader("Compare Odds Across Bookmakers")
        st.info("Feature coming soon! This will compare odds from multiple bookmakers.")
    
    with tab3:
        st.subheader("Your Betting History")
        
        # Bet tracker
        with st.form("bet_form"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                event = st.text_input("Event")
                stake = st.number_input("Stake ($)", min_value=0.0, step=1.0)
            
            with col2:
                bet_type = st.selectbox("Bet Type", ["Home", "Draw", "Away", "Over", "Under"])
                odds = st.number_input("Odds", min_value=1.0, step=0.1)
            
            with col3:
                bookmaker = st.text_input("Bookmaker")
                result = st.selectbox("Result", ["Pending", "Won", "Lost"])
            
            submitted = st.form_submit_button("Add Bet")
            
            if submitted and event:
                bet_data = {
                    'event': event,
                    'stake': stake,
                    'bet_type': bet_type,
                    'odds': odds,
                    'bookmaker': bookmaker,
                    'result': result,
                    'date': datetime.now().strftime('%Y-%m-%d %H:%M')
                }
                
                # Save to database
                db.save_bet(bet_data)
                st.success("Bet added successfully!")
        
        # Display bet history
        bets = db.get_all_bets()
        if bets:
            df_bets = pd.DataFrame(bets)
            st.dataframe(df_bets, use_container_width=True)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            total_bets = len(bets)
            won_bets = len([b for b in bets if b.get('result') == 'Won'])
            lost_bets = len([b for b in bets if b.get('result') == 'Lost'])
            
            with col1:
                st.metric("Total Bets", total_bets)
            with col2:
                st.metric("Won", won_bets)
            with col3:
                st.metric("Lost", lost_bets)
            with col4:
                win_rate = (won_bets / (won_bets + lost_bets) * 100) if (won_bets + lost_bets) > 0 else 0
                st.metric("Win Rate", f"{win_rate:.1f}%")
        else:
            st.info("No bets recorded yet. Add your first bet above!")

if __name__ == "__main__":
    main()
