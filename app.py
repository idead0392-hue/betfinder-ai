import streamlit as st
import pandas as pd
import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time

# Import our AI agents and bankroll manager
try:
    from picks_engine import PicksEngine
    from openai_daily_picks import OpenAIDailyPicksAgent
    from bankroll_manager import BankrollManager
except ImportError as e:
    st.error(f"Missing dependencies: {e}")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="BetFinder AI - Advanced Sports Betting Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state
if 'picks_history' not in st.session_state:
    st.session_state.picks_history = []
if 'current_picks' not in st.session_state:
    st.session_state.current_picks = []
if 'bankroll_manager' not in st.session_state:
    st.session_state.bankroll_manager = BankrollManager()

# CSV file for storing AI picks history with outcomes
PICKS_HISTORY_CSV = "ai_picks_history.csv"

def ensure_csv_exists():
    """Ensure the CSV file exists with proper headers"""
    if not os.path.exists(PICKS_HISTORY_CSV):
        # Create CSV with headers
        headers = [
            'timestamp', 'date', 'agent_type', 'pick_id', 'sport', 'competition', 
            'matchup', 'pick_description', 'pick_type', 'player_name', 'odds', 
            'confidence', 'expected_value', 'reasoning', 'outcome', 'bet_amount',
            'profit_loss', 'roi_percentage'
        ]
        df = pd.DataFrame(columns=headers)
        df.to_csv(PICKS_HISTORY_CSV, index=False)

def save_pick_to_csv(pick_data: Dict, outcome: str = "Pending", bet_amount: float = 0.0):
    """Save pick with outcome to CSV for historical tracking"""
    ensure_csv_exists()
    
    # Calculate profit/loss and ROI if outcome is determined
    profit_loss = 0.0
    roi_percentage = 0.0
    
    if outcome != "Pending" and bet_amount > 0:
        if outcome == "Win":
            # Calculate profit based on odds
            odds = pick_data.get('odds', -110)
            if odds > 0:
                profit_loss = bet_amount * (odds / 100)
            else:
                profit_loss = bet_amount * (100 / abs(odds))
            roi_percentage = (profit_loss / bet_amount) * 100
        elif outcome == "Loss":
            profit_loss = -bet_amount
            roi_percentage = -100.0
        # Push is 0 profit/loss
    
    # Prepare row data
    row_data = {
        'timestamp': datetime.now().isoformat(),
        'date': datetime.now().strftime('%Y-%m-%d'),
        'agent_type': pick_data.get('agent_type', 'Unknown'),
        'pick_id': pick_data.get('game_id', pick_data.get('pick_id', 'unknown')),
        'sport': pick_data.get('sport', ''),
        'competition': pick_data.get('competition', ''),
        'matchup': pick_data.get('matchup', ''),
        'pick_description': pick_data.get('pick', ''),
        'pick_type': pick_data.get('pick_type', ''),
        'player_name': pick_data.get('player_name', ''),
        'odds': pick_data.get('odds', 0),
        'confidence': pick_data.get('confidence', 0),
        'expected_value': pick_data.get('expected_value', 0),
        'reasoning': pick_data.get('reasoning', ''),
        'outcome': outcome,
        'bet_amount': bet_amount,
        'profit_loss': profit_loss,
        'roi_percentage': roi_percentage
    }
    
    # Read existing CSV and append new row
    try:
        df = pd.read_csv(PICKS_HISTORY_CSV)
        df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
        df.to_csv(PICKS_HISTORY_CSV, index=False)
        return True
    except Exception as e:
        st.error(f"Error saving to CSV: {e}")
        return False

def update_pick_outcome(pick_id: str, outcome: str, bet_amount: float = 0.0):
    """Update existing pick outcome in CSV"""
    if not os.path.exists(PICKS_HISTORY_CSV):
        return False
    
    try:
        df = pd.read_csv(PICKS_HISTORY_CSV)
        mask = df['pick_id'] == pick_id
        
        if mask.any():
            # Update outcome
            df.loc[mask, 'outcome'] = outcome
            df.loc[mask, 'bet_amount'] = bet_amount
            
            # Recalculate profit/loss and ROI
            for idx in df[mask].index:
                pick_odds = df.loc[idx, 'odds']
                if outcome == "Win" and bet_amount > 0:
                    if pick_odds > 0:
                        profit = bet_amount * (pick_odds / 100)
                    else:
                        profit = bet_amount * (100 / abs(pick_odds))
                    df.loc[idx, 'profit_loss'] = profit
                    df.loc[idx, 'roi_percentage'] = (profit / bet_amount) * 100
                elif outcome == "Loss" and bet_amount > 0:
                    df.loc[idx, 'profit_loss'] = -bet_amount
                    df.loc[idx, 'roi_percentage'] = -100.0
                else:
                    df.loc[idx, 'profit_loss'] = 0.0
                    df.loc[idx, 'roi_percentage'] = 0.0
            
            df.to_csv(PICKS_HISTORY_CSV, index=False)
            return True
    except Exception as e:
        st.error(f"Error updating CSV: {e}")
    
    return False

def load_picks_history() -> pd.DataFrame:
    """Load picks history from CSV"""
    if os.path.exists(PICKS_HISTORY_CSV):
        try:
            return pd.read_csv(PICKS_HISTORY_CSV)
        except Exception:
            pass
    return pd.DataFrame()

def display_pick_with_outcome_selector(pick: Dict, agent_type: str, pick_index: int):
    """Display pick with outcome selection radio buttons"""
    pick_id = pick.get('game_id', pick.get('pick_id', f"{agent_type}_{pick_index}"))
    
    with st.container():
        # Pick header
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            st.markdown(f"**{pick.get('sport', 'Unknown').title()}: {pick.get('matchup', 'Unknown Matchup')}**")
            st.markdown(f"*{pick.get('pick', 'No pick description')}*")
        
        with col2:
            confidence = pick.get('confidence', 0)
            confidence_color = "🟢" if confidence >= 80 else "🟡" if confidence >= 65 else "🔴"
            st.markdown(f"{confidence_color} **{confidence:.1f}%**")
        
        with col3:
            ev = pick.get('expected_value', 0)
            ev_color = "🟢" if ev >= 10 else "🟡" if ev >= 5 else "🔴"
            st.markdown(f"{ev_color} **+{ev:.1f}% EV**")
        
        # Pick details
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if pick.get('player_name'):
                st.markdown(f"**Player:** {pick.get('player_name')}")
            st.markdown(f"**Type:** {pick.get('pick_type', 'Standard').title()}")
            st.markdown(f"**Odds:** {pick.get('odds', 'N/A')}")
            
            if pick.get('reasoning'):
                with st.expander("📊 Analysis"):
                    st.write(pick.get('reasoning'))
        
        with col2:
            # Outcome selection
            outcome_key = f"outcome_{pick_id}_{pick_index}"
            bet_amount_key = f"bet_amount_{pick_id}_{pick_index}"
            
            st.markdown("**Track Outcome:**")
            outcome = st.radio(
                "Result",
                ["Pending", "Win", "Loss", "Push"],
                key=outcome_key,
                horizontal=True,
                label_visibility="collapsed"
            )
            
            # Bet amount input
            bet_amount = st.number_input(
                "Bet Amount ($)",
                min_value=0.0,
                value=0.0,
                step=5.0,
                key=bet_amount_key,
                help="Enter actual bet amount for P&L tracking"
            )
            
            # Save/Update button
            if st.button(f"Save Outcome", key=f"save_{pick_id}_{pick_index}"):
                # Add agent type to pick data for CSV storage
                pick_with_agent = pick.copy()
                pick_with_agent['agent_type'] = agent_type
                
                # Check if this pick already exists in CSV
                df = load_picks_history()
                existing = df[df['pick_id'] == pick_id]
                
                if len(existing) > 0:
                    # Update existing pick
                    success = update_pick_outcome(pick_id, outcome, bet_amount)
                    if success:
                        st.success("✅ Outcome updated!")
                        time.sleep(1)
                        st.rerun()
                else:
                    # Save new pick
                    success = save_pick_to_csv(pick_with_agent, outcome, bet_amount)
                    if success:
                        st.success("✅ Pick saved!")
                        time.sleep(1)
                        st.rerun()
        
        st.markdown("---")

# Main app title
st.title("🎯 BetFinder AI - Advanced Sports Betting Analytics")
st.markdown("*AI-Powered Pick Generation with Outcome Tracking & Performance Analytics*")

# Navigation tabs
tab_names = [
    "🏠 Dashboard", "🤖 AI Picks", "📊 Performance", "💰 Bankroll", 
    "📈 Analytics", "⚙️ Settings"
]
tabs = st.tabs(tab_names)

# Dashboard Tab
with tabs[0]:
    st.header("📊 Performance Dashboard")
    
    # Load and display recent picks history
    df_history = load_picks_history()
    
    if len(df_history) > 0:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        # Filter completed picks (not pending)
        completed_picks = df_history[df_history['outcome'] != 'Pending']
        
        with col1:
            total_picks = len(df_history)
            st.metric("Total Picks", total_picks)
        
        with col2:
            if len(completed_picks) > 0:
                wins = len(completed_picks[completed_picks['outcome'] == 'Win'])
                win_rate = (wins / len(completed_picks)) * 100
                st.metric("Win Rate", f"{win_rate:.1f}%")
            else:
                st.metric("Win Rate", "N/A")
        
        with col3:
            total_profit = completed_picks['profit_loss'].sum() if len(completed_picks) > 0 else 0
            st.metric("Total P&L", f"${total_profit:.2f}", 
                     delta=f"{total_profit:.2f}" if total_profit != 0 else None)
        
        with col4:
            if len(completed_picks) > 0 and completed_picks['bet_amount'].sum() > 0:
                total_roi = (total_profit / completed_picks['bet_amount'].sum()) * 100
                st.metric("ROI", f"{total_roi:.1f}%")
            else:
                st.metric("ROI", "N/A")
        
        st.markdown("---")
        
        # Recent picks table
        st.subheader("📋 Recent Picks")
        
        # Display options
        col1, col2 = st.columns([1, 3])
        with col1:
            show_filter = st.selectbox("Filter by:", ["All", "Pending", "Win", "Loss", "Push"])
        
        # Apply filter
        if show_filter != "All":
            display_df = df_history[df_history['outcome'] == show_filter]
        else:
            display_df = df_history
        
        # Sort by timestamp (newest first)
        display_df = display_df.sort_values('timestamp', ascending=False)
        
        # Display table with key columns
        if len(display_df) > 0:
            display_columns = [
                'date', 'agent_type', 'sport', 'matchup', 'pick_description', 
                'confidence', 'odds', 'outcome', 'bet_amount', 'profit_loss'
            ]
            st.dataframe(
                display_df[display_columns].head(20),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No picks match the selected filter.")
    
    else:
        st.info("No picks tracked yet. Generate some picks in the AI Picks tab!")

# AI Picks Tab
with tabs[1]:
    st.header("🤖 AI Pick Generation")
    
    # Agent selection
    col1, col2 = st.columns(2)
    
    with col1:
        agent_type = st.selectbox(
            "Select AI Agent",
            ["Advanced Picks Engine", "Simple OpenAI Agent"],
            help="Choose which AI agent to generate picks"
        )
    
    with col2:
        max_picks = st.slider("Number of Picks", 1, 10, 5)
    
    # Generate picks button
    if st.button("🎯 Generate AI Picks", type="primary"):
        with st.spinner("🤖 AI is analyzing markets and generating picks..."):
            try:
                if agent_type == "Advanced Picks Engine":
                    engine = PicksEngine()
                    picks = engine.generate_ai_powered_picks(max_picks=max_picks)
                    agent_key = "picks_engine"
                else:
                    agent = OpenAIDailyPicksAgent()
                    picks = agent.get_ai_daily_picks(max_picks=max_picks)
                    agent_key = "openai_simple"
                
                if picks:
                    st.session_state.current_picks = picks
                    st.session_state.current_agent = agent_key
                    st.success(f"✅ Generated {len(picks)} AI picks!")
                else:
                    st.error("❌ Failed to generate picks. Please try again.")
                    
            except Exception as e:
                st.error(f"❌ Error generating picks: {str(e)}")
    
    # Display current picks with outcome selectors
    if st.session_state.current_picks:
        st.markdown("---")
        st.subheader("🎲 Generated Picks")
        st.markdown("*Select outcomes and bet amounts to track performance*")
        
        agent_type_display = st.session_state.get('current_agent', 'unknown')
        
        for i, pick in enumerate(st.session_state.current_picks):
            display_pick_with_outcome_selector(pick, agent_type_display, i)

# Performance Tab
with tabs[2]:
    st.header("📈 Performance Analytics")
    
    df_history = load_picks_history()
    
    if len(df_history) > 0:
        # Agent performance comparison
        st.subheader("🤖 Agent Performance Comparison")
        
        agent_stats = df_history.groupby('agent_type').agg({
            'pick_id': 'count',
            'outcome': lambda x: (x == 'Win').sum(),
            'profit_loss': 'sum',
            'bet_amount': 'sum',
            'confidence': 'mean'
        }).round(2)
        
        agent_stats.columns = ['Total Picks', 'Wins', 'Total P&L', 'Total Bet', 'Avg Confidence']
        agent_stats['Win Rate %'] = ((agent_stats['Wins'] / agent_stats['Total Picks']) * 100).round(1)
        agent_stats['ROI %'] = ((agent_stats['Total P&L'] / agent_stats['Total Bet']) * 100).round(1)
        
        st.dataframe(agent_stats, use_container_width=True)
        
        # Sport performance
        st.subheader("🏀 Performance by Sport")
        
        sport_stats = df_history.groupby('sport').agg({
            'pick_id': 'count',
            'outcome': lambda x: (x == 'Win').sum(),
            'profit_loss': 'sum'
        }).round(2)
        
        sport_stats.columns = ['Total Picks', 'Wins', 'Total P&L']
        sport_stats['Win Rate %'] = ((sport_stats['Wins'] / sport_stats['Total Picks']) * 100).round(1)
        
        st.dataframe(sport_stats, use_container_width=True)
    
    else:
        st.info("No performance data available yet.")

# Bankroll Tab
with tabs[3]:
    st.header("💰 Bankroll Management")
    
    bankroll_mgr = st.session_state.bankroll_manager
    
    # Current bankroll status
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Bankroll", f"${bankroll_mgr.data['current_bankroll']:.2f}")
    
    with col2:
        st.metric("Starting Bankroll", f"${bankroll_mgr.data['starting_bankroll']:.2f}")
    
    with col3:
        profit = bankroll_mgr.data['current_bankroll'] - bankroll_mgr.data['starting_bankroll']
        st.metric("Profit/Loss", f"${profit:.2f}", delta=f"{profit:.2f}")
    
    st.markdown("---")
    
    # Bankroll settings
    st.subheader("⚙️ Bankroll Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        new_bankroll = st.number_input(
            "Set New Bankroll ($)",
            min_value=0.0,
            value=float(bankroll_mgr.data['current_bankroll']),
            step=10.0
        )
        
        if st.button("Update Bankroll"):
            bankroll_mgr.data['current_bankroll'] = new_bankroll
            bankroll_mgr.save_data()
            st.success("✅ Bankroll updated!")
            st.rerun()
    
    with col2:
        unit_size = st.number_input(
            "Unit Size ($)",
            min_value=1.0,
            value=float(bankroll_mgr.data['unit_size']),
            step=1.0
        )
        
        if st.button("Update Unit Size"):
            bankroll_mgr.data['unit_size'] = unit_size
            bankroll_mgr.save_data()
            st.success("✅ Unit size updated!")
            st.rerun()

# Analytics Tab
with tabs[4]:
    st.header("📈 Advanced Analytics")
    st.info("Advanced analytics and charts coming soon!")

# Settings Tab
with tabs[5]:
    st.header("⚙️ Settings")
    
    st.subheader("🔧 Application Settings")
    
    # Data management
    st.markdown("**Data Management:**")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear All Pick History", type="secondary"):
            if os.path.exists(PICKS_HISTORY_CSV):
                os.remove(PICKS_HISTORY_CSV)
                st.success("✅ Pick history cleared!")
                st.rerun()
    
    with col2:
        if st.button("💾 Export Pick History", type="secondary"):
            if os.path.exists(PICKS_HISTORY_CSV):
                df = pd.read_csv(PICKS_HISTORY_CSV)
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"picks_history_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning("No pick history to export.")
    
    st.markdown("---")
    
    # Agent settings
    st.subheader("🤖 Agent Configuration")
    st.info("Agent settings and API key management coming soon!")
