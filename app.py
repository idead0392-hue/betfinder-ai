import streamlit as st
import pandas as pd
import requests
import os
import time
# requests is used for custom API calls below
from lxml import etree

# Page configuration
st.set_page_config(
    page_title="BetFinder AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Initialize session state for caching
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}
if 'cache_timestamp' not in st.session_state:
    st.session_state.cache_timestamp = {}
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Cache duration (5 minutes)
CACHE_DURATION = 300

def is_cache_valid(cache_key):
    """Check if cached data is still valid"""
    if cache_key not in st.session_state.cache_timestamp:
        return False
    elapsed = time.time() - st.session_state.cache_timestamp[cache_key]
    return elapsed < CACHE_DURATION

def get_cached_data(cache_key):
    """Get data from cache if valid"""
    if is_cache_valid(cache_key):
        return st.session_state.data_cache.get(cache_key)
    return None

def set_cached_data(cache_key, data):
    """Store data in cache with timestamp"""
    st.session_state.data_cache[cache_key] = data
    st.session_state.cache_timestamp[cache_key] = time.time()

def load_api_data(url, cache_key, method='GET', data=None):
    """Load data from API with caching"""
    cached = get_cached_data(cache_key)
    if cached is not None:
        return cached
    
    try:
        if method == 'POST':
            response = requests.post(url, json=data or {}, timeout=10)
        else:
            response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            set_cached_data(cache_key, result)
            return result
    except:
        pass
    
    return None


# Helper: safe rerun wrapper for Streamlit versions without experimental_rerun
def safe_rerun():
    """Try to rerun the Streamlit script. If not available, show a message and stop execution.

    This avoids AttributeError on Streamlit builds that don't expose experimental_rerun.
    """
    try:
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
            return
    except Exception:
        # Fall through to graceful fallback
        pass

    # Graceful fallback: do nothing (allow the script to continue).
    # Many Streamlit versions will simply re-run on the next user interaction.
    return


def preload_sports_data():
    """Preload all sports data on app startup"""
    if st.session_state.data_loaded:
        return
    
    with st.spinner("Loading sports data..."):
        # Basketball data
        load_api_data("http://127.0.0.1:5001/api/basketball/props", "basketball_props")
        load_api_data("http://127.0.0.1:5001/api/basketball/odds", "basketball_odds", "POST")
        
        # Tennis data
        load_api_data("http://127.0.0.1:5001/api/tennis/competitions", "tennis_competitions")
        load_api_data("http://127.0.0.1:5001/api/tennis/odds", "tennis_odds", "POST")
        
        # Football data
        load_api_data("http://127.0.0.1:5001/api/football/competitions", "football_competitions")
        load_api_data("http://127.0.0.1:5001/api/football/odds", "football_odds", "POST")
        
        # Soccer data
        load_api_data("http://127.0.0.1:5001/api/soccer/competitions", "soccer_competitions")
        
        st.session_state.data_loaded = True

# Preload data on app start
preload_sports_data()

# Add refresh functionality
def refresh_all_data():
    """Clear cache and reload all sports data"""
    st.session_state.data_cache = {}
    st.session_state.cache_timestamp = {}
    st.session_state.data_loaded = False
    preload_sports_data()
    st.rerun()

# Header with refresh button
col1, col2 = st.columns([4, 1])
with col1:
    st.title("🎯 BetFinder AI")
with col2:
    if st.button("🔄 Refresh Data", help="Update all sports data"):
        refresh_all_data()


# --- RESTORED MULTI-TAB UI ---
tab_names = [
    "Home", "Stats", "Props", "Tennis", "Basketball", "Football", "Baseball",
    "Hockey", "Soccer", "Esports", "College Football"
]
tabs = st.tabs(tab_names)


# --- Session-based Yahoo OAuth Token Storage (Streamlit Cloud compatible) ---
if 'yahoo_access_token' not in st.session_state:
    st.session_state['yahoo_access_token'] = None
    st.session_state['yahoo_refresh_token'] = None
    st.session_state['yahoo_oauth_state'] = None
    st.session_state['yahoo_oauth_step'] = 0





# Home Tab (placeholder, no demo)
with tabs[2]:
    st.header("Props")
    st.write("Use a custom API that accepts a Bearer token to provide props data.")
    st.markdown("**Security:** Do not commit your tokens. Add them to `.streamlit/secrets.toml` as `NEW_API_TOKEN` or paste temporarily below.")

    provider = st.selectbox("Provider", ["None", "Custom (Bearer token)", "PandaScore"])

    if provider == "Custom (Bearer token)":
        endpoint = st.text_input("API endpoint (full URL)", value="https://api.example.com/v1/props")
        use_secret = st.checkbox("Use token from st.secrets['NEW_API_TOKEN']", value=True)
        token = None
        if use_secret:
            token = st.secrets.get("NEW_API_TOKEN") if hasattr(st, 'secrets') else None
            if not token:
                st.warning("`NEW_API_TOKEN` not found in `st.secrets`. You can paste a token below for testing.")
        else:
            token = st.text_input("Paste token (will not be saved)", type="password")

        if st.button("Test API"):
            if not endpoint:
                st.error("Please provide an API endpoint to test.")
            elif not token:
                st.error("No token available. Add `NEW_API_TOKEN` to `.streamlit/secrets.toml` or paste a token.")
            else:
                headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
                st.info(f"Calling {endpoint} with Bearer token (redacted in UI)")
                try:
                    resp = requests.get(endpoint, headers=headers, timeout=15)
                    st.write("Status:", resp.status_code)
                    ct = resp.headers.get('content-type','')
                    if 'application/json' in ct:
                        try:
                            st.json(resp.json())
                        except Exception:
                            st.code(resp.text)
                    else:
                        # Show text or XML
                        st.code(resp.text[:10000])
                except Exception as e:
                    st.error(f"Request failed: {e}")

    elif provider == "PandaScore":
        st.write("PandaScore API integration for esports data.")
        endpoint = st.selectbox("Endpoint", [
            "https://api.pandascore.co/videogames",
            "https://api.pandascore.co/matches",
            "https://api.pandascore.co/series",
            "https://api.pandascore.co/tournaments"
        ], index=0)
        use_secret = st.checkbox("Use token from st.secrets['NEW_API_TOKEN']", value=True)
        token = None
        if use_secret:
            token = st.secrets.get("NEW_API_TOKEN") if hasattr(st, 'secrets') else None
            if not token:
                st.warning("`NEW_API_TOKEN` not found in `st.secrets`. You can paste a token below for testing.")
        else:
            token = st.text_input("Paste token (will not be saved)", type="password")

        if st.button("Test PandaScore API"):
            if not token:
                st.error("No token available. Add `NEW_API_TOKEN` to `.streamlit/secrets.toml` or paste a token.")
            else:
                headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
                st.info(f"Calling {endpoint} with Bearer token (redacted in UI)")
                try:
                    resp = requests.get(endpoint, headers=headers, timeout=15)
                    st.write("Status:", resp.status_code)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.success("API call successful!")
                        if isinstance(data, list):
                            df = pd.DataFrame(data)
                            st.dataframe(df)
                        else:
                            st.json(data)
                    else:
                        st.error(f"API returned status {resp.status_code}")
                        st.code(resp.text[:2000])
                except Exception as e:
                    st.error(f"Request failed: {e}")

        st.markdown("---")
        st.write("If you want me to wire a specific API (StatPal, RapidAPI, etc.), tell me the provider name and I will add a ready-to-use integration that reads the token from `st.secrets`.")
    else:
        st.info("Select a provider to configure props data source.")

with tabs[3]:
    # Use cached tennis data
    all_tennis_competitions = get_cached_data("tennis_competitions") or {"data": []}
    tennis_odds_data = get_cached_data("tennis_odds") or {"data": []}
    
    # Extract data arrays
    all_tennis_competitions = all_tennis_competitions.get('data', [])
    tennis_odds_data = tennis_odds_data.get('data', [])
    
    # Auto-load matchups for a specific tennis competition (with caching)
    def load_tennis_matchups(competition_id):
        cache_key = f"tennis_matchups_{competition_id}"
        cached = get_cached_data(cache_key)
        if cached:
            return cached.get('data', [])
        
        try:
            api_url = f"http://127.0.0.1:5001/api/tennis/matchups/{competition_id}"
            response = requests.get(api_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                set_cached_data(cache_key, data)
                return data.get('data', [])
            return []
        except:
            return []
    
    # Helper function to find odds for a specific tennis game
    def find_tennis_game_odds(game_id, odds_list):
        for odds in odds_list:
            if isinstance(odds, dict) and odds.get('bfid') == game_id:
                return odds
        return None
    
    if all_tennis_competitions:
        # Create controls section
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Tournament dropdown
            tournament_options = ["All Tournaments"] + [
                f"{comp['competition'].get('name', 'Unknown')} ({comp.get('competitionRegion', 'Int')})"
                for comp in all_tennis_competitions 
                if isinstance(comp, dict) and 'competition' in comp
            ]
            selected_tournament = st.selectbox(
                "🏆 Select Tournament",
                tournament_options,
                index=0
            )
        
        with col2:
            # Search bar for player names
            search_query = st.text_input(
                "🔍 Search Player",
                placeholder="Enter player name...",
                help="Search for matches by player name"
            )
        
        # Collect all matches from all tournaments for search
        all_matches = []
        tournaments_with_no_matches = []
        
        for comp in all_tennis_competitions:
            if isinstance(comp, dict) and 'competition' in comp:
                comp_info = comp['competition']
                comp_name = comp_info.get('name', 'Unknown Tournament')
                comp_region = comp.get('competitionRegion', 'International')
                markets_count = comp.get('marketCount', 0)
                
                # Load matchups for this competition
                comp_matchups = load_tennis_matchups(comp_info.get('id'))
                
                if comp_matchups:
                    for matchup in comp_matchups:
                        if isinstance(matchup, dict):
                            home_player = matchup.get('homeTeam', {}).get('name', 'Player 1')
                            away_player = matchup.get('awayTeam', {}).get('name', 'Player 2')
                            start_time = matchup.get('startTime', 'TBD')
                            game_id = matchup.get('bfid')
                            
                            match_data = {
                                'tournament': comp_name,
                                'region': comp_region,
                                'markets': markets_count,
                                'home_player': home_player,
                                'away_player': away_player,
                                'start_time': start_time,
                                'game_id': game_id,
                                'matchup': matchup
                            }
                            all_matches.append(match_data)
                else:
                    # Track tournaments with no matches
                    tournaments_with_no_matches.append({
                        'name': comp_name,
                        'region': comp_region,
                        'markets': markets_count
                    })
        
        # Filter matches based on tournament selection and search query
        filtered_matches = all_matches
        
        # Filter by tournament
        if selected_tournament != "All Tournaments":
            tournament_name = selected_tournament.split(" (")[0]  # Extract tournament name without region
            filtered_matches = [m for m in filtered_matches if m['tournament'] == tournament_name]
        
        # Filter by player search
        if search_query:
            search_lower = search_query.lower()
            filtered_matches = [
                m for m in filtered_matches 
                if search_lower in m['home_player'].lower() or search_lower in m['away_player'].lower()
            ]
        
        # Display filtered matches
        if filtered_matches:
            st.write(f"� **{len(filtered_matches)} matches found**")
            
            # Group matches by tournament for better organization
            matches_by_tournament = {}
            for match in filtered_matches:
                tournament = match['tournament']
                if tournament not in matches_by_tournament:
                    matches_by_tournament[tournament] = []
                matches_by_tournament[tournament].append(match)
            
            # Display matches grouped by tournament
            for tournament, matches in matches_by_tournament.items():
                with st.expander(f"🎾 {tournament} ({len(matches)} matches)", expanded=len(matches_by_tournament) <= 3):
                    for match in matches:
                        col_a, col_b, col_c = st.columns([3, 3, 2])
                        
                        with col_a:
                            st.markdown(f"**{match['away_player']}**")
                            st.caption("Player 1")
                        
                        with col_b:
                            st.markdown(f"**{match['home_player']}**")
                            st.caption("Player 2")
                        
                        with col_c:
                            st.caption(f"⏰ {match['start_time']}")
                            st.caption(f"🌍 {match['region']}")
                        
                        # Find odds for this match
                        match_odds = find_tennis_game_odds(match['game_id'], tennis_odds_data) if match['game_id'] else None
                        
                        if match_odds and 'marketIds' in match_odds:
                            st.markdown("**📊 Betting Markets:**")
                            for market in match_odds['marketIds'][:2]:  # Show max 2 markets to save space
                                if isinstance(market, dict):
                                    market_name = market.get('marketName', 'Match Winner')
                                    total_matched = market.get('totalMatched', 0)
                                    st.caption(f"💰 {market_name}: ${total_matched:,.2f}")
                        else:
                            st.caption("📊 Betting markets available")
                        
                        st.divider()
        
        else:
            if search_query:
                st.info(f"🔍 No matches found for player '{search_query}'" + 
                       (f" in {selected_tournament}" if selected_tournament != "All Tournaments" else ""))
                st.caption("Try a different search term or select 'All Tournaments'")
            else:
                # Show detailed information about tournament availability
                st.info("🎾 No live tennis matches currently scheduled")
                
                # Show available tournaments for context
                if tournaments_with_no_matches:
                    with st.expander("📋 Available Tournaments (No Current Matches)", expanded=False):
                        col_a, col_b, col_c = st.columns(3)
                        
                        for i, tournament in enumerate(tournaments_with_no_matches):
                            col = [col_a, col_b, col_c][i % 3]
                            with col:
                                st.markdown(f"**{tournament['name']}**")
                                st.caption(f"📍 {tournament['region']}")
                                st.caption(f"📊 {tournament['markets']} markets")
                                st.markdown("---")
                
                st.markdown("""
                **Possible reasons:**
                - ⏰ Matches may be scheduled for different time zones
                - 🏆 Tournament may be between rounds or off-season
                - 📅 Check back during peak tennis season (Grand Slams, ATP/WTA tours)
                
                **Try these tournaments for active matches:**
                - 🎾 ATP Shanghai Masters (October)
                - 🎾 WTA Finals (October-November)
                - 🎾 Various Challenger events
                """)
        
        # Footer
        st.markdown("---")
        from datetime import datetime
        current_time = datetime.now().strftime("%H:%M:%S")
        st.caption(f"Tennis data automatically loaded from Sportbex API at {current_time} • Updates in real-time")
        
    else:
        st.warning("⚠️ Tennis competitions data not available")
        st.info("Unable to load tennis competitions. Please check the API connection.")

with tabs[4]:
    st.markdown('<div class="section-title">Basketball<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    
    # Use cached basketball data
    all_competitions = get_cached_data("basketball_props") or {"data": []}
    odds_data = get_cached_data("basketball_odds") or {"data": []}
    
    # Extract data arrays
    all_competitions = all_competitions.get('data', [])
    odds_data = odds_data.get('data', [])
    
    # Auto-load matchups for a specific competition (with caching)
    def load_matchups(competition_id):
        cache_key = f"basketball_matchups_{competition_id}"
        cached = get_cached_data(cache_key)
        if cached:
            return cached.get('data', [])
        
        try:
            api_url = f"http://127.0.0.1:5001/api/basketball/matchups/{competition_id}"
            response = requests.get(api_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                set_cached_data(cache_key, data)
                return data.get('data', [])
            return []
        except:
            return []
    
    # Find NBA and WNBA competitions
    nba_comp = None
    wnba_comp = None
    
    for comp in all_competitions:
        if isinstance(comp, dict) and 'competition' in comp:
            comp_name = comp['competition'].get('name', '').upper()
            if comp_name == 'NBA':
                nba_comp = comp
            elif comp_name == 'WNBA':
                wnba_comp = comp
    
    # Helper function to find odds for a specific game
    def find_game_odds(game_id, odds_list):
        for odds in odds_list:
            if isinstance(odds, dict) and odds.get('bfid') == game_id:
                return odds
        return None
    
    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)
    
    # NBA Column (Left)
    with col1:
        st.subheader("🏀 NBA")
        
        if nba_comp:
            comp_info = nba_comp['competition']
            markets_count = nba_comp.get('marketCount', 0)
            
            st.success(f"✅ {markets_count} betting markets available")
            
            # Load NBA matchups automatically
            nba_matchups = load_matchups(comp_info.get('id'))
            
            if nba_matchups:
                st.write(f"**{len(nba_matchups)} Upcoming Games:**")
                
                # Add a container for better organization when showing many games
                with st.container():
                    for i, matchup in enumerate(nba_matchups):  # Show all NBA games
                        if isinstance(matchup, dict):
                            home_team = matchup.get('homeTeam', {}).get('name', 'TBD')
                            away_team = matchup.get('awayTeam', {}).get('name', 'TBD')
                            start_time = matchup.get('startTime', 'TBD')
                            game_id = matchup.get('bfid')
                            
                            # Find odds for this game
                            game_odds = find_game_odds(game_id, odds_data) if game_id else None
                            
                            with st.expander(f"🏀 {away_team} @ {home_team}", expanded=False):
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.write(f"**Away:** {away_team}")
                                    st.write(f"**Home:** {home_team}")
                                with col_b:
                                    st.write(f"**Time:** {start_time}")
                                    st.write(f"**Status:** {matchup.get('status', 'Scheduled')}")
                                
                                # Display odds if available
                                if game_odds and 'marketIds' in game_odds:
                                    st.markdown("**📊 Betting Odds:**")
                                    for market in game_odds['marketIds']:
                                        if isinstance(market, dict):
                                            market_name = market.get('marketName', 'Unknown')
                                            total_matched = market.get('totalMatched', 0)
                                            st.write(f"• **{market_name}**: ${total_matched:,.2f} total matched")
                                elif game_odds:
                                    st.info("📊 Odds data available - processing...")
                                else:
                                    st.caption("📊 No odds currently available")
                                
                                if 'markets' in matchup and matchup['markets']:
                                    st.write(f"**Markets:** {len(matchup['markets'])} available")
                
            else:
                st.info("📅 No games scheduled at this time")
                st.caption("Check back later for upcoming games or explore the betting markets available")
                
        else:
            st.warning("⚠️ NBA data not available")
            st.info("Unable to load NBA competition data")
    
    # WNBA Column (Right)
    with col2:
        st.subheader("🏀 WNBA")
        
        if wnba_comp:
            comp_info = wnba_comp['competition']
            markets_count = wnba_comp.get('marketCount', 0)
            
            st.success(f"✅ {markets_count} betting markets available")
            
            # Load WNBA matchups automatically
            wnba_matchups = load_matchups(comp_info.get('id'))
            
            if wnba_matchups:
                st.write(f"**{len(wnba_matchups)} Upcoming Games:**")
                
                # Add a container for better organization when showing many games
                with st.container():
                    for i, matchup in enumerate(wnba_matchups):  # Show all WNBA games
                        if isinstance(matchup, dict):
                            home_team = matchup.get('homeTeam', {}).get('name', 'TBD')
                            away_team = matchup.get('awayTeam', {}).get('name', 'TBD')
                            start_time = matchup.get('startTime', 'TBD')
                            game_id = matchup.get('bfid')
                            
                            # Find odds for this game
                            game_odds = find_game_odds(game_id, odds_data) if game_id else None
                            
                            with st.expander(f"🏀 {away_team} @ {home_team}", expanded=False):
                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.write(f"**Away:** {away_team}")
                                    st.write(f"**Home:** {home_team}")
                                with col_b:
                                    st.write(f"**Time:** {start_time}")
                                    st.write(f"**Status:** {matchup.get('status', 'Scheduled')}")
                                
                                # Display odds if available
                                if game_odds and 'marketIds' in game_odds:
                                    st.markdown("**📊 Betting Odds:**")
                                    for market in game_odds['marketIds']:
                                        if isinstance(market, dict):
                                            market_name = market.get('marketName', 'Unknown')
                                            total_matched = market.get('totalMatched', 0)
                                            st.write(f"• **{market_name}**: ${total_matched:,.2f} total matched")
                                elif game_odds:
                                    st.info("📊 Odds data available - processing...")
                                else:
                                    st.caption("📊 No odds currently available")
                                
                                if 'markets' in matchup and matchup['markets']:
                                    st.write(f"**Markets:** {len(matchup['markets'])} available")
                
            else:
                st.info("📅 No games scheduled at this time")
                st.caption("Check back later for upcoming games or explore the betting markets available")
                
        else:
            st.warning("⚠️ WNBA data not available")
            st.info("Unable to load WNBA competition data")
    
    # Footer
    st.markdown("---")
    from datetime import datetime
    current_time = datetime.now().strftime("%H:%M:%S")
    st.caption(f"Data automatically loaded from Sportbex API at {current_time} • Updates in real-time")

# FOOTBALL TAB
with tabs[5]:
    st.markdown('<div class="section-title">Football<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    
    # Use cached football data
    all_football_competitions = get_cached_data("football_competitions") or {"data": []}
    football_odds_data = get_cached_data("football_odds") or {"data": []}
    
    # Extract data arrays
    all_football_competitions = all_football_competitions.get('data', [])
    football_odds_data = football_odds_data.get('data', [])
    
    # Auto-load matchups for a specific football competition (with caching)
    def load_football_matchups(competition_id):
        cache_key = f"football_matchups_{competition_id}"
        cached = get_cached_data(cache_key)
        if cached:
            return cached.get('data', [])
        
        try:
            api_url = f"http://127.0.0.1:5001/api/football/matchups/{competition_id}"
            response = requests.get(api_url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                set_cached_data(cache_key, data)
                return data.get('data', [])
            return []
        except:
            return []
    
    # Helper function to find odds for a specific football game
    def find_football_game_odds(game_id, odds_list):
        for odds in odds_list:
            if isinstance(odds, dict) and odds.get('bfid') == game_id:
                return odds
        return None
    
    if all_football_competitions:
        st.success(f"🏈 Found {len(all_football_competitions)} football competitions")
        
        # Display football competitions
        for i, comp in enumerate(all_football_competitions):
            if isinstance(comp, dict) and 'competition' in comp:
                comp_info = comp['competition']
                comp_name = comp_info.get('name', 'Unknown League')
                comp_region = comp.get('competitionRegion', 'USA')
                markets_count = comp.get('marketCount', 0)
                
                with st.expander(f"🏈 {comp_name}", expanded=i < 3):  # Expand first 3 leagues
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Region:** {comp_region}")
                        st.write(f"**Markets:** {markets_count} available")
                    
                    # Load matchups for this competition
                    comp_matchups = load_football_matchups(comp_info.get('id'))
                    
                    if comp_matchups:
                        st.write(f"**{len(comp_matchups)} Upcoming Games:**")
                        
                        for matchup in comp_matchups:
                            if isinstance(matchup, dict):
                                home_team = matchup.get('homeTeam', {}).get('name', 'Home Team')
                                away_team = matchup.get('awayTeam', {}).get('name', 'Away Team')
                                start_time = matchup.get('startTime', 'TBD')
                                game_id = matchup.get('bfid')
                                
                                # Find odds for this match
                                match_odds = find_football_game_odds(game_id, football_odds_data) if game_id else None
                                
                                with st.container():
                                    col_a, col_b, col_c = st.columns([2, 2, 1])
                                    with col_a:
                                        st.markdown(f"**{away_team}**")
                                        st.caption("Away")
                                    with col_b:
                                        st.markdown(f"**{home_team}**")
                                        st.caption("Home")
                                    with col_c:
                                        st.caption(f"⏰ {start_time}")
                                    
                                    if match_odds and 'marketIds' in match_odds:
                                        st.markdown("**📊 Betting Markets:**")
                                        for market in match_odds['marketIds'][:3]:  # Show max 3 markets
                                            if isinstance(market, dict):
                                                market_name = market.get('marketName', 'Moneyline')
                                                total_matched = market.get('totalMatched', 0)
                                                st.caption(f"💰 {market_name}: ${total_matched:,.2f} matched")
                                    elif comp_matchups:
                                        st.caption("📊 Betting odds available")
                                    
                                    st.markdown("---")
                    else:
                        st.info("📅 No games scheduled currently")
                        st.caption("Check back during football season for live games")
        
        # Footer
        st.markdown("---")
        from datetime import datetime
        current_time = datetime.now().strftime("%H:%M:%S")
        st.caption(f"Football data automatically loaded from Sportbex API at {current_time} • Updates in real-time")
        
    else:
        st.warning("⚠️ Football competitions data not available")
        st.info("Football season may be off-season. Check back during NFL/College Football season.")

# BASEBALL TAB
with tabs[6]:
    st.markdown('<div class="section-title">Baseball<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    
    st.info("⚾ Baseball API Integration Ready")
    st.markdown("""
    **Available for Implementation:**
    - MLB Regular Season & Playoffs
    - World Series Betting Markets
    - Player Props (Hits, RBIs, Home Runs)
    - Team Totals and Spreads
    
    *Contact support to enable Baseball API endpoints*
    """)
    
    # Sample placeholder data structure
    st.write("**Sample Data Structure:**")
    sample_baseball = {
        "leagues": ["MLB", "Minor League"],
        "markets": ["Moneyline", "Run Line", "Over/Under", "Player Props"],
        "teams": 30,
        "status": "Ready for API integration"
    }
    st.json(sample_baseball)

# HOCKEY TAB
with tabs[7]:
    st.markdown('<div class="section-title">Hockey<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    
    st.info("🏒 Hockey API Integration Ready")
    st.markdown("""
    **Available for Implementation:**
    - NHL Regular Season & Playoffs
    - Stanley Cup Betting Markets
    - Player Props (Goals, Assists, Shots)
    - Team Totals and Puck Line
    
    *Contact support to enable Hockey API endpoints*
    """)
    
    # Sample placeholder data structure
    st.write("**Sample Data Structure:**")
    sample_hockey = {
        "leagues": ["NHL", "International"],
        "markets": ["Moneyline", "Puck Line", "Over/Under", "Player Props"],
        "teams": 32,
        "status": "Ready for API integration"
    }
    st.json(sample_hockey)
    st.info("Replace this section with actual Hockey API data display.")

# SOCCER TAB
with tabs[8]:
    st.markdown('<div class="section-title">Soccer<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    
    # Auto-load soccer data on page load
    def load_soccer_data():
        try:
            api_url = "http://127.0.0.1:5001/api/soccer/competitions"
            response = requests.get(api_url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and 'data' in data and isinstance(data['data'], list):
                    return data['data']
            return []
        except:
            return []
    
    # Auto-load matchups for a specific soccer competition
    def load_soccer_matchups(competition_id):
        try:
            api_url = f"http://127.0.0.1:5001/api/soccer/matchups/{competition_id}"
            response = requests.get(api_url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                if isinstance(data, dict) and 'data' in data:
                    return data['data']
            return []
        except Exception as e:
            return []
    
    # Load all soccer competitions
    all_soccer_competitions = load_soccer_data()
    
    if all_soccer_competitions:
        st.success(f"⚽ Found {len(all_soccer_competitions)} soccer competitions")
        
        # Create two columns for better organization
        col1, col2 = st.columns(2)
        mid_point = len(all_soccer_competitions) // 2
        
        with col1:
            st.subheader("⚽ Major Leagues (Part 1)")
            for i, comp in enumerate(all_soccer_competitions[:mid_point]):
                if isinstance(comp, dict) and 'competition' in comp:
                    comp_info = comp['competition']
                    comp_name = comp_info.get('name', 'Unknown League')
                    comp_region = comp.get('competitionRegion', 'International')
                    markets_count = comp.get('marketCount', 0)
                    
                    with st.expander(f"⚽ {comp_name}", expanded=False):
                        st.write(f"**Region:** {comp_region}")
                        st.write(f"**Markets:** {markets_count} available")
                        
                        # Load matchups for this competition
                        comp_matchups = load_soccer_matchups(comp_info.get('id'))
                        
                        if comp_matchups:
                            st.write(f"**{len(comp_matchups)} Upcoming Matches:**")
                            for matchup in comp_matchups[:5]:  # Show max 5 matches
                                if isinstance(matchup, dict):
                                    home_team = matchup.get('homeTeam', {}).get('name', 'Home Team')
                                    away_team = matchup.get('awayTeam', {}).get('name', 'Away Team')
                                    start_time = matchup.get('startTime', 'TBD')
                                    
                                    st.markdown(f"**{away_team} vs {home_team}**")
                                    st.caption(f"⏰ {start_time}")
                                    
                                    if 'markets' in matchup and matchup['markets']:
                                        st.caption(f"📊 {len(matchup['markets'])} betting markets")
                                    
                                    st.markdown("---")
                        else:
                            st.info("📅 No matches scheduled currently")
        
        with col2:
            st.subheader("⚽ Major Leagues (Part 2)")
            for i, comp in enumerate(all_soccer_competitions[mid_point:]):
                if isinstance(comp, dict) and 'competition' in comp:
                    comp_info = comp['competition']
                    comp_name = comp_info.get('name', 'Unknown League')
                    comp_region = comp.get('competitionRegion', 'International')
                    markets_count = comp.get('marketCount', 0)
                    
                    with st.expander(f"⚽ {comp_name}", expanded=False):
                        st.write(f"**Region:** {comp_region}")
                        st.write(f"**Markets:** {markets_count} available")
                        
                        # Load matchups for this competition
                        comp_matchups = load_soccer_matchups(comp_info.get('id'))
                        
                        if comp_matchups:
                            st.write(f"**{len(comp_matchups)} Upcoming Matches:**")
                            for matchup in comp_matchups[:5]:  # Show max 5 matches
                                if isinstance(matchup, dict):
                                    home_team = matchup.get('homeTeam', {}).get('name', 'Home Team')
                                    away_team = matchup.get('awayTeam', {}).get('name', 'Away Team')
                                    start_time = matchup.get('startTime', 'TBD')
                                    
                                    st.markdown(f"**{away_team} vs {home_team}**")
                                    st.caption(f"⏰ {start_time}")
                                    
                                    if 'markets' in matchup and matchup['markets']:
                                        st.caption(f"📊 {len(matchup['markets'])} betting markets")
                                    
                                    st.markdown("---")
                        else:
                            st.info("📅 No matches scheduled currently")
        
        # Footer
        st.markdown("---")
        from datetime import datetime
        current_time = datetime.now().strftime("%H:%M:%S")
        st.caption(f"Soccer data automatically loaded from Sportbex API at {current_time} • Updates in real-time")
        
    else:
        st.warning("⚠️ Soccer competitions data not available")
        st.info("Unable to load soccer competitions. Please check the API connection.")

# ESPORTS TAB
with tabs[9]:
    st.markdown('<div class="section-title">Esports<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    
    st.info("🎮 Esports API Integration Ready")
    st.markdown("""
    **Available for Implementation:**
    - League of Legends (LCS, LEC, Worlds)
    - CS:GO/CS2 Major Tournaments
    - Dota 2 (The International, DPC)
    - Valorant Championship Series
    - Overwatch League
    
    *API endpoints available via RapidAPI integration*
    """)
    
    # Show existing esports integration status
    st.write("**Current Integration Status:**")
    esports_status = {
        "RapidAPI": "✅ Connected",
        "Endpoints": ["Team Info", "Tournament Data", "Season Info", "Match History"],
        "Games": ["LoL", "CS:GO", "Dota 2", "Valorant"],
        "status": "Partially integrated - extend for betting markets"
    }
    st.json(esports_status)
    
    st.markdown("""
    **Available Functions:**
    - `get_esport_team(team_id)` - Get team information
    - `get_esport_tournament(tournament_id)` - Get tournament details
    - `get_esport_season_info(tournament_id, season_id)` - Season information
    - `get_esport_season_last_matches(tournament_id, season_id, page)` - Recent matches
    """)

# COLLEGE FOOTBALL TAB
with tabs[10]:
    st.markdown('<div class="section-title">College Football<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    
    st.info("🏈 College Football API Integration Ready")
    st.markdown("""
    **Available for Implementation:**
    - NCAA Division I FBS
    - College Football Playoff
    - Bowl Games & Championships
    - Conference Championships
    - Player Props & Team Totals
    
    *Contact support to enable College Football API endpoints*
    """)
    
    # Sample placeholder data structure
    st.write("**Sample Data Structure:**")
    sample_college_football = {
        "conferences": ["SEC", "Big Ten", "ACC", "Big 12", "Pac-12"],
        "markets": ["Spread", "Moneyline", "Over/Under", "Player Props"],
        "teams": 130,
        "playoff_teams": 12,
        "status": "Ready for API integration"
    }
    st.json(sample_college_football)
    
    st.markdown("""
    **Key Features Ready:**
    - Conference standings and rankings
    - Bowl game predictions
    - Player performance metrics
    - Live betting during games
    """)




