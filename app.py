


import streamlit as st
import pandas as pd
import requests
import os
from requests_oauthlib import OAuth1Session
from lxml import etree


# --- RESTORED MULTI-TAB UI ---
tab_names = [
    "Home", "Stats", "Props", "Tennis", "Basketball", "Football", "Baseball",
    "Hockey", "Soccer", "Esports", "College Football"
]
tabs = st.tabs(tab_names)

# --- Yahoo OAuth in Sidebar (Global) ---
CONSUMER_KEY = st.secrets["YAHOO_CONSUMER_KEY"]
CONSUMER_SECRET = st.secrets["YAHOO_CONSUMER_SECRET"]
REQUEST_TOKEN_URL = "https://api.login.yahoo.com/oauth/v2/get_request_token"
AUTHORIZE_URL = "https://api.login.yahoo.com/oauth/v2/request_auth"
ACCESS_TOKEN_URL = "https://api.login.yahoo.com/oauth/v2/get_token"
CALLBACK_URI = "oob"  # For Streamlit, use oob/manual for now


# --- Session-based Yahoo OAuth Token Storage (Streamlit Cloud compatible) ---
if 'yahoo_access_token' not in st.session_state:
    st.session_state['yahoo_access_token'] = None
    st.session_state['yahoo_access_token_secret'] = None
    st.session_state['yahoo_resource_owner_key'] = None
    st.session_state['yahoo_resource_owner_secret'] = None
    st.session_state['yahoo_oauth_step'] = 0

sidebar = st.sidebar
sidebar.title("Yahoo Fantasy Login")
if 'yahoo_access_token' not in st.session_state:
    st.session_state['yahoo_access_token'] = None
    st.session_state['yahoo_access_token_secret'] = None
    st.session_state['yahoo_resource_owner_key'] = None
    st.session_state['yahoo_resource_owner_secret'] = None
    st.session_state['yahoo_oauth_step'] = 0


# Step 1: Start OAuth
if st.session_state['yahoo_oauth_step'] == 0:
    if sidebar.button("Login with Yahoo!"):
        yahoo = OAuth1Session(CONSUMER_KEY, client_secret=CONSUMER_SECRET, callback_uri=CALLBACK_URI)
        fetch_response = yahoo.fetch_request_token(REQUEST_TOKEN_URL)
        st.session_state['yahoo_resource_owner_key'] = fetch_response.get('oauth_token')
        st.session_state['yahoo_resource_owner_secret'] = fetch_response.get('oauth_token_secret')
        st.session_state['yahoo_oauth_step'] = 1
        st.experimental_rerun()

# Step 2: Show authorize URL and get verifier
elif st.session_state['yahoo_oauth_step'] == 1:
    auth_url = f"{AUTHORIZE_URL}?oauth_token={st.session_state['yahoo_resource_owner_key']}"
    sidebar.markdown(f"[Click here to authorize Yahoo! access]({auth_url})")
    verifier = sidebar.text_input("Paste the verifier code from Yahoo here:")
    if verifier:
        yahoo = OAuth1Session(
            CONSUMER_KEY,
            client_secret=CONSUMER_SECRET,
            resource_owner_key=st.session_state['yahoo_resource_owner_key'],
            resource_owner_secret=st.session_state['yahoo_resource_owner_secret'],
            verifier=verifier,
        )
        access_token_data = yahoo.fetch_access_token(ACCESS_TOKEN_URL)
        st.session_state['yahoo_access_token'] = access_token_data.get('oauth_token')
        st.session_state['yahoo_access_token_secret'] = access_token_data.get('oauth_token_secret')
        st.session_state['yahoo_oauth_step'] = 2
        # Store tokens in session only (no file write)
        sidebar.success("Yahoo! authentication complete.")
        st.experimental_rerun()

# Step 3: Authenticated
elif st.session_state['yahoo_oauth_step'] == 2:
    sidebar.success("Authenticated with Yahoo!")
    if sidebar.button("Logout Yahoo!"):
        st.session_state['yahoo_access_token'] = None
        st.session_state['yahoo_access_token_secret'] = None
        st.session_state['yahoo_resource_owner_key'] = None
        st.session_state['yahoo_resource_owner_secret'] = None
        st.session_state['yahoo_oauth_step'] = 0
    # No file to clear; session only
        st.experimental_rerun()




# Home Tab (placeholder, no demo)
with tabs[0]:
    st.header("🏟️ Sports News & Highlights")
    st.write("Welcome to BetFinder AI! Use the tabs above to explore player props, stats, and more.")



# Stats Tab (placeholder, no demo)
with tabs[1]:
    st.header("Projections")
    st.info("Stats integration coming soon.")






# Props Tab (Yahoo API only)
with tabs[2]:
    st.header("Props")
    if st.session_state['yahoo_oauth_step'] == 2:
        yahoo = OAuth1Session(
            CONSUMER_KEY,
            client_secret=CONSUMER_SECRET,
            resource_owner_key=st.session_state['yahoo_access_token'],
            resource_owner_secret=st.session_state['yahoo_access_token_secret'],
        )
        # Yahoo's closest to "props" is contest/games/players stats. We'll fetch NFL games and show a table of games as a starting point.
        resp = yahoo.get("https://fantasysports.yahooapis.com/fantasy/v2/game/nfl;out=leagues")
        if resp.status_code == 200:
            xml_root = etree.fromstring(resp.content)
            # Try to extract league/game info for display
            leagues = xml_root.findall('.//league')
            if leagues:
                league_rows = []
                for league in leagues:
                    lid = league.findtext('league_id')
                    lname = league.findtext('name')
                    ltype = league.findtext('league_type')
                    lseason = league.findtext('season')
                    league_rows.append({
                        'League ID': lid,
                        'Name': lname,
                        'Type': ltype,
                        'Season': lseason
                    })
                st.markdown("**Your Yahoo NFL Leagues:**")
                st.dataframe(pd.DataFrame(league_rows), use_container_width=True, hide_index=True)
                st.info("To show player props, connect to a league and fetch player stats. Further integration available on request.")
            else:
                st.info("No leagues found. Join a Yahoo Fantasy league to see props/stats.")
        else:
            st.error(f"Yahoo API error: {resp.status_code}")
    else:
        st.warning("Please log in with Yahoo to view props.")


# Tennis Tab (placeholder, no demo)
with tabs[3]:
    st.header("Tennis")
    st.info("Tennis integration coming soon.")


# Basketball Tab (placeholder, no demo)
with tabs[4]:
    st.header("Basketball")
    st.info("Basketball integration coming soon.")


# Football Tab (placeholder, no demo)
with tabs[5]:
    st.header("Football")
    st.info("Football integration coming soon.")


# Baseball Tab (placeholder, no demo)
with tabs[6]:
    st.header("Baseball")
    st.info("Baseball integration coming soon.")


# Hockey Tab (placeholder, no demo)
with tabs[7]:
    st.header("Hockey")
    st.info("Hockey integration coming soon.")


# Soccer Tab (placeholder, no demo)
with tabs[8]:
    st.header("Soccer")
    st.info("Soccer integration coming soon.")


# Esports Tab (placeholder, no demo)
with tabs[9]:
    st.header("Esports")
    st.info("Esports integration coming soon.")


# College Football Tab (placeholder, no demo)
with tabs[10]:
    st.header("College Football")
    st.info("College Football integration coming soon.")


    
# COLLEGE FOOTBALL TAB
with tabs[10]:
    st.write("**College Football Data (API Demo):**")
    st.info("Replace this section with actual College Football API data display.")






# TENNIS TAB
with tabs[3]:
    st.markdown('<div class="section-title">Tennis<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    st.write("**Tennis Data (API Demo):**")
    st.info("Replace this section with actual Tennis API data display.")

# BASKETBALL TAB
with tabs[4]:
    st.markdown('<div class="section-title">Basketball<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    st.write("**Basketball Data (API Demo):**")
    st.info("Replace this section with actual Basketball API data display.")

# FOOTBALL TAB
with tabs[5]:
    st.markdown('<div class="section-title">Football<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    st.write("**Football Data (API Demo):**")
    st.info("Replace this section with actual Football API data display.")

# BASEBALL TAB
with tabs[6]:
    st.markdown('<div class="section-title">Baseball<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    st.write("**Baseball Data (API Demo):**")
    st.info("Replace this section with actual Baseball API data display.")

# HOCKEY TAB
with tabs[7]:
    st.markdown('<div class="section-title">Hockey<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    st.write("**Hockey Data (API Demo):**")
    st.info("Replace this section with actual Hockey API data display.")

# SOCCER TAB
with tabs[8]:
    st.markdown('<div class="section-title">Soccer<span class="time">Live & Upcoming</span></div>', unsafe_allow_html=True)
    st.write("**Soccer Data (API Demo):**")
    st.info("Replace this section with actual Soccer API data display.")




