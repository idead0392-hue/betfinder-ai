


import streamlit as st
import pandas as pd
import requests
import os
import time
from requests_oauthlib import OAuth2Session
from oauthlib.oauth2 import WebApplicationClient
from lxml import etree


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


# --- RESTORED MULTI-TAB UI ---
tab_names = [
    "Home", "Stats", "Props", "Tennis", "Basketball", "Football", "Baseball",
    "Hockey", "Soccer", "Esports", "College Football"
]
tabs = st.tabs(tab_names)


# --- Yahoo OAuth2 in Sidebar (Global) ---
CLIENT_ID = st.secrets["YAHOO_CONSUMER_KEY"]
CLIENT_SECRET = st.secrets["YAHOO_CONSUMER_SECRET"]
AUTHORIZATION_BASE_URL = "https://api.login.yahoo.com/oauth2/request_auth"
TOKEN_URL = "https://api.login.yahoo.com/oauth2/get_token"
REDIRECT_URI = "oob"  # For Streamlit, use oob/manual for now

# --- Simple Yahoo API Rate Counter ---
if 'yahoo_rate_count' not in st.session_state:
    st.session_state['yahoo_rate_count'] = 0

def increment_yahoo_rate():
    st.session_state['yahoo_rate_count'] += 1

sidebar = st.sidebar
sidebar.title("Yahoo Fantasy Login")
sidebar.markdown(f"**Yahoo API Calls this session:** {st.session_state['yahoo_rate_count']}")


# --- Session-based Yahoo OAuth Token Storage (Streamlit Cloud compatible) ---
if 'yahoo_access_token' not in st.session_state:
    st.session_state['yahoo_access_token'] = None
    st.session_state['yahoo_refresh_token'] = None
    st.session_state['yahoo_oauth_state'] = None
    st.session_state['yahoo_oauth_step'] = 0





# Step 1: Start OAuth (only show login if not authenticated)
if st.session_state['yahoo_oauth_step'] == 0:
    if st.session_state['yahoo_access_token'] is None:
        if sidebar.button("Login with Yahoo!"):
            client = WebApplicationClient(CLIENT_ID)
            yahoo = OAuth2Session(client=client, redirect_uri=REDIRECT_URI)
            
            # Get authorization URL
            authorization_url, state = yahoo.authorization_url(AUTHORIZATION_BASE_URL)
            st.session_state['yahoo_oauth_state'] = state
            st.session_state['yahoo_oauth_step'] = 1
            safe_rerun()

# Step 2: Show authorize URL and get authorization code
elif st.session_state['yahoo_oauth_step'] == 1:
    client = WebApplicationClient(CLIENT_ID)
    yahoo = OAuth2Session(client=client, state=st.session_state['yahoo_oauth_state'], redirect_uri=REDIRECT_URI)
    
    authorization_url, _ = yahoo.authorization_url(AUTHORIZATION_BASE_URL)
    sidebar.markdown(f"[Click here to authorize Yahoo! access]({authorization_url})")
    auth_code = sidebar.text_input("Paste the authorization code from Yahoo here:")
    if auth_code:
        try:
            token = yahoo.fetch_token(TOKEN_URL, client_secret=CLIENT_SECRET, code=auth_code)
            st.session_state['yahoo_access_token'] = token.get('access_token')
            st.session_state['yahoo_refresh_token'] = token.get('refresh_token')
            st.session_state['yahoo_oauth_step'] = 2
            safe_rerun()
        except Exception as e:
            st.error(f"Failed to get access token: {e}")
        sidebar.success("Yahoo! authentication complete.")
        # safe_rerun already called on success; if we reach here show refresh hint
        st.info("If authentication succeeded, please refresh the page.")

# Step 3: Authenticated
elif st.session_state['yahoo_oauth_step'] == 2:
    sidebar.success("Authenticated with Yahoo!")
    if sidebar.button("Logout Yahoo!"):
        st.session_state['yahoo_access_token'] = None
        st.session_state['yahoo_refresh_token'] = None
        st.session_state['yahoo_oauth_state'] = None
        st.session_state['yahoo_oauth_step'] = 0
        # No file to clear; session only
        safe_rerun()




# Home Tab (placeholder, no demo)
with tabs[0]:
    st.header("🏟️ Sports News & Highlights")
    st.write("Welcome to BetFinder AI! Use the tabs above to explore player props, stats, and more.")



# Stats Tab (placeholder, no demo)
with tabs[1]:
    st.header("Projections")
    st.info("Stats integration coming soon.")


# Props Tab (placeholder)
with tabs[2]:
    st.header("Props")
    st.info("Props provider removed. Add a new data provider or re-enable the integration if needed.")


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




