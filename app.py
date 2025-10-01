import streamlit as st

st.set_page_config("BetFinder AI", layout="wide")
st.title("BetFinder AI - Sports Props Dashboard")
st.write("Welcome! Use the sidebar filters to explore value props.")

# Define navigation tabs with at least 3 elements
nav_tabs = ["Home", "Analytics", "Settings"]

# Create tabs in the Streamlit app
tabs = st.tabs(nav_tabs)

# Example content for each tab
with tabs[0]:
    st.header("Home")
    st.write("This is the Home tab.")

with tabs[1]:
    st.header("Analytics")
    st.write("This is the Analytics tab.")

with tabs[2]:
    st.header("Settings")
    st.write("This is the Settings tab.")

# ---- Placeholder for your dashboard ----
# Future: Add DataFrame, Odds API calls, projections here
st.markdown("---")
st.info("This is a working starter. Build your features from here!")