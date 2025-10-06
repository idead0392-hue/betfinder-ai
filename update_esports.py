import re

with open("app.py", "r") as f:
    content = f.read()

pattern = r"(with esports_tab:)(.*?)((?=^with \w+_tab:)|(?=^# End of tabs)|$)"
match = re.search(pattern, content, re.MULTILINE | re.DOTALL)

if match:
    new_section = """with esports_tab:
    st.header("🎮 Esports Betting Opportunities")
    
    try:
        games_data = fetch_esports_games()
        
        if games_data and 'data' in games_data and games_data['data']:
            st.success(f"Found {len(games_data['data'])} esports games")
            
            for idx, game in enumerate(games_data['data']):
                st.subheader(f"Game {idx + 1}")
                
                # Display raw API response for debugging
                st.write("**Raw API response:**")
                st.write(game)
                
                st.markdown("---")
        else:
            st.info("No esports games available at the moment")
            
    except Exception as e:
        st.error(f"Error fetching esports data: {str(e)}")
        import traceback
        st.code(traceback.format_exc())

"""

    content = content[: match.start()] + new_section + content[match.end() :]

    with open("app.py", "w") as f:
        f.write(content)

    print("Updated esports tab successfully!")
else:
    print("Could not find esports tab")
