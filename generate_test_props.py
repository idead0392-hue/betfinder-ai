#!/usr/bin/env python3
"""
Generate realistic test props data for BetFinder AI
"""
import pandas as pd

def generate_realistic_props():
    """Generate realistic current props data"""
    
    # NBA Props
    nba_props = [
        ("LeBron James", 26.5, "Points", "nba"),
        ("Anthony Davis", 12.5, "Rebounds", "nba"),
        ("Stephen Curry", 4.5, "3PT Made", "nba"),
        ("Giannis Antetokounmpo", 29.5, "Points", "nba"),
        ("Nikola Jokic", 9.5, "Assists", "nba"),
        ("Jayson Tatum", 27.5, "Points", "nba"),
        ("Luka Doncic", 8.5, "Assists", "nba"),
        ("Joel Embiid", 11.5, "Rebounds", "nba"),
        ("Damian Lillard", 24.5, "Points", "nba"),
        ("Kevin Durant", 25.5, "Points", "nba"),
    ]
    
    # NFL Props  
    nfl_props = [
        ("Patrick Mahomes", 285.5, "Passing Yards", "nfl"),
        ("Josh Allen", 275.5, "Passing Yards", "nfl"),
        ("Travis Kelce", 6.5, "Receptions", "nfl"),
        ("Tyreek Hill", 75.5, "Receiving Yards", "nfl"),
        ("Derrick Henry", 85.5, "Rushing Yards", "nfl"),
        ("Cooper Kupp", 7.5, "Receptions", "nfl"),
        ("Aaron Rodgers", 265.5, "Passing Yards", "nfl"),
        ("Davante Adams", 8.5, "Receptions", "nfl"),
    ]
    
    # MLB Props
    mlb_props = [
        ("Shohei Ohtani", 1.5, "Total Bases", "mlb"),
        ("Aaron Judge", 1.5, "Total Bases", "mlb"),
        ("Mookie Betts", 1.5, "Hits", "mlb"),
        ("Ronald Acuna Jr", 1.5, "Stolen Bases", "mlb"),
        ("Freddie Freeman", 1.5, "Hits", "mlb"),
    ]
    
    # NHL Props
    nhl_props = [
        ("Connor McDavid", 3.5, "Shots on Goal", "nhl"),
        ("Leon Draisaitl", 2.5, "Points", "nhl"),
        ("Nathan MacKinnon", 3.5, "Shots on Goal", "nhl"),
        ("Erik Karlsson", 0.5, "Goals", "nhl"),
        ("David Pastrnak", 2.5, "Shots on Goal", "nhl"),
    ]
    
    # Soccer Props
    soccer_props = [
        ("Lionel Messi", 0.5, "Goals", "epl"),
        ("Erling Haaland", 1.5, "Goals", "epl"),
        ("Kylian Mbappe", 1.5, "Shots on Target", "epl"),
        ("Kevin De Bruyne", 0.5, "Assists", "epl"),
    ]
    
    # Tennis Props
    tennis_props = [
        ("Novak Djokovic", 9.5, "Aces", "tennis"),
        ("Carlos Alcaraz", 8.5, "Aces", "tennis"),
        ("Iga Swiatek", 6.5, "Games Won", "tennis"),
    ]
    
    # Esports Props
    esports_props = [
        ("s1mple", 22.5, "Kills", "cs2"),
        ("ZywOo", 21.5, "Kills", "cs2"),
        ("Faker", 7.5, "Assists", "lol"),
        ("Canyon", 6.5, "Assists", "lol"),
        ("Miracle-", 250.5, "Last Hits", "dota2"),
        ("Puppey", 15.5, "Assists", "dota2"),
        ("TenZ", 18.5, "Kills", "valorant"),
        ("Aspas", 17.5, "Kills", "valorant"),
        ("Profit", 15.5, "Eliminations", "overwatch"),
        ("Leave", 14.5, "Eliminations", "overwatch"),
        ("jstn.", 1.5, "Goals", "rocket"),
        ("GarrettG", 2.5, "Saves", "rocket"),
    ]
    
    # College Football Props
    cfb_props = [
        ("Caleb Williams", 275.5, "Passing Yards", "cfb"),
        ("Drake Maye", 285.5, "Passing Yards", "cfb"),
        ("Marvin Harrison Jr", 85.5, "Receiving Yards", "cfb"),
        ("Rome Odunze", 75.5, "Receiving Yards", "cfb"),
    ]
    
    # Combine all props
    all_props = nba_props + nfl_props + mlb_props + nhl_props + soccer_props + tennis_props + esports_props + cfb_props
    
    # Create DataFrame
    df = pd.DataFrame(all_props, columns=['Name', 'Points', 'Prop', 'Sport'])
    
    return df

def main():
    print("Generating realistic test props...")
    df = generate_realistic_props()
    
    # Save to CSV
    output_file = "prizepicks_props.csv"
    df.to_csv(output_file, index=False)
    
    print(f"Generated {len(df)} props and saved to {output_file}")
    print(f"Sports covered: {', '.join(df['Sport'].unique())}")
    print("\nSample props:")
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()