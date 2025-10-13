# PrizePicks Integration Usage Guide

## Overview

The `player_prop_predictor.py` Streamlit app now supports fetching and displaying real props data from PrizePicks through the integrated `props_data_fetcher` and `prizepicks_provider` modules.

## Features

- **Real-time Props Data**: Fetches props from PrizePicks CSV cache (updated automatically)
- **Multi-Sport Support**: Basketball, Football, Baseball, Hockey, Tennis, Soccer, Esports (CS:GO, LoL, Dota2)
- **Interactive Filtering**: Filter by sport, stat type, player name, and team
- **Clean Display**: Props rendered in a structured DataFrame with key metrics
- **Sports Breakdown**: Visual metrics showing prop counts by sport

## How to Use

### 1. Run the Streamlit App

```bash
streamlit run player_prop_predictor.py
```

### 2. Select Data Source

In the sidebar, under "Data Source", select **PrizePicks** from the dropdown.

Available options:
- Mock Data (default demonstration data)
- SportbexProvider (if configured)
- **PrizePicks** (real props from CSV)

### 3. Apply Filters

Use the sidebar filters to narrow down props:

- **Sport**: Filter by specific sport (Basketball, Football, etc.)
- **Stat Type**: Filter by stat category (Points, Rebounds, etc.)
- **Search**: Search for specific players or teams
- **Date Range**: Filter by game dates

### 4. View Props

The main area displays:

- **Summary Metrics**: Total props count and breakdown by sport
- **Props Table**: Detailed table with columns:
  - Player Name
  - Sport/League
  - Stat Type
  - Line Value
  - Team
  - Matchup
  - Over/Under
  - Confidence Score
  - Odds

## Data Structure

Props fetched from PrizePicks include:

```python
{
    'player_name': 'LeBron James',
    'team': 'LAL',
    'stat_type': 'points',
    'line': 27.5,
    'odds': -110,
    'confidence': 70.0,
    'sport': 'basketball',
    'league': 'NBA',
    'matchup': 'LAL vs BOS',
    'over_under': None,
    'start_time': '2025-10-13 7:30 PM ET'
}
```

## Testing

Three test scripts are available:

### 1. Basic Integration Test

```bash
python test_prizepicks_integration.py
```

Tests the basic props fetching and field validation.

### 2. Render Logic Test

```bash
python test_render_props.py
```

Tests the DataFrame conversion and filter logic.

### 3. Full Demo

```bash
python demo_prizepicks_integration.py
```

Runs a comprehensive demo showing all features with sample output.

## CSV Data Source

The integration reads from `prizepicks_props.csv` which is automatically updated by the scraping process. The CSV contains:

- **5000+ props** across multiple sports
- **Real-time updates** via auto-scraper
- **Multiple stat types** per sport
- **Live and upcoming games**

## Implementation Details

### Key Functions

**`fetch_prizepicks_props(max_props: int)`**
- Fetches props from PrizePicks via props_data_fetcher
- Caches results for 5 minutes
- Filters out restricted esports stats
- Returns: `(List[Dict], Optional[str])` - props list and error message

**`render_props(props: List[Dict], max_display: int)`**
- Displays props in a clean Streamlit DataFrame
- Shows sports breakdown metrics
- Limits display to prevent UI overload
- Configurable column display

### Filtering

Props are filtered based on:
1. Sport/League matching
2. Stat type matching
3. Player name search (case-insensitive)
4. Team name search (case-insensitive)

### Esports Filtering

Esports props (CS:GO, LoL, Dota2) are automatically filtered to include only allowed stat types:
- `combined_map_1_2_kills`
- `combined_map_1_2_headshots`
- `fantasy_points`
- `combined_map_1_2_assists` (LoL only)

## Troubleshooting

### No Props Displayed

**Issue**: "No props data available from PrizePicks"

**Solutions**:
1. Verify `prizepicks_props.csv` exists in the project root
2. Check CSV has data: `wc -l prizepicks_props.csv`
3. Run scraper to refresh: `python prizepicks_scrape.py`

### Empty Sport Field

**Issue**: Props show empty sport column

**Explanation**: PrizePicks CSV uses League field primarily. The integration falls back to League when Sport is empty.

**Solution**: Filters check both `sport` and `league` fields.

### Time Validation Errors

**Issue**: Props filtered out due to time parsing

**Explanation**: The `fetch_prizepicks_props` function bypasses time validation since CSV time formats vary.

**Note**: Time filtering can be re-enabled by using `fetch_all_props()` after fixing time format parsing.

## Future Enhancements

Potential improvements:
- [ ] Real-time odds integration
- [ ] Historical prop performance tracking
- [ ] Advanced stat projections
- [ ] Bet sizing recommendations
- [ ] Multi-source prop comparison
- [ ] Automated prop alerts

## Related Files

- `player_prop_predictor.py` - Main Streamlit app
- `props_data_fetcher.py` - PrizePicks data fetching
- `prizepicks_provider.py` - Provider adapter
- `prizepicks_scraper.py` - Scraper stub
- `prizepicks_props.csv` - Data cache
- `test_*.py` - Test scripts

## Support

For issues or questions, refer to the main project README or open an issue on GitHub.
