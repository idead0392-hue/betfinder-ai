# PrizePicks Integration Guide

This guide explains how to use the PrizePicks props fetching and rendering functionality in BetFinder AI.

## Overview

The PrizePicks integration provides two main components:

1. **`PrizePicksProvider`**: A provider class for fetching props data from PrizePicks
2. **`render_props()`**: A Streamlit function for displaying props data in a clean, structured format

## Quick Start

### Basic Example

```python
import streamlit as st
from api_providers import PrizePicksProvider
from page_utils import render_props

# Initialize the provider
provider = PrizePicksProvider()

# Fetch props (optionally filtered by sport)
resp = provider.get_props(sport="basketball", max_props=25)

# Check if successful
if resp.success:
    # Extract the props data
    if isinstance(resp.data, dict) and "data" in resp.data:
        props_response = resp.data.get("data")
    else:
        props_response = resp.data
    
    # Render the props in Streamlit
    render_props(props_response, top_n=25)
else:
    st.error(f"Error: {resp.error_message}")
```

### Complete Integration Pattern

Here's the complete pattern from the problem statement, with proper error handling:

```python
import streamlit as st
from api_providers import PrizePicksProvider
from page_utils import render_props

# Check if provider is available
PROVIDER_AVAILABLE = True
try:
    from api_providers import PrizePicksProvider
except ImportError:
    PROVIDER_AVAILABLE = False
    PrizePicksProvider = None

# Example: fetch props and render them
props_response = None
if PROVIDER_AVAILABLE and PrizePicksProvider is not None:
    provider = PrizePicksProvider()
    resp = provider.get_props(sport="basketball")
    
    if getattr(resp, "success", False):
        props_response = resp.data.get("data") if isinstance(resp.data, dict) and "data" in resp.data else resp.data
    else:
        st.error(f"Provider error: {getattr(resp, 'error_message', 'Unknown')}")
else:
    # Demo data if provider missing
    props_response = [
        {
            "player_name": "LeBron James",
            "team": "LAL",
            "matchup": "LAL vs BOS",
            "stat_type": "points",
            "line": 27.5,
            "league": "NBA",
            "odds": -110,
            "confidence": 75.0,
            "expected_value": 5.2
        }
    ]

# Render the props
if props_response:
    render_props(props_response, top_n=25)
```

## API Reference

### PrizePicksProvider

#### `__init__()`

Initialize the PrizePicksProvider.

```python
provider = PrizePicksProvider()
```

#### `get_props(sport=None, max_props=1000)`

Fetch props data from PrizePicks.

**Parameters:**
- `sport` (str, optional): Filter props by sport (e.g., "basketball", "football")
- `max_props` (int): Maximum number of props to return (default: 1000)

**Returns:**
- `APIResponse` object with:
  - `success` (bool): Whether the request succeeded
  - `data` (dict): Props data with format `{"data": [...], "count": N}`
  - `error_message` (str): Error message if failed
  - `response_time` (float): Request duration in seconds
  - `status_code` (int): HTTP status code if applicable

**Example:**
```python
# Fetch all sports
resp = provider.get_props(max_props=50)

# Fetch basketball only
resp = provider.get_props(sport="basketball", max_props=25)
```

### render_props()

#### `render_props(props_data, top_n=25)`

Render props data in a clean, structured Streamlit display.

**Parameters:**
- `props_data` (List[Dict]): List of prop dictionaries
- `top_n` (int): Maximum number of props to display (default: 25)

**Expected Props Data Format:**
```python
{
    "player_name": str,      # Player name
    "team": str,            # Team abbreviation
    "stat_type": str,       # Stat type (e.g., "points", "assists")
    "line": float,          # Prop line value
    "matchup": str,         # Game matchup (optional)
    "league": str,          # League name (optional)
    "odds": int,            # Betting odds (optional)
    "confidence": float,    # Confidence % (optional)
    "expected_value": float # EV % (optional)
}
```

**Features:**
- Displays props in a Pandas DataFrame with formatted columns
- Shows summary statistics (total props, unique leagues, unique players)
- Automatic error handling with fallback display
- Clean formatting of numeric values

**Example:**
```python
props = [
    {
        "player_name": "Stephen Curry",
        "team": "GSW",
        "stat_type": "3-pointers made",
        "line": 4.5,
        "league": "NBA",
        "odds": -110,
        "confidence": 72.0,
        "expected_value": 4.8
    }
]

render_props(props, top_n=10)
```

## Demo Application

A complete demo application is available in `demo_render_props.py`. Run it with:

```bash
streamlit run demo_render_props.py
```

The demo includes:
- Live props fetching from PrizePicks
- Sport filtering
- Demo data viewer
- Complete usage examples
- Error handling demonstrations

## Testing

### Unit Tests

Run the integration tests:

```bash
python3 test_prizepicks_integration.py
```

### Streamlit Test App

Run the test Streamlit application:

```bash
streamlit run test_render_props.py
```

## Data Source

The PrizePicksProvider uses the `PropsDataFetcher` class which fetches data from:
1. **CSV Cache**: `prizepicks_props.csv` (fast, local cache)
2. **API**: PrizePicks public API (fallback if CSV is not available)

The data is automatically normalized to a consistent format for use throughout the application.

## Error Handling

The integration includes comprehensive error handling:

1. **Provider Initialization**: Gracefully handles missing dependencies
2. **API Failures**: Returns structured error responses
3. **Rendering Errors**: Falls back to simple text display if DataFrame rendering fails
4. **Missing Data**: Shows appropriate warnings when no props are available

## Support for Sports

The integration supports filtering by these sports:
- Basketball (NBA, WNBA, CBB)
- Football (NFL, CFB)
- Baseball (MLB)
- Hockey (NHL)
- Soccer
- Esports (CS:GO, League of Legends, Dota 2, Valorant, Apex)

## Integration with Existing Pages

To integrate render_props into an existing Streamlit page:

```python
from page_utils import render_props
from api_providers import PrizePicksProvider

# In your page's main function
provider = PrizePicksProvider()
resp = provider.get_props(sport="basketball")

if resp.success:
    props = resp.data.get("data") if isinstance(resp.data, dict) else resp.data
    render_props(props, top_n=50)
```

## Troubleshooting

### "PrizePicksProvider not available"
- Ensure `api_providers.py` is in your Python path
- Check that `props_data_fetcher.py` is available

### "No props data available"
- Check if `prizepicks_props.csv` exists and has data
- Verify network connectivity for API access
- Check the scraper is running (`prizepicks_scrape.py`)

### "Error rendering props"
- Verify props data has the expected format
- Check that required columns exist (player_name, stat_type, line)
- Review error message for specific details

## Future Enhancements

Potential improvements for the integration:
- Real-time data refresh with WebSocket support
- Advanced filtering (by team, player, stat type)
- Props comparison and analysis tools
- Historical props tracking
- Integration with betting recommendations

## Contributing

When adding new features:
1. Follow the existing error handling patterns
2. Add tests to `test_prizepicks_integration.py`
3. Update this documentation
4. Ensure backwards compatibility with existing code
