# SportbexProvider Usage Guide

The `SportbexProvider` class provides a comprehensive, production-ready interface to the Sportbex betting data API with built-in error handling, logging, request timeouts, and proper documentation.

## Features

- **Comprehensive Error Handling**: Automatic retry logic, timeout management, and standardized error responses
- **Flexible Configuration**: Environment variable support with sensible defaults
- **Sport Type Support**: Enum-based sport identification with automatic ID mapping
- **Multiple Endpoint Routing**: Handles different Sportbex API endpoint patterns automatically
- **Request Caching**: Built on the robust `BaseAPIProvider` with session management
- **Health Monitoring**: Built-in health check functionality
- **Detailed Logging**: Comprehensive logging for debugging and monitoring

## Supported Sports

- Tennis (ID: 2)
- Basketball (ID: 7522)
- American Football (ID: 6423)
- Soccer (ID: 1)
- Baseball (ID: 5)
- Hockey (ID: 6)
- Esports (ID: 7)
- College Football (ID: 8)

## Quick Start

### 1. Environment Configuration

Copy `.env.example` to `.env` and configure your API key:

```bash
cp .env.example .env
```

Edit `.env`:
```bash
SPORTBEX_API_KEY=your_actual_api_key_here
SPORTBEX_API_URL=https://trial-api.sportbex.com
```

### 2. Basic Usage

```python
from api_providers import SportbexProvider, SportType, create_sportbex_provider

# Method 1: Use factory function (reads from environment)
provider = create_sportbex_provider()

# Method 2: Direct instantiation
provider = SportbexProvider(api_key="your_api_key")

# Get tennis competitions
response = provider.get_competitions(sport=SportType.TENNIS)
if response.success:
    competitions = response.data.get('data', [])
    print(f"Found {len(competitions)} tennis competitions")
else:
    print(f"Error: {response.error_message}")
```

### 3. Advanced Usage

```python
# Get matchups for a specific competition
matchups = provider.get_matchups(
    sport=SportType.BASKETBALL, 
    competition_id="12345"
)

# Get odds with filtering
odds = provider.get_odds(
    event_ids=["event1", "event2"],
    market_types=["moneyline", "spread"]
)

# Health check
health = provider.health_check()
print(f"API Status: {'Healthy' if health.success else 'Error'}")
```

## API Methods

### `get_competitions(sport)`
Get competitions/leagues for a specific sport.

**Parameters:**
- `sport`: SportType enum or string identifier

**Returns:** APIResponse with competitions data

### `get_props(sport, competition_id=None)`
Get props/competitions data for a specific sport.

**Parameters:**
- `sport`: SportType enum or string identifier
- `competition_id`: Optional specific competition ID

**Returns:** APIResponse with props data

### `get_odds(event_ids=None, market_types=None, **kwargs)`
Get betting odds data.

**Parameters:**
- `event_ids`: List of specific event IDs
- `market_types`: List of market types to filter by
- `**kwargs`: Additional parameters

**Returns:** APIResponse with odds data

### `get_matchups(sport, competition_id)`
Get matchups/events for a specific competition.

**Parameters:**
- `sport`: SportType enum or string identifier
- `competition_id`: Competition ID to get matchups for

**Returns:** APIResponse with matchups data

### `health_check()`
Perform a health check on the Sportbex API.

**Returns:** APIResponse indicating API health status

## Error Handling

All methods return an `APIResponse` object with:

```python
@dataclass
class APIResponse:
    success: bool
    data: Optional[Any] = None
    error_message: Optional[str] = None
    status_code: Optional[int] = None
    headers: Optional[Dict[str, str]] = None
    response_time: Optional[float] = None
    provider: Optional[str] = None
```

Always check `response.success` before accessing `response.data`:

```python
response = provider.get_competitions(sport=SportType.TENNIS)
if response.success:
    # Process response.data
    pass
else:
    # Handle error: response.error_message
    pass
```

## Configuration Options

### Environment Variables

- `SPORTBEX_API_KEY`: Required API key for authentication
- `SPORTBEX_API_URL`: Optional custom base URL (defaults to trial API)

### Constructor Parameters

```python
SportbexProvider(
    api_key=None,           # API key (defaults to env var)
    base_url=None,          # Base URL (defaults to env var or default)
    timeout=20,             # Request timeout in seconds
    **kwargs                # Additional configuration
)
```

## Testing

Run the test suite:

```bash
python3 test_sportbex_provider.py
```

Run the example script:

```bash
python3 example_sportbex_usage.py
```

## Integration with Existing Code

The SportbexProvider can replace direct API calls in your existing code. For example, instead of:

```python
# Old direct API call
response = requests.get(
    f"https://trial-api.sportbex.com/api/other-sport/competitions/2",
    headers={'sportbex-api-key': api_key}
)
```

Use:

```python
# New provider-based call
provider = SportbexProvider(api_key=api_key)
response = provider.get_competitions(sport=SportType.TENNIS)
```

This provides better error handling, logging, retries, and maintainability.