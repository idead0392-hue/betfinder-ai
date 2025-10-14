# SPORTBEX_PROVIDER_GUIDE.md

## API Server Integration Examples

Below are example calls showing how api_server.py now delegates to SportbexProvider.

### Odds
- Endpoint: GET /api/odds
- Query params: sport, market, league
- Example: /api/odds?sport=basketball&market=spreads&league=NBA
- Behavior: Calls provider.get_odds and returns standardized JSON.

### Props
- Endpoint: GET /api/props
- Query params: sport, player, market, league
- Example: /api/props?sport=basketball&player=LeBron%20James&market=points&league=NBA
- Behavior: Calls provider.get_props and returns standardized JSON.

### Competitions
- Endpoint: GET /api/competitions
- Query params: sport, league
- Example: /api/competitions?sport=soccer&league=EPL
- Behavior: Calls provider.get_competitions and returns standardized JSON.

### Provider Metadata
- Endpoint: GET /api/meta/provider
- Response: { active_provider, capabilities }

### Legacy Compatibility
- /api/lines proxies to /api/odds
- /api/player_props proxies to /api/props

### Selecting Providers
- Send header X-Provider: sportbex (default) to choose provider. Future providers can be configured via ProviderFactory.

### Error Handling
- Upstream/provider errors are translated to JSON errors with HTTP 502.
- Response body includes an error message and detail.exception if available.
