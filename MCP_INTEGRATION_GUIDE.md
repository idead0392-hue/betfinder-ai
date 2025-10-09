# BetFinder AI MCP Server Integration Guide

## Overview

The BetFinder AI MCP Server provides OpenAI Agent Builder with access to comprehensive sports betting data and analysis tools. This server bridges the gap between real-time sports data and AI-powered decision making for betting workflows.

## Features

### 🔧 Tools Available

1. **get_live_odds**: Real-time betting odds across multiple sports
2. **get_ai_picks**: AI-generated betting recommendations with confidence scores  
3. **calculate_bet_size**: Kelly Criterion-based optimal bet sizing
4. **get_performance_metrics**: Comprehensive betting performance tracking
5. **analyze_value_bet**: Expected value analysis for betting opportunities

### 📊 Resource Endpoints

- `betfinder://sports/current-odds`: Live odds data feed
- `betfinder://ai/daily-picks`: Current AI recommendations
- `betfinder://bankroll/status`: Performance and bankroll metrics
- `betfinder://config/settings`: Server configuration and capabilities

## Installation & Setup

### 1. Install Dependencies

```bash
cd /workspaces/betfinder-ai/mcp-server
pip install -r requirements.txt
```

### 2. Environment Variables

Set up your Sportbex API key:

```bash
export SPORTBEX_API_KEY="your_api_key_here"
```

### 3. Run the Server

#### Development Mode
```bash
# Test with MCP Inspector
uv run mcp dev mcp-server/server.py

# Direct execution
python mcp-server/server.py
```

#### Debug Mode
Use VS Code's built-in debugger with the provided launch configuration in `.vscode/launch.json`.

### 4. Install in Claude Desktop

```bash
uv run mcp install mcp-server/server.py --name "BetFinder Sports Data"
```

## OpenAI Agent Builder Integration

### Agent Workflow Design

The MCP server enables several agent workflow patterns:

#### 1. **Automated Betting Analysis**
```
Trigger: Daily at 9 AM
├── get_live_odds(sport="basketball") 
├── get_ai_picks(sport="basketball", confidence_threshold=0.75)
├── For each pick:
│   ├── analyze_value_bet(odds, true_probability)
│   ├── calculate_bet_size(odds, confidence, bankroll)
│   └── [Human approval] → Place bet
└── get_performance_metrics()
```

#### 2. **Real-time Opportunity Scanner**
```
Trigger: Every 15 minutes
├── get_live_odds() → Check all sports
├── For each game:
│   ├── get_ai_picks(sport=detected_sport)
│   ├── analyze_value_bet() → Filter EV > 5%
│   └── If valuable → Alert + Human review
└── Log opportunities
```

#### 3. **Bankroll Management Workflow**
```
Trigger: Before each bet
├── get_performance_metrics() → Current bankroll status
├── calculate_bet_size() → Kelly Criterion sizing  
├── Risk assessment checks:
│   ├── Max bet percentage < 5%
│   ├── Recent performance trend
│   └── Drawdown limits
└── Approve/Reject bet
```

### Tool Usage Examples

#### Get Live Odds
```json
{
  "tool": "get_live_odds",
  "arguments": {
    "sport": "basketball",
    "market": "moneyline"
  }
}
```

#### AI Picks with High Confidence
```json
{
  "tool": "get_ai_picks", 
  "arguments": {
    "sport": "football",
    "confidence_threshold": 0.8
  }
}
```

#### Calculate Optimal Bet Size
```json
{
  "tool": "calculate_bet_size",
  "arguments": {
    "odds": 2.1,
    "confidence": 0.75,
    "bankroll": 1000
  }
}
```

#### Value Bet Analysis
```json
{
  "tool": "analyze_value_bet",
  "arguments": {
    "odds": 1.9,
    "true_probability": 0.6
  }
}
```

## Agent Builder Configuration

### 1. Server Connection

Add the MCP server to your OpenAI Agent Builder configuration:

```json
{
  "servers": {
    "betfinder-sports": {
      "type": "stdio", 
      "command": "python",
      "args": ["/path/to/mcp-server/server.py"],
      "env": {
        "SPORTBEX_API_KEY": "your_api_key"
      }
    }
  }
}
```

### 2. Agent Instructions

Use these instructions for your agent:

```
You are a professional sports betting analysis agent with access to BetFinder AI tools.

CAPABILITIES:
- Real-time odds from Sportbex API
- AI-generated betting picks with confidence scores
- Kelly Criterion bet sizing with risk management
- Performance tracking and bankroll analysis
- Expected value calculations

WORKFLOW:
1. Always check get_performance_metrics() before making betting decisions
2. Use get_live_odds() to get current market data
3. Generate picks with get_ai_picks() using minimum 70% confidence
4. Analyze value with analyze_value_bet() - require positive EV
5. Calculate bet size with calculate_bet_size() - never exceed 5% of bankroll
6. Present recommendations with reasoning and risk assessment

RISK MANAGEMENT:
- Never recommend bets without positive expected value
- Limit individual bets to 5% of bankroll maximum
- Monitor recent performance trends
- Require human approval for bets over 2% of bankroll
```

### 3. Guardrails & Decision Nodes

Implement these decision nodes in your agent workflow:

- **Value Check**: EV must be > 0
- **Size Limit**: Bet size < 5% of bankroll  
- **Confidence Gate**: Confidence > 70%
- **Human Approval**: Required for bets > 2% of bankroll
- **Performance Check**: Recent win rate > 40%

## Production Deployment

### Monitoring

The server provides comprehensive logging and health checks:

- **Health Endpoint**: Check server status
- **Performance Metrics**: Track API response times
- **Error Handling**: Graceful fallback to mock data
- **Request Logging**: Audit trail for all betting decisions

### Scaling

For high-volume usage:

- Deploy server with load balancer
- Use Redis for caching live odds data
- Implement rate limiting for API calls
- Set up monitoring dashboards

### Security

- Store API keys in environment variables
- Use HTTPS for all communications
- Implement request authentication
- Log all betting decisions for audit

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all BetFinder AI components are available
2. **API Timeouts**: Check Sportbex API key and network connectivity
3. **Memory Usage**: Monitor server performance under load

### Debug Mode

Enable detailed logging:

```bash
export LOG_LEVEL=DEBUG
python mcp-server/server.py
```

### Testing

Run the test suite:

```bash
python -m pytest mcp-server/tests/
```

## Examples

See the `examples/` directory for complete OpenAI Agent Builder workflows using this MCP server.

## Support

For issues and feature requests, see the project documentation or create an issue in the repository.