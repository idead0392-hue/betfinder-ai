# Copilot Instructions for BetFinder AI MCP Server

## Project Overview
This workspace contains a comprehensive Model Context Protocol (MCP) server for BetFinder AI that provides sports data integration for OpenAI Agent Builder workflows.

## MCP Server Implementation
- **Location**: `/mcp-server/server.py` 
- **Type**: Python-based MCP server using the official Python SDK
- **Purpose**: Expose BetFinder AI functionality to OpenAI Agent Builder and other MCP clients

## Key Components

### Sports Data Tools
1. **get_live_odds**: Real-time betting odds from Sportbex API
2. **get_ai_picks**: AI-generated betting recommendations with confidence scores
3. **calculate_bet_size**: Kelly Criterion-based bet sizing with risk management
4. **get_performance_metrics**: Comprehensive bankroll and performance tracking
5. **analyze_value_bet**: Expected value analysis for betting opportunities
### Sports Data Tools
1. **get_live_odds**: Real-time betting odds from PrizePicks data
2. **get_ai_picks**: AI-generated betting recommendations with confidence scores
3. **calculate_bet_size**: Kelly Criterion-based bet sizing with risk management
4. **get_performance_metrics**: Comprehensive bankroll and performance tracking
5. **analyze_value_bet**: Expected value analysis for betting opportunities

### Resource Endpoints
- `betfinder://sports/current-odds`: Live odds data
- `betfinder://ai/daily-picks`: AI recommendations  
- `betfinder://bankroll/status`: Performance metrics
- `betfinder://config/settings`: Server configuration

### Integration Points
### Integration Points
- **PrizePicks Provider**: Real-time player prop data via automated scraping
- **Picks Engine**: AI-powered betting analysis
- **Bankroll Manager**: Professional money management with Kelly Criterion
- **Risk Assessment**: Multi-factor risk evaluation system
## Development Guidelines

### MCP Server Development
- Follow Model Context Protocol specification 2025-06-18
- Use structured JSON responses for all tool outputs
- Implement proper error handling and logging
- Support both mock and live data modes for testing

### Testing the MCP Server
- Debug in VS Code using the MCP configuration in `.vscode/mcp.json`
- Test tools using MCP Inspector: `uv run mcp dev mcp-server/server.py`
- Install in Claude Desktop: `uv run mcp install mcp-server/server.py`

### OpenAI Agent Builder Integration
- Tools expose BetFinder AI functionality as structured functions
- Resources provide real-time data access for agent workflows
- Error handling ensures robust agent operation
- Supports both autonomous and human-in-the-loop betting workflows

## Documentation References
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk)
- [MCP Specification](https://modelcontextprotocol.io/specification/2025-06-18/)
- [OpenAI Agent Builder](https://platform.openai.com/docs/agent-builder)

## Architecture Notes
The MCP server acts as a bridge between OpenAI Agent Builder and BetFinder AI's sports betting analysis capabilities, enabling automated workflows while maintaining professional-grade risk management and data integrity.
## Architecture Notes
The MCP server acts as a bridge between OpenAI Agent Builder and BetFinder AI's sports betting analysis capabilities, leveraging real-time PrizePicks data through automated scraping to enable intelligent betting workflows while maintaining professional-grade risk management and data integrity.