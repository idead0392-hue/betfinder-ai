# OpenAI Agent Router System Documentation

## Overview

The OpenAI Agent Router system provides seamless integration between BetFinder AI's prop analysis and specialized OpenAI Assistants. It routes each sport's prop analysis requests to the appropriate AI agent, ensuring expert-level analysis with sport-specific knowledge.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   BetFinder AI  │────│  Agent Router   │────│ OpenAI Agents   │
│    Frontend     │    │   Integration   │    │   (Assistants)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
    ┌─────────┐           ┌─────────────┐         ┌─────────────┐
    │ Props   │           │ Data        │         │ NBA Agent   │
    │ Data    │           │ Formatters  │         │ NFL Agent   │
    │ CSV     │           │             │         │ Soccer Agent│
    └─────────┘           └─────────────┘         │ ...         │
                                 │                └─────────────┘
                          ┌─────────────┐
                          │ Local Agent │
                          │ Fallback    │
                          └─────────────┘
```

## Components

### 1. OpenAI Agent Router (`openai_agent_router.py`)
- **Primary Function**: Routes prop analysis to specialized OpenAI Assistants
- **Key Features**:
  - Sport detection and agent mapping
  - Structured prompt generation
  - OpenAI Assistants API integration
  - Error handling and fallback support

### 2. Sport Data Formatters (`sport_data_formatters.py`)
- **Primary Function**: Format prop data for sport-specific analysis
- **Supported Sports**:
  - NBA Basketball
  - NFL Football
  - Soccer/Football
  - CS:GO, League of Legends, Dota 2, Valorant, Overwatch
  - Tennis, Baseball, Hockey, Golf

### 3. Agent Integration (`agent_integration.py`)
- **Primary Function**: Bridge between new OpenAI system and existing agents
- **Features**:
  - Dual-mode operation (OpenAI + Local fallback)
  - Performance tracking
  - Seamless switching between modes

### 4. Configuration Manager (`agent_config.py`)
- **Primary Function**: Manage API keys, Assistant IDs, and settings
- **Features**:
  - Dynamic agent configuration
  - Environment variable support
  - Configuration validation

## Setup Instructions

### 1. Install Dependencies
```bash
pip install openai streamlit pandas numpy
```

### 2. Configure OpenAI API Key
```bash
# Option 1: Environment variable
export OPENAI_API_KEY="your-openai-api-key-here"

# Option 2: In the app interface
# Use the sidebar "🚀 Initialize AI Agents" button
```

### 3. Configure Assistant IDs
Edit `agent_config.json` or set environment variables:
```bash
export NBA_ASSISTANT_ID="asst_your_nba_assistant_id"
export NFL_ASSISTANT_ID="asst_your_nfl_assistant_id"
export SOCCER_ASSISTANT_ID="asst_your_soccer_assistant_id"
# ... etc for other sports
```

### 4. Run the Application
```bash
streamlit run app_redesign.py --server.port 8502 --server.address 0.0.0.0
```

## Usage Guide

### Basic Usage

1. **Start the App**: Launch the Streamlit application
2. **Select Analysis Mode**: Use the sidebar toggle to switch between OpenAI and Local modes
3. **Choose Sport**: Select from the sport tabs (NBA, NFL, etc.)
4. **View Analysis**: See AI-generated picks with confidence scores and reasoning

### Advanced Configuration

#### Creating OpenAI Assistants

1. **Go to OpenAI Platform**: https://platform.openai.com/assistants
2. **Create New Assistant** for each sport with these settings:
   - **Model**: GPT-4 Turbo
   - **Instructions**: Sport-specific betting analysis prompt
   - **Tools**: None (optional: Code Interpreter for statistics)

Example NBA Assistant Instructions:
```
You are an expert NBA prop betting analyst. Analyze player props using:

1. Statistical trends (season, recent games, matchups)
2. Situational factors (rest, injuries, game importance)
3. Historical performance vs opponent
4. Line value assessment

Provide structured JSON responses with:
- Value assessment (positive/negative EV)
- Confidence level (1-100)
- Key reasoning points
- Recommended action (bet/pass)

Focus on data-driven analysis with clear explanations.
```

#### Configuration File Format

```json
{
  "router_config": {
    "openai_api_key": "",
    "default_timeout": 30,
    "max_retries": 3,
    "enable_fallback": true,
    "enable_logging": true
  },
  "agent_configs": {
    "NBA_Analysis_Agent": {
      "name": "NBA_Analysis_Agent",
      "assistant_id": "asst_your_actual_id",
      "sport": "basketball",
      "description": "NBA prop analysis",
      "enabled": true,
      "model": "gpt-4-1106-preview"
    }
  }
}
```

## Sport Agent Mapping

| Sport | Agent Name | Assistant ID Env Var |
|-------|------------|---------------------|
| Basketball/NBA | NBA_Analysis_Agent | NBA_ASSISTANT_ID |
| Football/NFL | NFL_Analysis_Agent | NFL_ASSISTANT_ID |
| College Football | CFB_Analysis_Agent | CFB_ASSISTANT_ID |
| Soccer | Soccer_Analysis_Agent | SOCCER_ASSISTANT_ID |
| Hockey/NHL | NHL_Analysis_Agent | NHL_ASSISTANT_ID |
| Baseball/MLB | MLB_Analysis_Agent | MLB_ASSISTANT_ID |
| CS:GO | CSGO_Analysis_Agent | CSGO_ASSISTANT_ID |
| League of Legends | LoL_Analysis_Agent | LOL_ASSISTANT_ID |
| Dota 2 | Dota2_Analysis_Agent | DOTA2_ASSISTANT_ID |
| Valorant | Valorant_Analysis_Agent | VALORANT_ASSISTANT_ID |
| Overwatch | Overwatch_Analysis_Agent | OVERWATCH_ASSISTANT_ID |
| Tennis | Tennis_Analysis_Agent | TENNIS_ASSISTANT_ID |
| Golf | Golf_Analysis_Agent | GOLF_ASSISTANT_ID |

## API Integration

### Programmatic Usage

```python
from agent_integration import create_agent_manager

# Initialize agent manager
manager = create_agent_manager(
    use_openai=True,
    openai_api_key="your-api-key"
)

# Analyze props for a specific sport
props_data = [
    {
        "Name": "LeBron James",
        "Prop": "Over 24.5 Points", 
        "Points": 24.5,
        "team": "LAL",
        "opponent": "GSW"
    }
]

result = manager.analyze_props("basketball", props_data)

if result["success"]:
    picks = result["picks"]
    for pick in picks:
        print(f"Player: {pick['player_name']}")
        print(f"Confidence: {pick['confidence']}%")
        print(f"Expected Value: {pick['expected_value']}%")
```

### Data Structure

#### Input Props Format
```python
{
    "Name": "Player Name",
    "Prop": "Over/Under X.X Stat",
    "Points": 24.5,  # Line value
    "team": "Team Code",
    "opponent": "Opponent Code",
    "odds": -110  # Optional
}
```

#### Output Analysis Format
```python
{
    "success": True,
    "picks": [
        {
            "player_name": "Player Name",
            "stat_type": "Points",
            "line": 24.5,
            "over_under": "Over",
            "confidence": 78,
            "expected_value": 12.5,
            "reasoning": "Analysis explanation...",
            "prizepicks_classification": "DECENT"
        }
    ],
    "analysis_metadata": {
        "sport": "basketball",
        "method": "openai",
        "timestamp": "2025-01-01T12:00:00",
        "duration_ms": 1250
    }
}
```

## Testing

### Run Test Suite
```bash
python test_agent_router.py
```

### Test Components
- ✅ Sport detection and agent mapping
- ✅ Data formatting for all sports
- ✅ OpenAI API integration (mock)
- ✅ Local agent fallback
- ✅ Error handling
- ✅ Performance tracking

## Troubleshooting

### Common Issues

#### 1. "OpenAI API key is required"
**Solution**: Set the OPENAI_API_KEY environment variable or configure it in the app

#### 2. "No agent found for sport: X"
**Solution**: Check that the sport name matches the supported sports list

#### 3. OpenAI requests failing
**Solutions**:
- Verify API key is valid
- Check Assistant IDs are correct
- Ensure sufficient API quota
- System will fallback to local agents automatically

#### 4. Local agents not working
**Solution**: Check that the sport_agents.py file is properly imported

### Performance Monitoring

Access performance metrics via the sidebar:
- Total requests processed
- Success rate by sport
- OpenAI vs Local usage percentage
- Response time statistics

### Logs and Debugging

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Extension and Customization

### Adding New Sports

1. **Add Sport Formatter**:
```python
class NewSportFormatter(BaseSportFormatter):
    def __init__(self):
        super().__init__("NewSport")
        # Define sport-specific formatting
```

2. **Update Configuration**:
```json
{
  "NewSport_Analysis_Agent": {
    "name": "NewSport_Analysis_Agent",
    "assistant_id": "asst_newsport_id",
    "sport": "newsport",
    "enabled": true
  }
}
```

3. **Create OpenAI Assistant** with sport-specific instructions

### Custom Analysis Logic

Override the `_analyze_with_openai` method in `EnhancedSportAgent` for custom processing.

## Best Practices

1. **Assistant Instructions**: Make them sport-specific and detailed
2. **Error Handling**: Always implement fallback strategies
3. **Performance**: Monitor API usage and costs
4. **Testing**: Test with real data before production
5. **Monitoring**: Track success rates and response times

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review test results for component status
3. Examine logs for detailed error information
4. Verify configuration settings

## License

This system is part of the BetFinder AI project and follows the same licensing terms.