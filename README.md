# BetFinder AI

An intelligent sports betting analysis tool that helps users find valuable betting opportunities using AI-powered predictions and data analysis.

## Description

BetFinder AI is a Python-based application that analyzes sports betting markets to identify potentially profitable betting opportunities. The system uses machine learning algorithms and statistical analysis to evaluate odds and provide insights to users.

## Features

- Real-time sports betting data analysis
- AI-powered prediction algorithms
- SQLite database for storing betting data
- Web interface for easy interaction
- Support for multiple sports and betting markets

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/idead0392-hue/betfinder-ai.git
cd betfinder-ai
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

3. Initialize the database:
```bash
python bet_db.py
```

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your web browser and navigate to the local server address displayed in the terminal (typically `http://localhost:5000`)

3. Use the web interface to:
   - View current betting opportunities
   - Analyze odds and predictions
   - Track betting history

## Project Structure

```
betfinder-ai/
├── app.py              # Main application file
├── bet_db.py           # Database management
├── requirements.txt    # Python dependencies
└── README.md          # Project documentation
```

## Technologies Used

- Python
- Flask (Web framework)
- SQLite (Database)
- Machine Learning libraries

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Disclaimer

This tool is for educational and informational purposes only. Betting involves risk, and users should gamble responsibly. Past performance does not guarantee future results.

## League of Legends Data Integration

Agents can optionally use Leaguepedia-derived data via the `leaguepedia_parser_thomasbarrepitous` package, wrapped by `lol_data.py` for safe, cached access.

- Toggle external calls: set environment variable `DISABLE_EXTERNAL_LOL_DATA=1` to disable network usage and force safe fallbacks.
- Available insights include recent player match history (KDA, kill participation, gold share), team performance trends, tournament standings, and champion meta summaries.
- The LeagueOfLegendsAgent will gracefully degrade if data is unavailable or the toggle is enabled.

## AutoML Integration

BetFinder AI now features automatic machine learning powered by the `auto-machine-learning` package. Every agent can automatically train, select, and predict with best-in-class models using historical props and results data.

### Features
- **Automatic Model Selection**: AutoML automatically selects, tunes, and optimizes the best model for each agent's data
- **Data Preparation**: Converts picks_ledger format to ML-ready features (confidence, odds, stat types, outcomes)
- **Model Persistence**: Trained models are saved and reloaded across sessions
- **Graceful Fallbacks**: Falls back to weight-based approach when AutoML is disabled or unavailable

### Usage

**Enable/Disable AutoML:**
```bash
# Enable AutoML (default)
export DISABLE_AUTOML=0

# Disable AutoML (use fallback weights)
export DISABLE_AUTOML=1
```

**Agent Training:**
Agents automatically train when initialized if sufficient historical data is available (minimum 10 completed picks).

**Data Export for External ML:**
```python
from automl_engine import export_picks_to_ml_format
from picks_ledger import picks_ledger

# Export all picks to CSV
picks = picks_ledger.get_all_picks()
df = export_picks_to_ml_format(picks, "agent_props_and_results.csv")
```

**Custom Objectives:**
- Classification: hit/miss, win/loss (default)
- Regression: points, profit, expected value (modify target_column in AutoMLEngine)

### Agent Integration
Each agent now uses AutoML predictions for:
- Prop recommendation scoring
- Confidence estimation
- Expected value calculations
- Pick filtering and ranking

## Memory & Personalization System

BetFinder AI features intelligent memory and personalization powered by `mem0ai`. Every agent learns from user patterns and adapts recommendations automatically.

### Features
- **User Pattern Recognition**: Tracks success rates by prop type, sport, and betting style
- **Personalized Confidence Adjustment**: Increases/decreases confidence based on user's historical performance
- **Preference Learning**: Stores and applies user feedback and betting preferences
- **Session-Based Learning**: Agents learn from betting sessions and outcomes
- **Cross-Agent Insights**: Share learnings across all sports and agents

### Configuration
```bash
# Enable memory (default, requires OPENAI_API_KEY)
export DISABLE_MEMORY=0

# Disable memory (uses safe fallback storage)
export DISABLE_MEMORY=1

# Optional: Set OpenAI API key for full mem0ai features
export OPENAI_API_KEY="your-key-here"
```

### Usage Examples

**Store Pick Results:**
```python
# Agents automatically learn from outcomes
agent.store_pick_result(user_id, prop, "won", confidence=85.0, reasoning="Strong performance")
```

**Get Personalized Insights:**
```python
insights = agent.get_personalized_insights(user_id, prop)
pattern = insights['stat_pattern']['pattern']  # 'positive', 'negative', 'neutral'
```

**Enhanced Recommendations:**
```python
# Automatically adjust confidence based on user history
enhanced_pick = agent.enhance_pick_with_memory(user_id, original_pick)
```

**User Preferences:**
```python
agent.store_user_feedback(user_id, "favorite_stat", "points", "User prefers scoring props")
```

### Memory Categories
- **Pick Results**: Win/loss outcomes with confidence and reasoning
- **User Preferences**: Favorite stats, risk tolerance, betting style  
- **Agent Learnings**: Strategy insights and performance patterns
- **Cross-Agent Insights**: Trends that apply across multiple sports

### Fallback Mode
When mem0ai is unavailable or disabled, the system uses local fallback storage that provides the same functionality with simple in-memory storage.
