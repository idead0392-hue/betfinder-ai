# api_server.py - refactor to use SportbexProvider
from flask import Flask, request, jsonify
from typing import Any, Dict
from sportbex_provider import SportbexProvider
from api_providers import ProviderFactory

app = Flask(__name__)

# Create provider via factory for future multi-provider support
provider_name = request.headers.get('X-Provider', 'sportbex') if request else 'sportbex'
provider = ProviderFactory.create(provider_name)

# Helper for consistent error responses

def make_error(message: str, status: int = 500, detail: Dict[str, Any] | None = None):
    payload = {"error": message}
    if detail:
        payload["detail"] = detail
    return jsonify(payload), status

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "provider": provider_name})

# Odds endpoints
@app.route('/api/odds', methods=['GET'])
def get_odds():
    try:
        sport = request.args.get('sport')
        market = request.args.get('market')
        league = request.args.get('league')
        res = provider.get_odds(sport=sport, market=market, league=league)
        return jsonify(res)
    except Exception as e:
        return make_error("Failed to fetch odds", 502, {"exception": str(e)})

# Props endpoints
@app.route('/api/props', methods=['GET'])
def get_props():
    try:
        sport = request.args.get('sport')
        league = request.args.get('league')
        player = request.args.get('player')
        market = request.args.get('market')
        res = provider.get_props(sport=sport, league=league, player=player, market=market)
        return jsonify(res)
    except Exception as e:
        return make_error("Failed to fetch props", 502, {"exception": str(e)})

# Competitions / schedule endpoints
@app.route('/api/competitions', methods=['GET'])
def get_competitions():
    try:
        sport = request.args.get('sport')
        league = request.args.get('league')
        res = provider.get_competitions(sport=sport, league=league)
        return jsonify(res)
    except Exception as e:
        return make_error("Failed to fetch competitions", 502, {"exception": str(e)})

# Backward-compat legacy endpoints mapping
@app.route('/api/lines', methods=['GET'])
def get_lines_legacy():
    try:
        # Map to provider odds
        sport = request.args.get('sport')
        market = request.args.get('market')
        league = request.args.get('league')
        res = provider.get_odds(sport=sport, market=market, league=league)
        return jsonify(res)
    except Exception as e:
        return make_error("Failed to fetch lines", 502, {"exception": str(e)})

@app.route('/api/player_props', methods=['GET'])
def get_player_props_legacy():
    try:
        sport = request.args.get('sport')
        league = request.args.get('league')
        player = request.args.get('player')
        market = request.args.get('market')
        res = provider.get_props(sport=sport, league=league, player=player, market=market)
        return jsonify(res)
    except Exception as e:
        return make_error("Failed to fetch player props", 502, {"exception": str(e)})

# Documentation route to aid migration
@app.route('/api/meta/provider', methods=['GET'])
def provider_meta():
    try:
        return jsonify({
            "active_provider": provider_name,
            "capabilities": provider.capabilities() if hasattr(provider, 'capabilities') else {}
        })
    except Exception as e:
        return make_error("Failed to fetch provider metadata", 500, {"exception": str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
