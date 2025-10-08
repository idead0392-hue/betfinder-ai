from flask import Flask, jsonify, request
import requests

# Sportbex API key
SPORTBEX_API_KEY = 'NZLDw8ZXFv0O8elaPq0wjbP4zxb2gCwJDsArWQUF'


from ai_integration_test import (
    get_esport_team,
    get_esport_tournament,
    get_esport_season_info,
    get_esport_season_last_matches,
    RAPIDAPI_KEY, RAPIDAPI_HOST
)

app = Flask(__name__)


@app.route("/")
def home():
    """Serve a simple homepage. If an `index.html` file exists in the project root, return its contents.

    This keeps the change minimal and avoids adding template dependencies.
    """
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return "<h1>Betfinder API</h1><p>Welcome. Use /api/... endpoints.</p>", 200

@app.route("/api/team/<team_id>")
def team(team_id):
    return get_esport_team(team_id, RAPIDAPI_KEY, RAPIDAPI_HOST)

@app.route("/api/tournament/<tournament_id>")
def tournament(tournament_id):
    return get_esport_tournament(tournament_id, RAPIDAPI_KEY, RAPIDAPI_HOST)

@app.route("/api/season/<tournament_id>/<season_id>")
def season(tournament_id, season_id):
    return get_esport_season_info(tournament_id, season_id, RAPIDAPI_KEY, RAPIDAPI_HOST)

@app.route("/api/seasonlast/<tournament_id>/<season_id>/<page>")
def season_last_matches(tournament_id, season_id, page):
    return get_esport_season_last_matches(tournament_id, season_id, page, RAPIDAPI_KEY, RAPIDAPI_HOST)

@app.route("/api/basketball/props")
def basketball_props():
    """Get basketball props data from Sportbex API"""
    try:
        url = "https://trial-api.sportbex.com/api/other-sport/competitions/7522"
        headers = {
            'sportbex-api-key': SPORTBEX_API_KEY
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({
                "error": f"Sportbex API returned status {response.status_code}",
                "details": response.text[:500]
            }), response.status_code
            
    except requests.exceptions.RequestException as e:
        return jsonify({
            "error": "Failed to fetch basketball props",
            "details": str(e)
        }), 500

@app.route("/api/basketball/matchups/<competition_id>")
def basketball_matchups(competition_id):
    """Get basketball matchups/events data from Sportbex API"""
    try:
        url = f"https://trial-api.sportbex.com/api/other-sport/event/7522/{competition_id}"
        headers = {
            'sportbex-api-key': SPORTBEX_API_KEY
        }
        
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({
                "error": f"Sportbex API returned status {response.status_code}",
                "details": response.text[:500],
                "url": url
            }), response.status_code
            
    except requests.exceptions.RequestException as e:
        return jsonify({
            "error": "Failed to fetch basketball matchups",
            "details": str(e),
            "url": f"https://trial-api.sportbex.com/api/other-sport/event/7522/{competition_id}"
        }), 500

@app.route("/api/basketball/odds", methods=['POST'])
def basketball_odds():
    """Get basketball odds data from Sportbex API"""
    try:
        url = "https://trial-api.sportbex.com/api/other-sport/event-odds"
        headers = {
            'sportbex-api-key': SPORTBEX_API_KEY,
            'Content-Type': 'application/json'
        }
        
                # Get the request body from the client, or use empty dict as default
        request_data = request.get_json() or {}
        
        response = requests.post(url, headers=headers, json=request_data, timeout=15)
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({
                "error": f"Sportbex API returned status {response.status_code}",
                "details": response.text[:500],
                "url": url,
                "request_data": request_data
            }), response.status_code
            
    except requests.exceptions.RequestException as e:
        return jsonify({
            "error": "Failed to fetch basketball odds",
            "details": str(e),
            "url": "https://trial-api.sportbex.com/api/other-sport/event-odds"
        }), 500

# Tennis endpoints
@app.route('/api/tennis/competitions', methods=['GET'])
def tennis_competitions():
    """Get tennis competitions data from Sportbex API"""
    try:
        url = "https://trial-api.sportbex.com/api/other-sport/competitions/2"
        headers = {
            'sportbex-api-key': SPORTBEX_API_KEY
        }
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({
                "error": f"Sportbex API returned status {response.status_code}",
                "details": response.text[:500],
                "url": url
            }), response.status_code
            
    except requests.exceptions.RequestException as e:
        return jsonify({
            "error": "Failed to fetch tennis competitions",
            "details": str(e),
            "url": "https://trial-api.sportbex.com/api/other-sport/competitions/2"
        }), 500

@app.route('/api/tennis/matchups/<competition_id>', methods=['GET'])
def tennis_matchups(competition_id):
    """Get tennis matchups for a specific competition"""
    try:
        url = "https://trial-api.sportbex.com/api/other-sport/match-ups"
        headers = {
            'sportbex-api-key': SPORTBEX_API_KEY,
            "Content-Type": "application/json"
        }
        request_data = {"competition_id": competition_id}
        response = requests.post(url, headers=headers, json=request_data, timeout=15)
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({
                "error": f"Sportbex API returned status {response.status_code}",
                "details": response.text[:500],
                "url": url,
                "request_data": request_data
            }), response.status_code
            
    except requests.exceptions.RequestException as e:
        return jsonify({
            "error": "Failed to fetch tennis matchups",
            "details": str(e),
            "url": "https://trial-api.sportbex.com/api/other-sport/match-ups"
        }), 500

@app.route('/api/tennis/odds', methods=['POST'])
def tennis_odds():
    """Get tennis odds data from Sportbex API"""
    try:
        url = "https://trial-api.sportbex.com/api/other-sport/event-odds"
        headers = {
            'sportbex-api-key': SPORTBEX_API_KEY,
            "Content-Type": "application/json"
        }
        request_data = request.get_json()
        response = requests.post(url, headers=headers, json=request_data, timeout=15)
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({
                "error": f"Sportbex API returned status {response.status_code}",
                "details": response.text[:500],
                "url": url,
                "request_data": request_data
            }), response.status_code
            
    except requests.exceptions.RequestException as e:
        return jsonify({
            "error": "Failed to fetch tennis odds",
            "details": str(e),
            "url": "https://trial-api.sportbex.com/api/other-sport/event-odds"
        }), 500

# Football endpoints (American Football - ID: 6423)
@app.route('/api/football/competitions', methods=['GET'])
def football_competitions():
    """Get football competitions data from Sportbex API"""
    try:
        url = "https://trial-api.sportbex.com/api/other-sport/competitions/6423"
        headers = {
            'sportbex-api-key': SPORTBEX_API_KEY
        }
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({
                "error": f"Sportbex API returned status {response.status_code}",
                "details": response.text[:500],
                "url": url
            }), response.status_code
            
    except requests.exceptions.RequestException as e:
        return jsonify({
            "error": "Failed to fetch football competitions",
            "details": str(e),
            "url": "https://trial-api.sportbex.com/api/other-sport/competitions/6423"
        }), 500

@app.route('/api/football/matchups/<competition_id>', methods=['GET'])
def football_matchups(competition_id):
    """Get football matchups for a specific competition"""
    try:
        url = f"https://trial-api.sportbex.com/api/other-sport/event/6423/{competition_id}"
        headers = {
            'sportbex-api-key': SPORTBEX_API_KEY
        }
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({
                "error": f"Sportbex API returned status {response.status_code}",
                "details": response.text[:500],
                "url": url
            }), response.status_code
            
    except requests.exceptions.RequestException as e:
        return jsonify({
            "error": "Failed to fetch football matchups",
            "details": str(e),
            "url": f"https://trial-api.sportbex.com/api/other-sport/event/6423/{competition_id}"
        }), 500

@app.route('/api/football/odds', methods=['POST'])
def football_odds():
    """Get football odds data from Sportbex API"""
    try:
        url = "https://trial-api.sportbex.com/api/other-sport/event-odds"
        headers = {
            'sportbex-api-key': SPORTBEX_API_KEY,
            "Content-Type": "application/json"
        }
        request_data = request.get_json() or {}
        response = requests.post(url, headers=headers, json=request_data, timeout=15)
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({
                "error": f"Sportbex API returned status {response.status_code}",
                "details": response.text[:500],
                "url": url,
                "request_data": request_data
            }), response.status_code
            
    except requests.exceptions.RequestException as e:
        return jsonify({
            "error": "Failed to fetch football odds",
            "details": str(e),
            "url": "https://trial-api.sportbex.com/api/other-sport/event-odds"
        }), 500

# Soccer endpoints (ID: 1)
@app.route('/api/soccer/competitions', methods=['GET'])
def soccer_competitions():
    """Get soccer competitions data from Sportbex API"""
    try:
        url = "https://trial-api.sportbex.com/api/sportbex/competitions/1"
        headers = {
            'sportbex-api-key': SPORTBEX_API_KEY
        }
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({
                "error": f"Sportbex API returned status {response.status_code}",
                "details": response.text[:500],
                "url": url
            }), response.status_code
            
    except requests.exceptions.RequestException as e:
        return jsonify({
            "error": "Failed to fetch soccer competitions",
            "details": str(e),
            "url": "https://trial-api.sportbex.com/api/sportbex/competitions/1"
        }), 500

@app.route('/api/soccer/matchups/<competition_id>', methods=['GET'])
def soccer_matchups(competition_id):
    """Get soccer matchups for a specific competition"""
    try:
        url = f"https://trial-api.sportbex.com/api/sportbex/event/1/{competition_id}"
        headers = {
            'sportbex-api-key': SPORTBEX_API_KEY
        }
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({
                "error": f"Sportbex API returned status {response.status_code}",
                "details": response.text[:500],
                "url": url
            }), response.status_code
            
    except requests.exceptions.RequestException as e:
        return jsonify({
            "error": "Failed to fetch soccer matchups",
            "details": str(e),
            "url": f"https://trial-api.sportbex.com/api/sportbex/event/1/{competition_id}"
        }), 500

@app.route('/api/soccer/odds/<event_id>/<market_id>', methods=['GET'])
def soccer_odds(event_id, market_id):
    """Get soccer odds data from Sportbex API"""
    try:
        url = f"https://trial-api.sportbex.com/api/sportbex/event-odds/{event_id}/{market_id}"
        headers = {
            'sportbex-api-key': SPORTBEX_API_KEY
        }
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            return jsonify(response.json())
        else:
            return jsonify({
                "error": f"Sportbex API returned status {response.status_code}",
                "details": response.text[:500],
                "url": url
            }), response.status_code
            
    except requests.exceptions.RequestException as e:
        return jsonify({
            "error": "Failed to fetch soccer odds",
            "details": str(e),
            "url": f"https://trial-api.sportbex.com/api/sportbex/event-odds/{event_id}/{market_id}"
        }), 500

if __name__ == "__main__":
    app.run(debug=False, port=5001, host='0.0.0.0')
