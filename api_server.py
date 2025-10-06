from flask import Flask
from ai_integration_test import (
    get_esport_team,
    get_esport_tournament,
    get_esport_season_info,
    get_esport_season_last_matches,
    RAPIDAPI_KEY, RAPIDAPI_HOST
)

app = Flask(__name__)

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

if __name__ == "__main__":
    app.run(debug=True)
