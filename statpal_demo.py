import requests

API_KEY = "60476ba91e9f5e9696625fbabc2554f48c115a3099ed877060ccb6cfe5a59a0a"

# You can change the sport and version as needed
API_VERSION = "v1"
SPORT_NAME = "soccer"

url = f"https://statpal.io/api/{API_VERSION}/{SPORT_NAME}/livescores"
params = {"access_key": API_KEY}

try:
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    print("Live Scores Response:")
    print(data)
except requests.exceptions.RequestException as e:
    print(f"Error fetching live scores: {e}")
