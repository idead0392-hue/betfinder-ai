import os
from requests_oauthlib import OAuth1Session
from lxml import etree

# Yahoo Fantasy Sports API credentials
CONSUMER_KEY = os.getenv("YAHOO_CONSUMER_KEY", "dj0yJmk9V0FGMzNaZUNVckNIJmQ9WVdrOU1XRjJOazF2Vm0wbWNHbzlNQT09JnM9Y29uc3VtZXJzZWNyZXQmc3Y9MCZ4PTRk")
CONSUMER_SECRET = os.getenv("YAHOO_CONSUMER_SECRET", "9d43ec9caa17a1b36671014dac8253728f1a0fe0")

REQUEST_TOKEN_URL = "https://api.login.yahoo.com/oauth/v2/get_request_token"
AUTHORIZE_URL = "https://api.login.yahoo.com/oauth/v2/request_auth"
ACCESS_TOKEN_URL = "https://api.login.yahoo.com/oauth/v2/get_token"

CALLBACK_URI = "oob"  # Out-of-band for CLI/desktop, change for webapp

# Step 1: Get request token
try:
    yahoo = OAuth1Session(CONSUMER_KEY, client_secret=CONSUMER_SECRET, callback_uri=CALLBACK_URI)
    fetch_response = yahoo.fetch_request_token(REQUEST_TOKEN_URL)
    resource_owner_key = fetch_response.get('oauth_token')
    resource_owner_secret = fetch_response.get('oauth_token_secret')
except Exception as e:
    print(f"Error: Failed to get request token from Yahoo: {e}")
    print("Please check your YAHOO_CONSUMER_KEY and YAHOO_CONSUMER_SECRET credentials.")
    exit(1)

# Step 2: Authorize
print("Go to the following URL to authorize:")
print(f"{AUTHORIZE_URL}?oauth_token={resource_owner_key}")
verifier = input("Paste the verifier code here: ").strip()

# Step 3: Get access token
try:
    yahoo = OAuth1Session(
        CONSUMER_KEY,
        client_secret=CONSUMER_SECRET,
        resource_owner_key=resource_owner_key,
        resource_owner_secret=resource_owner_secret,
        verifier=verifier,
    )
    access_token_data = yahoo.fetch_access_token(ACCESS_TOKEN_URL)
    access_token = access_token_data.get('oauth_token')
    access_token_secret = access_token_data.get('oauth_token_secret')
except Exception as e:
    print(f"Error: Failed to get access token: {e}")
    print("Please verify the verifier code and try again.")
    exit(1)

# Step 4: Make an API call (example: get NFL game info)
try:
    yahoo = OAuth1Session(
        CONSUMER_KEY,
        client_secret=CONSUMER_SECRET,
        resource_owner_key=access_token,
        resource_owner_secret=access_token_secret,
    )
    response = yahoo.get("https://fantasysports.yahooapis.com/fantasy/v2/game/nfl")
    if response.status_code == 200:
        xml_root = etree.fromstring(response.content)
        print(etree.tostring(xml_root, pretty_print=True).decode())
    else:
        print(f"Error: Yahoo API returned status code {response.status_code}")
        print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: Failed to fetch data from Yahoo API: {e}")
