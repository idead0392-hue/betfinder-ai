"""
Local debug helper to reproduce Yahoo OAuth1 request token fetch.
This script reads YAHOO_CONSUMER_KEY and YAHOO_CONSUMER_SECRET from environment variables
and attempts to fetch a request token from Yahoo. It prints full response details which
are useful when Streamlit redacts the error message.

Usage:
    export YAHOO_CONSUMER_KEY=your_key
    export YAHOO_CONSUMER_SECRET=your_secret
    python yahoo_oauth_debug.py

WARNING: Do not commit your keys. Keep them in environment variables or a local .env
that is not checked into source control.
"""

import os
import sys
from requests_oauthlib import OAuth1Session

REQUEST_TOKEN_URL = "https://api.login.yahoo.com/oauth/v2/get_request_token"
CALLBACK_URI = "oob"

def main():
    key = os.getenv('YAHOO_CONSUMER_KEY')
    secret = os.getenv('YAHOO_CONSUMER_SECRET')
    if not key or not secret:
        print("Missing YAHOO_CONSUMER_KEY or YAHOO_CONSUMER_SECRET in environment.")
        sys.exit(2)

    yahoo = OAuth1Session(key, client_secret=secret, callback_uri=CALLBACK_URI)
    max_retries = 3
    backoff = 1
    for attempt in range(1, max_retries + 1):
        try:
            fetch_response = yahoo.fetch_request_token(REQUEST_TOKEN_URL)
            print("Request token fetched successfully:")
            for k, v in fetch_response.items():
                print(f"  {k}: {v}")
            break
        except Exception as e:
            print(f"Attempt {attempt} failed: {type(e)}: {e}")
            r = getattr(e, 'response', None)
            if r is not None:
                status = getattr(r, 'status_code', 'unknown')
                print("Response status:", status)
                # show Retry-After if provided
                ra = r.headers.get('Retry-After') if hasattr(r, 'headers') else None
                if ra:
                    print("Retry-After header:", ra)
                print("Response headers:")
                for hk, hv in r.headers.items():
                    print(f"  {hk}: {hv}")
                txt = getattr(r, 'text', '')
                if isinstance(txt, str):
                    print("Response body (truncated to 2000 chars):")
                    print(txt[:2000])
                # If rate limited, wait accordingly
                if status == 429 and attempt < max_retries:
                    wait = None
                    if ra:
                        try:
                            wait = int(ra)
                        except Exception:
                            wait = None
                    if wait is None:
                        wait = backoff
                    wait = min(wait, 60)
                    print(f"Rate limited. Waiting {wait}s before retrying...")
                    import time
                    time.sleep(wait)
                    backoff *= 2
                    continue
            else:
                print("No response object attached to exception. Full exception:\n", e)
            sys.exit(1)

if __name__ == '__main__':
    main()
