# Yahoo OAuth Error Handling

## Overview

This document describes the error handling improvements made to the Yahoo OAuth flow in the BetFinder AI application.

## Problem

Previously, when Yahoo OAuth authentication failed (e.g., due to invalid credentials, network issues, or API errors), the application would crash with an unhandled `TokenRequestDenied` exception. This resulted in a poor user experience and made debugging difficult.

## Solution

Error handling has been added to all critical OAuth operations:

### 1. Request Token Fetch (app.py, lines 49-58)

```python
try:
    yahoo = OAuth1Session(CONSUMER_KEY, client_secret=CONSUMER_SECRET, callback_uri=CALLBACK_URI)
    fetch_response = yahoo.fetch_request_token(REQUEST_TOKEN_URL)
    # ... store tokens ...
except Exception as e:
    sidebar.error(f"Failed to initiate Yahoo OAuth: {str(e)}")
    sidebar.info("Please check your Yahoo API credentials and try again.")
```

### 2. Access Token Fetch (app.py, lines 66-86)

```python
try:
    yahoo = OAuth1Session(...)
    access_token_data = yahoo.fetch_access_token(ACCESS_TOKEN_URL)
    # ... store tokens ...
except Exception as e:
    sidebar.error(f"Failed to fetch access token: {str(e)}")
    sidebar.info("Please verify your verifier code and try again.")
    # Provides option to restart OAuth flow
```

### 3. Yahoo API Calls (app.py, lines 123-147)

```python
try:
    yahoo = OAuth1Session(...)
    resp = yahoo.get("https://fantasysports.yahooapis.com/...")
    # ... process response ...
except Exception as e:
    st.error(f"Error fetching Yahoo data: {str(e)}")
    st.info("Your session may have expired. Please log out and log in again.")
```

### 4. CLI Script (yahoo_oauth_cli.py)

Similar error handling has been added to the CLI OAuth script with appropriate error messages and exit codes.

## Benefits

1. **No More Crashes**: The application gracefully handles OAuth errors instead of crashing
2. **Better User Experience**: Users receive clear, actionable error messages
3. **Easier Debugging**: Error messages help identify the root cause (credentials, network, etc.)
4. **Recovery Options**: Users can restart the OAuth flow or retry without restarting the app

## Testing

Error handling has been tested with:
- Invalid credentials scenarios
- Network error simulation
- Token expiration scenarios
- Success cases to ensure normal flow is not affected

## Configuration

Ensure your Yahoo API credentials are properly configured in `.streamlit/secrets.toml`:

```toml
YAHOO_CONSUMER_KEY = "your_consumer_key"
YAHOO_CONSUMER_SECRET = "your_consumer_secret"
```

## Common Errors

| Error | Possible Cause | Solution |
|-------|---------------|----------|
| "Failed to initiate Yahoo OAuth" | Invalid credentials | Check YAHOO_CONSUMER_KEY and YAHOO_CONSUMER_SECRET |
| "Failed to fetch access token" | Invalid verifier code | Re-authorize and enter the correct verifier |
| "Error fetching Yahoo data" | Expired session | Log out and log in again |
| 401 Unauthorized | Invalid/expired credentials | Re-authorize with Yahoo |
