# BetFinder AI - Source Code Audit Report

**Date:** October 8, 2025  
**Repository:** betfinder-ai  
**Branch:** copilot/vscode1759760048042  
**Audit Scope:** app.py, bet_db.py, api_providers.py, api_server.py, ai_integration_test.py

---

## Executive Summary

This audit reveals a functional but architecturally fragmented codebase with **critical security vulnerabilities** that require immediate attention. The application shows good database design and caching concepts but suffers from hard-coded API keys, inconsistent error handling, and monolithic structure that impedes maintainability.

### Critical Issues Found:
- 🚨 **CRITICAL**: Hard-coded API keys exposed in source code
- ⚠️ **HIGH**: Silent error handling masking failures
- ⚠️ **MEDIUM**: No cache size limits causing potential memory leaks
- ⚠️ **MEDIUM**: Monolithic app.py file (1600+ lines) mixing concerns

---

## 1. Props API Integration Implementation

### Current Props Tab Implementation (app.py lines 314-434)

```python
with tabs[2]:
    st.header("Props")
    st.write("Use a custom API that accepts a Bearer token to provide props data.")
    st.markdown("**Security:** Do not commit your tokens. Add them to `.streamlit/secrets.toml` as `NEW_API_TOKEN` or paste temporarily below.")

    provider = st.selectbox("Provider", ["None", "Custom (Bearer token)", "PandaScore"])

    if provider == "Custom (Bearer token)":
        endpoint = st.text_input("API endpoint (full URL)", value="https://api.example.com/v1/props")
        use_secret = st.checkbox("Use token from st.secrets['NEW_API_TOKEN']", value=True)
        token = None
        if use_secret:
            token = st.secrets.get("NEW_API_TOKEN") if hasattr(st, 'secrets') else None
            if not token:
                st.warning("`NEW_API_TOKEN` not found in `st.secrets`. You can paste a token below for testing.")
        else:
            token = st.text_input("Paste token (will not be saved)", type="password")

        col_btn, col_clear = st.columns([1, 1])
        with col_btn:
            if st.button("Load Props"):
                if not endpoint:
                    st.error("Please provide an API endpoint to load props from.")
                elif not token:
                    st.error("No token available. Add `NEW_API_TOKEN` to `.streamlit/secrets.toml` or paste a token.")
                else:
                    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
                    st.info(f"Calling {endpoint} with Bearer token (redacted in UI)")
                    try:
                        resp = requests.get(endpoint, headers=headers, timeout=20)
                        st.write("Status:", resp.status_code)
                        if resp.status_code == 200:
                            try:
                                data = resp.json()
                            except Exception:
                                data = resp.text
                            set_cached_data("custom_props", data)
                            st.success("Props loaded and cached (key: custom_props). Preview below.")
                        else:
                            st.error(f"API returned status {resp.status_code}")
                            st.code(resp.text[:2000])
                    except Exception as e:
                        st.error(f"Request failed: {e}")

        with col_clear:
            if st.button("Clear Cached Props"):
                if "custom_props" in st.session_state.data_cache:
                    st.session_state.data_cache.pop("custom_props", None)
                    st.session_state.cache_timestamp.pop("custom_props", None)
                    st.success("Cleared cached custom props.")
                else:
                    st.info("No cached custom props to clear.")

        # Show cached props if present
        cached = get_cached_data("custom_props")
        if cached is not None:
            st.markdown("### Cached Props Preview")
            try:
                if isinstance(cached, list):
                    df = pd.DataFrame(cached)
                    st.dataframe(df)
                elif isinstance(cached, dict):
                    # Show flattened preview for dicts
                    df = pd.json_normalize(cached)
                    st.dataframe(df)
                else:
                    st.code(str(cached)[:10000])
            except Exception:
                # Fallback to JSON/text
                if isinstance(cached, (dict, list)):
                    st.json(cached)
                else:
                    st.code(str(cached)[:10000])

        st.markdown("---")
        st.caption("Tip: store your token in `.streamlit/secrets.toml` as `NEW_API_TOKEN = \"your_token_here\"` to avoid pasting it into the UI.")

    elif provider == "PandaScore":
        # PandaScore implementation...
        
    else:
        st.info("Select a provider to configure props data source.")
```

**Analysis:**
- ✅ **Good**: Proper Bearer token handling
- ✅ **Good**: Integration with caching system  
- ✅ **Good**: UI feedback and error messages
- ⚠️ **Issue**: Mixing UI logic with API calls
- ⚠️ **Issue**: No input validation for endpoint URLs

---

## 2. Caching Helper Functions (app.py lines 30-70)

```python
# Cache duration (5 minutes)
CACHE_DURATION = 300

def is_cache_valid(cache_key):
    """Check if cached data is still valid"""
    if cache_key not in st.session_state.cache_timestamp:
        return False
    elapsed = time.time() - st.session_state.cache_timestamp[cache_key]
    return elapsed < CACHE_DURATION

def get_cached_data(cache_key):
    """Get data from cache if valid"""
    if is_cache_valid(cache_key):
        return st.session_state.data_cache.get(cache_key)
    return None

def set_cached_data(cache_key, data):
    """Store data in cache with timestamp"""
    st.session_state.data_cache[cache_key] = data
    st.session_state.cache_timestamp[cache_key] = time.time()

def load_api_data(url, cache_key, method='GET', data=None):
    """Load data from API with caching"""
    cached = get_cached_data(cache_key)
    if cached is not None:
        return cached
    
    try:
        if method == 'POST':
            response = requests.post(url, json=data or {}, timeout=10)
        else:
            response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            set_cached_data(cache_key, result)
            return result
    except:
        pass
    
    return None
```

**Critical Issues:**
- 🚨 **CRITICAL**: Silent failure with bare `except:` clause (line 65)
- ⚠️ **HIGH**: No cache size limits - potential memory leak
- ⚠️ **MEDIUM**: No cache persistence across sessions
- ⚠️ **MEDIUM**: Fixed 5-minute TTL not configurable

---

## 3. Hard-Coded API Keys (IMMEDIATE SECURITY REMEDIATION REQUIRED)

### api_server.py (Line 5)
```python
# Sportbex API key
SPORTBEX_API_KEY = 'NZLDw8ZXFv0O8elaPq0wjbP4zxb2gCwJDsArWQUF'
```

### ai_integration_test.py (Line 9)
```python
# RapidAPI Configuration
RAPIDAPI_KEY = "4ac75ea836mshd4804cec3a1eea0p1085f3jsn0b73793a40af"
RAPIDAPI_HOST = "esportapi1.p.rapidapi.com"
```

**🚨 CRITICAL SECURITY VULNERABILITY:**
These API keys are exposed in the source code and committed to git history. **Immediate action required:**

1. **Revoke these API keys** in the respective provider dashboards
2. **Generate new keys** and store in environment variables
3. **Remove from git history** using `git filter-branch` or similar
4. **Add to .gitignore** any secrets files

---

## 4. Error Handling Analysis

### Poor Patterns Found:

**Silent Failures (Multiple Locations):**
```python
# app.py line 65 - Bare except clause
except:
    pass

# app.py lines 467, 761, 970 - Similar patterns
except:
    return []
```

**Inconsistent Error Responses:**
```python
# Good pattern (api_server.py)
except requests.exceptions.RequestException as e:
    return jsonify({
        "error": "Failed to fetch basketball props",
        "details": str(e)
    }), 500

# Poor pattern (app.py) - Generic handling
except Exception as e:
    st.error(f"Request failed: {e}")
```

---

## 5. Architectural Assessment

### Current Structure Issues:

**Monolithic app.py (1624 lines):**
- Mixed concerns: UI, API calls, caching, visualization
- No separation of business logic
- Difficult to test and maintain

**Unused/Empty Files:**
- `api_providers.py` - Empty file suggesting incomplete refactoring

**Good Patterns Found:**
- `bet_db.py` - Well-structured database layer with proper connection management
- Consistent use of context managers for database operations
- Good separation in database module

---

## 6. Detailed Recommendations

### Priority 1: Security Fixes (IMMEDIATE)
```bash
# 1. Remove hard-coded keys
git filter-branch --force --index-filter \
  'git rm --cached --ignore-unmatch api_server.py ai_integration_test.py' \
  --prune-empty --tag-name-filter cat -- --all

# 2. Create .env file
cat > .env << EOF
SPORTBEX_API_KEY=your_new_key_here
RAPIDAPI_KEY=your_new_key_here
RAPIDAPI_HOST=esportapi1.p.rapidapi.com
EOF

# 3. Update code to use environment variables
```

**Code changes needed:**
```python
# api_server.py - Replace hard-coded key
import os
from dotenv import load_dotenv

load_dotenv()
SPORTBEX_API_KEY = os.getenv('SPORTBEX_API_KEY')
if not SPORTBEX_API_KEY:
    raise ValueError("SPORTBEX_API_KEY environment variable required")
```

### Priority 2: Error Handling Standardization
```python
# Create error handling decorator
import logging
from functools import wraps

def handle_api_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.Timeout:
            logging.error(f"Timeout in {func.__name__}")
            return {"error": "API timeout", "retry": True}
        except requests.exceptions.ConnectionError:
            logging.error(f"Connection error in {func.__name__}")
            return {"error": "Connection failed", "retry": True}
        except Exception as e:
            logging.error(f"Unexpected error in {func.__name__}: {e}")
            return {"error": "Internal error", "retry": False}
    return wrapper
```

### Priority 3: Caching Modernization
```python
# Replace session-based caching
from cachetools import TTLCache
import redis

class CacheManager:
    def __init__(self, max_size=1000, ttl=300):
        self.local_cache = TTLCache(maxsize=max_size, ttl=ttl)
        self.redis_client = redis.Redis() if os.getenv('REDIS_URL') else None
    
    def get(self, key):
        # Try local first, then Redis
        value = self.local_cache.get(key)
        if value is None and self.redis_client:
            value = self.redis_client.get(key)
            if value:
                self.local_cache[key] = value
        return value
    
    def set(self, key, value, ttl=None):
        self.local_cache[key] = value
        if self.redis_client:
            self.redis_client.setex(key, ttl or 300, value)
```

### Priority 4: API Provider Abstraction
```python
# Implement api_providers.py
from abc import ABC, abstractmethod

class BaseAPIProvider(ABC):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.session = requests.Session()
        
    @abstractmethod
    def get_props(self, **kwargs):
        pass
        
    def _make_request(self, url, headers=None, **kwargs):
        # Standardized request handling with retries
        pass

class SportbexProvider(BaseAPIProvider):
    def get_props(self, sport_id: str):
        headers = {'sportbex-api-key': self.api_key}
        return self._make_request(f"https://trial-api.sportbex.com/api/other-sport/competitions/{sport_id}", headers)

class PandaScoreProvider(BaseAPIProvider):
    def get_props(self, endpoint: str):
        headers = {'Authorization': f'Bearer {self.api_key}'}
        return self._make_request(endpoint, headers)
```

### Priority 5: Code Structure Refactoring
```
betfinder-ai/
├── src/
│   ├── api/
│   │   ├── providers/
│   │   │   ├── __init__.py
│   │   │   ├── base.py          # BaseAPIProvider
│   │   │   ├── sportbex.py      # SportbexProvider
│   │   │   └── pandascore.py    # PandaScoreProvider
│   │   ├── cache.py             # CacheManager
│   │   └── client.py            # HTTP client wrapper
│   ├── ui/
│   │   ├── components/          # Reusable UI components
│   │   ├── pages/
│   │   │   ├── props.py         # Props tab logic
│   │   │   ├── tennis.py        # Tennis tab logic
│   │   │   └── basketball.py    # Basketball tab logic
│   │   └── utils.py             # UI utilities
│   ├── models/                  # Data models and schemas
│   ├── config.py                # Configuration management
│   └── main.py                  # Entry point
├── tests/
│   ├── test_api/
│   ├── test_ui/
│   └── test_cache/
├── .env.example                 # Template for environment variables
├── .gitignore                   # Include .env, secrets.toml
└── requirements.txt
```

---

## 7. Testing Strategy

### Current Testing Status: ❌ **No tests found**

**Recommended test structure:**
```python
# tests/test_api/test_providers.py
import pytest
from src.api.providers.sportbex import SportbexProvider

def test_sportbex_provider_init():
    provider = SportbexProvider("test_key")
    assert provider.api_key == "test_key"

def test_sportbex_get_props(mock_requests):
    # Test API provider implementations
    pass

# tests/test_cache/test_cache_manager.py
def test_cache_ttl_expiry():
    # Test cache expiration logic
    pass

def test_cache_size_limits():
    # Test memory management
    pass
```

---

## 8. Immediate Action Checklist

### Critical (Do Today):
- [ ] **Revoke exposed API keys** in provider dashboards
- [ ] **Generate new API keys** with restricted permissions
- [ ] **Create .env file** and move keys out of source code
- [ ] **Add .env to .gitignore**
- [ ] **Update deployment scripts** to use environment variables

### High Priority (This Week):
- [ ] **Replace bare except clauses** with specific exception handling
- [ ] **Add structured logging** throughout the application
- [ ] **Implement cache size limits** to prevent memory leaks
- [ ] **Add input validation** for all API endpoints
- [ ] **Create basic unit tests** for core functions

### Medium Priority (Next Sprint):
- [ ] **Refactor app.py** into smaller modules
- [ ] **Implement API provider abstraction**
- [ ] **Add Redis caching** for production environments
- [ ] **Create comprehensive test suite**
- [ ] **Add API rate limiting**

### Long Term:
- [ ] **Database migration system**
- [ ] **API documentation** with OpenAPI/Swagger
- [ ] **CI/CD pipeline** with security scanning
- [ ] **Performance monitoring** and alerting
- [ ] **Container deployment** with Docker

---

## 9. Security Compliance Notes

**Current Security Posture: POOR**
- Hard-coded secrets in source code
- No input sanitization
- No rate limiting
- No audit logging

**Required for Production:**
- Environment-based secret management
- Input validation and sanitization  
- Rate limiting and DDoS protection
- Comprehensive audit logging
- Regular security scanning
- Dependency vulnerability monitoring

---

**Report Generated:** October 8, 2025  
**Next Review Date:** October 15, 2025 (post-critical fixes)