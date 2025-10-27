"""
API Providers Base Classes

This module provides base classes and utilities for API provider implementations
used throughout the BetFinder AI platform.
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum

# Optional re-export to support tests importing SportbexProvider from here
try:
    from sportbex_provider import SportbexProvider  # type: ignore
except Exception:
    SportbexProvider = None  # Will be imported lazily in factory if needed

logger = logging.getLogger(__name__)

class SportType(Enum):
    """Enumeration of supported sport types"""
    BASKETBALL = "basketball"
    FOOTBALL = "football" 
    TENNIS = "tennis"
    BASEBALL = "baseball"
    SOCCER = "soccer"
    HOCKEY = "hockey"

@dataclass
class APIResponse:
    """Standardized API response container"""
    success: bool
    data: Any = None
    error_message: Optional[str] = None
    status_code: Optional[int] = None
    response_time: Optional[float] = None
    
    @classmethod
    def success_response(cls, data: Any, response_time: float = None, status_code: int = 200):
        """Create a successful response"""
        return cls(
            success=True,
            data=data,
            response_time=response_time,
            status_code=status_code
        )
    
    @classmethod
    def error_response(cls, error_message: str, status_code: int = None, response_time: float = None):
        """Create an error response"""
        return cls(
            success=False,
            error_message=error_message,
            status_code=status_code,
            response_time=response_time
        )

@dataclass 
class RequestConfig:
    """Configuration for API requests"""
    timeout: int = 30
    retries: int = 3
    retry_delay: float = 1.0
    headers: Optional[Dict[str, str]] = None
    
class BaseAPIProvider(ABC):
    """Abstract base class for API providers"""
    
    def __init__(self, api_key: str, base_url: str, provider_name: str, **kwargs):
        """Initialize the base API provider"""
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.provider_name = provider_name
        self.request_config = RequestConfig(**kwargs)
        self.logger = logging.getLogger(f"{__name__}.{provider_name}")
        
        # Initialize headers
        self.headers = {
            'Content-Type': 'application/json',
            'User-Agent': f'BetFinder-AI/{provider_name}-Provider'
        }
        
    def health_check(self) -> APIResponse:
        """Perform a health check on the API"""
        try:
            start_time = time.time()
            # Simple health check - override in subclasses for specific implementations
            response_time = time.time() - start_time
            
            return APIResponse.success_response(
                data={"status": "healthy", "provider": self.provider_name, "response_time": response_time},
                response_time=response_time
            )
        except Exception as e:
            return APIResponse.error_response(f"Health check failed: {str(e)}")
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> APIResponse:
        """Make an HTTP request to the API"""
        try:
            import requests
            
            url = f"{self.base_url}/{endpoint.lstrip('/')}"
            start_time = time.time()
            
            # Prepare request arguments
            request_kwargs = {
                'headers': self.headers,
                'timeout': self.request_config.timeout
            }
            request_kwargs.update(kwargs)
            
            # Make the request
            if method.upper() == 'GET':
                response = requests.get(url, **request_kwargs)
            elif method.upper() == 'POST':
                response = requests.post(url, **request_kwargs)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response_time = time.time() - start_time
            
            # Check if request was successful
            response.raise_for_status()
            
            # Try to parse JSON response
            try:
                data = response.json()
            except ValueError:
                data = response.text
            
            return APIResponse.success_response(
                data=data,
                response_time=response_time,
                status_code=response.status_code
            )
            
        except Exception as e:
            response_time = time.time() - start_time if 'start_time' in locals() else None
            
            # Safely extract status code from exception
            status_code = None
            if hasattr(e, 'response') and e.response is not None:
                if hasattr(e.response, 'status_code'):
                    status_code = e.response.status_code
                elif isinstance(e.response, dict):
                    status_code = e.response.get('status_code')
            
            return APIResponse.error_response(
                error_message=str(e),
                response_time=response_time,
                status_code=status_code
            )
    
    @abstractmethod
    def get_competitions(self, sport: SportType = None) -> APIResponse:
        """Get available competitions/events"""
        pass
        
    @abstractmethod
    def get_odds(self, sport: SportType = None, market: str = None) -> APIResponse:
        """Get betting odds"""
        pass
        
    @abstractmethod
    def get_props(self, sport: SportType = None) -> APIResponse:
        """Get player/game props"""
        pass

if TYPE_CHECKING:
    pass

def create_sportbex_provider(api_key: str = None):
    """Factory function to create a SportbexProvider instance"""
    if api_key is None:
        api_key = 'NZLDw8ZXFv0O8elaPq0wjbP4zxb2gCwJDsArWQUF'
    
    from sportbex_provider import SportbexProvider
    return SportbexProvider(api_key=api_key)


class PrizePicksProvider:
    """
    Provider wrapper for PrizePicks data fetching.
    Uses PropsDataFetcher to get normalized PrizePicks props.
    """
    
    def __init__(self):
        """Initialize PrizePicksProvider"""
        try:
            from props_data_fetcher import PropsDataFetcher
            self.fetcher = PropsDataFetcher()
        except Exception as e:
            logger.error(f"Failed to initialize PropsDataFetcher: {e}")
            self.fetcher = None
    
    def get_props(self, sport: str = None, max_props: int = 1000) -> APIResponse:
        """
        Fetch PrizePicks props data.
        
        Args:
            sport: Optional sport filter (not currently used by fetcher)
            max_props: Maximum number of props to return
            
        Returns:
            APIResponse with props data or error
        """
        if not self.fetcher:
            return APIResponse.error_response("PropsDataFetcher not available")
        
        try:
            start_time = time.time()
            props = self.fetcher.fetch_prizepicks_props(max_items=max_props)
            response_time = time.time() - start_time
            
            # Filter by sport if specified
            if sport and isinstance(props, list):
                sport_lower = str(sport).lower()
                props = [p for p in props if str(p.get('sport', '')).lower() == sport_lower or 
                        str(p.get('league', '')).lower() == sport_lower]
            
            return APIResponse.success_response(
                data={"data": props, "count": len(props)},
                response_time=response_time
            )
        except Exception as e:
            return APIResponse.error_response(f"Failed to fetch PrizePicks props: {str(e)}")
