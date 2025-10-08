#!/usr/bin/env python3
"""
BetFinder AI - API Provider Abstraction Layer

This module provides a comprehensive abstraction layer for different sports betting
and esports API providers. It standardizes the interface for retrieving props data,
competitions, odds, and other betting-related information across multiple providers.

The design follows the Strategy pattern, allowing easy addition of new providers
without modifying existing code.

Author: BetFinder AI Team
Date: October 2025
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


# Configure module-level logging
logger = logging.getLogger(__name__)


class APIProviderError(Exception):
    """Base exception for API provider errors"""
    pass


class AuthenticationError(APIProviderError):
    """Raised when API authentication fails"""
    pass


class RateLimitError(APIProviderError):
    """Raised when API rate limit is exceeded"""
    pass


class APIConnectionError(APIProviderError):
    """Raised when API connection fails"""
    pass


class DataFormat(Enum):
    """Enumeration of supported data formats"""
    JSON = "json"
    XML = "xml"
    TEXT = "text"


@dataclass
class APIResponse:
    """
    Standardized API response container
    
    This class wraps all API responses in a consistent format, making it easier
    to handle responses from different providers uniformly.
    """
    success: bool
    data: Optional[Any] = None
    error_message: Optional[str] = None
    status_code: Optional[int] = None
    headers: Optional[Dict[str, str]] = None
    response_time: Optional[float] = None
    provider: Optional[str] = None
    
    def __post_init__(self):
        """Validate response data after initialization"""
        if not self.success and not self.error_message:
            self.error_message = "Unknown error occurred"


@dataclass
class RequestConfig:
    """Configuration for API requests"""
    timeout: int = 30
    max_retries: int = 3
    backoff_factor: float = 0.3
    retry_status_codes: List[int] = None
    
    def __post_init__(self):
        """Set default retry status codes if not provided"""
        if self.retry_status_codes is None:
            self.retry_status_codes = [500, 502, 503, 504]


class BaseAPIProvider(ABC):
    """
    Abstract base class for all API providers
    
    This class defines the common interface that all API providers must implement.
    It provides shared functionality for HTTP requests, error handling, caching,
    and response standardization.
    
    Key Features:
    - Automatic retry logic with exponential backoff
    - Request/response logging for debugging
    - Rate limiting protection
    - Standardized error handling
    - Response caching capabilities
    - Request timeout management
    
    Attributes:
        provider_name (str): Human-readable name of the provider
        base_url (str): Base URL for the API
        api_key (str): API authentication key
        headers (Dict[str, str]): Default headers for all requests
        session (requests.Session): HTTP session with retry configuration
        request_config (RequestConfig): Request configuration settings
        last_request_time (float): Timestamp of last API request (for rate limiting)
        min_request_interval (float): Minimum seconds between requests
    """
    
    def __init__(
        self, 
        api_key: str, 
        base_url: str = "",
        provider_name: str = "",
        request_config: Optional[RequestConfig] = None,
        min_request_interval: float = 0.1
    ):
        """
        Initialize the API provider
        
        Args:
            api_key: Authentication key for the API
            base_url: Base URL for API endpoints
            provider_name: Human-readable provider name
            request_config: Configuration for HTTP requests
            min_request_interval: Minimum seconds between requests (rate limiting)
            
        Raises:
            ValueError: If api_key is empty or None
        """
        if not api_key:
            raise ValueError(f"{provider_name or 'API Provider'} requires a valid API key")
            
        self.provider_name = provider_name or self.__class__.__name__
        self.base_url = base_url.rstrip('/')  # Remove trailing slash
        self.api_key = api_key
        self.request_config = request_config or RequestConfig()
        self.last_request_time = 0.0
        self.min_request_interval = min_request_interval
        
        # Initialize default headers (to be customized by subclasses)
        self.headers = {
            'User-Agent': f'BetFinder-AI/{self.provider_name}',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        # Configure HTTP session with retry strategy
        self.session = self._create_session()
        
        logger.info(f"Initialized {self.provider_name} API provider")
    
    def _create_session(self) -> requests.Session:
        """
        Create and configure a requests session with retry logic
        
        Returns:
            Configured requests.Session object
        """
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.request_config.max_retries,
            status_forcelist=self.request_config.retry_status_codes,
            backoff_factor=self.request_config.backoff_factor,
            allowed_methods=["HEAD", "GET", "OPTIONS", "POST"]
        )
        
        # Mount adapter with retry strategy
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _enforce_rate_limit(self):
        """
        Enforce rate limiting between API requests
        
        This method ensures that requests are spaced appropriately to avoid
        hitting provider rate limits.
        """
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.3f} seconds")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _prepare_headers(self, additional_headers: Optional[Dict[str, str]] = None) -> Dict[str, str]:
        """
        Prepare headers for API request
        
        Args:
            additional_headers: Additional headers to include
            
        Returns:
            Complete headers dictionary
        """
        headers = self.headers.copy()
        if additional_headers:
            headers.update(additional_headers)
        return headers
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        additional_headers: Optional[Dict[str, str]] = None,
        timeout: Optional[int] = None
    ) -> APIResponse:
        """
        Make an HTTP request to the API
        
        This is the core method that handles all HTTP communication with the API.
        It includes error handling, logging, and response standardization.
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (relative to base_url)
            params: URL parameters
            data: Form data
            json_data: JSON data for request body
            additional_headers: Extra headers for this request
            timeout: Request timeout (overrides default)
            
        Returns:
            APIResponse object with standardized response data
        """
        # Enforce rate limiting
        self._enforce_rate_limit()
        
        # Prepare URL and headers
        url = f"{self.base_url}/{endpoint.lstrip('/')}" if self.base_url else endpoint
        headers = self._prepare_headers(additional_headers)
        timeout = timeout or self.request_config.timeout
        
        # Log request details (excluding sensitive data)
        logger.debug(f"Making {method} request to {url}")
        if params:
            logger.debug(f"Request params: {params}")
        
        start_time = time.time()
        
        try:
            # Make the HTTP request
            response = self.session.request(
                method=method,
                url=url,
                headers=headers,
                params=params,
                data=data,
                json=json_data,
                timeout=timeout
            )
            
            response_time = time.time() - start_time
            
            # Log response details
            logger.debug(f"Response status: {response.status_code}, time: {response_time:.3f}s")
            
            # Handle different response status codes
            if response.status_code == 200:
                return self._handle_success_response(response, response_time)
            elif response.status_code == 401:
                raise AuthenticationError(f"Authentication failed for {self.provider_name}")
            elif response.status_code == 429:
                raise RateLimitError(f"Rate limit exceeded for {self.provider_name}")
            else:
                return self._handle_error_response(response, response_time)
                
        except requests.exceptions.Timeout:
            logger.error(f"Request timeout for {self.provider_name} after {timeout}s")
            return APIResponse(
                success=False,
                error_message=f"Request timeout after {timeout} seconds",
                provider=self.provider_name
            )
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error for {self.provider_name}: {e}")
            return APIResponse(
                success=False,
                error_message=f"Connection error: {str(e)}",
                provider=self.provider_name
            )
        except (AuthenticationError, RateLimitError) as e:
            logger.error(f"API error for {self.provider_name}: {e}")
            return APIResponse(
                success=False,
                error_message=str(e),
                provider=self.provider_name
            )
        except Exception as e:
            logger.error(f"Unexpected error for {self.provider_name}: {e}")
            return APIResponse(
                success=False,
                error_message=f"Unexpected error: {str(e)}",
                provider=self.provider_name
            )
    
    def _handle_success_response(self, response: requests.Response, response_time: float) -> APIResponse:
        """
        Handle successful API response
        
        Args:
            response: The HTTP response object
            response_time: Time taken for the request
            
        Returns:
            APIResponse with parsed data
        """
        try:
            # Try to parse as JSON first
            data = response.json()
        except ValueError:
            # Fall back to text if JSON parsing fails
            data = response.text
        
        return APIResponse(
            success=True,
            data=data,
            status_code=response.status_code,
            headers=dict(response.headers),
            response_time=response_time,
            provider=self.provider_name
        )
    
    def _handle_error_response(self, response: requests.Response, response_time: float) -> APIResponse:
        """
        Handle error API response
        
        Args:
            response: The HTTP response object
            response_time: Time taken for the request
            
        Returns:
            APIResponse with error information
        """
        try:
            error_data = response.json()
            error_message = error_data.get('error', error_data.get('message', f"API error: {response.status_code}"))
        except ValueError:
            error_message = f"API error {response.status_code}: {response.text[:200]}"
        
        logger.warning(f"API error for {self.provider_name}: {error_message}")
        
        return APIResponse(
            success=False,
            error_message=error_message,
            status_code=response.status_code,
            headers=dict(response.headers),
            response_time=response_time,
            provider=self.provider_name
        )
    
    # Abstract methods that must be implemented by subclasses
    
    @abstractmethod
    def get_competitions(self, sport_id: Optional[str] = None) -> APIResponse:
        """
        Get list of available competitions/tournaments
        
        Args:
            sport_id: Optional sport identifier to filter competitions
            
        Returns:
            APIResponse containing competition data
        """
        pass
    
    @abstractmethod
    def get_props(self, **kwargs) -> APIResponse:
        """
        Get props/betting data
        
        Args:
            **kwargs: Provider-specific parameters
            
        Returns:
            APIResponse containing props data
        """
        pass
    
    @abstractmethod
    def get_odds(self, **kwargs) -> APIResponse:
        """
        Get odds data for events
        
        Args:
            **kwargs: Provider-specific parameters
            
        Returns:
            APIResponse containing odds data
        """
        pass
    
    # Optional methods that can be overridden by subclasses
    
    def get_events(self, competition_id: Optional[str] = None) -> APIResponse:
        """
        Get list of events/matches
        
        Args:
            competition_id: Optional competition identifier
            
        Returns:
            APIResponse containing events data
        """
        return APIResponse(
            success=False,
            error_message=f"get_events not implemented for {self.provider_name}",
            provider=self.provider_name
        )
    
    def get_teams(self, **kwargs) -> APIResponse:
        """
        Get team information
        
        Args:
            **kwargs: Provider-specific parameters
            
        Returns:
            APIResponse containing team data
        """
        return APIResponse(
            success=False,
            error_message=f"get_teams not implemented for {self.provider_name}",
            provider=self.provider_name
        )
    
    def health_check(self) -> APIResponse:
        """
        Check if the API provider is healthy and responding
        
        Returns:
            APIResponse indicating provider health status
        """
        try:
            # Most APIs have a simple endpoint for health checks
            # Subclasses should override this with provider-specific health check
            return APIResponse(
                success=True,
                data={"status": "healthy", "provider": self.provider_name},
                provider=self.provider_name
            )
        except Exception as e:
            return APIResponse(
                success=False,
                error_message=f"Health check failed: {str(e)}",
                provider=self.provider_name
            )
    
    def __str__(self) -> str:
        """String representation of the provider"""
        return f"{self.provider_name} API Provider"
    
    def __repr__(self) -> str:
        """Developer representation of the provider"""
        return f"{self.__class__.__name__}(provider_name='{self.provider_name}', base_url='{self.base_url}')"


# Example implementation helper functions (to be used by concrete providers)

def standardize_competition_data(raw_data: Any, provider_name: str) -> List[Dict[str, Any]]:
    """
    Standardize competition data across different providers
    
    Args:
        raw_data: Raw competition data from API
        provider_name: Name of the provider
        
    Returns:
        List of standardized competition dictionaries
    """
    # This would contain logic to normalize competition data
    # Implementation depends on specific provider formats
    logger.debug(f"Standardizing competition data from {provider_name}")
    
    if isinstance(raw_data, list):
        return raw_data
    elif isinstance(raw_data, dict) and 'data' in raw_data:
        return raw_data['data']
    else:
        return []


def standardize_odds_data(raw_data: Any, provider_name: str) -> List[Dict[str, Any]]:
    """
    Standardize odds data across different providers
    
    Args:
        raw_data: Raw odds data from API
        provider_name: Name of the provider
        
    Returns:
        List of standardized odds dictionaries
    """
    # This would contain logic to normalize odds data
    # Implementation depends on specific provider formats
    logger.debug(f"Standardizing odds data from {provider_name}")
    
    if isinstance(raw_data, list):
        return raw_data
    elif isinstance(raw_data, dict) and 'data' in raw_data:
        return raw_data['data']
    else:
        return []


# Sport type enumeration for Sportbex API
class SportType(Enum):
    """Enumeration of supported sports for Sportbex API."""
    TENNIS = "2"
    BASKETBALL = "7522" 
    AMERICAN_FOOTBALL = "6423"
    SOCCER = "1"
    BASEBALL = "5"
    HOCKEY = "6"
    ESPORTS = "7"
    COLLEGE_FOOTBALL = "8"


class SportbexProvider(BaseAPIProvider):
    """
    Sportbex API provider implementation.
    
    This provider interfaces with the Sportbex betting data API, providing access to
    competitions, odds, props, and matchup data across multiple sports.
    
    The provider handles Sportbex-specific authentication, endpoint routing,
    and data formatting while maintaining the standard BaseAPIProvider interface.
    
    Environment Variables:
        SPORTBEX_API_KEY: Required API key for authentication
        SPORTBEX_API_URL: Optional custom base URL (defaults to trial API)
    
    Supported Sports:
        - Tennis (ID: 2)
        - Basketball (ID: 7522)
        - American Football (ID: 6423)
        - Soccer (ID: 1)
        - Baseball (ID: 5)
        - Hockey (ID: 6)
        - Esports (ID: 7)
        - College Football (ID: 8)
    
    Key Features:
        - Automatic sport ID mapping
        - Multiple API endpoint routing based on sport type
        - Comprehensive error handling and logging
        - Request timeout and retry management
        - Health check functionality
    """
    
    def __init__(
        self, 
        api_key: Optional[str] = None, 
        base_url: Optional[str] = None, 
        timeout: int = 20,
        **kwargs
    ):
        """
        Initialize Sportbex provider.
        
        Args:
            api_key: Sportbex API key (defaults to SPORTBEX_API_KEY env var)
            base_url: API base URL (defaults to SPORTBEX_API_URL env var or default URL)
            timeout: Request timeout in seconds (default: 20)
            **kwargs: Additional configuration parameters
            
        Raises:
            ValueError: If API key is not provided and not found in environment
        """
        import os
        
        # Load configuration from environment variables
        api_key = api_key or os.getenv('SPORTBEX_API_KEY')
        if not api_key:
            raise ValueError(
                "SPORTBEX_API_KEY is required. Set it as an environment variable or pass it directly."
            )
        
        base_url = base_url or os.getenv(
            'SPORTBEX_API_URL', 
            'https://trial-api.sportbex.com'
        )
        
        # Create request config with appropriate timeout
        request_config = RequestConfig(
            timeout=timeout,
            max_retries=3,
            backoff_factor=0.5,
            retry_status_codes=[500, 502, 503, 504, 429]
        )
        
        # Initialize base provider
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            provider_name="Sportbex",
            request_config=request_config,
            min_request_interval=0.2,  # 200ms between requests
            **kwargs
        )
        
        # Sportbex-specific headers
        self.headers.update({
            'sportbex-api-key': self.api_key,
            'Accept': 'application/json',
            'User-Agent': 'BetFinder-AI/Sportbex-1.0'
        })
        
        logger.info(f"Initialized SportbexProvider with base URL: {self.base_url}")
    
    def _get_sport_id(self, sport: Union[SportType, str, int]) -> str:
        """
        Convert sport parameter to Sportbex sport ID.
        
        Args:
            sport: Sport type enum, string name, or numeric ID
            
        Returns:
            Sportbex sport ID string
            
        Raises:
            ValueError: If sport type is not supported
        """
        if isinstance(sport, SportType):
            return sport.value
        elif isinstance(sport, str):
            # Try to find matching enum by name
            sport_upper = sport.upper().replace(' ', '_')
            for sport_type in SportType:
                if sport_type.name == sport_upper:
                    return sport_type.value
            # If not found, assume it's already an ID
            return sport
        elif isinstance(sport, int):
            return str(sport)
        else:
            raise ValueError(f"Unsupported sport type: {type(sport)}. Use SportType enum, string name, or numeric ID.")
    
    def _determine_endpoint_type(self, sport_id: str) -> str:
        """
        Determine which API endpoint type to use based on sport ID.
        
        Sportbex uses different endpoint patterns for different sports:
        - Soccer: /api/sportbex/competitions/{sport_id}
        - Baseball/Hockey/Esports/College Football: /api/sportbex/sports/{sport_id}/competitions
        - Tennis/Basketball/American Football: /api/other-sport/competitions/{sport_id}
        
        Args:
            sport_id: Sportbex sport identifier
            
        Returns:
            Endpoint type: 'sportbex', 'sportbex_sports', or 'other_sport'
        """
        if sport_id == '1':  # Soccer
            return 'sportbex'
        elif sport_id in ['5', '6', '7', '8']:  # Baseball, Hockey, Esports, College Football
            return 'sportbex_sports'
        else:  # Tennis, Basketball, American Football
            return 'other_sport'
    
    def get_competitions(self, sport_id: Optional[str] = None, sport: Optional[Union[SportType, str]] = None) -> APIResponse:
        """
        Get competitions/leagues for a specific sport.
        
        Args:
            sport_id: Direct sport ID (legacy parameter)
            sport: Sport type enum or string identifier
            
        Returns:
            APIResponse with competitions data
        """
        try:
            # Handle both legacy sport_id and new sport parameter
            if sport is not None:
                sport_id = self._get_sport_id(sport)
            elif sport_id is None:
                raise ValueError("Either sport_id or sport parameter must be provided")
            
            endpoint_type = self._determine_endpoint_type(sport_id)
            
            # Build appropriate endpoint
            if endpoint_type == 'sportbex':
                endpoint = f"api/sportbex/competitions/{sport_id}"
            elif endpoint_type == 'sportbex_sports':
                endpoint = f"api/sportbex/sports/{sport_id}/competitions"
            else:  # other_sport
                endpoint = f"api/other-sport/competitions/{sport_id}"
            
            logger.debug(f"Getting competitions for sport {sport_id} using endpoint: {endpoint}")
            
            response = self._make_request('GET', endpoint)
            
            if response.success:
                logger.info(f"Successfully retrieved competitions for sport {sport_id}")
                # Standardize the response data
                if isinstance(response.data, dict) and 'data' in response.data:
                    response.data = standardize_competition_data(response.data, self.provider_name)
            else:
                logger.error(f"Failed to retrieve competitions for sport {sport_id}: {response.error_message}")
            
            return response
            
        except Exception as e:
            error_msg = f"Error getting competitions for sport {sport_id}: {str(e)}"
            logger.error(error_msg)
            return APIResponse(
                success=False,
                error_message=error_msg,
                provider=self.provider_name
            )
    
    def get_props(self, sport: Union[SportType, str], competition_id: Optional[str] = None, **kwargs) -> APIResponse:
        """
        Get props/competitions data for a specific sport.
        
        Args:
            sport: Sport type or identifier
            competition_id: Optional specific competition ID
            **kwargs: Additional parameters
            
        Returns:
            APIResponse with props/competitions data
        """
        try:
            sport_id = self._get_sport_id(sport)
            
            if competition_id:
                # Get specific competition details
                endpoint = f"api/other-sport/competitions/{sport_id}/{competition_id}"
            else:
                # Get all competitions for sport (same as get_competitions)
                return self.get_competitions(sport=sport)
            
            logger.debug(f"Getting props for sport {sport_id}, competition {competition_id}")
            
            response = self._make_request('GET', endpoint)
            
            if response.success:
                logger.info(f"Successfully retrieved props for sport {sport_id}")
            else:
                logger.error(f"Failed to retrieve props for sport {sport_id}: {response.error_message}")
            
            return response
            
        except Exception as e:
            error_msg = f"Error getting props for sport {sport}: {str(e)}"
            logger.error(error_msg)
            return APIResponse(
                success=False,
                error_message=error_msg,
                provider=self.provider_name
            )
    
    def get_odds(self, event_ids: Optional[List[str]] = None, market_types: Optional[List[str]] = None, **kwargs) -> APIResponse:
        """
        Get betting odds data.
        
        Args:
            event_ids: List of specific event IDs to get odds for
            market_types: List of market types to filter by
            **kwargs: Additional parameters (sport_type, competition_ids, etc.)
            
        Returns:
            APIResponse with odds data
        """
        try:
            # Prepare request data
            request_data = {}
            if event_ids:
                request_data['event_ids'] = event_ids
            if market_types:
                request_data['market_types'] = market_types
            
            # Add any additional kwargs to request data
            request_data.update(kwargs)
            
            # Determine endpoint based on request data or default
            if any(key in request_data for key in ['sport_type', 'competition_ids']):
                # Use sportbex endpoint for filtered requests
                endpoint = "api/sportbex/events/odds"
            else:
                # Use other-sport endpoint for general requests
                endpoint = "api/other-sport/event-odds"
            
            logger.debug(f"Getting odds using endpoint: {endpoint}")
            
            response = self._make_request('POST', endpoint, json_data=request_data)
            
            if response.success:
                logger.info("Successfully retrieved odds data")
                # Standardize the response data
                if isinstance(response.data, dict) and 'data' in response.data:
                    response.data = standardize_odds_data(response.data, self.provider_name)
            else:
                logger.error(f"Failed to retrieve odds data: {response.error_message}")
            
            return response
            
        except Exception as e:
            error_msg = f"Error getting odds: {str(e)}"
            logger.error(error_msg)
            return APIResponse(
                success=False,
                error_message=error_msg,
                provider=self.provider_name
            )
    
    def get_matchups(self, sport: Union[SportType, str], competition_id: str, **kwargs) -> APIResponse:
        """
        Get matchups/events for a specific competition.
        
        Args:
            sport: Sport type or identifier
            competition_id: Competition ID to get matchups for
            **kwargs: Additional parameters
            
        Returns:
            APIResponse with matchups data
        """
        try:
            sport_id = self._get_sport_id(sport)
            endpoint_type = self._determine_endpoint_type(sport_id)
            
            # Build appropriate endpoint based on sport type
            if endpoint_type == 'sportbex':  # Soccer
                endpoint = f"api/sportbex/event/{sport_id}/{competition_id}"
            elif endpoint_type == 'sportbex_sports':  # Baseball, Hockey, Esports, College Football
                endpoint = f"api/sportbex/competitions/{competition_id}/events"
            elif sport_id == '2':  # Tennis uses special match-ups endpoint
                endpoint = "api/other-sport/match-ups"
                request_data = {"competition_id": competition_id}
                return self._make_request('POST', endpoint, json_data=request_data)
            else:  # Basketball, American Football
                endpoint = f"api/other-sport/event/{sport_id}/{competition_id}"
            
            logger.debug(f"Getting matchups for competition {competition_id} using endpoint: {endpoint}")
            
            response = self._make_request('GET', endpoint)
            
            if response.success:
                logger.info(f"Successfully retrieved matchups for competition {competition_id}")
            else:
                logger.error(f"Failed to retrieve matchups for competition {competition_id}: {response.error_message}")
            
            return response
            
        except Exception as e:
            error_msg = f"Error getting matchups for competition {competition_id}: {str(e)}"
            logger.error(error_msg)
            return APIResponse(
                success=False,
                error_message=error_msg,
                provider=self.provider_name
            )
    
    def health_check(self) -> APIResponse:
        """
        Perform a health check on the Sportbex API.
        
        Returns:
            APIResponse indicating API health status
        """
        try:
            # Use tennis competitions as a lightweight health check
            response = self.get_competitions(sport=SportType.TENNIS)
            
            if response.success:
                logger.info("Sportbex API health check passed")
                return APIResponse(
                    success=True, 
                    data={
                        "status": "healthy", 
                        "provider": self.provider_name,
                        "response_time": response.response_time,
                        "base_url": self.base_url
                    },
                    provider=self.provider_name
                )
            else:
                logger.warning(f"Sportbex API health check failed: {response.error_message}")
                return APIResponse(
                    success=False, 
                    error_message=f"Health check failed: {response.error_message}",
                    provider=self.provider_name
                )
                
        except Exception as e:
            error_msg = f"Health check error: {str(e)}"
            logger.error(error_msg)
            return APIResponse(
                success=False,
                error_message=error_msg,
                provider=self.provider_name
            )


# Factory function for easy provider instantiation
def create_sportbex_provider(api_key: Optional[str] = None, **kwargs) -> SportbexProvider:
    """
    Factory function to create a SportbexProvider instance.
    
    Args:
        api_key: Optional API key (will use environment variable if not provided)
        **kwargs: Additional configuration parameters
        
    Returns:
        Configured SportbexProvider instance
        
    Raises:
        ValueError: If API key is not available
    """
    return SportbexProvider(api_key=api_key, **kwargs)
