#!/usr/bin/env python3
"""
Sportbex API Provider Implementation

This module implements the SportbexProvider class, which extends BaseAPIProvider
to provide access to Sportbex betting data and odds.

Example Usage:
    from api_providers import SportbexProvider
    
    provider = SportbexProvider(api_key="your_api_key")
    competitions = provider.get_competitions(sport_id="7522")
    if competitions.success:
        print(f"Found {len(competitions.data)} competitions")
"""

import os
from typing import Optional, Dict, Any
from api_providers import BaseAPIProvider, APIResponse, RequestConfig


class SportbexProvider(BaseAPIProvider):
    """
    Sportbex API provider implementation
    
    This class provides access to Sportbex betting data including competitions,
    odds, props, and events for various sports.
    
    API Documentation: https://sportbex.com/api/docs
    """
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize Sportbex API provider
        
        Args:
            api_key: Sportbex API key (if None, will read from SPORTBEX_API_KEY env var)
            **kwargs: Additional arguments passed to BaseAPIProvider
        """
        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.getenv('SPORTBEX_API_KEY')
        
        # Initialize base provider
        super().__init__(
            api_key=api_key,
            base_url="https://trial-api.sportbex.com/api",
            provider_name="Sportbex",
            **kwargs
        )
        
        # Customize headers for Sportbex API
        self.headers.update({
            'sportbex-api-key': self.api_key
        })
    
    def get_competitions(self, sport_id: Optional[str] = None) -> APIResponse:
        """
        Get list of available competitions from Sportbex
        
        Args:
            sport_id: Sport identifier (e.g., "7522" for basketball)
            
        Returns:
            APIResponse containing competition data
        """
        if not sport_id:
            return APIResponse(
                success=False,
                error_message="sport_id is required for Sportbex competitions",
                provider=self.provider_name
            )
        
        # Determine the correct endpoint based on sport_id
        if sport_id in ["7522"]:  # Basketball and other sports using other-sport API
            endpoint = f"other-sport/competitions/{sport_id}"
        else:  # Soccer and traditional sports using sportbex API
            endpoint = f"sportbex/competitions/{sport_id}"
        
        return self._make_request("GET", endpoint)
    
    def get_props(self, sport_id: str = "7522", **kwargs) -> APIResponse:
        """
        Get props data from Sportbex
        
        Args:
            sport_id: Sport identifier (default: "7522" for basketball)
            **kwargs: Additional parameters
            
        Returns:
            APIResponse containing props data
        """
        # Props are typically available through competitions endpoint
        return self.get_competitions(sport_id=sport_id)
    
    def get_odds(self, event_ids: Optional[list] = None, **kwargs) -> APIResponse:
        """
        Get odds data from Sportbex
        
        Args:
            event_ids: List of event IDs to get odds for
            **kwargs: Additional parameters
            
        Returns:
            APIResponse containing odds data
        """
        # Sportbex odds endpoint expects POST with event data
        endpoint = "other-sport/event-odds"
        
        # Prepare request data
        request_data = kwargs.get('data', {})
        if event_ids:
            request_data['event_ids'] = event_ids
        
        return self._make_request("POST", endpoint, json_data=request_data)
    
    def get_events(self, competition_id: Optional[str] = None, sport_id: str = "7522") -> APIResponse:
        """
        Get events/matchups for a specific competition
        
        Args:
            competition_id: Competition identifier
            sport_id: Sport identifier
            
        Returns:
            APIResponse containing events data
        """
        if not competition_id:
            return APIResponse(
                success=False,
                error_message="competition_id is required for Sportbex events",
                provider=self.provider_name
            )
        
        # Use the appropriate endpoint based on sport
        if sport_id in ["7522"]:  # Basketball using other-sport API
            endpoint = f"other-sport/event/{sport_id}/{competition_id}"
        else:  # Soccer using sportbex API
            endpoint = f"sportbex/event/{sport_id}/{competition_id}"
        
        return self._make_request("GET", endpoint)
    
    def health_check(self) -> APIResponse:
        """
        Check Sportbex API health by making a simple request
        
        Returns:
            APIResponse indicating API health status
        """
        try:
            # Try to get basketball competitions as a health check
            response = self.get_competitions(sport_id="7522")
            
            if response.success:
                return APIResponse(
                    success=True,
                    data={
                        "status": "healthy",
                        "provider": self.provider_name,
                        "api_responsive": True,
                        "competitions_available": len(response.data.get('data', [])) if response.data else 0
                    },
                    provider=self.provider_name
                )
            else:
                return APIResponse(
                    success=False,
                    error_message=f"Health check failed: {response.error_message}",
                    provider=self.provider_name
                )
                
        except Exception as e:
            return APIResponse(
                success=False,
                error_message=f"Health check error: {str(e)}",
                provider=self.provider_name
            )


# Factory function for easy provider creation
def create_sportbex_provider(api_key: Optional[str] = None) -> SportbexProvider:
    """
    Factory function to create a Sportbex provider instance
    
    Args:
        api_key: Optional API key (will use environment variable if not provided)
        
    Returns:
        Configured SportbexProvider instance
        
    Raises:
        ValueError: If no API key is available
    """
    return SportbexProvider(api_key=api_key)


# Example usage and testing
if __name__ == "__main__":
    import logging
    
    # Configure logging for testing
    logging.basicConfig(level=logging.DEBUG)
    
    try:
        # Create provider instance
        provider = create_sportbex_provider()
        print(f"Created provider: {provider}")
        
        # Test health check
        health = provider.health_check()
        print(f"Health check: {health.success}")
        if health.data:
            print(f"Health data: {health.data}")
        
        # Test getting competitions
        competitions = provider.get_competitions(sport_id="7522")
        print(f"Competitions success: {competitions.success}")
        if competitions.success and competitions.data:
            data = competitions.data.get('data', [])
            print(f"Found {len(data)} basketball competitions")
            
            # Test getting events for first competition
            if data:
                first_comp_id = data[0].get('competition', {}).get('id')
                if first_comp_id:
                    events = provider.get_events(competition_id=first_comp_id)
                    print(f"Events success: {events.success}")
                    if events.success and events.data:
                        events_data = events.data.get('data', [])
                        print(f"Found {len(events_data)} events for competition {first_comp_id}")
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please set SPORTBEX_API_KEY environment variable")
    except Exception as e:
        print(f"Unexpected error: {e}")