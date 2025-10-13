from page_utils import display_sport_page
from sport_agents import EsportsAgent

class ApexAgent(EsportsAgent):
    """Apex Legends specific agent"""
    
    def __init__(self):
        super().__init__("apex")
    
    def _analyze_esports_specific_factors(self, prop):
        """Analyze Apex Legends specific factors"""
        return {
            'legend': 'unknown',
            'playstyle': 'balanced',
            'score': 5.0,
            'reasoning': 'Apex Legends analysis - kills and placement focused'
        }

display_sport_page('apex', 'Apex Legends', ApexAgent)