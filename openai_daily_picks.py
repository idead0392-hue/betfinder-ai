import os
import time
import logging
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenAIDailyPicksAgent:
    """
    OpenAI-powered daily picks agent for sports betting analysis.
    Uses GPT-5 for all clients.
    """
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        self.client = OpenAI(api_key=self.api_key)
        self.model = "gpt-5"  # Enable GPT-5 for all clients
        self.max_retries = 3
        self.base_delay = 1

    def get_ai_daily_picks(self, max_picks: int = 5) -> List[Dict]:
        """
        Get AI-generated daily picks using GPT-5.
        Returns a list of pick dicts with keys: sport, team, odds, confidence.
        Implements retry logic and comprehensive error handling.
        """
        prompt = (
            f"You are a sports betting AI. Generate {max_picks} picks. "
            "For each pick, return sport, team, odds, and confidence (0-1). "
            "Respond in valid JSON list format."
        )
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    temperature=0.7,
                    timeout=30
                )
                content = response.choices[0].message.content.strip()
                picks = self._parse_response(content)
                return picks
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                if attempt == self.max_retries - 1:
                    raise
                delay = self.base_delay * (2 ** attempt)
                logger.warning(f"Retrying in {delay}s...")
                time.sleep(delay)
        return []

    def _parse_response(self, content: str) -> List[Dict]:
        """
        Parse the OpenAI response and validate the structure.
        """
        import json
        try:
            picks = json.loads(content)
            if not isinstance(picks, list):
                raise ValueError("Response is not a list")
            for pick in picks:
                for key in ["sport", "team", "odds", "confidence"]:
                    if key not in pick:
                        raise ValueError(f"Missing key: {key}")
            return picks
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            return []

# Global instance
openai_agent = OpenAIDailyPicksAgent()
