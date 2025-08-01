"""Recommendation processing module."""
import json

from app.dao.history import get_watch_history
from app.schemas.recommendations.movies import MovieRecommendationsResponse
from app.utils.logger import get_logger
from app.utils.openai_client import get_openai_chat_completion
from app.utils.prompt_registry import PromptRegistry

logger = get_logger(__name__)


class MovieRecommender:
    """Processor for generating movie recommendations based on watch history."""

    def __init__(self):
        self.prompt_registry = PromptRegistry("app/prompts/recommend")

    def load_watch_history(self, type: str = "movie"):
        """Load watch history from the database."""
        try:
            watch_history = get_watch_history(type=type)
            logger.info(f"Loaded {len(watch_history)} watch history items.")
            return watch_history
        except Exception as e:
            logger.error(f"Failed to load watch history: {e}")
            return []

    def get_recommendation_prompt(self, watch_history):
        """Generate a prompt for movie recommendations based on watch history."""
        prompt_template = self.prompt_registry.load_prompt_template(
            "movie_recommender", "1")
        return prompt_template.render(watch_history=watch_history)

    def generate_recommendations(self):
        """Generate movie recommendations based on watch history."""
        watch_history = self.load_watch_history("movie")
        prompt = self.get_recommendation_prompt(watch_history)
        logger.info(f"Generated recommendation prompt: {prompt[:1000]}...")
        messages = [
            {"role": "user", "content": prompt}
        ]
        try:
            response = get_openai_chat_completion(
                "gpt-4.1-nano",
                messages=messages,
                response_format={"type": "json_object"}
            )
            completion_text = response.choices[0].message.content
            recommendations = json.loads(
                completion_text).get("recommendations", [])
        except Exception as e:
            logger.error(
                f"Failed to parse structured OpenAI response. Exception: {repr(e)}")
            recommendations = []
            raise
        return MovieRecommendationsResponse(recommendations=recommendations)
