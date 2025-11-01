"""Recommendation processing module."""
import json
from pyexpat import model

from app.dao.history import get_watch_history
from app.schemas.recommendations.movies import MovieRecommendationsResponse
from app.utils.logger import get_logger
from app.utils.openai_client import get_openai_chat_completion
from app.utils.oss_loader import load_oss_model
from app.utils.prompt_registry import PromptRegistry

logger = get_logger(__name__)


class MovieRecommender:
    """Processor for generating movie recommendations based on watch history."""

    def __init__(self):
        self.prompt_registry = PromptRegistry("app/prompts/recommend")

    def _load_oss_model(self):
        """Load OSS model."""
        self.oss_model, self.oss_tokenizer = load_oss_model()

    def format_watch_history(self, watch_history: list[dict], type: str = "movie"):
        """Format watch history."""
        formatted_watch_history = []
        for item in watch_history:
            if item.get("action") == "watch":
                formatted_watch_history.append(
                    {
                        k: item[k] for k in
                        ["title", "year", "watch_count", "watched_at", "earliest_watched_at", "latest_watched_at"]
                    }
                )
        return formatted_watch_history

    def load_watch_history(self, type: str = "movie"):
        """Load watch history from the database."""
        try:
            watch_history = get_watch_history(type=type)
            watch_history = self.format_watch_history(watch_history=watch_history, type=type)
            logger.info(f"Loaded {len(watch_history)} watch history items.")
            return watch_history
        except Exception as e:
            logger.error(f"Failed to load watch history: {e}")
            return []

    def get_recommendation_prompt(self, watch_history: list[dict], type: str = "movie", recommend_count: int = 5):
        """Generate a prompt for movie recommendations based on watch history."""
        prompt_template = self.prompt_registry.load_prompt_template(
            f"{type}_recommender", "1")
        return prompt_template.render(watch_history=watch_history, recommend_count=recommend_count)

    def generate_recommendations(self, type: str = "movie", recommend_count: int = 5):
        """Generate movie recommendations based on watch history."""
        watch_history = self.load_watch_history(type=type)
        prompt = self.get_recommendation_prompt(watch_history=watch_history, type=type, recommend_count=recommend_count)
        logger.info(f"Generated {type} recommendation prompt: {prompt[:2000]}...")
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

    def generate_recommendations_oss(self, type: str = "movie", recommend_count: int = 5):
        watch_history = self.load_watch_history("movie")
        self._load_oss_model()
        prompt = self.get_recommendation_prompt(watch_history=watch_history, type=type, recommend_count=recommend_count)
        logger.info(f"Generated {type} recommendation prompt: {prompt[:2000]}...")
        messages = [
            {"role": "user", "content": prompt},
        ]
        try:
            inputs = self.oss_tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.oss_model.device)

            outputs = self.oss_model.generate(**inputs, max_new_tokens=4096)
            recommendations = self.oss_tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:]).get("recommendations", [])
            return MovieRecommendationsResponse(recommendations=recommendations)
        except Exception as e:
            logger.error(
                f"Failed to parse OSS response. Exception: {repr(e)}")
            raise
