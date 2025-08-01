"""OpenAI API client for making requests to the OpenAI service."""
import openai
import time

from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def get_openai_client():
    """Configure and return the OpenAI Python client instance."""
    return openai.OpenAI(api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_API_BASE_URL)


def get_openai_chat_completion(model, messages, **kwargs):
    """Get chat completion from OpenAI API using the official openai Python module. Accepts extra payload params via kwargs."""
    client = get_openai_client()
    max_retries = 5
    backoff = 1
    for _ in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                **kwargs
            )
            return response
        except Exception as e:
            if hasattr(e, "status_code") and e.status_code == 429:
                logger.warning(
                    f"Rate limited by OpenAI API (429). Retrying in {backoff} seconds...")
                time.sleep(backoff)
                backoff *= 2
                continue
            logger.error(f"OpenAI API request failed: {e}")
            raise
    logger.error("Max retries exceeded for OpenAI API request.")
    raise Exception("Max retries exceeded for OpenAI API request.")
