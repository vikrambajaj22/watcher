"""OpenAI API client for making requests to the OpenAI service."""

import time

import openai

from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


def get_openai_client():
    """Configure and return the OpenAI Python client instance."""
    return openai.OpenAI(
        api_key=settings.OPENAI_API_KEY, base_url=settings.OPENAI_API_BASE_URL
    )


def get_openai_chat_completion(model, messages, **kwargs):
    """Get chat completion from OpenAI API using the official openai Python module. Accepts extra payload params via kwargs."""
    client = get_openai_client()
    max_retries = 5
    backoff = 1
    for _ in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
            return response
        except openai.RateLimitError:
            logger.warning(
                "Rate limited by OpenAI API (429). Retrying in %s seconds...", backoff
            )
            time.sleep(backoff)
            backoff *= 2
            continue
        except Exception as e:
            logger.error("OpenAI API request failed: %s", repr(e), exc_info=True)
            raise
    logger.error("Max retries exceeded for OpenAI API request.")
    raise Exception("Max retries exceeded for OpenAI API request.")
