import requests
import logging

from app.config.settings import settings
from app.dao.history import store_watch_history
from app.utils.logger import get_logger

logger = get_logger(__name__)


def sync_trakt_history():
    all_history = []
    page = 1
    per_page = 100  # trakt's max limit per page is 100
    while True:
        params = {"page": page, "limit": per_page}
        response = requests.get(settings.TRAKT_HISTORY_API_URL, headers=settings.trakt_headers, params=params)
        logger.info(f"Syncing Trakt history: {response.status_code} (page {page})")
        if response.status_code != 200:
            break
        page_data = response.json()
        logger.info(f"Fetched {len(page_data)} items on page {page}")
        if not page_data:
            break
        all_history.extend(page_data)
        if len(page_data) < per_page:
            break  # last page
        page += 1
    logger.info(f"Total history items fetched: {len(all_history)}")
    if all_history:
        store_watch_history(all_history)
