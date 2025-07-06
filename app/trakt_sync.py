import requests

from app.config.settings import settings
from app.dao.history import store_watch_history


def sync_trakt_history():
    response = requests.get(settings.TRAKT_API_URL, headers=settings.trakt_headers)
    if response.status_code == 200:
        store_watch_history(response.json())
