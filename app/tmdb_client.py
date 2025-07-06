import requests

from app.config.settings import settings


def get_metadata(tmdb_id):
    url = f"{settings.TMDB_API_URL}/movie/{tmdb_id}?api_key={settings.TMDB_API_KEY}"
    return requests.get(url).json()
