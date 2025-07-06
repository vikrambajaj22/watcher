import requests

from app.config.settings import settings


def get_metadata(tmdb_id, media_type="movie"):
    if media_type not in ("movie", "tv"):
        raise ValueError("media_type must be 'movie' or 'tv'")
    url = f"{settings.TMDB_API_URL}/{media_type}/{tmdb_id}?api_key={settings.TMDB_API_KEY}"
    return requests.get(url).json()
