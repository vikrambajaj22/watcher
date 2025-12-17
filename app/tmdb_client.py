import requests
import urllib.parse

from app.config.settings import settings


def get_metadata(tmdb_id, media_type="movie"):
    if media_type not in ("movie", "tv"):
        raise ValueError("media_type must be 'movie' or 'tv'")
    url = f"{settings.TMDB_API_URL}/{media_type}/{tmdb_id}?api_key={settings.TMDB_API_KEY}&append_to_response=credits,keywords"
    return requests.get(url).json()


def search_by_title(title: str, media_type: str = "movie") -> dict:
    """Search TMDB for a title (movie or tv) and return the first matching metadata dict, or None.

    This calls the /search/{media_type} endpoint and then fetches full metadata for the top hit
    (so callers get complete fields for embedding).
    """
    if media_type not in ("movie", "tv"):
        raise ValueError("media_type must be 'movie' or 'tv'")
    if not title:
        return None
    q = urllib.parse.quote_plus(title)
    search_url = f"{settings.TMDB_API_URL}/search/{media_type}?api_key={settings.TMDB_API_KEY}&query={q}&page=1"
    resp = requests.get(search_url)
    if resp.status_code != 200:
        return None
    data = resp.json()
    results = data.get("results") or []
    if not results:
        return None
    # pick the top result
    top = results[0]
    # fetch full metadata for the top result (append credits/keywords for embedding)
    tmdb_id = top.get("id")
    if not tmdb_id:
        return None
    return get_metadata(tmdb_id, media_type=media_type)
