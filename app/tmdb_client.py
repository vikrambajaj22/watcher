from typing import List, Dict, Any, Optional

import requests
import urllib.parse

from app.config.settings import settings


def get_metadata(tmdb_id, media_type="movie"):
    if media_type not in ("movie", "tv"):
        raise ValueError("media_type must be 'movie' or 'tv'")
    url = f"{settings.TMDB_API_URL}/{media_type}/{tmdb_id}?api_key={settings.TMDB_API_KEY}&append_to_response=credits,keywords"
    resp = requests.get(url)
    resp.raise_for_status()
    return resp.json()


def search_multi(query: str, limit: int = 6) -> List[Dict[str, Any]]:
    """Search TMDB across movies and TV shows, returning top results."""
    if not query or not query.strip():
        return []
    q = urllib.parse.quote_plus(query.strip())
    url = f"{settings.TMDB_API_URL}/search/multi?api_key={settings.TMDB_API_KEY}&query={q}&page=1"
    resp = requests.get(url, timeout=15)
    if resp.status_code != 200:
        return []
    data = resp.json()
    hits = [
        r for r in (data.get("results") or [])
        if r.get("media_type") in ("movie", "tv") and r.get("id")
    ][:limit]
    out = []
    for r in hits:
        title_str = r.get("title") or r.get("name") or ""
        year_raw = r.get("release_date") or r.get("first_air_date") or ""
        out.append({
            "id": int(r["id"]),
            "title": title_str,
            "media_type": r["media_type"],
            "year": year_raw[:4] if len(year_raw) >= 4 else None,
            "poster_path": r.get("poster_path"),
        })
    return out


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
