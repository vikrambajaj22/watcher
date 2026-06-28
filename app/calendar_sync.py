"""Upcoming TV episodes for shows the user is watching, from the Trakt calendar."""
from datetime import date

import requests
from cachetools import TTLCache

from app.config.settings import settings
from app.dao.history import get_watch_history
from app.trakt_sync import _ensure_valid_token
from app.utils.logger import get_logger

logger = get_logger(__name__)

_CACHE: TTLCache = TTLCache(maxsize=8, ttl=3600)


def get_upcoming_episodes(days: int = 14) -> list[dict]:
    """Episodes airing in the next ``days`` days for the user's shows, poster-enriched."""
    cached = _CACHE.get(days)
    if cached is not None:
        return cached

    _ensure_valid_token()
    start = date.today().isoformat()
    url = f"{settings.TRAKT_CALENDAR_SHOWS_API_URL}/{start}/{days}"
    resp = requests.get(url, headers=settings.trakt_headers)
    if resp.status_code != 200:
        logger.warning("Failed to fetch Trakt calendar: %s", resp.status_code)
        return []

    posters = {
        s.get("id"): s.get("poster_path")
        for s in get_watch_history(media_type="tv")
        if s.get("id")
    }

    upcoming = []
    for item in resp.json() or []:
        show = item.get("show") or {}
        episode = item.get("episode") or {}
        tmdb_id = (show.get("ids") or {}).get("tmdb")
        if not tmdb_id:
            continue
        upcoming.append(
            {
                "tmdb_id": tmdb_id,
                "show_title": show.get("title"),
                "poster_path": posters.get(tmdb_id),
                "season": episode.get("season"),
                "episode": episode.get("number"),
                "episode_title": episode.get("title"),
                "first_aired": item.get("first_aired"),
            }
        )

    _CACHE[days] = upcoming
    logger.info("Fetched %s upcoming episodes from Trakt calendar", len(upcoming))
    return upcoming


def clear_calendar_cache() -> None:
    _CACHE.clear()
