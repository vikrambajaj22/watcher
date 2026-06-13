from datetime import datetime, timezone

import requests

from app.config.settings import settings
from app.dao.watchlist import (
    bulk_upsert_watchlist,
    get_watchlist,
    remove_watchlist_item,
    remove_watchlist_items_by_ids,
    upsert_watchlist_item,
)
from app.tmdb_client import get_metadata
from app.utils.logger import get_logger

logger = get_logger(__name__)

_TRAKT_LISTS_BASE = "https://api.trakt.tv/users/me/lists"


def _list_items_url(slug: str) -> str:
    return f"{_TRAKT_LISTS_BASE}/{slug}/items"


def _enrich(tmdb_id: int, media_type: str) -> dict:
    try:
        meta = get_metadata(tmdb_id, media_type=media_type)
        genres = [g["name"] for g in (meta.get("genres") or []) if g.get("name")]
        return {
            "title": meta.get("title") or meta.get("name"),
            "poster_path": meta.get("poster_path"),
            "overview": meta.get("overview"),
            "release_date": meta.get("release_date") or meta.get("first_air_date"),
            "genres": genres,
        }
    except Exception as e:
        logger.warning("Could not enrich TMDB %s/%s: %s", media_type, tmdb_id, repr(e))
        return {}


def pull_watchlist() -> dict:
    """Sync Trakt custom lists → local MongoDB cache. Returns {added, removed} counts."""
    now = datetime.now(timezone.utc).isoformat()
    added = 0
    removed = 0

    list_configs = [
        ("movie", settings.TRAKT_MOVIE_LIST_ID, "movie"),
        ("tv", settings.TRAKT_TV_LIST_ID, "show"),
    ]

    for media_type, slug, trakt_key in list_configs:
        try:
            entries = []
            page = 1
            while True:
                resp = requests.get(
                    _list_items_url(slug),
                    headers=settings.trakt_headers,
                    params={"limit": 1000, "page": page},
                )
                if resp.status_code != 200:
                    logger.warning("Trakt list %s fetch failed: %s", slug, resp.status_code)
                    break
                page_items = resp.json() or []
                entries.extend(page_items)
                page_count = int(resp.headers.get("X-Pagination-Page-Count", 1))
                if page >= page_count:
                    break
                page += 1
            logger.info("Fetched %d items from Trakt list %s", len(entries), slug)
            existing_ids = {item["tmdb_id"] for item in get_watchlist(media_type)}
            incoming_ids: set[int] = set()
            new_docs: list[dict] = []

            for entry in entries:
                data = entry.get(trakt_key) or {}
                tmdb_id = (data.get("ids") or {}).get("tmdb")
                if not tmdb_id:
                    continue
                tmdb_id = int(tmdb_id)
                incoming_ids.add(tmdb_id)
                enriched = _enrich(tmdb_id, media_type)
                doc = {"tmdb_id": tmdb_id, "media_type": media_type, "synced_at": now, **enriched}
                new_docs.append(doc)
                if tmdb_id not in existing_ids:
                    added += 1

            bulk_upsert_watchlist(new_docs)
            stale = existing_ids - incoming_ids
            removed += remove_watchlist_items_by_ids(stale, media_type)

        except Exception as e:
            logger.error("pull_watchlist error for %s: %s", slug, repr(e), exc_info=True)

    return {"added": added, "removed": removed}


def add_to_watchlist(
    tmdb_id: int,
    media_type: str,
    title: str | None = None,
    poster_path: str | None = None,
    overview: str | None = None,
    release_date: str | None = None,
) -> dict:
    """Add item to Trakt custom list and local cache. Returns the cached document."""
    slug = settings.TRAKT_MOVIE_LIST_ID if media_type == "movie" else settings.TRAKT_TV_LIST_ID
    trakt_key = "movies" if media_type == "movie" else "shows"

    resp = requests.post(
        _list_items_url(slug),
        json={trakt_key: [{"ids": {"tmdb": tmdb_id}}]},
        headers=settings.trakt_headers,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Trakt add failed: {resp.status_code} {resp.text[:200]}")

    if not title:
        enriched = _enrich(tmdb_id, media_type)
    else:
        enriched = {
            "title": title,
            "poster_path": poster_path,
            "overview": overview,
            "release_date": release_date,
        }

    doc = {
        "tmdb_id": tmdb_id,
        "media_type": media_type,
        "synced_at": datetime.now(timezone.utc).isoformat(),
        **enriched,
    }
    upsert_watchlist_item(doc)
    return doc


def remove_from_watchlist(tmdb_id: int, media_type: str) -> None:
    """Remove item from Trakt custom list and local cache."""
    slug = settings.TRAKT_MOVIE_LIST_ID if media_type == "movie" else settings.TRAKT_TV_LIST_ID
    trakt_key = "movies" if media_type == "movie" else "shows"

    resp = requests.post(
        f"{_list_items_url(slug)}/remove",
        json={trakt_key: [{"ids": {"tmdb": tmdb_id}}]},
        headers=settings.trakt_headers,
    )
    if resp.status_code not in (200, 201):
        raise RuntimeError(f"Trakt remove failed: {resp.status_code} {resp.text[:200]}")

    remove_watchlist_item(tmdb_id, media_type)
