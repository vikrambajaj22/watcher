from app.db import watch_history_collection, tmdb_metadata_collection
from app.utils.logger import get_logger
import time
from cachetools import TTLCache
import copy

logger = get_logger(__name__)

# Simple in-memory TTL cache for watch history: key = (media_type or 'all') + '|' + str(include_posters)
_HISTORY_CACHE = TTLCache(maxsize=128, ttl=300)


def store_watch_history(data):
    if isinstance(data, list):
        watch_history_collection.delete_many({})
        watch_history_collection.insert_many(data)
        # invalidate cache when history changes
        try:
            _HISTORY_CACHE.clear()
        except Exception:
            pass
        logger.info("Watch history stored successfully.")
    else:
        logger.error("Invalid data format for storing watch history. Expected a list.")
        raise ValueError("Data must be a list of watch history items.")


def get_watch_history(media_type=None, include_posters: bool = True):
    """Return watch history. If include_posters=True, batch-enrich poster_path from tmdb metadata.

    Args:
        media_type: optional filter ('movie' or 'tv')
        include_posters: when False, skip the poster enrichment (faster)
    """
    cache_key = f"{media_type or 'all'}|{include_posters}"
    try:
        cached = _HISTORY_CACHE.get(cache_key)
        if cached is not None:
            # return a deep copy to prevent accidental mutation of cached data
            return copy.deepcopy(cached)
    except Exception:
        pass

    start = time.time()
    query = {"media_type": media_type} if media_type else {}
    # fetch history in one query
    history = list(watch_history_collection.find(query, {"_id": 0}))
    db_fetch_time = time.time() - start

    if include_posters and history:
        t0 = time.time()
        ids_set = set()
        for item in history:
            tmdb_id = item.get("id") or (item.get("ids") or {}).get("tmdb")
            if tmdb_id:
                try:
                    ids_set.add(int(tmdb_id))
                except Exception:
                    pass
        ids_list = list(ids_set)

        if ids_list:
            try:
                # single query to fetch poster paths and media_type for all ids
                cursor = tmdb_metadata_collection.find({"id": {"$in": ids_list}}, {"_id": 0, "id": 1, "media_type": 1, "poster_path": 1})
                # build two maps: exact (id,media_type) -> poster, and fallback id->poster (first seen)
                poster_map_exact = {}
                poster_map_fallback = {}
                for d in cursor:
                    try:
                        _id = int(d.get("id"))
                    except Exception:
                        continue
                    mtype = (d.get("media_type") or "").lower()
                    ppath = d.get("poster_path")
                    if ppath:
                        poster_map_exact[(_id, mtype)] = ppath
                        # keep a fallback if not set yet
                        if _id not in poster_map_fallback:
                            poster_map_fallback[_id] = ppath
            except Exception:
                poster_map_exact = {}
                poster_map_fallback = {}

            # attach poster_path to history items where available, prefer exact media_type match
            for item in history:
                tmdb_id = item.get("id") or (item.get("ids") or {}).get("tmdb")
                if not tmdb_id:
                    continue
                try:
                    key_id = int(tmdb_id)
                except Exception:
                    continue
                media = (item.get("media_type") or "").lower()
                p = poster_map_exact.get((key_id, media))
                if not p:
                    p = poster_map_fallback.get(key_id)
                if p:
                    item["poster_path"] = p
        enrich_time = time.time() - t0
    else:
        enrich_time = 0.0

    total_time = time.time() - start
    logger.info("Watch history retrieved successfully. items=%s db_time=%.3fs enrich_time=%.3fs total=%.3fs",
                len(history), db_fetch_time, enrich_time, total_time)

    # cache the result (deepcopy to keep cached copy immutable)
    try:
        _HISTORY_CACHE[cache_key] = copy.deepcopy(history)
    except Exception:
        pass

    return history


def clear_history_cache() -> bool:
    """Clear the in-memory watch history cache. Returns True on success."""
    try:
        _HISTORY_CACHE.clear()
        logger.info("Watch history cache cleared via clear_history_cache()")
        return True
    except Exception as e:
        logger.warning("Failed to clear history cache: %s", repr(e))
        return False
