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
        # deduplicate based on (id, media_type) before storing
        seen_keys = {}
        deduplicated = []
        for item in data:
            key = (item.get("id"), item.get("media_type"))
            if key not in seen_keys:
                seen_keys[key] = item
                deduplicated.append(item)

        if len(data) != len(deduplicated):
            logger.warning(
                "Deduplicating watch history before storage: %d items reduced to %d unique items",
                len(data), len(deduplicated)
            )

        watch_history_collection.delete_many({})
        watch_history_collection.insert_many(deduplicated)
        # invalidate cache when history changes
        try:
            _HISTORY_CACHE.clear()
        except Exception:
            pass
        logger.info("Watch history stored successfully: %d unique items.", len(deduplicated))
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

    # deduplicate based on (id, media_type) - database may have duplicates
    if history:
        seen_keys = {}
        deduplicated = []
        for item in history:
            key = (item.get("id"), item.get("media_type"))
            if key not in seen_keys:
                seen_keys[key] = item
                deduplicated.append(item)
        if len(history) != len(deduplicated):
            logger.warning("Removed %d duplicate entries from watch history (db has duplicates)", len(history) - len(deduplicated))
        history = deduplicated

    if include_posters and history:
        t0 = time.time()
        # use (id, media_type) tuples since IDs are NOT unique across media types
        id_media_pairs = set()
        for item in history:
            tmdb_id = item.get("id") or (item.get("ids") or {}).get("tmdb")
            media = item.get("media_type")
            if tmdb_id and media:
                try:
                    id_media_pairs.add((int(tmdb_id), media.lower()))
                except Exception:
                    pass

        # Extract unique IDs for the query (still need to fetch all with same ID but different media_type)
        unique_ids = list(set(pair[0] for pair in id_media_pairs))

        if unique_ids:
            try:
                # single query to fetch poster paths and media_type for all ids
                cursor = tmdb_metadata_collection.find({"id": {"$in": unique_ids}}, {"_id": 0, "id": 1, "media_type": 1, "poster_path": 1})
                # build exact map keyed by (id, media_type) only
                poster_map_exact = {}
                for d in cursor:
                    try:
                        _id = int(d.get("id"))
                    except Exception:
                        continue
                    mtype = (d.get("media_type") or "").lower()
                    ppath = d.get("poster_path")
                    if ppath:
                        poster_map_exact[(_id, mtype)] = ppath
            except Exception:
                poster_map_exact = {}

            # attach poster_path to history items where available, require exact media_type match
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
