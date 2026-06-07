from app.db import watch_history_collection
from app.utils.logger import get_logger
import time
from cachetools import TTLCache
import copy
from pymongo import ReplaceOne

logger = get_logger(__name__)

_HISTORY_CACHE = TTLCache(maxsize=128, ttl=300)


def store_watch_history(data):
    if isinstance(data, list):
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
                len(data),
                len(deduplicated),
            )

        ops = [
            ReplaceOne(
                {"id": item.get("id"), "media_type": item.get("media_type")},
                item,
                upsert=True,
            )
            for item in deduplicated
        ]
        if ops:
            watch_history_collection.bulk_write(ops, ordered=False)
        incoming_keys = {(item.get("id"), item.get("media_type")) for item in deduplicated}
        existing = watch_history_collection.find({}, {"_id": 0, "id": 1, "media_type": 1})
        stale_filters = []
        for doc in existing:
            if (doc.get("id"), doc.get("media_type")) not in incoming_keys:
                stale_filters.append({"id": doc["id"], "media_type": doc["media_type"]})
        if stale_filters:
            for sf in stale_filters:
                watch_history_collection.delete_one(sf)
        try:
            _HISTORY_CACHE.clear()
        except Exception:
            pass
        logger.info("Watch history stored successfully: %d unique items.", len(deduplicated))
    else:
        logger.error("Invalid data format for storing watch history. Expected a list.")
        raise ValueError("Data must be a list of watch history items.")


def get_watch_history(media_type=None, include_posters: bool = True):
    """Return watch history from MongoDB. Posters are stored directly on each document during sync."""
    cache_key = f"{media_type or 'all'}|{include_posters}"
    try:
        cached = _HISTORY_CACHE.get(cache_key)
        if cached is not None:
            return copy.deepcopy(cached)
    except Exception:
        pass

    start = time.time()
    query = {"media_type": media_type} if media_type else {}
    history = list(watch_history_collection.find(query, {"_id": 0}))

    if history:
        seen_keys = {}
        deduplicated = []
        for item in history:
            key = (item.get("id"), item.get("media_type"))
            if key not in seen_keys:
                seen_keys[key] = item
                deduplicated.append(item)
        if len(history) != len(deduplicated):
            logger.warning(
                "Removed %d duplicate entries from watch history",
                len(history) - len(deduplicated),
            )
        history = deduplicated

    logger.info(
        "Watch history retrieved: items=%s time=%.3fs",
        len(history),
        time.time() - start,
    )

    try:
        _HISTORY_CACHE[cache_key] = copy.deepcopy(history)
    except Exception:
        pass

    return history


def clear_history_cache() -> bool:
    try:
        _HISTORY_CACHE.clear()
        logger.info("Watch history cache cleared")
        return True
    except Exception as e:
        logger.warning("Failed to clear history cache: %s", repr(e))
        return False
