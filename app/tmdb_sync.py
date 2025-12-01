from datetime import datetime, timezone
import time
from typing import Optional
import concurrent.futures

import requests

from app.config.settings import settings
from app.db import tmdb_metadata_collection, sync_meta_collection, tmdb_failures_collection
from app.utils.logger import get_logger

from app.embeddings import embed_item_and_store
from dateutil import parser as _dateutil_parser

logger = get_logger(__name__)

# TMDB provides bulk data options via their "/movie/now_playing", "/movie/popular", etc.
# but better for keeping fresh is using the /movie/changes and /tv/changes endpoints which return ids changed since a timestamp.

CHUNK_SIZE = 100  # number of IDs to fetch details for in one batch


def _get_last_sync_timestamp(media_type: str) -> Optional[int]:
    doc = sync_meta_collection.find_one({"_id": f"tmdb_{media_type}_last_sync"})
    if not doc:
        return None
    return doc.get("last_sync")


def _set_last_sync_timestamp(media_type: str, ts: int):
    sync_meta_collection.update_one(
        {"_id": f"tmdb_{media_type}_last_sync"},
        {"$set": {"last_sync": ts}},
        upsert=True
    )


def _fetch_changes(media_type: str, start_time: Optional[int] = None, page: int = 1):
    url = f"{settings.TMDB_API_URL}/{media_type}/changes"
    params = {"api_key": settings.TMDB_API_KEY, "page": page}
    if start_time:
        # TMDB changes endpoint expects a date string (YYYY-MM-DD). Convert epoch to UTC date.
        try:
            date_str = datetime.fromtimestamp(start_time, tz=timezone.utc).strftime("%Y-%m-%d")
            params["start_date"] = date_str
        except Exception:
            params["start_date"] = start_time
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    return resp.json()


# failure threshold before marking as permanently failed
FAILURE_THRESHOLD = 3


def _is_failed(tmdb_id: int, media_type: str) -> bool:
    """Return True if the tmdb_id+media_type is marked as permanently failed."""
    doc = tmdb_failures_collection.find_one({"id": tmdb_id, "media_type": media_type, "permanent": True})
    return bool(doc)


def _mark_failure(tmdb_id: int, media_type: str, reason: str = ""):
    """Increment failure count for an id. If threshold reached, mark as permanent.

    Stored document shape:
    {
        "id": 12345,
        "media_type": "movie",
        "count": 1,
        "last_failed_at": 1690000000,
        "last_reason": "status_404",
        "permanent": False
    }
    """
    try:
        now = int(time.time())
        tmdb_failures_collection.update_one(
            {"id": tmdb_id, "media_type": media_type},
            {
                "$set": {"last_failed_at": now, "last_reason": reason},
                "$inc": {"count": 1}
            },
            upsert=True
        )
        doc = tmdb_failures_collection.find_one({"id": tmdb_id, "media_type": media_type})
        if doc and doc.get("count", 0) >= FAILURE_THRESHOLD:
            tmdb_failures_collection.update_one({"id": tmdb_id, "media_type": media_type}, {"$set": {"permanent": True}})
            logger.info("Marking TMDB %s %s as permanently failed after %s attempts", media_type, tmdb_id, doc.get("count"))
    except Exception as e:
        logger.warning("Failed to mark failure for %s %s: %s", media_type, tmdb_id, repr(e), exc_info=True)


def _fetch_details(media_type: str, tmdb_ids: list[int]):
    results = []
    for tmdb_id in tmdb_ids:
        try:
            if _is_failed(tmdb_id, media_type):
                logger.info("Skipping TMDB %s %s because it's marked as permanently failed", media_type, tmdb_id)
                continue
            url = f"{settings.TMDB_API_URL}/{media_type}/{tmdb_id}?api_key={settings.TMDB_API_KEY}"
            r = requests.get(url)
            if r.status_code == 200:
                results.append(r.json())
                # on success, clear any previous failure records for this id
                try:
                    tmdb_failures_collection.delete_many({"id": tmdb_id, "media_type": media_type})
                except Exception as e:
                    logger.warning("Failed to clear failure records for %s %s: %s", media_type, tmdb_id, repr(e), exc_info=True)
            else:
                logger.warning("Failed to fetch %s details for %s: %s", media_type, tmdb_id, r.status_code)
                _mark_failure(tmdb_id, media_type, reason=f"status_{r.status_code}")
        except Exception as e:
            logger.warning("Error fetching details for %s: %s", tmdb_id, repr(e), exc_info=True)
            _mark_failure(tmdb_id, media_type, reason=repr(e))
            continue
    return results


def sync_tmdb_changes(media_type: str = "movie", window_seconds: int = 60 * 60 * 24 * 7, embed_updated: bool = True):
    """Sync TMDB changes using the /changes endpoint.

    - media_type: "movie" or "tv"
    - window_seconds: how far back to check changes if there's no stored last sync (default 7 days)
    - embed_updated: if True and embeddings are available, compute embeddings for updated items (background threads)
    """
    assert media_type in ("movie", "tv"), f"media_type {media_type} not supported for syncing TMDB changes."
    start_ts = _get_last_sync_timestamp(media_type)
    if not start_ts:
        # default to now - window_seconds
        start_ts = int(time.time()) - window_seconds

    page = 1
    total_processed = 0
    max_ts = start_ts
    to_embed = []

    while True:
        data = _fetch_changes(media_type, start_time=start_ts, page=page)
        if not data or not data.get("results"):
            break
        ids = [item.get("id") for item in data["results"] if item.get("id")]
        # fetch details in chunks
        for i in range(0, len(ids), CHUNK_SIZE):
            chunk = ids[i:i + CHUNK_SIZE]
            details = _fetch_details(media_type, chunk)
            for det in details:
                if not det:
                    continue
                # upsert into collection by tmdb id
                tmdb_metadata_collection.update_one(
                    {"id": det.get("id"), "media_type": media_type},
                    {"$set": dict(det, media_type=media_type)},
                    upsert=True
                )
                total_processed += 1
                updated_at = det.get("updated_at")
                # TMDB returns updated_at as ISO string; best-effort to find a newer timestamp
                if updated_at:
                    try:
                        ts = int(_dateutil_parser.isoparse(updated_at).timestamp())
                        if ts > max_ts:
                            max_ts = ts
                    except Exception:
                        pass
                if embed_updated:
                    to_embed.append(det)
        # paging
        if data.get("page") >= data.get("total_pages"):
            break
        page += 1

    if total_processed:
        _set_last_sync_timestamp(media_type, max_ts)
        logger.info("Processed %s %s items from TMDB changes; set last_sync=%s", total_processed, media_type, max_ts)
    else:
        logger.info("No %s changes processed.", media_type)

    # process embeddings for updated items in a small thread pool to avoid blocking the sync too long
    if embed_updated and to_embed:
        try:
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                for item in to_embed:
                    try:
                        futures.append(executor.submit(embed_item_and_store, item))
                    except Exception as e:
                        logger.warning("Failed to submit embedding job for %s: %s", item.get("id"), repr(e), exc_info=True)
                # wait a short while for tasks to complete; don't block forever
                for f in futures:
                    try:
                        f.result(timeout=30)
                    except Exception as e:
                        logger.warning("Embedding job error or timeout: %s", repr(e), exc_info=True)
            logger.info("Submitted and waited for %s embedding tasks", len(futures))
        except Exception as e:
            logger.warning("Embedding step failed: %s", repr(e), exc_info=True)


def full_tmdb_popular_sync(media_type: str = "movie", pages: int = 5):
    """Fetch popular movies/TV shows pages and store metadata. Also update `is_popular` flag.

    - pages: how many pages of popular results to fetch (20 results per page typically)
    """
    assert media_type in ("movie", "tv")
    total = 0
    seen_ids = set()
    for page in range(1, pages + 1):
        url = f"{settings.TMDB_API_URL}/{media_type}/popular"
        params = {"api_key": settings.TMDB_API_KEY, "page": page}
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
        for item in data.get("results", []):
            item["media_type"] = media_type
            tmdb_metadata_collection.update_one({"id": item.get("id"), "media_type": media_type}, {"$set": item}, upsert=True)
            # mark as popular and set timestamp
            tmdb_metadata_collection.update_one({"id": item.get("id"), "media_type": media_type}, {"$set": {"is_popular": True, "popular_updated_at": datetime.utcnow().isoformat()}}, upsert=True)
            seen_ids.add(item.get("id"))
            total += 1

    # unset is_popular for items not seen in this run (but that were previously marked popular)
    if seen_ids:
        cursor = tmdb_metadata_collection.find({"media_type": media_type, "is_popular": True}, {"_id": 0, "id": 1})
        for doc in cursor:
            if doc.get("id") not in seen_ids:
                tmdb_metadata_collection.update_one({"id": doc.get("id"), "media_type": media_type}, {"$set": {"is_popular": False}})

    logger.info("Inserted/updated %s popular %s items and updated popularity flags", total, media_type)
