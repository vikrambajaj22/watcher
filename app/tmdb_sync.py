import time
from datetime import datetime, timezone
from typing import Optional
import gzip
import json
from datetime import timedelta
import os

import requests
from dateutil import parser as _dateutil_parser
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests.exceptions import RequestException

from app.config.settings import settings
from app.db import (
    sync_meta_collection,
    tmdb_failures_collection,
    tmdb_metadata_collection,
)
from app.embeddings import EMBED_MODEL_NAME, _process_batch
from app.utils.logger import get_logger

logger = get_logger(__name__)

# TMDB provides bulk data options via their "/movie/now_playing", "/movie/popular", etc.
# but better for keeping fresh is using the /movie/changes and /tv/changes endpoints which return ids changed since a timestamp.

CHUNK_SIZE = 100  # number of IDs to fetch details for in one batch

# logging cadence controls
DISCOVER_LOG_EVERY_PAGES = 50  # log every N discover pages
EXPORT_LOG_EVERY_IDS = 10000  # log every N ids processed from export
CHUNK_LOG_EVERY = 10  # log after every N chunks processed

# embedding batch controls
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "256"))
# when the queued embeddings reach this threshold, submit batches incrementally
EMBED_BATCH_SUBMIT_THRESHOLD = int(os.getenv("EMBED_BATCH_SUBMIT_THRESHOLD", "512"))
# if desired, batch submissions can be parallelized by workers
EMBED_BATCH_MAX_WORKERS = int(os.getenv("EMBED_BATCH_MAX_WORKERS", "2"))


def _get_last_sync_timestamp(media_type: str) -> Optional[int]:
    doc = sync_meta_collection.find_one({"_id": f"tmdb_{media_type}_last_sync"})
    if not doc:
        return None
    return doc.get("last_sync")


def _set_last_sync_timestamp(media_type: str, ts: int):
    sync_meta_collection.update_one(
        {"_id": f"tmdb_{media_type}_last_sync"},
        {"$set": {"last_sync": ts}},
        upsert=True,
    )


def _fetch_changes(media_type: str, start_time: Optional[int] = None, page: int = 1):
    url = f"{settings.TMDB_API_URL}/{media_type}/changes"
    params = {"api_key": settings.TMDB_API_KEY, "page": page}
    if start_time:
        # TMDB changes endpoint expects a date string (YYYY-MM-DD). Convert epoch to UTC date.
        try:
            date_str = datetime.fromtimestamp(start_time, tz=timezone.utc).strftime(
                "%Y-%m-%d"
            )
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
    doc = tmdb_failures_collection.find_one(
        {"id": tmdb_id, "media_type": media_type, "permanent": True}
    )
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
                "$inc": {"count": 1},
            },
            upsert=True,
        )
        doc = tmdb_failures_collection.find_one(
            {"id": tmdb_id, "media_type": media_type}
        )
        if doc and doc.get("count", 0) >= FAILURE_THRESHOLD:
            tmdb_failures_collection.update_one(
                {"id": tmdb_id, "media_type": media_type}, {"$set": {"permanent": True}}
            )
            logger.info(
                "Marking TMDB %s %s as permanently failed after %s attempts",
                media_type,
                tmdb_id,
                doc.get("count"),
            )
    except Exception as e:
        logger.warning(
            "Failed to mark failure for %s %s: %s",
            media_type,
            tmdb_id,
            repr(e),
            exc_info=True,
        )


def _fetch_details(media_type: str, tmdb_ids: list[int], skip_failed_filter: bool = False):
    results = []
    for tmdb_id in tmdb_ids:
        try:
            if not skip_failed_filter and _is_failed(tmdb_id, media_type):
                logger.info(
                    "Skipping TMDB %s %s because it's marked as permanently failed",
                    media_type,
                    tmdb_id,
                )
                continue

            url = f"{settings.TMDB_API_URL}/{media_type}/{tmdb_id}"
            params = {
                "api_key": settings.TMDB_API_KEY,
                "append_to_response": "credits,keywords",
            }
            r = requests.get(url, params=params, timeout=10)

            if r.status_code == 200:
                data = r.json()
                results.append(data)
                logger.debug("Fetched details for %s %s (payload_keys=%s)", media_type, tmdb_id, list(data.keys()) if isinstance(data, dict) else None)
                # on success, clear any previous failure records for this id
                try:
                    tmdb_failures_collection.delete_many(
                        {"id": tmdb_id, "media_type": media_type}
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to clear failure records for %s %s: %s",
                        media_type,
                        tmdb_id,
                        repr(e),
                        exc_info=True,
                    )
            else:
                logger.warning(
                    "Failed to fetch %s details for %s: %s",
                    media_type,
                    tmdb_id,
                    r.status_code,
                )
                _mark_failure(tmdb_id, media_type, reason=f"status_{r.status_code}")
        except Exception as e:
            logger.warning(
                "Error fetching details for %s: %s", tmdb_id, repr(e), exc_info=True
            )
            _mark_failure(tmdb_id, media_type, reason=repr(e))
            continue
    return results


def _fetch_all_ids_by_discover(media_type: str) -> list[int]:
    """Fetch all TMDB IDs for a media_type using the /discover endpoint.

    This will page through the discover results until all pages are retrieved.
    This can be large and may hit rate limits; the function logs progress.
    """
    assert media_type in ("movie", "tv")
    url = f"{settings.TMDB_API_URL}/discover/{media_type}"
    params = {"api_key": settings.TMDB_API_KEY, "page": 1}
    ids: list[int] = []
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.error("Failed to fetch discover %s page 1: %s", media_type, repr(e), exc_info=True)
        return ids

    total_pages = data.get("total_pages", 1) or 1
    for item in data.get("results", []):
        if item.get("id"):
            ids.append(item.get("id"))

    # TMDB limits page parameter to [1..500]. Cap pages to avoid Invalid page errors.
    try:
        total_pages = int(total_pages)
    except Exception:
        total_pages = 1

    max_pages_allowed = 500
    if total_pages > max_pages_allowed:
        logger.warning(
            "TMDB discover reports %s pages for %s but API max is %s; capping to %s",
            total_pages,
            media_type,
            max_pages_allowed,
            max_pages_allowed,
        )
    capped_pages = min(total_pages, max_pages_allowed)

    logger.info("Discover: media_type=%s total_pages_reported=%s capped_to=%s", media_type, total_pages, capped_pages)
    pages_processed = 1
    for page in range(2, capped_pages + 1):
        params["page"] = int(page)
        try:
            r = requests.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            for item in data.get("results", []):
                if item.get("id"):
                    ids.append(item.get("id"))
        except Exception as e:
            logger.warning(
                "Failed to fetch discover %s page %s: %s", media_type, page, repr(e), exc_info=True
            )
            continue
        pages_processed += 1
        # periodic progress log to show long-running discover paging
        if pages_processed % DISCOVER_LOG_EVERY_PAGES == 0:
            logger.info(
                "Discover progress: media_type=%s processed_pages=%s/%s collected_ids=%s",
                media_type,
                pages_processed,
                capped_pages,
                len(ids),
            )

    ids = list(dict.fromkeys(ids))
    logger.info("Discover finished: collected %s %s ids", len(ids), media_type)
    return ids


def _sync_from_export(media_type: str, days_back_limit: int = 7, embed_updated: bool = True) -> Optional[dict]:
    """Stream TMDB daily export files and process IDs in chunks.

    Tries today's export and falls back up to `days_back_limit` days. Returns
    a summary dict with the same shape as other sync paths. On failure (no
    export found) returns None.
    """
    assert media_type in ("movie", "tv")
    prefix = "movie" if media_type == "movie" else "tv_series"
    today = datetime.now(timezone.utc).date()

    total_processed = 0
    max_ts = int(time.time())
    to_embed: list[dict] = []

    # create a requests Session with retry/backoff for transient network errors
    def _session_with_retries(retries: int = 3, backoff: float = 0.5) -> requests.Session:
        s = requests.Session()
        retry = Retry(
            total=retries,
            read=retries,
            connect=retries,
            backoff_factor=backoff,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET", "POST"),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        return s

    session = _session_with_retries(retries=3, backoff=0.8)

    for days_back in range(0, days_back_limit):
        d = today - timedelta(days=days_back)
        date_str = d.strftime("%m_%d_%Y")
        url = f"http://files.tmdb.org/p/exports/{prefix}_ids_{date_str}.json.gz"
        logger.info("Attempting TMDB export URL: %s", url)
        EXPORT_DOWNLOAD_RETRIES = 3
        resp = None
        last_exc = None
        for attempt in range(1, EXPORT_DOWNLOAD_RETRIES + 1):
            try:
                resp = session.get(url, stream=True, timeout=30)
                # if non-200, let the outer logic treat it as not available
                break
            except RequestException as e:
                last_exc = e
                logger.warning(
                    "Export request attempt %s/%s failed for %s: %s",
                    attempt,
                    EXPORT_DOWNLOAD_RETRIES,
                    url,
                    repr(e),
                    exc_info=True,
                )
                time.sleep(attempt * 1.0)
                resp = None
                continue
        if resp is None:
            logger.warning("Failed to request export %s after %s attempts: %s", url, EXPORT_DOWNLOAD_RETRIES, repr(last_exc))
            continue

        # stream and process the gzipped newline-delimited JSON file in CHUNK_SIZE batches
        STREAM_RETRIES = 2
        stream_success = False
        for stream_attempt in range(1, STREAM_RETRIES + 1):
            try:
                logger.info("Downloaded TMDB export %s; processing... (attempt %s/%s)", url, stream_attempt, STREAM_RETRIES)
                resp.raw.decode_content = True
                gz = gzip.GzipFile(fileobj=resp.raw)
                # iterate over gzip bytes lines and decode each line
                chunk_ids: list[int] = []
                # counters for incremental embedding submissions during export
                embed_submitted = 0
                embed_succeeded = 0
                embed_failed = 0
                embed_timed_out = 0

                for raw_line in gz:
                    try:
                        line = raw_line.decode("utf-8").strip()
                    except Exception:
                        continue
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        tid = obj.get("id")
                        if tid:
                            chunk_ids.append(int(tid))
                    except Exception:
                        continue

                    if len(chunk_ids) >= CHUNK_SIZE:
                        details = _fetch_details(media_type, chunk_ids, skip_failed_filter=True)
                        logger.debug("Export fetch details: media_type=%s requested_ids=%s returned_details=%s", media_type, len(chunk_ids), len(details))
                        if not details:
                            logger.warning(
                                "No details returned for export chunk (media_type=%s requested_ids=%s). These IDs may be marked failed or TMDB returned no data.",
                                media_type,
                                len(chunk_ids),
                            )
                        processed, max_seen, queued = _process_details_batch(details, media_type, embed_updated)
                        total_processed += processed
                        # collect queued embedding items into the shared to_embed list (cumulative)
                        if queued:
                            to_embed.extend(queued)
                        if max_seen and max_seen > max_ts:
                            max_ts = max_seen
                        logger.debug(
                            "Export chunk: media_type=%s chunk_size=%s processed=%s cumulative_queued_embeddings=%s",
                            media_type,
                            len(chunk_ids),
                            processed,
                            len(to_embed),
                        )
                        chunk_ids = []
                        # log progress for large exports
                        if total_processed % EXPORT_LOG_EVERY_IDS == 0:
                            logger.info(
                                "Export progress: media_type=%s processed_ids=%s cumulative_queued_embeddings=%s",
                                media_type,
                                total_processed,
                                len(to_embed),
                            )
                        if processed and (total_processed // CHUNK_SIZE) % CHUNK_LOG_EVERY == 0:
                            logger.info(
                                "Export chunk processed: media_type=%s total_processed=%s cumulative_queued_embeddings=%s",
                                media_type,
                                total_processed,
                                len(to_embed),
                            )

                        # if queued embeddings exceed the threshold, process them incrementally now
                        if embed_updated and len(to_embed) >= EMBED_BATCH_SUBMIT_THRESHOLD:
                            logger.info("Embedding threshold reached (cumulative_queued=%s). Submitting batches...", len(to_embed))
                            emb_summary = _submit_embedding_tasks(to_embed)
                            embed_submitted += emb_summary.get("submitted", 0)
                            embed_succeeded += emb_summary.get("succeeded", 0)
                            embed_failed += emb_summary.get("failed", 0)
                            embed_timed_out += emb_summary.get("timed_out", 0)
                            # drain queued items after submission
                            to_embed = []

                if chunk_ids:
                    details = _fetch_details(media_type, chunk_ids, skip_failed_filter=True)
                    processed, max_seen, queued = _process_details_batch(details, media_type, embed_updated)
                    total_processed += processed
                    if queued:
                        to_embed.extend(queued)
                    if max_seen and max_seen > max_ts:
                        max_ts = max_seen
                    logger.debug(
                        "Export final chunk: media_type=%s chunk_size=%s processed=%s cumulative_queued_embeddings=%s",
                        media_type,
                        len(chunk_ids),
                        processed,
                        len(to_embed),
                    )

                # if any queued remain and exceed threshold, process them now
                if embed_updated and len(to_embed) >= EMBED_BATCH_SUBMIT_THRESHOLD:
                    logger.info("Submitting remaining queued embeddings (cumulative_queued=%s)", len(to_embed))
                    emb_summary = _submit_embedding_tasks(to_embed)
                    embed_submitted += emb_summary.get("submitted", 0)
                    embed_succeeded += emb_summary.get("succeeded", 0)
                    embed_failed += emb_summary.get("failed", 0)
                    embed_timed_out += emb_summary.get("timed_out", 0)
                    to_embed = []

                logger.info("Processed export %s: processed=%s cumulative_queued_embeddings=%s", url, total_processed, len(to_embed))

                # final submission for any remaining queued embeddings
                summary = {"total_processed": total_processed, "embed_queued": len(to_embed), "embed_submitted": embed_submitted, "embed_succeeded": embed_succeeded, "embed_failed": embed_failed, "embed_timed_out": embed_timed_out}
                if embed_updated and to_embed:
                    # process remaining queued items
                    emb_summary = _submit_embedding_tasks(to_embed)
                    summary["embed_submitted"] += emb_summary.get("submitted", 0)
                    summary["embed_succeeded"] += emb_summary.get("succeeded", 0)
                    summary["embed_failed"] += emb_summary.get("failed", 0)
                    summary["embed_timed_out"] += emb_summary.get("timed_out", 0)

                stream_success = True
                return summary

            except Exception as e:
                logger.warning("Stream attempt %s/%s failed for %s: %s", stream_attempt, STREAM_RETRIES, url, repr(e), exc_info=True)
                # if there are more stream attempts, try to re-request the URL (session has retry policy)
                if stream_attempt < STREAM_RETRIES:
                    try:
                        time.sleep(stream_attempt * 1.0)
                        resp = session.get(url, stream=True, timeout=30)
                    except RequestException as re:
                        logger.warning("Re-request failed during stream retry for %s: %s", url, repr(re), exc_info=True)
                        resp = None
                        break
                else:
                    logger.warning("Failed to stream/process export %s after %s attempts: %s", url, STREAM_RETRIES, repr(e))
        if not stream_success:
            # try next date
            continue

    logger.info("No TMDB export found for %s in the past %s days", media_type, days_back_limit)
    return None


def _process_details_batch(details: list[dict], media_type: str, embed_updated: bool) -> tuple[int, int, list]:
    """Process a list of detail objects: upsert into Mongo, collect items to embed, and
    return (processed_count, max_ts_seen, queued_items) where max_ts_seen is an int timestamp or 0
    and queued_items is a list of dicts to be embedded (each includes media_type).
    This avoids mutating caller state and makes queuing explicit.
    """
    processed = 0
    max_ts_seen = 0
    queued_items: list[dict] = []
    for det in details:
        if not det:
            continue
        # ensure the detail payload contains a valid id before processing
        tid = det.get("id") if isinstance(det, dict) else None
        if not tid:
            logger.warning("Skipping detail without id (media_type=%s) - malformed TMDB response: %s", media_type, det)
            continue

        # fetch existing doc (if any) BEFORE overwriting, so we can decide whether an embedding is needed
        existing_doc = None
        try:
            existing_doc = tmdb_metadata_collection.find_one(
                {"id": tid, "media_type": media_type}, {"_id": 0, "embedding_meta": 1, "embedding": 1}
            )
        except Exception:
            existing_doc = None

        try:
            tmdb_metadata_collection.update_one(
                {"id": tid, "media_type": media_type},
                {"$set": dict(det, media_type=media_type)},
                upsert=True,
            )
            processed += 1
        except Exception as e:
            logger.warning(
                "Failed to upsert %s %s: %s",
                media_type,
                det.get("id"),
                repr(e),
                exc_info=True,
            )
            continue

        updated_at = det.get("updated_at")
        if updated_at:
            try:
                ts = int(_dateutil_parser.isoparse(updated_at).timestamp())
                if ts > max_ts_seen:
                    max_ts_seen = ts
            except Exception:
                pass

        if embed_updated:
            try:
                # determine if embedding is needed:
                needs_embedding = False
                # if there is an existing doc with embedding_meta, and model matches, skip
                try:
                    if existing_doc:
                        emb = existing_doc.get("embedding")
                        emb_meta = existing_doc.get("embedding_meta") or {}
                        emb_model = emb_meta.get("embedding_model")
                        if not emb or not emb_meta:
                            needs_embedding = True
                        elif emb_model != EMBED_MODEL_NAME:
                            needs_embedding = True
                        else:
                            needs_embedding = False
                    else:
                        # no existing doc -> if incoming payload already has embedding (unlikely) skip, otherwise embed
                        if not det.get("embedding"):
                            needs_embedding = True
                except Exception:
                    needs_embedding = True

                if needs_embedding:
                    queued_item = dict(det, media_type=media_type)
                    queued_items.append(queued_item)
                    # info-level on first queued item (visible in normal logs)
                    if len(queued_items) == 1:
                        logger.info("Queued first embedding item in batch for media_type=%s (id=%s)", media_type, queued_item.get("id"))
                    else:
                        logger.debug("Queued embedding for %s (media_type=%s). batch_queued=%s", queued_item.get("id"), media_type, len(queued_items))
                else:
                    logger.debug("Skipping embedding for %s (media_type=%s) - embedding present and up-to-date", tid, media_type)
            except Exception as e:
                logger.warning("Failed to prepare embedding for %s: %s", det.get("id"), repr(e), exc_info=True)

    return processed, max_ts_seen, queued_items


def _submit_embedding_tasks(to_embed: list) -> dict:
    """Process embeddings in batches using the embeddings module's _process_batch.
    Returns a summary dict with keys: submitted, succeeded, failed, timed_out.
    This function validates results by checking the DB for written embeddings after each batch.
    """
    summary = {"submitted": 0, "succeeded": 0, "failed": 0, "timed_out": 0}
    if not to_embed:
        return summary

    total = len(to_embed)
    logger.info("Starting embedding processing for %s items (batch_size=%s)", total, EMBED_BATCH_SIZE)

    try:
        for idx in range(0, total, EMBED_BATCH_SIZE):
            batch = to_embed[idx : idx + EMBED_BATCH_SIZE]
            batch_ids = [int(item.get("id")) for item in batch if item.get("id")]
            submitted = len(batch)
            try:
                _process_batch(batch)
                # verify how many docs now have embeddings written
                succeeded = 0
                try:
                    # build a set of (id, media_type) pairs from the batch
                    pairs = []
                    for item in batch:
                        try:
                            _id = int(item.get("id"))
                        except Exception:
                            continue
                        m = str(item.get("media_type") or "").lower()
                        pairs.append((_id, m))
                    # for efficiency, group by id and fetch candidates, then verify per-pair
                    ids_in_batch = list({p[0] for p in pairs})
                    if ids_in_batch:
                        cursor = tmdb_metadata_collection.find({"id": {"$in": ids_in_batch}, "embedding": {"$exists": True}}, {"_id": 0, "id": 1, "media_type": 1})
                        found = set()
                        for d in cursor:
                            try:
                                did = int(d.get("id"))
                            except Exception:
                                continue
                            mtype = str(d.get("media_type") or "").lower()
                            found.add((did, mtype))
                        # count how many pairs in 'pairs' are present in found
                        for p in pairs:
                            if p in found:
                                succeeded += 1
                except Exception:
                    # fallback: best effort, try id-only count
                    try:
                        written = tmdb_metadata_collection.count_documents({"id": {"$in": batch_ids}, "embedding": {"$exists": True}})
                        succeeded = int(written)
                    except Exception:
                        succeeded = 0

                failed = submitted - succeeded
                summary["submitted"] += submitted
                summary["succeeded"] += succeeded
                summary["failed"] += failed

                logger.info(
                    "Embedding batch processed: batch_index=%s batch_size=%s cumulative_remaining=%s submitted=%s succeeded=%s failed=%s",
                    idx // EMBED_BATCH_SIZE,
                    submitted,
                    max(0, total - (idx + submitted)),
                    submitted,
                    succeeded,
                    failed,
                )
            except Exception as e:
                # mark all in this batch as failed
                summary["submitted"] += submitted
                summary["failed"] += submitted
                logger.warning(
                    "Embedding batch failed for ids=%s: %s",
                    batch_ids[:10],
                    repr(e),
                    exc_info=True,
                )
    except Exception as e:
        logger.warning("Embedding processing aborted: %s", repr(e), exc_info=True)

    return summary


def sync_tmdb_changes(
    media_type: str = "movie",
    window_seconds: int = 60 * 60 * 24 * 7,
    embed_updated: bool = True,
):
    """Sync TMDB changes using the /changes endpoint.

    - media_type: "movie" or "tv"
    - window_seconds: how far back to check changes when running an incremental /changes sync
      (only used when a stored last-sync timestamp exists). Default: 7 days.
    - embed_updated: if True and embeddings are available, compute embeddings for updated items (background threads)

    If there's no last sync timestamp stored, performs an initial full sync via TMDB exports or the /discover endpoint.
    """
    assert media_type in ("movie", "tv"), (
        f"media_type {media_type} not supported for syncing TMDB changes."
    )
    start_ts = _get_last_sync_timestamp(media_type)

    if not start_ts:
        logger.info(
            "No last sync timestamp found; attempting initial full sync via TMDB export for %s",
            media_type,
        )
        export_summary = None
        try:
            export_summary = _sync_from_export(media_type, days_back_limit=7, embed_updated=embed_updated)
        except Exception:
            logger.exception("Export-based initial sync failed; falling back to discover full sync")

        if export_summary is not None:
            return export_summary

        logger.info("Falling back to discover-based initial sync for %s", media_type)
        ids = _fetch_all_ids_by_discover(media_type)
        total_processed = 0
        max_ts = int(time.time())
        to_embed: list[dict] = []

        # fetch details in chunks and upsert using helper
        chunks_processed = 0
        for i in range(0, len(ids), CHUNK_SIZE):
            chunk = ids[i : i + CHUNK_SIZE]
            details = _fetch_details(media_type, chunk)
            processed, max_seen, queued = _process_details_batch(details, media_type, embed_updated)
            total_processed += processed
            if queued:
                to_embed.extend(queued)
            chunks_processed += 1
            if max_seen and max_seen > max_ts:
                max_ts = max_seen
            # periodic progress log for initial discover ingest
            if chunks_processed % CHUNK_LOG_EVERY == 0:
                logger.info(
                    "Initial discover progress: media_type=%s chunks_processed=%s total_ids=%s processed_items=%s",
                    media_type,
                    chunks_processed,
                    len(ids),
                    total_processed,
                )

        # set last sync to the max observed timestamp (or now)
        _set_last_sync_timestamp(media_type, max_ts)
        logger.info(
            "Initial full discover sync processed %s %s items; set last_sync=%s",
            total_processed,
            media_type,
            max_ts,
        )

        # process embeddings if requested
        summary = {
            "total_processed": total_processed,
            "embed_queued": len(to_embed),
            "embed_submitted": 0,
            "embed_succeeded": 0,
            "embed_failed": 0,
            "embed_timed_out": 0,
        }
        if embed_updated and to_embed:
            emb_summary = _submit_embedding_tasks(to_embed)
            summary["embed_submitted"] = emb_summary.get("submitted", 0)
            summary["embed_succeeded"] = emb_summary.get("succeeded", 0)
            summary["embed_failed"] = emb_summary.get("failed", 0)
            summary["embed_timed_out"] = emb_summary.get("timed_out", 0)
        return summary

    # at this point we have a stored last-sync timestamp -> run incremental /changes.
    # use window_seconds as a safety buffer: start the /changes query from (last_sync - window_seconds)
    try:
        changes_start_ts = int(start_ts) - int(window_seconds)
    except Exception:
        changes_start_ts = int(start_ts)
    page = 1
    total_processed = 0
    max_ts = start_ts
    to_embed = []

    while True:
        # use the changes_start_ts (last_sync - window_seconds) to ensure we pick up any lagging updates
        data = _fetch_changes(media_type, start_time=changes_start_ts, page=page)
        if not data or not data.get("results"):
            break
        ids = [item.get("id") for item in data["results"] if item.get("id")]
        # log page-level progress for /changes
        try:
            logger.info(
                "Changes page: media_type=%s page=%s/%s ids_on_page=%s",
                media_type,
                data.get("page"),
                data.get("total_pages"),
                len(ids),
            )
        except Exception:
            logger.info("Changes page: media_type=%s page=%s ids_on_page=%s", media_type, page, len(ids))
        # fetch details in chunks
        chunks_processed = 0
        for i in range(0, len(ids), CHUNK_SIZE):
            chunk = ids[i : i + CHUNK_SIZE]
            details = _fetch_details(media_type, chunk)
            # filter out unchanged items first
            details_to_process = []
            for det in details:
                if not det:
                    continue
                try:
                    # prefer exact match on (id, media_type) to avoid cross-media collisions
                    existing = tmdb_metadata_collection.find_one(
                        {"id": det.get("id"), "media_type": media_type}, {"_id": 0, "updated_at": 1}
                    )
                except Exception:
                    existing = None

                skip_due_to_unchanged = False
                det_updated_at = det.get("updated_at")
                if existing and existing.get("updated_at") and det_updated_at:
                    try:
                        existing_ts = int(
                            _dateutil_parser.isoparse(existing.get("updated_at")).timestamp()
                        )
                        det_ts = int(_dateutil_parser.isoparse(det_updated_at).timestamp())
                        if existing_ts == det_ts:
                            skip_due_to_unchanged = True
                    except Exception:
                        skip_due_to_unchanged = False

                if skip_due_to_unchanged:
                    total_processed += 1
                    # still try to update max_ts if present
                    try:
                        if det_updated_at:
                            ts = int(_dateutil_parser.isoparse(det_updated_at).timestamp())
                            if ts > max_ts:
                                max_ts = ts
                    except Exception:
                        pass
                    continue

                details_to_process.append(det)

            # process the filtered list using helper
            processed, max_seen, queued = _process_details_batch(details_to_process, media_type, embed_updated)
            total_processed += processed
            if queued:
                to_embed.extend(queued)
            chunks_processed += 1
            # periodic progress log for changes processing
            if chunks_processed % CHUNK_LOG_EVERY == 0:
                logger.info(
                    "Changes progress: media_type=%s changes_page=%s chunks_processed=%s processed_items=%s",
                    media_type,
                    data.get("page"),
                    chunks_processed,
                    total_processed,
                )
        # paging
        if data.get("page") >= data.get("total_pages"):
            break
        page += 1

    if total_processed:
        _set_last_sync_timestamp(media_type, max_ts)
        logger.info(
            "Processed %s %s items from TMDB changes; set last_sync=%s",
            total_processed,
            media_type,
            max_ts,
        )
    else:
        logger.info("No %s changes processed.", media_type)

    # process embeddings for updated items in a small thread pool to avoid blocking the sync too long
    summary = {
        "total_processed": total_processed,
        "embed_queued": len(to_embed),
        "embed_submitted": 0,
        "embed_succeeded": 0,
        "embed_failed": 0,
        "embed_timed_out": 0,
    }

    if embed_updated and to_embed:
        emb_summary = _submit_embedding_tasks(to_embed)
        summary["embed_submitted"] = emb_summary.get("submitted", 0)
        summary["embed_succeeded"] = emb_summary.get("succeeded", 0)
        summary["embed_failed"] = emb_summary.get("failed", 0)
        summary["embed_timed_out"] = emb_summary.get("timed_out", 0)
        logger.info(
            "Submitted and processed %s embedding tasks (submitted=%s, succeeded=%s, failed=%s, timed_out=%s)",
            len(to_embed),
            summary["embed_submitted"],
            summary["embed_succeeded"],
            summary["embed_failed"],
            summary["embed_timed_out"],
        )
    return summary


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
            tmdb_metadata_collection.update_one(
                {"id": item.get("id"), "media_type": media_type},
                {"$set": item},
                upsert=True,
            )
            # mark as popular and set timestamp
            tmdb_metadata_collection.update_one(
                {"id": item.get("id"), "media_type": media_type},
                {
                    "$set": {
                        "is_popular": True,
                        "popular_updated_at": datetime.now(timezone.utc).isoformat(),
                    }
                },
                upsert=True,
            )
            seen_ids.add(item.get("id"))
            total += 1

    # unset is_popular for items not seen in this run (but that were previously marked popular)
    if seen_ids:
        cursor = tmdb_metadata_collection.find(
            {"media_type": media_type, "is_popular": True}, {"_id": 0, "id": 1}
        )
        for doc in cursor:
            if doc.get("id") not in seen_ids:
                tmdb_metadata_collection.update_one(
                    {"id": doc.get("id"), "media_type": media_type},
                    {"$set": {"is_popular": False}},
                )

    logger.info(
        "Inserted/updated %s popular %s items and updated popularity flags",
        total,
        media_type,
    )
