"""Shared KNN result processing helpers.

Provides `process_knn_results` to turn FAISS/vector-store results into
annotated candidate documents by fetching metadata from Mongo and applying
media-type filtering, watched-id exclusion, and exact (id,media_type)
resolution.
"""
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from app.db import tmdb_metadata_collection
from app.utils.logger import get_logger

logger = get_logger(__name__)


def _normalize_title(doc: Dict[str, Any]) -> str:
    if not doc:
        return ""
    return (
        doc.get("title")
        or doc.get("name")
        or doc.get("original_title")
        or doc.get("original_name")
        or ""
    )


def process_knn_results(
    vs_res: Iterable[Tuple],
    k: int,
    exclude_id: Optional[int] = None,
    requested_media_type: Optional[str] = None,
    watched_ids: Optional[Set[str]] = None,
    projection: Optional[Dict] = None,
) -> List[Dict[str, Any]]:
    """Turn vector-store results into candidate metadata dicts.

    Args:
        vs_res: iterable of results from FAISS/vector store. Each item is expected to be
            (id, media_type, score).
        k: max number of results to return (after filtering)
        exclude_id: optional TMDB id to exclude from results
        requested_media_type: optional 'movie'|'tv'|'all' media filter
        watched_ids: optional set of string ids to exclude (e.g., user's watched ids)
        projection: optional Mongo projection to pass to find()

    Returns:
        list of candidate dicts with keys: id, title, media_type, score, poster_path, overview
    """
    if watched_ids is None:
        watched_ids = set()

    # default projection
    if projection is None:
        projection = {"_id": 0, "id": 1, "title": 1, "name": 1, "original_title": 1, "original_name": 1, "media_type": 1, "poster_path": 1, "overview": 1}

    # collect unique (id, media_type) entries while also gathering a set of ids for DB fetch
    ids_set = set()
    ids = []
    seen_entries = set()
    for entry in vs_res:
        if not (isinstance(entry, (list, tuple)) and len(entry) == 3):
            continue
        try:
            tid = int(entry[0])
        except Exception:
            continue
        media_type_entry = str(entry[1] or "").lower()
        key = (tid, media_type_entry)
        # deduplicate by (id, media_type); keep insertion order in ids list for DB fetch
        if key not in seen_entries:
            seen_entries.add(key)
            if tid not in ids_set:
                ids_set.add(tid)
                ids.append(tid)
    if not ids:
        return []

    docs = list(tmdb_metadata_collection.find({"id": {"$in": ids}}, projection))
    # build maps for exact (id, media_type) and fallback by id
    docs_by_id_media = {}
    docs_by_id = {}
    for d in docs:
        try:
            _id = int(d.get("id"))
        except Exception:
            continue
        mtype = str(d.get("media_type") or "").lower()
        docs_by_id_media[(_id, mtype)] = d
        if _id not in docs_by_id:
            docs_by_id[_id] = d

    results: List[Dict[str, Any]] = []
    requested_media_norm = str(requested_media_type).lower() if requested_media_type else None

    for entry in vs_res:
        # expect (id, media_type, score)
        if not (isinstance(entry, (list, tuple)) and len(entry) == 3):
            continue
        tid, media_type_entry, score = entry
        try:
            tid_int = int(tid)
        except Exception:
            continue

        # exclusions
        if exclude_id is not None and int(tid_int) == int(exclude_id):
            continue
        if str(tid_int) in watched_ids:
            continue

        # resolve doc: prefer exact (id, media_type) if present
        doc = None
        if media_type_entry:
            doc = docs_by_id_media.get((tid_int, str(media_type_entry).lower()))
        if not doc:
            doc = docs_by_id.get(tid_int)
        media = (doc.get("media_type") or "").lower() if doc else (str(media_type_entry).lower() if media_type_entry else "")

        # media filtering
        if requested_media_norm and requested_media_norm != "all" and media != requested_media_norm:
            continue

        candidate = {
            "id": int(tid_int),
            "title": _normalize_title(doc) if doc else "",
            "media_type": (doc.get("media_type") if doc else None),
            "score": float(score),
            "poster_path": doc.get("poster_path") if doc else None,
            "overview": doc.get("overview") if doc else None,
        }

        results.append(candidate)
        if len(results) >= k:
            break

    logger.info("process_knn_results: returned %s candidates (requested k=%s)", len(results), k)
    return results
