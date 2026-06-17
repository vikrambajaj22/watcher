"""Shared 'find similar titles' logic, used by both the /similar API route and the chat tools."""
from typing import Optional

from app.dao.history import get_watch_history
from app.schemas.api import SimilarResponse, SimilarResultItem
from app.tmdb_client import get_metadata, search_by_title
from app.tmdb_discover import fetch_cross_type_similar, fetch_similar_and_recommendations
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SimilarError(Exception):
    """Raised when the source title cannot be resolved or inputs are invalid."""


def compute_similar(
    *,
    media_type: str,
    tmdb_id: Optional[int] = None,
    title: Optional[str] = None,
    k: int = 20,
    cross_type: bool = False,
    filter_to_history: bool = False,
) -> SimilarResponse:
    """Resolve a source title (by id or name) and return similar titles via TMDB.

    filter_to_history=True restricts results to titles already in the user's watch history.
    cross_type=True discovers titles of the opposite media type. Provide exactly one of
    tmdb_id or title.
    """
    if media_type not in ("movie", "tv"):
        raise SimilarError("media_type must be 'movie' or 'tv'")
    if tmdb_id is None and not title:
        raise SimilarError("one of tmdb_id or title must be provided")

    source_title: Optional[str] = None
    md: dict = {}
    if tmdb_id is None:
        md = search_by_title(title, media_type=media_type)
        if not md or not md.get("id"):
            raise SimilarError(f"'{title}' not found on TMDB")
        tmdb_id = int(md["id"])
        source_title = md.get("title") or md.get("name") or title
    else:
        try:
            md = get_metadata(tmdb_id, media_type=media_type)
            source_title = md.get("title") or md.get("name")
        except Exception as e:
            logger.warning("similar: get_metadata failed for %s/%s: %s", media_type, tmdb_id, e)

    history_set: Optional[set] = None
    if filter_to_history:
        history = get_watch_history(include_posters=False)
        history_set = {
            (int(h["id"]), str(h.get("media_type", "")))
            for h in history
            if h.get("id") is not None
        }

    per_endpoint = max(10, (k + 1) // 2) if history_set is None else 40
    if cross_type:
        raw = fetch_cross_type_similar(tmdb_id, media_type, per_endpoint=per_endpoint)
    else:
        raw = fetch_similar_and_recommendations(tmdb_id, media_type, per_endpoint=per_endpoint)

    seen: set = set()
    results: list[SimilarResultItem] = []
    for item in raw:
        iid = int(item["id"])
        if iid in seen or iid == tmdb_id:
            continue
        if history_set is not None and (iid, str(item.get("media_type", ""))) not in history_set:
            continue
        seen.add(iid)
        results.append(SimilarResultItem(
            id=iid,
            title=item.get("title"),
            media_type=item.get("media_type"),
            poster_path=item.get("poster_path"),
            overview=item.get("overview"),
            release_date=item.get("release_date"),
        ))
        if len(results) >= k:
            break

    if history_set is not None and (tmdb_id, media_type) in history_set:
        results.insert(0, SimilarResultItem(
            id=tmdb_id,
            title=md.get("title") or md.get("name") or source_title,
            media_type=media_type,
            poster_path=md.get("poster_path"),
            overview=md.get("overview"),
            release_date=md.get("release_date") or md.get("first_air_date"),
        ))

    return SimilarResponse(source_title=source_title, results=results)
