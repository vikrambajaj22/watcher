"""TMDB discover / similar / recommendations helpers for LLM-driven recommend."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

from app.config.settings import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_GENRE_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
_GENRE_CACHE_TTL = 86400.0

_DISCOVER_ALLOWED = frozenset(
    {
        "with_genres",
        "without_genres",
        "with_keywords",
        "without_keywords",
        "with_original_language",
        "without_original_language",
        "primary_release_date.gte",
        "primary_release_date.lte",
        "first_air_date.gte",
        "first_air_date.lte",
        "with_runtime.gte",
        "with_runtime.lte",
        "vote_count.gte",
        "vote_count.lte",
        "vote_average.gte",
        "vote_average.lte",
        "sort_by",
        "certification_country",
        "certification",
        "watch_region",
    }
)


def _api_get(path: str, params: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    params = dict(params or {})
    params.setdefault("api_key", settings.TMDB_API_KEY)
    url = f"{settings.TMDB_API_URL.rstrip('/')}/{path.lstrip('/')}"
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code != 200:
            logger.warning("TMDB GET %s -> %s", path, resp.status_code)
            return None
        return resp.json()
    except Exception as e:
        logger.warning("TMDB GET %s failed: %s", path, repr(e))
        return None


def get_genre_list(media_type: str) -> List[Dict[str, Any]]:
    """Return [{id, name}, ...] for movie or tv."""
    if media_type not in ("movie", "tv"):
        return []
    now = time.time()
    cached = _GENRE_CACHE.get(media_type)
    if cached and now - cached[0] < _GENRE_CACHE_TTL:
        return cached[1]
    data = _api_get(f"genre/{media_type}/list")
    genres = (data or {}).get("genres") or []
    _GENRE_CACHE[media_type] = (now, genres)
    return genres


def format_genre_cheat_sheet() -> str:
    lines = []
    for mt in ("movie", "tv"):
        genres = get_genre_list(mt)
        if not genres:
            continue
        pairs = ", ".join(f"{g['id']}={g['name']}" for g in genres[:24] if g.get("id"))
        lines.append(f"{mt}: {pairs}")
    return "\n".join(lines)


def _normalize_discover_item(item: Dict[str, Any], media_type: str) -> Dict[str, Any]:
    title = item.get("title") or item.get("name") or ""
    return {
        "id": int(item["id"]),
        "title": title,
        "media_type": media_type,
        "poster_path": item.get("poster_path"),
        "overview": item.get("overview"),
        "release_date": item.get("release_date") or item.get("first_air_date"),
    }


def discover(
    media_type: str,
    params: Dict[str, Any],
    *,
    max_pages: int = 2,
) -> List[Dict[str, Any]]:
    """Run GET /discover/{movie|tv} with validated params."""
    if media_type not in ("movie", "tv"):
        return []
    safe: Dict[str, Any] = {}
    for k, v in (params or {}).items():
        if k in _DISCOVER_ALLOWED and v is not None and str(v).strip() != "":
            safe[k] = str(v).strip()
    if "sort_by" not in safe:
        safe["sort_by"] = "popularity.desc"

    out: List[Dict[str, Any]] = []
    pages = max(1, min(int(max_pages), 5))
    for page in range(1, pages + 1):
        q = {**safe, "page": page}
        data = _api_get(f"discover/{media_type}", q)
        if not data:
            break
        for item in data.get("results") or []:
            if item.get("id"):
                out.append(_normalize_discover_item(item, media_type))
        if page >= (data.get("total_pages") or 1):
            break
        time.sleep(0.05)
    return out


def fetch_similar_and_recommendations(
    tmdb_id: int,
    media_type: str,
    *,
    per_endpoint: int = 10,
) -> List[Dict[str, Any]]:
    """Merge /similar and /recommendations for one title."""
    if media_type not in ("movie", "tv"):
        return []
    out: List[Dict[str, Any]] = []
    for suffix in ("similar", "recommendations"):
        data = _api_get(f"{media_type}/{tmdb_id}/{suffix}", {"page": 1})
        if not data:
            continue
        for item in (data.get("results") or [])[:per_endpoint]:
            if item.get("id"):
                out.append(_normalize_discover_item(item, media_type))
    return out


def merge_candidates(
    *sources: List[Dict[str, Any]],
    watched_ids: Set[Tuple[int, str]],
    cap: int = 80,
) -> List[Dict[str, Any]]:
    seen: Set[Tuple[int, str]] = set()
    merged: List[Dict[str, Any]] = []
    for src in sources:
        for c in src:
            try:
                tid = int(c["id"])
            except (KeyError, TypeError, ValueError):
                continue
            mt = str(c.get("media_type") or "").lower()
            if not mt:
                continue
            key = (tid, mt)
            if key in seen or key in watched_ids:
                continue
            seen.add(key)
            merged.append(c)
            if len(merged) >= cap:
                return merged
    return merged
