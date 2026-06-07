"""TMDB discover / similar / recommendations helpers for LLM-driven recommend."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from cachetools import TTLCache

from app.config.settings import settings
from app.tmdb_client import get_metadata
from app.utils.logger import get_logger
from app.utils.openai_client import get_openai_client

logger = get_logger(__name__)

_GENRE_CACHE: Dict[str, Tuple[float, List[Dict[str, Any]]]] = {}
_GENRE_CACHE_TTL = 86400.0

# Cache similar/recommendations results keyed by (tmdb_id, media_type) — 6-hour TTL
_SIMILAR_CACHE: TTLCache = TTLCache(maxsize=512, ttl=21600)

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
        "with_people",
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


_RRF_K = 60


def fetch_similar_and_recommendations(
    tmdb_id: int,
    media_type: str,
    *,
    per_endpoint: int = 10,
) -> List[Dict[str, Any]]:
    """Merge /similar and /recommendations via Reciprocal Rank Fusion."""
    if media_type not in ("movie", "tv"):
        return []
    cache_key = (tmdb_id, media_type, per_endpoint, "rrf")
    cached = _SIMILAR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    items: Dict[int, Dict[str, Any]] = {}
    scores: Dict[int, float] = {}

    for suffix in ("similar", "recommendations"):
        data = _api_get(f"{media_type}/{tmdb_id}/{suffix}", {"page": 1})
        if not data:
            continue
        for rank, item in enumerate((data.get("results") or [])[:per_endpoint], start=1):
            if not item.get("id"):
                continue
            iid = int(item["id"])
            scores[iid] = scores.get(iid, 0.0) + 1.0 / (_RRF_K + rank)
            if iid not in items:
                items[iid] = _normalize_discover_item(item, media_type)

    out = [items[iid] for iid in sorted(scores, key=scores.__getitem__, reverse=True)]
    _SIMILAR_CACHE[cache_key] = out
    return out


def _llm_thematic_keywords(title: str, overview: str, genres: List[str], media_type: str) -> List[str]:
    """Ask the LLM for thematic descriptors to drive cross-type TMDB keyword search."""
    prompt = (
        f"Given this {media_type}:\n"
        f"Title: {title}\n"
        f"Genres: {', '.join(genres)}\n"
        f"Overview: {overview}\n\n"
        "List 5 short thematic keywords or phrases that best capture its essence "
        "(themes, setting, mood, narrative elements) for searching a film/TV database.\n"
        'Respond ONLY with JSON: {"keywords": ["word1", "word2", "word3", "word4", "word5"]}'
    )
    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=80,
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content or "{}")
        return [str(k).strip() for k in (result.get("keywords") or []) if k][:5]
    except Exception as e:
        logger.warning("LLM keyword generation failed: %s", repr(e))
        return []


def _resolve_tmdb_keyword_id(term: str) -> Optional[str]:
    """Resolve a free-text term to a TMDB keyword ID via /search/keyword."""
    data = _api_get("search/keyword", {"query": term})
    if not data:
        return None
    results = data.get("results") or []
    if results and results[0].get("id"):
        return str(results[0]["id"])
    return None


def fetch_cross_type_similar(
    tmdb_id: int,
    source_media_type: str,
    *,
    per_endpoint: int = 20,
) -> List[Dict[str, Any]]:
    """Discover titles of the opposite type using LLM-generated thematic keywords.

    LLM produces descriptive terms from the source title's overview; each term is
    resolved to a TMDB keyword ID via /search/keyword, then /discover runs per keyword.
    Results are merged with RRF so titles matching multiple themes rank higher.
    Falls back to genre-based discover if no keyword IDs resolve.
    """
    if source_media_type not in ("movie", "tv"):
        return []
    result_type = "tv" if source_media_type == "movie" else "movie"
    cache_key = (tmdb_id, source_media_type, per_endpoint, "cross_type_v2")
    cached = _SIMILAR_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        md = get_metadata(tmdb_id, media_type=source_media_type)
    except Exception:
        return []
    if not md:
        return []

    title = md.get("title") or md.get("name") or ""
    overview = md.get("overview") or ""
    genre_names = [g["name"] for g in (md.get("genres") or [])[:3] if g.get("name")]
    genre_ids = [str(g["id"]) for g in (md.get("genres") or [])[:3] if g.get("id")]

    base_params: Dict[str, Any] = {"sort_by": "popularity.desc", "vote_count.gte": "100"}

    terms = _llm_thematic_keywords(title, overview, genre_names, source_media_type)
    keyword_ids = [kid for t in terms if (kid := _resolve_tmdb_keyword_id(t))]

    if keyword_ids:
        scores: Dict[int, float] = {}
        items: Dict[int, Dict[str, Any]] = {}
        kw_params = {**base_params}
        if genre_ids:
            kw_params["with_genres"] = "|".join(genre_ids)
        for kid in keyword_ids:
            for rank, item in enumerate(
                discover(result_type, {**kw_params, "with_keywords": kid}, max_pages=1)[:10],
                start=1,
            ):
                iid = int(item["id"])
                scores[iid] = scores.get(iid, 0.0) + 1.0 / (_RRF_K + rank)
                items.setdefault(iid, item)
        out = [items[iid] for iid in sorted(scores, key=scores.__getitem__, reverse=True)]
    elif genre_ids:
        out = discover(result_type, {**base_params, "with_genres": ",".join(genre_ids)}, max_pages=2)
        if not out:
            out = discover(result_type, {**base_params, "with_genres": genre_ids[0]}, max_pages=2)
    else:
        out = []

    out = out[:per_endpoint]
    _SIMILAR_CACHE[cache_key] = out
    return out


def fetch_taste_keyword_candidates(
    taste_summary: str,
    media_type: str,
    *,
    per_keyword: int = 10,
) -> List[Dict[str, Any]]:
    """Discover candidates by resolving taste-profile themes to TMDB keyword IDs."""
    prompt = (
        f"Given this viewer taste profile:\n{taste_summary}\n\n"
        "List 6 short thematic keywords or phrases that best capture recurring themes, "
        "moods, settings, or narrative elements (e.g. 'heist', 'time travel', 'family drama'). "
        'Respond ONLY with JSON: {"keywords": ["word1", "word2", ...]}'
    )
    try:
        client = get_openai_client()
        resp = client.chat.completions.create(
            model="gpt-4.1-nano",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=100,
            response_format={"type": "json_object"},
        )
        result = json.loads(resp.choices[0].message.content or "{}")
        terms = [str(k).strip() for k in (result.get("keywords") or []) if k][:6]
    except Exception as e:
        logger.warning("Taste keyword generation failed: %s", repr(e))
        return []

    keyword_ids = [kid for t in terms if (kid := _resolve_tmdb_keyword_id(t))]
    if not keyword_ids:
        return []

    target_types = ["movie", "tv"] if media_type == "all" else [media_type]
    scores: Dict[int, float] = {}
    items: Dict[int, Dict[str, Any]] = {}

    for mt in target_types:
        for kid in keyword_ids:
            results = discover(mt, {"with_keywords": kid, "vote_count.gte": "50", "sort_by": "popularity.desc"}, max_pages=1)
            for rank, item in enumerate(results[:per_keyword], start=1):
                try:
                    iid = int(item["id"])
                except (KeyError, TypeError, ValueError):
                    continue
                scores[iid] = scores.get(iid, 0.0) + 1.0 / (_RRF_K + rank)
                items.setdefault(iid, item)

    return [items[iid] for iid in sorted(scores, key=scores.__getitem__, reverse=True)]


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
