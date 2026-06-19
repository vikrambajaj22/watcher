import json
from typing import Any, Dict, List, Optional, Set, Tuple

from cachetools import TTLCache

from app.dao.history import get_watch_history
from app.schemas.api import DescribeResponse, DiscoverFilters, DiscoverItem
from app.tmdb_discover import _RRF_K, _api_get, discover, get_genre_list
from app.utils.logger import get_logger
from app.utils.openai_client import get_openai_client
from app.utils.prompt_registry import PromptRegistry

logger = get_logger(__name__)

_DESCRIBE_CACHE: TTLCache = TTLCache(maxsize=128, ttl=3600)
_MODEL = "gpt-4.1-nano"
_registry = PromptRegistry("app/prompts/discover")


def clear_describe_cache() -> None:
    try:
        _DESCRIBE_CACHE.clear()
    except Exception:
        pass


def _extract_filters(query: str) -> DiscoverFilters:
    movie_genres = [g["name"] for g in get_genre_list("movie") if g.get("name")]
    tv_genres = [g["name"] for g in get_genre_list("tv") if g.get("name")]
    valid_genres = ", ".join(sorted(set(movie_genres + tv_genres)))
    template = _registry.load_prompt_template("extract_filters", 1)
    prompt = template.render(query=query, valid_genres=valid_genres)
    client = get_openai_client()
    resp = client.chat.completions.create(
        model=_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=300,
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(resp.choices[0].message.content or "{}")
        return DiscoverFilters.model_validate(data)
    except Exception:
        return DiscoverFilters()


def _genre_ids_from_names(names: List[str], media_type: str) -> List[str]:
    if not names:
        return []
    target_types = ["movie", "tv"] if media_type == "both" else [media_type]
    name_lower = {n.lower() for n in names}
    ids: List[str] = []
    for mt in target_types:
        for g in get_genre_list(mt):
            if g.get("name", "").lower() in name_lower and g.get("id"):
                gid = str(g["id"])
                if gid not in ids:
                    ids.append(gid)
    return ids


def _resolve_person_id(name: str) -> Optional[str]:
    data = _api_get("search/person", {"query": name})
    if not data:
        return None
    results = data.get("results") or []
    if results and results[0].get("id"):
        return str(results[0]["id"])
    return None


def _resolve_keyword_id(term: str) -> Optional[str]:
    data = _api_get("search/keyword", {"query": term})
    if not data:
        return None
    results = data.get("results") or []
    if results and results[0].get("id"):
        return str(results[0]["id"])
    return None


def _year_params(mt: str, year_from: Optional[int], year_to: Optional[int]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if year_from:
        params["primary_release_date.gte" if mt == "movie" else "first_air_date.gte"] = f"{year_from}-01-01"
    if year_to:
        params["primary_release_date.lte" if mt == "movie" else "first_air_date.lte"] = f"{year_to}-12-31"
    return params


def _discover_for_type(
    mt: str, base_params: Dict[str, Any], keyword_ids: List[str], want: int
) -> List[Dict[str, Any]]:
    """Discover candidates for one media type.

    Each keyword is queried separately (genres/people AND one keyword) and merged
    via RRF so titles matching more themes rank higher. AND-ing every keyword into
    a single query over-constrains and can collapse to a handful of results, so a
    keyword-free pass broadens the tail when the themed set is thin.
    """
    if not keyword_ids:
        return discover(mt, base_params, max_pages=2)

    scores: Dict[int, float] = {}
    items: Dict[int, Dict[str, Any]] = {}
    for kid in keyword_ids:
        for rank, item in enumerate(discover(mt, {**base_params, "with_keywords": kid}, max_pages=1), start=1):
            iid = int(item["id"])
            scores[iid] = scores.get(iid, 0.0) + 1.0 / (_RRF_K + rank)
            items.setdefault(iid, item)
    ranked = [items[iid] for iid in sorted(scores, key=scores.__getitem__, reverse=True)]

    if len(ranked) < want:
        for item in discover(mt, base_params, max_pages=2):
            iid = int(item["id"])
            if iid not in items:
                items[iid] = item
                ranked.append(item)
    return ranked


def _get_watched_ids() -> Set[Tuple[int, str]]:
    history = get_watch_history(media_type=None, include_posters=False)
    watched: Set[Tuple[int, str]] = set()
    for h in history:
        try:
            tid = int(h.get("id") or h.get("tmdb_id") or 0)
            mt = str(h.get("media_type") or "").lower()
            if tid and mt in ("movie", "tv"):
                watched.add((tid, mt))
        except (TypeError, ValueError):
            pass
    return watched


def discover_by_description(query: str, limit: int = 20, *, media_type: Optional[str] = None) -> DescribeResponse:
    """LLM-extract filters from a natural language query, then run TMDB Discover."""
    cache_key = (query.strip().lower(), limit, media_type)
    cached = _DESCRIBE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    filters = _extract_filters(query)
    if media_type and media_type in ("movie", "tv", "both"):
        filters.media_type = media_type
    media_type = filters.media_type

    genre_ids = _genre_ids_from_names(filters.genres, media_type)
    person_ids = [pid for n in filters.cast[:3] if (pid := _resolve_person_id(n))]
    keyword_ids = [kid for t in filters.keywords[:6] if (kid := _resolve_keyword_id(t))]

    base_params: Dict[str, Any] = {"sort_by": "vote_count.desc", "vote_count.gte": "50"}
    if genre_ids:
        base_params["with_genres"] = "|".join(genre_ids[:3])
    if person_ids:
        base_params["with_people"] = ",".join(person_ids)

    watched_ids = _get_watched_ids()
    target_types = ["movie", "tv"] if media_type == "both" else [media_type]

    all_results: List[Dict[str, Any]] = []
    for mt in target_types:
        mt_params = {**base_params, **_year_params(mt, filters.year_from, filters.year_to)}
        all_results.extend(_discover_for_type(mt, mt_params, keyword_ids, limit))

    seen: Set[Tuple[int, str]] = set()
    unwatched: List[Dict[str, Any]] = []
    watched: List[Dict[str, Any]] = []
    for r in all_results:
        try:
            tid = int(r["id"])
        except (TypeError, ValueError, KeyError):
            continue
        mt = str(r.get("media_type") or "").lower()
        key = (tid, mt)
        if key in seen:
            continue
        seen.add(key)
        if key in watched_ids:
            watched.append({**r, "watched": True})
        else:
            unwatched.append(r)

    combined = unwatched + watched
    result = DescribeResponse(
        results=[DiscoverItem(**r) for r in combined[:limit]],
        filters=filters,
    )
    _DESCRIBE_CACHE[cache_key] = result
    return result
