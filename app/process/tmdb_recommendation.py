"""LLM taste planner + TMDB discover/recommendations — no FAISS or tmdb_metadata."""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Set, Tuple

from app.dao.history import get_watch_history
from app.schemas.recommendations.recommendations import (
    Recommendation,
    RecommendationsResponse,
)
from app.tmdb_discover import (
    discover,
    fetch_similar_and_recommendations,
    format_genre_cheat_sheet,
    merge_candidates,
)
from app.utils.logger import get_logger
from app.utils.openai_client import get_openai_chat_completion
from app.utils.prompt_registry import PromptRegistry

logger = get_logger(__name__)

PLANNER_MODEL = "gpt-4.1-nano"
PICKER_MODEL = "gpt-4.1-nano"


def _history_key(item: Dict[str, Any]) -> Optional[Tuple[int, str]]:
    try:
        tid = int(item.get("id") or item.get("tmdb_id"))
        mt = str(item.get("media_type") or "").lower()
        if mt in ("movie", "tv"):
            return (tid, mt)
    except (TypeError, ValueError):
        pass
    return None


def _rank_history(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def score(item: Dict[str, Any]) -> float:
        eng = float(item.get("rewatch_engagement") or 0)
        wc = float(item.get("watch_count") or 0)
        latest = str(item.get("latest_watched_at") or "")
        recency = 1.0 if latest else 0.0
        return eng * 2.0 + wc + recency

    return sorted(items, key=score, reverse=True)


def _format_history_line(item: Dict[str, Any]) -> str:
    title = item.get("title") or item.get("name") or f"id={item.get('id')}"
    year = item.get("year") or ""
    mt = item.get("media_type") or "?"
    wc = item.get("watch_count")
    eng = item.get("rewatch_engagement")
    parts = [f'"{title}"']
    if year:
        parts.append(f"({year})")
    parts.append(f"[{mt}]")
    if wc is not None:
        parts.append(f"watches={wc}")
    if eng is not None:
        parts.append(f"engagement={eng}")
    return " ".join(parts)


def _default_discover_queries(media_type: str) -> List[Dict[str, Any]]:
    """Fallback when planner JSON fails."""
    queries = []
    if media_type in ("movie", "all"):
        queries.append(
            {
                "media_type": "movie",
                "params": {"vote_count.gte": "200", "sort_by": "popularity.desc"},
                "pages": 2,
            }
        )
    if media_type in ("tv", "all"):
        queries.append(
            {
                "media_type": "tv",
                "params": {"vote_count.gte": "50", "sort_by": "popularity.desc"},
                "pages": 2,
            }
        )
    return queries


def _parse_planner_json(text: str) -> Dict[str, Any]:
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("planner response not an object")
    return data


def _run_planner(
    history_lines: List[str],
    media_type: str,
) -> Dict[str, Any]:
    registry = PromptRegistry("app/prompts/recommend")
    template = registry.load_prompt_template("taste_planner", 1)
    prompt = template.render(
        media_scope=media_type,
        genre_cheat_sheet=format_genre_cheat_sheet(),
        history_lines=history_lines[:40],
    )
    response = get_openai_chat_completion(
        PLANNER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return _parse_planner_json(response.choices[0].message.content)


def _fetch_candidates(
    plan: Dict[str, Any],
    seeds: List[Tuple[int, str]],
    watched_ids: Set[Tuple[int, str]],
    media_type: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    sources: List[List[Dict[str, Any]]] = []
    meta: Dict[str, Any] = {"discover_calls": 0, "seed_calls": 0}

    for q in plan.get("discover_queries") or []:
        if not isinstance(q, dict):
            continue
        mt = str(q.get("media_type") or "").lower()
        if mt not in ("movie", "tv"):
            continue
        if media_type != "all" and mt != media_type:
            continue
        params = q.get("params") if isinstance(q.get("params"), dict) else {}
        pages = int(q.get("pages") or 2)
        sources.append(discover(mt, params, max_pages=pages))
        meta["discover_calls"] += 1

    for tid, mt in seeds:
        if media_type != "all" and mt != media_type:
            continue
        sources.append(fetch_similar_and_recommendations(tid, mt))
        meta["seed_calls"] += 1

    merged = merge_candidates(*sources, watched_ids=watched_ids, cap=80)
    meta["candidate_count"] = len(merged)
    return merged, meta


def _format_candidates_for_picker(candidates: List[Dict[str, Any]]) -> List[str]:
    lines = []
    for i, c in enumerate(candidates, start=1):
        title = c.get("title") or "Unknown"
        mt = c.get("media_type") or ""
        year = ""
        rd = c.get("release_date")
        if rd and len(str(rd)) >= 4:
            year = str(rd)[:4]
        line = f'{i}. "{title}"'
        if year:
            line += f" ({year})"
        line += f" [{mt}]"
        lines.append(line)
    return lines


def _pick_recommendations(
    watch_history: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    recommend_count: int,
    media_type: str,
) -> List[Dict[str, Any]]:
    registry = PromptRegistry("app/prompts/recommend")
    template = registry.load_prompt_template("tmdb_picker", 1)

    history_lines = [_format_history_line(h) for h in _rank_history(watch_history)[:35]]

    prompt = template.render(
        recommend_count=recommend_count,
        watch_history_formatted=history_lines,
        candidates_formatted=_format_candidates_for_picker(candidates),
    )
    response = get_openai_chat_completion(
        PICKER_MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content).get("recommendations") or []


def _map_picks_to_response(
    picks: List[Dict[str, Any]],
    candidates: List[Dict[str, Any]],
    recommend_count: int,
) -> List[Recommendation]:
    index_map = {i + 1: c for i, c in enumerate(candidates)}
    valid: List[Recommendation] = []
    for p in picks:
        try:
            idx = int(p.get("index"))
        except (TypeError, ValueError):
            continue
        cand = index_map.get(idx)
        if not cand:
            continue
        valid.append(
            Recommendation(
                id=str(cand["id"]),
                title=cand.get("title") or "",
                reasoning=p.get("reasoning") if isinstance(p.get("reasoning"), str) else "",
                media_type=cand.get("media_type"),
                metadata={
                    "poster_path": cand.get("poster_path"),
                    "overview": cand.get("overview"),
                    "id": cand["id"],
                },
            )
        )

    already = {r.id for r in valid}
    for c in candidates:
        cid = str(c.get("id"))
        if cid in already:
            continue
        valid.append(
            Recommendation(
                id=cid,
                title=c.get("title") or "",
                reasoning="Popular TMDB match (fallback).",
                media_type=c.get("media_type"),
                metadata={
                    "poster_path": c.get("poster_path"),
                    "overview": c.get("overview"),
                    "id": c.get("id"),
                },
            )
        )
        if len(valid) >= recommend_count:
            break
    return valid[:recommend_count]


class TmdbRecommender:
    """Experimental recommend path: watch history + LLM + TMDB API only."""

    def generate(
        self,
        media_type: str = "all",
        recommend_count: int = 5,
    ) -> Tuple[RecommendationsResponse, Dict[str, Any]]:
        t0 = time.time()
        debug: Dict[str, Any] = {"timings_ms": {}, "media_type": media_type}

        history_mt = None if media_type == "all" else media_type
        raw_history = get_watch_history(media_type=history_mt, include_posters=False)
        if not raw_history:
            return RecommendationsResponse(recommendations=[]), {
                **debug,
                "error": "empty_watch_history",
            }

        ranked = _rank_history(raw_history)
        watched_ids: Set[Tuple[int, str]] = set()
        for item in raw_history:
            k = _history_key(item)
            if k:
                watched_ids.add(k)

        history_lines = [_format_history_line(h) for h in ranked[:40]]
        seeds = []
        for item in ranked[:3]:
            k = _history_key(item)
            if k:
                seeds.append(k)
        debug["seed_ids"] = [{"id": s[0], "media_type": s[1]} for s in seeds]

        plan: Dict[str, Any]
        t_plan = time.time()
        try:
            plan = _run_planner(history_lines, media_type)
        except Exception as e:
            logger.warning("taste planner failed, using defaults: %s", repr(e))
            plan = {
                "discover_queries": _default_discover_queries(media_type),
                "taste_summary": f"fallback ({repr(e)})",
            }
        debug["timings_ms"]["planner"] = int((time.time() - t_plan) * 1000)
        debug["plan"] = plan

        if not plan.get("discover_queries"):
            plan["discover_queries"] = _default_discover_queries(media_type)

        t_fetch = time.time()
        candidates, fetch_meta = _fetch_candidates(
            plan, seeds, watched_ids, media_type
        )
        debug["timings_ms"]["tmdb_fetch"] = int((time.time() - t_fetch) * 1000)
        debug.update(fetch_meta)

        if not candidates:
            return RecommendationsResponse(recommendations=[]), {
                **debug,
                "error": "no_candidates",
            }

        t_pick = time.time()
        try:
            picks = _pick_recommendations(ranked, candidates, recommend_count, media_type)
        except Exception as e:
            logger.error("tmdb picker failed: %s", repr(e), exc_info=True)
            picks = []
        debug["timings_ms"]["picker"] = int((time.time() - t_pick) * 1000)

        recs = _map_picks_to_response(picks, candidates, recommend_count)
        debug["timings_ms"]["total"] = int((time.time() - t0) * 1000)
        return RecommendationsResponse(recommendations=recs), debug
