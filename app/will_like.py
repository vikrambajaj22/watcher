import json
from typing import Optional

from cachetools import TTLCache

from app.tmdb_client import get_metadata, search_by_title
from app.dao.history import get_watch_history
from app.schemas.api import ItemSummary, WillLikeResponse
from app.taste_profile import get_taste_text
from app.utils.openai_client import get_openai_client
from app.utils.logger import get_logger
from app.utils.prompt_registry import PromptRegistry

logger = get_logger(__name__)

WILL_LIKE_MODEL = "gpt-4.1-nano"
_registry = PromptRegistry("app/prompts/will_like")

# Cache LLM predictions keyed by (resolved_id, media_type) — 1-hour TTL
_WILL_LIKE_CACHE: TTLCache = TTLCache(maxsize=256, ttl=3600)


class WillLikeError(Exception):
    pass


def clear_will_like_cache() -> None:
    try:
        _WILL_LIKE_CACHE.clear()
    except Exception:
        pass


def _format_history(history: list) -> str:
    lines = []
    for item in history[:50]:
        title = item.get("title") or item.get("name") or f"id={item.get('id')}"
        mt = item.get("media_type", "")
        lines.append(f"- {title} ({mt})")
    return "\n".join(lines) if lines else "(no history)"


def compute_will_like(
    tmdb_id: Optional[int], title: Optional[str], media_type: str
) -> WillLikeResponse:
    if media_type not in {"movie", "tv"}:
        raise WillLikeError("media_type must be 'movie' or 'tv'")

    md = None
    if tmdb_id is not None:
        try:
            md = get_metadata(tmdb_id, media_type=media_type)
        except Exception:
            md = None
    if md is None and title:
        try:
            md = search_by_title(title, media_type=media_type)
        except Exception:
            md = None

    if not md or not md.get("id"):
        raise WillLikeError("item not found")

    resolved_id = int(md["id"])
    resolved_title = md.get("title") or md.get("name") or title
    resolved_overview = md.get("overview", "")
    resolved_poster = md.get("poster_path")
    genres = [g.get("name") for g in (md.get("genres") or []) if g.get("name")]

    history = get_watch_history(media_type=None, include_posters=False)
    if history and any(
        int(h.get("id", 0)) == resolved_id and h.get("media_type") == media_type
        for h in history
    ):
        return WillLikeResponse(
            will_like=False,
            score=1.0,
            explanation="You have already watched this item.",
            already_watched=True,
            item=ItemSummary(id=resolved_id, title=resolved_title, media_type=media_type, overview=resolved_overview, poster_path=resolved_poster),
        )

    if not history:
        raise WillLikeError("no watch history — sync your history first")

    cache_key = (resolved_id, media_type)
    cached = _WILL_LIKE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    try:
        taste_text = get_taste_text()
    except Exception:
        taste_text = _format_history(history)

    item_label = "movie" if media_type == "movie" else "TV show"
    genre_str = ", ".join(genres) if genres else "unknown"

    template = _registry.load_prompt_template("predict", 1)
    prompt = template.render(
        item_label=item_label,
        taste_text=taste_text,
        resolved_title=resolved_title,
        genre_str=genre_str,
        resolved_overview=resolved_overview,
    )

    client = get_openai_client()
    resp = client.chat.completions.create(
        model=WILL_LIKE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=150,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or "{}"
    try:
        result = json.loads(content)
    except Exception:
        raise WillLikeError("LLM returned invalid JSON")

    confidence = float(result.get("score", 0.5))
    will_like = confidence >= 0.5
    reasoning = str(result.get("reasoning", ""))

    result_out = WillLikeResponse(
        will_like=will_like,
        score=confidence,
        explanation=reasoning,
        already_watched=False,
        item=ItemSummary(id=resolved_id, title=resolved_title, media_type=media_type, overview=resolved_overview, poster_path=resolved_poster),
    )
    _WILL_LIKE_CACHE[cache_key] = result_out
    return result_out
