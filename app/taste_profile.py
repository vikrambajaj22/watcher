import json
import math
from collections import Counter
from typing import Any, Dict, List

from cachetools import TTLCache

from app.dao.history import get_watch_history
from app.schemas.api import TasteProfile
from app.utils.openai_client import get_openai_client
from app.utils.logger import get_logger

logger = get_logger(__name__)

TASTE_PROFILE_MODEL = "gpt-4.1-nano"
_CACHE: TTLCache = TTLCache(maxsize=4, ttl=3600)

_SKIP_KEYWORDS = {
    "based on novel or book", "based on comic", "based on true story",
    "independent film", "sequel", "remake", "duringcreditsstinger",
    "aftercreditsstinger", "female protagonist", "male protagonist",
    "superhero", "anti-hero", "ensemble cast", "voice over narration",
}


def _rank_score(item: Dict[str, Any]) -> float:
    eng = float(item.get("rewatch_engagement") or 0)
    wc = float(item.get("watch_count") or 0)
    return eng * 2.0 + math.log1p(wc)


def _top_keywords(history: list, sample: int = 30, top_n: int = 8) -> List[str]:
    ranked = sorted(history, key=_rank_score, reverse=True)[:sample]
    counts: Counter = Counter()
    for item in ranked:
        for kw in item.get("tmdb_keywords") or []:
            if kw not in _SKIP_KEYWORDS:
                counts[kw] += 1
    return [kw for kw, _ in counts.most_common(top_n)]


def _format_history(history: list) -> str:
    lines = []
    for item in history[:80]:
        title = item.get("title") or item.get("name") or f"id={item.get('id')}"
        mt = item.get("media_type", "")
        lines.append(f"- {title} ({mt})")
    return "\n".join(lines) if lines else "(no history)"


def compute_taste_profile() -> TasteProfile:
    cached = _CACHE.get("profile")
    if cached is not None:
        return cached

    history = get_watch_history(media_type=None, include_posters=False)
    if not history:
        raise ValueError("no watch history — sync your history first")

    history_text = _format_history(history)
    themes = _top_keywords(history)

    prompt = (
        "You are a film and TV critic analyzing someone's viewing taste. "
        "The history includes both movies (tagged 'movie') and TV shows (tagged 'tv'). "
        "Based on the watch history below, generate a concise taste profile that reflects both.\n\n"
        f"Watch history ({len(history)} titles):\n{history_text}\n\n"
        "Respond ONLY with valid JSON matching this exact shape:\n"
        '{"signature": "a punchy 5-10 word phrase capturing their overall taste across movies and TV", '
        '"summary": "2-3 sentence paragraph describing their taste in movies and TV, addressed to the user using \'you\'", '
        '"genres": ["up to 5 genres they clearly favor across movies and TV"], '
        '"avoid": ["up to 3 things they seem to avoid or rarely watch"]}\n'
        "Be specific and grounded in the actual titles — don't be generic."
    )

    client = get_openai_client()
    resp = client.chat.completions.create(
        model=TASTE_PROFILE_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=300,
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or "{}"
    try:
        result = json.loads(content)
    except Exception:
        raise ValueError("LLM returned invalid JSON")

    out = TasteProfile(
        signature=str(result.get("signature", "")),
        summary=str(result.get("summary", "")),
        genres=[str(g) for g in result.get("genres", [])],
        themes=themes,
        avoid=[str(a) for a in result.get("avoid", [])],
        history_count=len(history),
    )
    _CACHE["profile"] = out
    return out


def clear_taste_cache() -> None:
    try:
        _CACHE.clear()
    except Exception:
        pass


def get_taste_text() -> str:
    profile = compute_taste_profile()
    genres = ", ".join(profile.genres) if profile.genres else "varied"
    themes = ", ".join(profile.themes) if profile.themes else "varied"
    return (
        f"Taste signature: {profile.signature}\n"
        f"Summary: {profile.summary}\n"
        f"Genres: {genres}\n"
        f"Recurring themes: {themes}"
    )
