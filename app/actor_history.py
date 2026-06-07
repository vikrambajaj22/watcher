from typing import Any, Dict, Set, Tuple

from app.tmdb_client import get_person_credits, search_person
from app.dao.history import get_watch_history
from app.schemas.api import ActorHistoryItem, ActorHistoryResponse, PersonSummary


def get_actor_history(name: str) -> ActorHistoryResponse:
    """Return watched titles featuring a named actor or director."""
    person_data = search_person(name)
    if not person_data:
        return ActorHistoryResponse()

    credits = get_person_credits(person_data["id"])
    history = get_watch_history(media_type=None, include_posters=True)

    watched_map: Dict[Tuple[int, str], Dict[str, Any]] = {}
    for h in history:
        try:
            tid = int(h.get("id") or h.get("tmdb_id") or 0)
            mt = str(h.get("media_type") or "").lower()
            if tid and mt in ("movie", "tv"):
                watched_map[(tid, mt)] = h
        except (TypeError, ValueError):
            pass

    seen: Set[int] = set()
    items: list[ActorHistoryItem] = []
    for role in (credits.get("cast") or []) + (credits.get("crew") or []):
        try:
            rid = int(role["id"])
        except (TypeError, ValueError, KeyError):
            continue
        if rid in seen:
            continue
        mt = str(role.get("media_type") or "").lower()
        if mt not in ("movie", "tv"):
            continue
        h = watched_map.get((rid, mt))
        if not h:
            continue
        seen.add(rid)
        items.append(ActorHistoryItem(
            id=rid,
            title=role.get("title") or role.get("name"),
            media_type=mt,
            poster_path=h.get("poster_path"),
            character=role.get("character") or role.get("job"),
            department=role.get("department") or "Acting",
            watched_at=h.get("latest_watched_at"),
        ))

    return ActorHistoryResponse(person=PersonSummary.model_validate(person_data), items=items)
