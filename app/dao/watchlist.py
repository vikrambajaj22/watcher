from pymongo import ReplaceOne

from app.db import watchlist_collection
from app.utils.logger import get_logger

logger = get_logger(__name__)


def get_watchlist(media_type: str | None = None) -> list[dict]:
    query = {"media_type": media_type} if media_type else {}
    items = list(watchlist_collection.find(query, {"_id": 0}))
    # Sort by Trakt added date; fall back to sync time for records synced before listed_at existed.
    items.sort(key=lambda i: i.get("listed_at") or i.get("synced_at") or "", reverse=True)
    return items


def upsert_watchlist_item(item: dict) -> None:
    watchlist_collection.replace_one(
        {"tmdb_id": item["tmdb_id"], "media_type": item["media_type"]},
        item,
        upsert=True,
    )


def remove_watchlist_item(tmdb_id: int, media_type: str) -> bool:
    result = watchlist_collection.delete_one({"tmdb_id": tmdb_id, "media_type": media_type})
    return result.deleted_count > 0


def bulk_upsert_watchlist(items: list[dict]) -> None:
    if not items:
        return
    ops = [
        ReplaceOne(
            {"tmdb_id": item["tmdb_id"], "media_type": item["media_type"]},
            item,
            upsert=True,
        )
        for item in items
    ]
    watchlist_collection.bulk_write(ops, ordered=False)


def remove_watchlist_items_by_ids(tmdb_ids: set[int], media_type: str) -> int:
    if not tmdb_ids:
        return 0
    result = watchlist_collection.delete_many(
        {"media_type": media_type, "tmdb_id": {"$in": list(tmdb_ids)}}
    )
    return result.deleted_count


def clear_watchlist_items_in_history(watched_by_type: dict[str, set[int]]) -> int:
    """Remove watchlist items that now appear in watch history."""
    removed = 0
    for media_type, ids in watched_by_type.items():
        for raw_tmdb_id in ids or set():
            try:
                tmdb_id = int(raw_tmdb_id)
            except (TypeError, ValueError):
                logger.warning("Skipping invalid TMDB id %r for local removal", raw_tmdb_id)
                continue
            result = watchlist_collection.delete_one(
                {"media_type": media_type, "tmdb_id": tmdb_id}
            )
            if result.deleted_count:
                removed += 1
    return removed


def remove_watchlist_items_from_trakt(watched_by_type: dict[str, set[int]]) -> int:
    """Remove watched items from the corresponding Trakt watchlists in a batch."""
    if not watched_by_type:
        return 0

    from app.watchlist_sync import remove_many_from_watchlist

    removed = 0
    for media_type, ids in watched_by_type.items():
        tmdb_ids: list[int] = []
        for raw_tmdb_id in sorted(ids or set()):
            try:
                tmdb_ids.append(int(raw_tmdb_id))
            except (TypeError, ValueError):
                logger.warning("Skipping invalid TMDB id %r for Trakt removal", raw_tmdb_id)
        if tmdb_ids:
            try:
                removed += remove_many_from_watchlist(tmdb_ids, media_type)
            except Exception as exc:
                logger.warning(
                    "Could not batch-remove watched %s items from Trakt watchlist: %s",
                    media_type,
                    repr(exc),
                )
    return removed
