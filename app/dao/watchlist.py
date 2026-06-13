from pymongo import ReplaceOne

from app.db import watchlist_collection
from app.utils.logger import get_logger

logger = get_logger(__name__)


def get_watchlist(media_type: str | None = None) -> list[dict]:
    query = {"media_type": media_type} if media_type else {}
    return list(watchlist_collection.find(query, {"_id": 0}).sort("synced_at", -1))


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
        if ids:
            result = watchlist_collection.delete_many(
                {"media_type": media_type, "tmdb_id": {"$in": list(ids)}}
            )
            removed += result.deleted_count
    return removed
