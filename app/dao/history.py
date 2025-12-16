from app.db import watch_history_collection, tmdb_metadata_collection
from app.utils.logger import get_logger

logger = get_logger(__name__)


def store_watch_history(data):
    if isinstance(data, list):
        watch_history_collection.delete_many({})
        watch_history_collection.insert_many(data)
        logger.info("Watch history stored successfully.")
    else:
        logger.error("Invalid data format for storing watch history. Expected a list.")
        raise ValueError("Data must be a list of watch history items.")


def get_watch_history(media_type=None):
    query = {"media_type": media_type} if media_type else {}
    history = list(watch_history_collection.find(query, {"_id": 0}))
    # enrich with poster_path from TMDB metadata
    for item in history:
        tmdb_id = item.get("id") or item.get("ids", {}).get("tmdb")
        if tmdb_id:
            tmdb_doc = tmdb_metadata_collection.find_one(
                {"id": tmdb_id, "media_type": item.get("media_type")},
                {"_id": 0, "poster_path": 1}
            )
            if tmdb_doc and tmdb_doc.get("poster_path"):
                item["poster_path"] = tmdb_doc["poster_path"]

    logger.info("Watch history retrieved successfully.")
    return history
