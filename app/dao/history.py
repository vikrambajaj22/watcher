from app.db import watch_history_collection
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

def get_watch_history(type=None):
    query = {"type": type} if type else {}
    history = list(watch_history_collection.find(query, {"_id": 0}))
    logger.info("Watch history retrieved successfully.")
    return history
