from pymongo import MongoClient

from app.config.settings import settings

client = MongoClient(settings.MONGODB_URI)
db = client.get_database(settings.MONGODB_DB_NAME)
watch_history_collection = db["watch_history"]
tmdb_metadata_collection = db["tmdb_metadata"]
sync_meta_collection = db["sync_meta"]
tmdb_failures_collection = db["tmdb_failures"]
