from pymongo import MongoClient

from app.config.settings import settings

client = MongoClient(settings.MONGODB_URI)
db = client.get_database(settings.MONGODB_DB_NAME)
watch_history_collection = db["watch_history"]
