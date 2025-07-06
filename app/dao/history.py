from app.db import watch_history_collection


def store_watch_history(data):
    if isinstance(data, list):
        watch_history_collection.delete_many({})
        watch_history_collection.insert_many(data)

def get_watch_history():
    return list(watch_history_collection.find({}, {"_id": 0}))
