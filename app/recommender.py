from app.dao.history import get_watch_history


def recommend():
    history = get_watch_history()
    return ["Movie A", "Show B"]
